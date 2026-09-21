"""Global LSTM — the deep-learning model on the "make" side of the comparison.

One network is trained across every series of the dataset and then applied to each
series' own history, the same shape of deployment as `lightgbm_global`. The head emits
all quantiles at once and is trained on pinball loss (the MQ-RNN recipe, Wen et al.
2017) so the decision layer gets a genuine predictive distribution rather than a point
forecast dressed up with residual spread — a built model should not be handicapped
against the foundation models on exactly the axis the newsvendor cares about.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from .base import Forecaster

CONTEXT = 52          # one seasonal cycle of weekly history
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.1
STEPS = 5000          # gradient-step budget, so small and large panels both train fully
BATCH = 256
LR = 1e-3


class LSTMGlobal(Forecaster):
    name = "lstm_global"
    supported_regimes = ["statistical"]
    seed = C.SEED         # class attribute: a robustness run can vary it on its own

    def _context_len(self, panel):
        shortest = int(panel.groupby("series_id").size().min())
        return max(2 * self.horizon, min(CONTEXT, shortest - self.horizon))

    def _windows(self, panel):
        """Sliding (context, horizon) pairs, each scaled by its own context mean.

        Weekly demand spans four orders of magnitude between datasets and two within
        one, so the network sees every window in units of its own recent average and
        the scale is put back on the forecast afterwards.
        """
        x, y = [], []
        for _, g in panel.groupby("series_id", sort=True):
            series = g.sort_values("t")["y"].to_numpy(dtype=np.float32)
            for i in range(self.context, len(series) - self.horizon + 1):
                window = series[i - self.context:i]
                scale = window.mean()
                if scale <= 0:  # nothing sold in the whole context: no signal to learn
                    continue
                x.append(window / scale)
                y.append(series[i:i + self.horizon] / scale)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def _pinball(self, pred, target):
        import torch

        levels = torch.tensor(self._levels, dtype=torch.float32)
        diff = target.unsqueeze(-1) - pred                      # (batch, horizon, n_quantiles)
        return torch.maximum(levels * diff, (levels - 1) * diff).mean()

    def fit(self, train_panel=None):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.context = self._context_len(train_panel)
        x, y = self._windows(train_panel)
        if len(x) == 0:
            raise ValueError("lstm_global: no usable training window in this panel.")

        # ascending, so a sort across the output axis is enough to keep them from crossing
        self._levels = sorted(self.quantile_levels)
        n_q = len(self._levels)
        self.encoder = nn.LSTM(1, HIDDEN, LAYERS, batch_first=True,
                               dropout=DROPOUT if LAYERS > 1 else 0.0)
        self.head = nn.Linear(HIDDEN, self.horizon * n_q)
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=LR)

        x_t = torch.from_numpy(x).unsqueeze(-1)
        y_t = torch.from_numpy(y)
        rng = np.random.default_rng(self.seed)
        batches = max(1, len(x_t) // BATCH)
        epochs = 1 if self.smoke else -(-STEPS // batches)
        for _ in range(epochs):
            for batch in np.array_split(rng.permutation(len(x_t)), batches):
                idx = torch.from_numpy(batch)
                out, _ = self.encoder(x_t[idx])
                pred = self.head(out[:, -1]).view(-1, self.horizon, n_q)
                loss = self._pinball(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.encoder.eval()
        self.head.eval()
        return self

    def predict_quantiles(self, history):
        import torch

        window = np.asarray(history, dtype=np.float32)[-self.context:]
        if len(window) < self.context:  # pad short histories with their earliest value
            window = np.concatenate([np.full(self.context - len(window), window[0]), window])
        scale = window.mean()
        if scale <= 0:
            return {q: np.zeros(self.horizon) for q in self._levels}

        with torch.no_grad():
            out, _ = self.encoder(torch.from_numpy(window / scale).view(1, -1, 1))
            pred = self.head(out[:, -1]).view(self.horizon, -1).numpy() * scale
        # the quantile levels are independent outputs and may cross; sorting restores a
        # monotone predictive distribution
        pred = np.sort(np.clip(pred, 0, None), axis=1)
        return {q: pred[:, i] for i, q in enumerate(self._levels)}
