import numpy as np
import torch
from torch import nn

from .. import config as C
from .base import Forecaster, scaled_windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT = 52
LAYERS = 2
DROPOUT = 0.1
STEPS = 5000
BATCH = 256


class Net(nn.Module):
    def __init__(self, hidden, outputs):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden, LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        out, _ = self.encoder(x.unsqueeze(-1))
        return self.head(out[:, -1])


class LSTMGlobal(Forecaster):
    grids = {"statistical": [{"hidden": n, "lr": r}
                             for n in (32, 64, 128, 256)
                             for r in (3e-4, 1e-3, 3e-3, 1e-2)]}

    def fit(self, train_panel):
        torch.manual_seed(C.SEED)
        x, y = scaled_windows(train_panel.to_numpy(np.float32), CONTEXT, self.horizon)
        x_t = torch.tensor(x, device=DEVICE)
        y_t = torch.tensor(y, device=DEVICE)
        levels = torch.tensor(self.quantile_levels, device=DEVICE)

        n_outputs = self.horizon * len(self.quantile_levels)
        self.net = Net(self.params["hidden"], n_outputs).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])

        for _ in range(STEPS):
            batch = torch.randint(len(x_t), (BATCH,), device=DEVICE)
            pred = self.net(x_t[batch]).view(BATCH, self.horizon, -1)
            diff = y_t[batch, :, None] - pred
            pinball = torch.maximum(levels * diff, (levels - 1) * diff)
            loss = pinball.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        self.net.eval()
        return self

    def predict_quantiles(self, history):
        context = history.to_numpy(np.float32)[:, -CONTEXT:]
        scale = 1 + context.mean(axis=1, keepdims=True)
        inputs = torch.tensor(context / scale, device=DEVICE)

        with torch.no_grad():
            pred = self.net(inputs)
        pred = pred.view(len(context), self.horizon, -1)
        pred = pred.cpu().numpy()

        quantiles = np.sort(pred, axis=-1)
        quantiles = quantiles * scale[..., None]
        return np.clip(quantiles, 0, None)
