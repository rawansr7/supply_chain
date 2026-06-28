"""TimesFM 2.0 (Google) adapter — zero-shot and fine-tune.

Open model: google/timesfm-2.0-500m-pytorch.
Install (zero-shot):   pip install "timesfm[torch]==1.3.0"
Fine-tune ALSO needs:
    pip install wandb                       # finetuning_torch.py imports it at module top
    # copy v1/src/finetuning/finetuning_torch.py from the timesfm repo into THIS folder
    # (analysis/tsfm_inventory/models/finetuning_torch.py) — it is NOT in the pip wheel.

Verified against the timesfm v1 source. Independent tests find 2.0 >= 2.5, so we use 2.0.
The model emits the 9 deciles (0.1..0.9); arbitrary quantile levels are snapped to the
nearest decile. GPU strongly recommended for fine-tune (full-parameter, 500M params).
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster

REPO_ID = "google/timesfm-2.0-500m-pytorch"
OUTPUT_PATCH_LEN = 128            # the model's supervised output patch (NOT the business horizon)
CONTEXT_LEN = 192                 # multiple of 32, <= 2048
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFM(Forecaster):
    name = "timesfm"
    supported_regimes = ["zero_shot", "fine_tune"]
    needs_gpu = True
    num_epochs = 5              # fixed default (no tuning); timesfm has no canonical recipe. smoke uses 1
    batch_size = 64
    learning_rate = 1e-4       # the library default

    def _backend(self):
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"   # NB: 'gpu', never 'cuda'

    def _build_wrapper(self):
        import timesfm
        # num_layers=50 and use_positional_embedding=False are REQUIRED for the 500m checkpoint.
        return timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self._backend(), per_core_batch_size=32,
                horizon_len=OUTPUT_PATCH_LEN, num_layers=50,
                use_positional_embedding=False, context_len=CONTEXT_LEN),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=REPO_ID))

    def fit(self, train_panel, dataset):
        self._tfm = self._build_wrapper()
        if self.regime == "zero_shot":
            return self

        # fine_tune: official v1 full-parameter recipe, global over all series.
        import torch
        from os import path
        from torch.utils.data import ConcatDataset, Dataset
        from huggingface_hub import snapshot_download
        from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
        try:
            from finetuning_torch import FinetuningConfig, TimesFMFinetuner
        except ImportError as e:                      # copied repo file is missing
            raise ImportError(
                "timesfm fine-tune needs finetuning_torch.py copied from the timesfm repo "
                "(v1/src/finetuning/finetuning_torch.py) into analysis/tsfm_inventory/models/, "
                "plus `pip install wandb`.") from e

        class _WindowDS(Dataset):
            """Sliding (context, output-patch) windows — the official 4-tuple."""
            def __init__(self, series, ctx, out):
                s = np.asarray(series, dtype=np.float32); tot = ctx + out
                self.samples = [(s[i:i + ctx], s[i + ctx:i + ctx + out])
                                for i in range(0, len(s) - tot + 1)]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                xc, xf = self.samples[idx]
                xc = torch.tensor(xc, dtype=torch.float32)
                xf = torch.tensor(xf, dtype=torch.float32)
                return xc, torch.zeros_like(xc), torch.tensor([0], dtype=torch.long), xf

        cfg = self._tfm._model_config
        model = PatchedTimeSeriesDecoder(cfg)
        ckpt = path.join(snapshot_download(REPO_ID), "torch_model.ckpt")
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        out_patch = cfg.horizon_len                   # 128 — supervised target length
        window = CONTEXT_LEN + out_patch
        parts = []
        for _, g in train_panel.groupby("series_id", sort=False):
            s = g.sort_values("t")["y"].to_numpy(dtype=np.float32)
            if len(s) >= window + 1:
                parts.append(_WindowDS(s, CONTEXT_LEN, out_patch))
        if not parts:
            raise ValueError(
                f"timesfm fine-tune: no series long enough for context+{out_patch}+1 "
                f"({window + 1} weeks).")
        ds = ConcatDataset(parts)

        use_gpu = torch.cuda.is_available()
        config = FinetuningConfig(
            batch_size=self.batch_size, num_epochs=1 if self.smoke else self.num_epochs,
            learning_rate=self.learning_rate, weight_decay=0.01, freq_type=0,
            use_quantile_loss=True,                   # train the 9 quantile heads
            device="cuda" if use_gpu else "cpu", use_wandb=False)
        TimesFMFinetuner(model, config).finetune(train_dataset=ds, val_dataset=ds)
        self._tfm._model = model.to(self._tfm._device).eval()   # rebind fine-tuned weights
        return self

    def predict_quantiles(self, history):
        _mean, full_fc = self._tfm.forecast(
            inputs=[np.asarray(history, dtype=np.float32)], freq=[0])
        full = np.asarray(full_fc)[0]                 # (H_model, 10): col 0 mean, 1..9 deciles
        out = {}
        for q in self.quantile_levels:
            i = int(np.argmin([abs(q - d) for d in DECILES]))   # snap to nearest decile
            out[q] = np.clip(full[:self.horizon, 1 + i], 0, None)
        return out
