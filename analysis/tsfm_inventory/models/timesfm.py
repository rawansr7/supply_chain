from __future__ import annotations

import numpy as np

from .base import Forecaster

REPO_ID = "google/timesfm-2.0-500m-pytorch"
CONTEXT_LEN = 192
OUTPUT_PATCH_LEN = 128
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFM(Forecaster):
    name = "timesfm"
    supported_regimes = ["zero_shot"]
    needs_gpu = True

    def fit(self, train_panel=None):
        import timesfm
        import torch

        self._tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                # backend must be "gpu"/"cpu", never "cuda"; the other three are
                # required by the 500m checkpoint.
                backend="gpu" if torch.cuda.is_available() else "cpu",
                per_core_batch_size=32, horizon_len=OUTPUT_PATCH_LEN,
                num_layers=50, use_positional_embedding=False, context_len=CONTEXT_LEN),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=REPO_ID))
        return self

    def predict_quantiles(self, history):
        _mean, full = self._tfm.forecast(inputs=[np.asarray(history, dtype=np.float32)], freq=[0])
        full = np.asarray(full)[0]
        out = {}
        for q in self.quantile_levels:
            decile = int(np.argmin([abs(q - d) for d in DECILES]))
            out[q] = np.clip(full[:self.horizon, 1 + decile], 0, None)
        return out
