import tempfile

import numpy as np
import torch
from chronos import BaseChronosPipeline

from .. import config as C
from .base import Forecaster, listed

REPO_ID = "amazon/chronos-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA = [{"finetune_mode": "lora", "learning_rate": r} for r in (1e-5, 1e-4, 1e-3)]
FULL = [{"finetune_mode": "full", "learning_rate": r} for r in (1e-6, 1e-5, 3e-5)]


class Chronos2(Forecaster):
    grids = {"zero_shot": [{}], "fine_tune": LORA + FULL}

    def fit(self, train_panel):
        self._pipeline = BaseChronosPipeline.from_pretrained(REPO_ID, device_map=DEVICE)
        if self.regime == "fine_tune":
            inputs = listed(train_panel)
            with tempfile.TemporaryDirectory() as out:
                self._pipeline = self._pipeline.fit(
                    inputs,
                    self.horizon,
                    output_dir=out,
                    seed=C.SEED,
                    disable_tqdm=True,
                    remove_printer_callback=True,
                    **self.params,
                )
        return self

    def predict_quantiles(self, history):
        inputs = listed(history)
        quantiles, _mean = self._pipeline.predict_quantiles(
            inputs,
            prediction_length=self.horizon,
            quantile_levels=self.quantile_levels,
        )
        forecast = torch.stack(quantiles)
        forecast = forecast[:, 0]
        forecast = forecast.float().cpu().numpy()
        return np.clip(forecast, 0, None)
