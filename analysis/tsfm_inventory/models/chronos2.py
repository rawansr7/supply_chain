import tempfile

import numpy as np
import torch
from chronos import BaseChronosPipeline

from .. import config as C
from .base import Forecaster, listed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Chronos2(Forecaster):
    grids = {"zero_shot": [{}],
             "fine_tune": [{"finetune_mode": "lora", "learning_rate": r} for r in (1e-5, 1e-4, 1e-3)]
                          + [{"finetune_mode": "full", "learning_rate": r} for r in (1e-6, 1e-5, 3e-5)]}

    def fit(self, Y):
        self.pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
        if self.regime == "fine_tune":
            with tempfile.TemporaryDirectory() as out:
                self.pipeline = self.pipeline.fit(listed(Y), self.horizon, output_dir=out, seed=C.SEED,
                                                  disable_tqdm=True, remove_printer_callback=True, **self.params)
        return self

    def predict(self, Y):
        quantiles, _ = self.pipeline.predict_quantiles(listed(Y), prediction_length=self.horizon,
                                                       quantile_levels=self.levels)
        return np.clip(torch.stack(quantiles)[:, 0].float().cpu().numpy(), 0, None)
