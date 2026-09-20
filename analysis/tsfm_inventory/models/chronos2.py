from __future__ import annotations

import numpy as np

from .base import Forecaster

REPO_ID = "amazon/chronos-2"


class Chronos2(Forecaster):
    name = "chronos2"
    supported_regimes = ["zero_shot", "fine_tune"]
    needs_gpu = True
    finetune_mode = "lora"
    num_steps = 1000
    batch_size = 32

    def fit(self, train_panel=None):
        import torch
        from chronos import BaseChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = BaseChronosPipeline.from_pretrained(REPO_ID, device_map=device)

        if self.regime == "zero_shot":
            self._pipeline = pipeline
            return self

        inputs = [g.sort_values("t")["y"].to_numpy(dtype=np.float32)
                  for _, g in train_panel.groupby("series_id", sort=False)]
        inputs = [series for series in inputs if len(series) > self.horizon]
        if not inputs:
            raise ValueError("chronos2 fine-tune: no series longer than the horizon.")

        # fit() returns a NEW fine-tuned pipeline; it does not mutate the base one.
        self._pipeline = pipeline.fit(
            inputs=inputs,
            prediction_length=self.horizon,
            finetune_mode=self.finetune_mode,
            learning_rate=1e-4 if self.finetune_mode == "lora" else 1e-5,
            num_steps=50 if self.smoke else self.num_steps,
            batch_size=self.batch_size,
            logging_steps=100)
        return self

    def predict_quantiles(self, history):
        quantiles, _mean = self._pipeline.predict_quantiles(
            inputs=[np.asarray(history, dtype=np.float32)],
            prediction_length=self.horizon,
            quantile_levels=self.quantile_levels)
        forecast = quantiles[0].squeeze(0).cpu().numpy()
        return {q: np.clip(forecast[:, i], 0, None)
                for i, q in enumerate(self.quantile_levels)}
