"""Chronos-2 (Amazon) adapter — zero-shot and fine-tune.

Open model on Hugging Face: amazon/chronos-2 (Oct 2025). GPU recommended; CPU works
(slow for fine-tune). Install:
    pip install "chronos-forecasting>=2.1.0"     # LoRA fine-tune also needs: pip install peft

Verified against chronos-forecasting main (v2.3.0): BaseChronosPipeline.from_pretrained
returns a Chronos2Pipeline whose .fit(...) returns a NEW fine-tuned pipeline, and whose
.predict_quantiles(inputs=[...]) returns (quantiles_list, mean_list).
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster


class Chronos2(Forecaster):
    name = "chronos2"
    supported_regimes = ["zero_shot", "fine_tune"]
    needs_gpu = True

    # Fine-tune knobs. LoRA is the only realistic path without a GPU.
    finetune_mode = "lora"      # "lora" or "full"
    num_steps = 1000            # official default config (no tuning); smoke uses 50
    batch_size = 32

    def _device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train_panel, dataset):
        from chronos import BaseChronosPipeline  # -> Chronos2Pipeline for amazon/chronos-2
        base = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=self._device())

        if self.regime == "zero_shot":
            self._pipeline = base
            return self

        # fine_tune: global fine-tune across all series (one 1D float array per series).
        inputs = [g.sort_values("t")["y"].to_numpy(dtype=np.float32)
                  for _, g in train_panel.groupby("series_id", sort=False)]
        inputs = [a for a in inputs if len(a) > self.horizon]
        if not inputs:
            raise ValueError("chronos2 fine-tune: no series longer than the horizon.")

        use_lora = self.finetune_mode == "lora"
        steps = 50 if self.smoke else self.num_steps
        # IMPORTANT: fit() returns a NEW pipeline and does NOT mutate `base`. Assign it.
        # On CPU keep fp32 (do not pass bf16/fp16 trainer flags).
        self._pipeline = base.fit(
            inputs=inputs,
            prediction_length=self.horizon,                  # must match horizon at predict time
            finetune_mode="lora" if use_lora else "full",
            learning_rate=1e-4 if use_lora else 1e-5,        # notebook: 1e-4 LoRA / 1e-5 full
            num_steps=steps,
            batch_size=self.batch_size,
            logging_steps=100,                               # -> HF TrainingArguments
        )
        return self

    def predict_quantiles(self, history):
        qlevels = list(self.quantile_levels)
        # predict_quantiles takes a list of 1D arrays (one per series); we pass one.
        quantiles_list, _mean = self._pipeline.predict_quantiles(
            inputs=[np.asarray(history, dtype=np.float32)],
            prediction_length=self.horizon,
            quantile_levels=qlevels,
        )
        # quantiles_list[0]: (n_variates=1, horizon, n_quantiles); last-axis order == qlevels.
        arr = quantiles_list[0].squeeze(0).cpu().numpy()     # -> (horizon, n_quantiles)
        return {q: np.clip(arr[:, i], 0, None) for i, q in enumerate(qlevels)}
