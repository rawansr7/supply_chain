"""Lag-Llama adapter — zero-shot and fine-tune (gluonts-based).

NOT on PyPI. Install from source + download the checkpoint:
    git clone https://github.com/time-series-foundation-models/lag-llama
    pip install -r lag-llama/requirements.txt
    huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir .
Run with the lag_llama package importable and lag-llama.ckpt in the working dir (or edit CKPT_PATH).

Verified against lag-llama main + gluonts:
  zero_shot = build a predictor from the checkpoint without training;
  fine_tune = construct LagLlamaEstimator(ckpt_path=...) then call .train() (returns a predictor).
Architecture kwargs MUST be read from the checkpoint or the load mismatches.
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster

CKPT_PATH = "lag-llama.ckpt"
CONTEXT_LENGTH = 32              # pretraining length; a robustness check at 64 gave no gain
FREQ = "W"
START = "2020-01-06"            # any fixed Monday; only relative positions matter


class LagLlama(Forecaster):
    name = "lag_llama"
    supported_regimes = ["zero_shot", "fine_tune"]
    needs_gpu = True
    lr = 5e-4                   # official Colab finetune demo values (no tuning)
    max_epochs = 50            # smoke uses 1
    batch_size = 64

    def _device(self):
        import torch
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def fit(self, train_panel, dataset):
        import torch
        from gluonts.dataset.common import ListDataset
        from lag_llama.gluon.estimator import LagLlamaEstimator

        # The Lag-Llama checkpoint is a Lightning ckpt holding gluonts objects. torch>=2.6
        # defaults weights_only=True, which rejects it — both for our read below AND for the
        # estimator's internal load_from_checkpoint. Default it back to False (trusted local
        # ckpt) for this process.
        if not getattr(torch.load, "_ll_patched", False):
            _orig_load = torch.load

            def _load(*a, **k):
                k.setdefault("weights_only", False)
                return _orig_load(*a, **k)
            _load._ll_patched = True
            torch.load = _load

        device = self._device()
        # architecture hyperparameters live in the checkpoint; they must match on load.
        args = torch.load(CKPT_PATH, map_location=device)["hyper_parameters"]["model_kwargs"]
        common = dict(
            prediction_length=self.horizon, context_length=CONTEXT_LENGTH,
            input_size=args["input_size"], n_layer=args["n_layer"],
            n_embd_per_head=args["n_embd_per_head"], n_head=args["n_head"],
            scaling=args["scaling"], time_feat=args["time_feat"],
            nonnegative_pred_samples=True, num_parallel_samples=100, device=device)

        if self.regime == "zero_shot":
            est = LagLlamaEstimator(ckpt_path=CKPT_PATH, **common)
            lm = est.create_lightning_module()
            tf = est.create_transformation()
            self.predictor = est.create_predictor(tf, lm)
        else:  # fine_tune — global over all warm series
            entries = []
            for sid, g in train_panel.sort_values("t").groupby("series_id"):
                y = g["y"].to_numpy(dtype="float32")
                if len(y) >= CONTEXT_LENGTH + self.horizon:   # else windows get dropped
                    entries.append({"start": START, "target": y, "item_id": str(sid)})
            if not entries:
                raise ValueError("lag_llama fine-tune: no series >= context+horizon.")
            train_ds = ListDataset(entries, freq=FREQ)
            est = LagLlamaEstimator(
                ckpt_path=CKPT_PATH, aug_prob=0, lr=self.lr, batch_size=self.batch_size,
                trainer_kwargs={"max_epochs": 1 if self.smoke else self.max_epochs,
                                "accelerator": "gpu" if device.type == "cuda" else "cpu",
                                "devices": 1},
                **common)
            # train() loads the ckpt weights, fine-tunes, and RETURNS a PyTorchPredictor.
            self.predictor = est.train(train_ds, cache_data=True, shuffle_buffer_length=1000)

        self._freq, self._start = FREQ, START
        return self

    def predict_quantiles(self, history):
        from gluonts.dataset.common import ListDataset
        from gluonts.evaluation import make_evaluation_predictions

        target = np.asarray(history, dtype="float32")
        # make_evaluation_predictions forecasts the LAST prediction_length steps, so append
        # `horizon` placeholders to forecast beyond the real history.
        padded = np.concatenate([target, np.zeros(self.horizon, dtype="float32")])
        ds = ListDataset([{"start": self._start, "target": padded, "item_id": "0"}],
                         freq=self._freq)
        forecast_it, _ = make_evaluation_predictions(
            dataset=ds, predictor=self.predictor, num_samples=100)
        forecast = next(iter(forecast_it))            # SampleForecast (num_samples, horizon)
        return {q: np.clip(np.asarray(forecast.quantile(q), dtype="float64"), 0, None)
                for q in self.quantile_levels}
