from __future__ import annotations

import numpy as np

from .base import Forecaster

CKPT_PATH = "lag-llama.ckpt"
CONTEXT_LENGTH = 32
FREQ = "W"
START = "2020-01-06"


def _allow_full_unpickle(torch):
    if getattr(torch.load, "_patched", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        # the checkpoint holds gluonts objects, so weights_only=True rejects it
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    load._patched = True
    torch.load = load


class LagLlama(Forecaster):
    name = "lag_llama"
    supported_regimes = ["zero_shot"]
    needs_gpu = True

    def fit(self, train_panel=None):
        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator

        _allow_full_unpickle(torch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = torch.load(CKPT_PATH, map_location=device)["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
            ckpt_path=CKPT_PATH, prediction_length=self.horizon,
            context_length=CONTEXT_LENGTH, input_size=args["input_size"],
            n_layer=args["n_layer"], n_embd_per_head=args["n_embd_per_head"],
            n_head=args["n_head"], scaling=args["scaling"], time_feat=args["time_feat"],
            nonnegative_pred_samples=True, num_parallel_samples=100, device=device)
        self.predictor = estimator.create_predictor(estimator.create_transformation(),
                                                    estimator.create_lightning_module())
        return self

    def predict_quantiles(self, history):
        from gluonts.dataset.common import ListDataset
        from gluonts.evaluation import make_evaluation_predictions

        # make_evaluation_predictions forecasts the LAST horizon steps, so pad the history
        padded = np.concatenate([np.asarray(history, dtype="float32"),
                                 np.zeros(self.horizon, dtype="float32")])
        ds = ListDataset([{"start": START, "target": padded, "item_id": "0"}], freq=FREQ)
        forecast_it, _ = make_evaluation_predictions(dataset=ds, predictor=self.predictor,
                                                     num_samples=100)
        forecast = next(iter(forecast_it))
        return {q: np.clip(np.asarray(forecast.quantile(q), dtype="float64"), 0, None)
                for q in self.quantile_levels}
