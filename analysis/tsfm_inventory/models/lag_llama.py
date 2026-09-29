import functools
import tempfile

import numpy as np
import torch
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

from .. import config as C
from .base import Forecaster

torch.load = functools.partial(torch.load, weights_only=False)
FREQ = "W"
SAMPLES = 100
EPOCHS = 50
PREDICTION_BATCH = 16


def checkpoint_args():
    checkpoint = torch.load(C.LAG_LLAMA_CKPT, map_location="cpu")
    return checkpoint["hyper_parameters"]["model_kwargs"]


ARGS = checkpoint_args()


def dataset(panel):
    starts = panel.notna().idxmax(axis=1)
    rows = panel.to_numpy(np.float32)
    entries = []
    for start, y in zip(starts, rows):
        entries.append({"start": start, "target": y[~np.isnan(y)]})
    return ListDataset(entries, freq=FREQ)


class LagLlama(Forecaster):
    grids = {"zero_shot": [{"context_length": c} for c in (32, 64, 128, 256)],
             "fine_tune": [{"context_length": c, "lr": r}
                           for c in (32, 64, 128)
                           for r in (1e-4, 5e-4)]}

    def estimator(self, **kwargs):
        context = self.params["context_length"]
        factor = max(1.0, (context + self.horizon) / ARGS["context_length"])
        return LagLlamaEstimator(
            ckpt_path=str(C.LAG_LLAMA_CKPT),
            prediction_length=self.horizon,
            input_size=ARGS["input_size"],
            n_layer=ARGS["n_layer"],
            n_embd_per_head=ARGS["n_embd_per_head"],
            n_head=ARGS["n_head"],
            scaling=ARGS["scaling"],
            time_feat=ARGS["time_feat"],
            rope_scaling={"type": "linear", "factor": factor},
            nonnegative_pred_samples=True,
            num_parallel_samples=SAMPLES,
            batch_size=64,
            **self.params,
            **kwargs,
        )

    def fit(self, train_panel):
        torch.manual_seed(C.SEED)
        np.random.seed(C.SEED)
        if self.regime == "zero_shot":
            estimator = self.estimator()
            transformation = estimator.create_transformation()
            module = estimator.create_lightning_module()
            self.predictor = estimator.create_predictor(transformation, module)
        else:
            with tempfile.TemporaryDirectory() as out:
                trainer = {
                    "max_epochs": EPOCHS,
                    "default_root_dir": out,
                    "logger": False,
                    "enable_progress_bar": False,
                }
                estimator = self.estimator(aug_prob=0.0, trainer_kwargs=trainer)
                self.predictor = estimator.train(dataset(train_panel), cache_data=True,
                                                 shuffle_buffer_length=1000)
        self.predictor.batch_size = PREDICTION_BATCH
        return self

    def predict_quantiles(self, history):
        torch.manual_seed(C.SEED)
        forecast_it = self.predictor.predict(dataset(history))
        quantiles = []
        for forecast in forecast_it:
            per_level = [forecast.quantile(q) for q in self.quantile_levels]
            quantiles.append(np.stack(per_level, axis=-1))
        return np.stack(quantiles)
