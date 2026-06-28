"""Moving-average baseline: flat forecast = mean of the last `window` weeks.

Quantiles come from in-sample one-step moving-average residuals.
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster, quantiles_from_residuals


class MovingAverage(Forecaster):
    name = "moving_average"
    supported_regimes = ["statistical"]

    def __init__(self, *args, window: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def predict_quantiles(self, history):
        w = min(self.window, len(history)) or 1
        level = float(np.mean(history[-w:]))
        point = np.full(self.horizon, level)
        # in-sample one-step residuals of the same moving average
        if len(history) > w:
            preds = np.array([history[i - w:i].mean() for i in range(w, len(history))])
            resid = history[w:] - preds
        else:
            resid = np.array([0.0, 0.0])
        return quantiles_from_residuals(point, resid, self.quantile_levels)
