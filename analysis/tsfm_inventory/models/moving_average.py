from __future__ import annotations

import numpy as np

from .base import Forecaster, quantiles_from_residuals


class MovingAverage(Forecaster):
    name = "moving_average"
    supported_regimes = ["statistical"]
    window = 8

    def predict_quantiles(self, history):
        w = min(self.window, len(history)) or 1
        point = np.full(self.horizon, float(np.mean(history[-w:])))
        if len(history) > w:
            preds = np.array([history[i - w:i].mean() for i in range(w, len(history))])
            resid = history[w:] - preds
        else:
            resid = np.zeros(2)
        return quantiles_from_residuals(point, resid, self.quantile_levels)
