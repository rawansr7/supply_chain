from __future__ import annotations

import numpy as np

from .base import Forecaster, quantiles_from_residuals


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"
    supported_regimes = ["statistical"]

    def predict_quantiles(self, history):
        m, h = self.seasonality, self.horizon
        if len(history) >= m:
            point = np.array([history[-m + (i % m)] for i in range(h)], dtype=float)
            resid = history[m:] - history[:-m]
        else:
            point = np.full(h, float(history[-1]) if len(history) else 0.0)
            resid = np.diff(history)
        return quantiles_from_residuals(point, resid, self.quantile_levels)
