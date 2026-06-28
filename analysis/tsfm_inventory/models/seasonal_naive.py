"""Seasonal-naive baseline: next value = value one season ago.

The standard, hard-to-beat baseline for seasonal demand. Quantiles come from the
in-sample seasonal residuals.
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster, quantiles_from_residuals


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"
    supported_regimes = ["statistical"]

    def predict_quantiles(self, history):
        m = self.seasonality
        h = self.horizon
        if len(history) >= m:
            point = np.array([history[-m + (i % m)] for i in range(h)], dtype=float)
            resid = history[m:] - history[:-m]
        else:                                  # too short for a full season
            point = np.full(h, float(history[-1]) if len(history) else 0.0)
            resid = np.diff(history) if len(history) > 1 else np.array([0.0, 0.0])
        return quantiles_from_residuals(point, resid, self.quantile_levels)
