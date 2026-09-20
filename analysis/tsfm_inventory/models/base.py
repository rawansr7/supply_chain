from __future__ import annotations

import numpy as np


def quantiles_from_residuals(point, residuals, quantile_levels):
    if len(residuals) < 2:
        residuals = np.zeros(2)
    return {q: np.clip(point + np.quantile(residuals, q), 0, None) for q in quantile_levels}


class Forecaster:
    name = "base"
    supported_regimes = ["zero_shot"]
    needs_gpu = False

    def __init__(self, regime, horizon, quantile_levels, seasonality, smoke=False):
        if regime not in self.supported_regimes:
            raise ValueError(f"{self.name} does not support regime {regime!r}; "
                             f"supported: {self.supported_regimes}")
        self.regime = regime
        self.horizon = horizon
        self.quantile_levels = quantile_levels
        self.seasonality = seasonality
        self.smoke = smoke

    def fit(self, train_panel=None):
        return self

    def predict_quantiles(self, history):
        raise NotImplementedError
