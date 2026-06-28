"""The one interface every forecaster implements.

A model is created with its regime, then:
  - .fit(train_panel, dataset)         trains if the regime needs it (no-op otherwise)
  - .predict_quantiles(history)        forecasts one series -> {quantile: array(horizon)}

Regimes (kept deliberately simple — two for foundation models, one for baselines):
  zero_shot  : use the pretrained model out of the box, on the series' own history
  fine_tune  : fine-tune on the dataset's training panel, then forecast
  statistical: the single regime the classical/ML baselines run in
"""
from __future__ import annotations

import numpy as np


def quantiles_from_residuals(point: np.ndarray, residuals: np.ndarray,
                             quantile_levels: list[float]) -> dict[float, np.ndarray]:
    """Turn a point forecast into quantiles using the empirical residual distribution.

    Demand is non-negative, so quantiles are clipped at 0. This is how the classical
    baselines produce the predictive distribution the newsvendor decision needs.
    """
    if len(residuals) < 2:
        residuals = np.array([0.0, 0.0])
    return {q: np.clip(point + np.quantile(residuals, q), 0, None)
            for q in quantile_levels}


class Forecaster:
    name = "base"
    supported_regimes = ["zero_shot"]
    needs_gpu = False

    def __init__(self, regime, horizon, quantile_levels, seasonality, smoke=False, **kwargs):
        if regime not in self.supported_regimes:
            raise ValueError(
                f"{self.name} does not support regime {regime!r}; "
                f"supported: {self.supported_regimes}")
        self.regime = regime
        self.horizon = horizon
        self.quantile_levels = quantile_levels
        self.seasonality = seasonality
        self.smoke = smoke      # tiny/fast fine-tune for plumbing checks (not a real result)

    def fit(self, train_panel, dataset):
        """Train if needed. Default: no-op (zero-shot / per-series models)."""
        return self

    def predict_quantiles(self, history):
        """Return {quantile_level: np.ndarray(horizon)} for one series."""
        raise NotImplementedError
