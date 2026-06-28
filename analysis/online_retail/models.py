"""Forecasting models.

GlobalLGBM trains ONE LightGBM across all products (the modern global-model
approach: products share statistical strength, and static product/text features
distinguish them). It produces:
  - a point forecast        (objective = regression / L2)  -> accuracy metrics
  - a quantile forecast at the newsvendor critical ratio    -> stocking decision

A SeasonalNaive baseline is included for the M5-style scaling and as a sanity
floor every learned model must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from . import config as C
from .metrics import critical_ratio

_LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    n_jobs=-1,
    verbosity=-1,
    random_state=C.SEED,
)


class GlobalLGBM:
    """Global LightGBM forecaster with optional quantile head."""

    def __init__(self, feat_cols: list[str], quantile: float | None = None):
        self.feat_cols = feat_cols
        self.quantile = quantile if quantile is not None else critical_ratio()
        self.point_model = LGBMRegressor(objective="regression", **_LGBM_PARAMS)
        self.q_model = LGBMRegressor(objective="quantile", alpha=self.quantile, **_LGBM_PARAMS)

    def fit(self, train: pd.DataFrame, target: str = "sold"):
        X, y = train[self.feat_cols], train[target]
        self.point_model.fit(X, y)
        self.q_model.fit(X, y)
        return self

    def predict_point(self, test: pd.DataFrame) -> np.ndarray:
        return np.clip(self.point_model.predict(test[self.feat_cols]), 0, None)

    def predict_order(self, test: pd.DataFrame) -> np.ndarray:
        """Critical-ratio quantile -> the newsvendor order quantity."""
        return np.clip(self.q_model.predict(test[self.feat_cols]), 0, None)


class SeasonalNaive:
    """Predict demand = value 52 weeks ago (falls back to last value / 0)."""

    def fit(self, train, target="sold"):
        return self

    @staticmethod
    def predict_point(test: pd.DataFrame) -> np.ndarray:
        pred = test.get("sold_lag_52")
        if pred is None:
            pred = test.get("sold_lag_1", pd.Series(0, index=test.index))
        pred = pred.fillna(test.get("sold_lag_1", 0))
        return np.clip(pred.fillna(0).values, 0, None)
