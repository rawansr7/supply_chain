"""Global LightGBM baseline: one gradient-boosted model across all series.

The standard ML baseline. Uses autoregressive lag features and forecasts recursively.
Quantiles come from the global training-residual distribution. lightgbm is imported
lazily so smoke runs (baselines only) need it not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, quantiles_from_residuals


class LightGBMGlobal(Forecaster):
    name = "lightgbm_global"
    supported_regimes = ["statistical"]

    def _lags(self):
        return sorted({1, 2, 3, 4, self.seasonality})

    def _design(self, df: pd.DataFrame):
        g = df.groupby("series_id")["y"]
        feats = {f"lag_{L}": g.shift(L) for L in self._lags()}
        feats["roll4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
        X = pd.DataFrame(feats, index=df.index)
        return X

    def fit(self, train_panel, dataset):
        import lightgbm as lgb   # lazy
        df = train_panel.sort_values(["series_id", "t"]).copy()
        X = self._design(df)
        y = df["y"]
        keep = X.dropna().index
        X, y = X.loc[keep], y.loc[keep]
        self.feat_cols = list(X.columns)
        self.model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                       num_leaves=31, random_state=51, verbose=-1)
        self.model.fit(X, y)
        self.residuals = (y - self.model.predict(X)).to_numpy()
        return self

    def predict_quantiles(self, history):
        buf = list(history.astype(float))
        lags = self._lags()
        point = []
        for _ in range(self.horizon):
            row = {f"lag_{L}": (buf[-L] if len(buf) >= L else buf[0]) for L in lags}
            row["roll4"] = float(np.mean(buf[-4:])) if buf else 0.0
            x = pd.DataFrame([row])[self.feat_cols]
            yhat = max(float(self.model.predict(x)[0]), 0.0)
            point.append(yhat)
            buf.append(yhat)
        return quantiles_from_residuals(np.array(point), self.residuals, self.quantile_levels)
