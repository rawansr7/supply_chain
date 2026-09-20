from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .base import Forecaster, quantiles_from_residuals


class LightGBMGlobal(Forecaster):
    name = "lightgbm_global"
    supported_regimes = ["statistical"]

    def _lags(self):
        return sorted({1, 2, 3, 4, self.seasonality})

    def _design(self, df):
        g = df.groupby("series_id")["y"]
        feats = {f"lag_{lag}": g.shift(lag) for lag in self._lags()}
        feats["roll4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
        return pd.DataFrame(feats, index=df.index)

    def fit(self, train_panel=None):
        import lightgbm as lgb

        df = train_panel.sort_values(["series_id", "t"])
        X = self._design(df)
        keep = X.dropna().index
        X, y = X.loc[keep], df["y"].loc[keep]

        self.feat_cols = list(X.columns)
        self.model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                       num_leaves=31, random_state=C.SEED, verbose=-1)
        self.model.fit(X, y)
        self.residuals = (y - self.model.predict(X)).to_numpy()
        return self

    def predict_quantiles(self, history):
        buf = list(history.astype(float))
        lags = self._lags()
        point = []
        for _ in range(self.horizon):
            row = {f"lag_{lag}": (buf[-lag] if len(buf) >= lag else buf[0]) for lag in lags}
            row["roll4"] = float(np.mean(buf[-4:]))
            x = pd.DataFrame([row])[self.feat_cols]
            yhat = max(float(self.model.predict(x)[0]), 0.0)
            point.append(yhat)
            buf.append(yhat)
        return quantiles_from_residuals(np.array(point), self.residuals, self.quantile_levels)
