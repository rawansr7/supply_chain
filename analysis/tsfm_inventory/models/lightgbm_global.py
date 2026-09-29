import lightgbm as lgb
import numpy as np

from .. import config as C
from .base import Forecaster, scaled_windows

CONTEXT = 52
TREES = 500


class LightGBMGlobal(Forecaster):
    grids = {"statistical": [{"num_leaves": n, "learning_rate": r} for n in (15, 63, 255) for r in (0.02, 0.05, 0.1)]}

    def rows(self, X):
        return np.hstack([np.tile(X, (self.horizon, 1)), np.repeat(np.arange(self.horizon), len(X))[:, None]])

    def fit(self, Y):
        X, F = scaled_windows(Y.to_numpy(), CONTEXT, self.horizon)
        self.models = [lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=TREES, random_state=C.SEED,
                                         verbose=-1, **self.params).fit(self.rows(X), F.T.ravel())
                       for q in self.levels]
        return self

    def predict(self, Y):
        context = Y.to_numpy()[:, -CONTEXT:]
        scale = 1 + context.mean(1, keepdims=True)
        rows = self.rows(context / scale)
        Q = np.stack([m.predict(rows).reshape(self.horizon, -1).T for m in self.models], -1)
        return np.clip(np.sort(Q, -1) * scale[..., None], 0, None)
