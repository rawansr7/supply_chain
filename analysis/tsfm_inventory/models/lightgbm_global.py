import lightgbm as lgb
import numpy as np

from .. import config as C
from .base import Forecaster, scaled_windows

CONTEXT = 52
TREES = 500


class LightGBMGlobal(Forecaster):
    grids = {"statistical": [{"num_leaves": n, "learning_rate": r}
                             for n in (15, 63, 255)
                             for r in (0.02, 0.05, 0.1)]}

    def rows(self, X):
        features = np.tile(X, (self.horizon, 1))
        steps = np.repeat(np.arange(self.horizon), len(X))
        return np.hstack([features, steps[:, None]])

    def fit(self, train_panel):
        X, y = scaled_windows(train_panel.to_numpy(), CONTEXT, self.horizon)
        features = self.rows(X)
        targets = y.T.ravel()

        self.models = []
        for q in self.quantile_levels:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=TREES,
                                      random_state=C.SEED, verbose=-1, **self.params)
            model.fit(features, targets)
            self.models.append(model)
        return self

    def predict_quantiles(self, history):
        context = history.to_numpy()[:, -CONTEXT:]
        scale = 1 + context.mean(axis=1, keepdims=True)
        features = self.rows(context / scale)

        per_level = []
        for model in self.models:
            pred = model.predict(features)
            pred = pred.reshape(self.horizon, -1).T
            per_level.append(pred)

        quantiles = np.stack(per_level, axis=-1)
        quantiles = np.sort(quantiles, axis=-1)
        quantiles = quantiles * scale[..., None]
        return np.clip(quantiles, 0, None)
