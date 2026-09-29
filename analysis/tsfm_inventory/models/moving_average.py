import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Forecaster, quantiles_from_residuals


class MovingAverage(Forecaster):
    grids = {"statistical": [{"window": w} for w in (2, 4, 6, 8, 13, 26, 52)]}

    def predict_quantiles(self, history):
        history = history.to_numpy()
        w = self.params["window"]
        h = self.horizon

        windows = sliding_window_view(history, w, axis=1)
        means = windows.mean(axis=2)

        futures = sliding_window_view(history[:, w:], h, axis=1)
        residuals = futures - means[:, :-h, None]

        point = np.repeat(means[:, -1:], h, axis=1)
        return quantiles_from_residuals(point, residuals, self.quantile_levels)
