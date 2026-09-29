import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Forecaster, empirical_quantiles


class MovingAverage(Forecaster):
    grids = {"statistical": [{"window": w} for w in (2, 4, 6, 8, 13, 26, 52)]}

    def predict(self, Y):
        Y, w, h = Y.to_numpy(), self.params["window"], self.horizon
        means = sliding_window_view(Y, w, axis=1).mean(2)
        errors = sliding_window_view(Y[:, w:], h, axis=1) - means[:, :-h, None]
        return empirical_quantiles(np.repeat(means[:, -1:], h, 1), errors, self.levels)
