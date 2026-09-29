import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Forecaster:
    grids = {}

    def __init__(self, regime, horizon, levels, **params):
        self.regime = regime
        self.horizon = horizon
        self.levels = levels
        self.params = params

    def fit(self, Y):
        return self

    def predict(self, Y):
        raise NotImplementedError


def listed(Y):
    return [y[~np.isnan(y)] for y in Y.to_numpy(np.float32)]


def empirical_quantiles(point, errors, levels):
    return np.clip(point[..., None] + np.moveaxis(np.nanquantile(errors, levels, axis=1), 0, -1), 0, None)


def scaled_windows(Y, context, horizon):
    W = sliding_window_view(Y, context + horizon, axis=1).reshape(-1, context + horizon)
    W = W[~np.isnan(W).any(1)]
    scale = 1 + W[:, :context].mean(1, keepdims=True)
    return W[:, :context] / scale, W[:, context:] / scale
