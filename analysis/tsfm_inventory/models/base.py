import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Forecaster:
    grids = {}

    def __init__(self, regime, horizon, quantile_levels, **params):
        self.regime = regime
        self.horizon = horizon
        self.quantile_levels = quantile_levels
        self.params = params

    def fit(self, train_panel):
        return self

    def predict_quantiles(self, history):
        raise NotImplementedError


def listed(panel):
    rows = panel.to_numpy(np.float32)
    return [y[~np.isnan(y)] for y in rows]


def quantiles_from_residuals(point, residuals, quantile_levels):
    residual_quantiles = np.nanquantile(residuals, quantile_levels, axis=1)
    residual_quantiles = np.moveaxis(residual_quantiles, 0, -1)
    quantiles = point[..., None] + residual_quantiles
    return np.clip(quantiles, 0, None)


def scaled_windows(panel, context, horizon):
    width = context + horizon
    windows = sliding_window_view(panel, width, axis=1)
    windows = windows.reshape(-1, width)
    complete = ~np.isnan(windows).any(axis=1)
    windows = windows[complete]

    inputs = windows[:, :context]
    targets = windows[:, context:]
    scale = 1 + inputs.mean(axis=1, keepdims=True)
    return inputs / scale, targets / scale
