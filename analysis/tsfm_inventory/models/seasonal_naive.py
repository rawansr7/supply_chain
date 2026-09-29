from .. import config as C
from .base import Forecaster, quantiles_from_residuals


class SeasonalNaive(Forecaster):
    grids = {"statistical": [{}]}

    def predict_quantiles(self, history):
        history = history.to_numpy()
        m = C.SEASONALITY

        point = history[:, -m:-m + self.horizon]
        residuals = history[:, m:] - history[:, :-m]
        residuals = residuals[..., None]
        return quantiles_from_residuals(point, residuals, self.quantile_levels)
