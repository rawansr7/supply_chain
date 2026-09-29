from .. import config as C
from .base import Forecaster, empirical_quantiles


class SeasonalNaive(Forecaster):
    grids = {"statistical": [{}]}

    def predict(self, Y):
        Y, m = Y.to_numpy(), C.SEASON
        return empirical_quantiles(Y[:, -m:-m + self.horizon], (Y[:, m:] - Y[:, :-m])[..., None], self.levels)
