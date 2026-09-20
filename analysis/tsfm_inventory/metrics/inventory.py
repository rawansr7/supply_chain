from __future__ import annotations

import numpy as np

from .. import config as C


def order_from_quantiles(quantile_forecast):
    level = min(quantile_forecast, key=lambda q: abs(q - C.CRITICAL_RATIO))
    return quantile_forecast[level]


def cost(order, demand):
    return (C.HOLDING_COST * np.maximum(order - demand, 0)
            + C.STOCKOUT_COST * np.maximum(demand - order, 0))


def fill_rate(order, demand):
    total = demand.sum()
    return float(np.minimum(order, demand).sum() / total) if total > 0 else float("nan")
