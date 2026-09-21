from __future__ import annotations

import numpy as np


def order_from_quantiles(quantile_forecast, costs):
    """Newsvendor order: the critical-ratio quantile of the predictive distribution."""
    level = min(quantile_forecast, key=lambda q: abs(q - costs.critical_ratio))
    return quantile_forecast[level]


def cost(order, demand, costs):
    return (costs.holding * np.maximum(order - demand, 0)
            + costs.stockout * np.maximum(demand - order, 0))


def fill_rate(order, demand):
    total = demand.sum()
    return float(np.minimum(order, demand).sum() / total) if total > 0 else float("nan")
