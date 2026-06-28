"""Decision-centric (inventory) metrics kept for the thesis: cost per unit and fill rate.

The forecast is turned into a stocking order with the newsvendor rule, then scored by
what that order costs and how much demand it meets.

newsvendor: order the critical-ratio quantile of demand, CR = Cu / (Cu + Co).
Given order Q and realised demand D per week:
    overage  = max(Q - D, 0)   (held, costs Co each)
    underage = max(D - Q, 0)   (short, costs Cu each)
    cost     = Co*overage + Cu*underage
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def order_from_quantiles(quantile_forecast: dict[float, np.ndarray],
                         critical_ratio: float = C.CRITICAL_RATIO) -> np.ndarray:
    """Pick the forecast quantile nearest the critical ratio as the order quantity."""
    q = min(quantile_forecast, key=lambda x: abs(x - critical_ratio))
    return quantile_forecast[q]


def costs(order: np.ndarray, demand: np.ndarray,
          holding: float = C.HOLDING_COST, stockout: float = C.STOCKOUT_COST):
    overage = np.maximum(order - demand, 0)
    underage = np.maximum(demand - order, 0)
    cost = holding * overage + stockout * underage
    return cost, overage, underage


def fill_rate(order: np.ndarray, demand: np.ndarray) -> float:
    """Fraction of total demand met from stock."""
    total = demand.sum()
    return float(np.minimum(order, demand).sum() / total) if total > 0 else float("nan")
