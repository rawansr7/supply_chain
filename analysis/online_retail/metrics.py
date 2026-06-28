"""Evaluation metrics: accuracy + decision-centric (inventory) + significance.

Accuracy
--------
  - MAE, RMSE                : standard point-error.
  - RMSSE / MASE             : scaled errors (M5-style). Each product's error is
                               divided by the in-sample naive one-step error, so
                               products on different scales are comparable and the
                               aggregate is not dominated by a few high-volume SKUs.

Decision-centric (the thesis's business-meaningful evaluation)
--------------------------------------------------------------
  We turn each forecast into a single-period newsvendor stocking decision and
  score the *decision*, not the forecast:
      order q*  = forecast of the critical-ratio quantile (cr = Cu/(Cu+Co))
      cost      = Co * max(q-d,0)  +  Cu * max(d-q,0)
  We report achieved service level, fill rate, and average cost per unit demand.
  This answers: does a text-driven accuracy gain actually lower inventory cost?

Significance
------------
  Diebold-Mariano test on per-product loss differentials between two models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def rmsse(eval_df: pd.DataFrame, train_panel: pd.DataFrame) -> float:
    """Root Mean Scaled Squared Error, averaged over products (M5 style).

    eval_df:    columns [StockCode, y_true, y_pred] over the test weeks.
    train_panel: in-sample weekly panel used to compute the naive scaling denom.
    """
    scales = _naive_scale(train_panel)
    per = []
    for code, g in eval_df.groupby("StockCode"):
        s = scales.get(code, np.nan)
        if not np.isfinite(s) or s <= 0:
            continue
        num = np.mean((g["y_true"].values - g["y_pred"].values) ** 2)
        per.append(np.sqrt(num / s))
    return float(np.mean(per)) if per else float("nan")


def _naive_scale(train_panel: pd.DataFrame) -> dict:
    """Per-product mean squared one-step naive error on the in-sample period."""
    scales = {}
    for code, g in train_panel.sort_values("week").groupby("StockCode"):
        y = g["sold"].dropna().values
        if len(y) < 2:
            continue
        scales[code] = float(np.mean(np.diff(y) ** 2))
    return scales


# ---------------------------------------------------------------------------
# Decision-centric / inventory
# ---------------------------------------------------------------------------
def critical_ratio(cu: float = None, co: float = None) -> float:
    cu = C.STOCKOUT_COST if cu is None else cu
    co = C.HOLDING_COST if co is None else co
    return cu / (cu + co)


def newsvendor_costs(y_true, q_order, cu: float = None, co: float = None) -> dict:
    """Score stocking decisions q_order against realised demand y_true."""
    cu = C.STOCKOUT_COST if cu is None else cu
    co = C.HOLDING_COST if co is None else co
    d = np.asarray(y_true, dtype=float)
    q = np.asarray(q_order, dtype=float)
    over = np.maximum(q - d, 0.0)
    under = np.maximum(d - q, 0.0)
    cost = co * over + cu * under
    total_demand = d.sum()
    return {
        "avg_cost_per_period": float(cost.mean()),
        "cost_per_unit_demand": float(cost.sum() / total_demand) if total_demand > 0 else float("nan"),
        "service_level": float(np.mean(q >= d)),        # P(no stockout)
        "fill_rate": float(1 - under.sum() / total_demand) if total_demand > 0 else float("nan"),
        "avg_overage_units": float(over.mean()),
        "avg_underage_units": float(under.mean()),
    }


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """Paired DM-style test on loss differentials d = loss_a - loss_b.

    Negative mean(d) => model A has lower loss (A better). Uses a paired t-test on
    per-product mean differentials (a robust, simple variant suitable across SKUs).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"mean_diff": float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "n": len(d)}
    t, p = stats.ttest_1samp(d, 0.0)
    return {"mean_diff": float(d.mean()), "t_stat": float(t), "p_value": float(p), "n": int(len(d))}
