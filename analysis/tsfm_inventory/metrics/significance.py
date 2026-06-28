"""Statistical significance: is one model's loss really lower than another's?

A paired test on per-series losses (the Diebold-Mariano idea, implemented as a paired
t-test on the per-series loss differential).
"""
from __future__ import annotations

import numpy as np


def diebold_mariano(loss_a, loss_b) -> dict:
    """Paired test of loss_a vs loss_b over the same series.

    Returns mean_diff (a - b; negative => a is better), t_stat, p_value, n.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    out = {"mean_diff": float(np.mean(d)), "t_stat": float("nan"),
           "p_value": float("nan"), "n": int(len(d))}
    if len(d) < 2 or np.allclose(d, 0):
        return out
    from scipy import stats   # lazy import
    t, p = stats.ttest_rel(a, b)
    out["t_stat"], out["p_value"] = float(t), float(p)
    return out
