from __future__ import annotations

import numpy as np


def diebold_mariano(loss_a, loss_b):
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    out = {"mean_diff": float(np.mean(d)), "t_stat": float("nan"),
           "p_value": float("nan"), "n": len(d)}
    if len(d) >= 2 and not np.allclose(d, 0):
        from scipy import stats
        t, p = stats.ttest_rel(a, b)
        out["t_stat"], out["p_value"] = float(t), float(p)
    return out
