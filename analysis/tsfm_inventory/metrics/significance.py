from __future__ import annotations

import numpy as np

from .. import config as C

N_RESAMPLES = 10_000


def paired_bootstrap(cost_a, cost_b, demand, n_resamples=N_RESAMPLES, seed=None):
    """Resample series to put an interval on the difference in pooled cost per unit.

    The headline metric divides total cost by total demand, so a paired test over
    per-series costs would answer a different question from the one the table reports —
    and on a uniform sample of series, where demand spans four orders of magnitude, it
    would be decided by the few largest. Drawing whole series with replacement asks the
    question the table asks: would another 300 series from this catalogue rank these two
    models the same way?
    """
    a = np.asarray(cost_a, dtype=float)
    b = np.asarray(cost_b, dtype=float)
    d = np.asarray(demand, dtype=float)
    empty = {"diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
             "p_value": float("nan"), "n": len(a)}
    if len(a) == 0 or d.sum() <= 0:  # nothing in common to compare (e.g. two panels)
        return empty
    observed = float((a.sum() - b.sum()) / d.sum())

    rng = np.random.default_rng(C.SEED if seed is None else seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    total = d[idx].sum(axis=1)
    diff = (a[idx].sum(axis=1) - b[idx].sum(axis=1)) / np.where(total > 0, total, np.nan)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return empty

    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"diff": observed, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(2 * min((diff <= 0).mean(), (diff >= 0).mean())),
            "n": len(a)}
