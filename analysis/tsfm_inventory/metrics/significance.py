from __future__ import annotations

import numpy as np

from .. import config as C

N_RESAMPLES = 10_000


def paired_bootstrap(statistic, a, b, n_resamples=N_RESAMPLES, seed=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = {"a": float("nan"), "b": float("nan"), "diff": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan"), "n": len(a)}
    if len(a) == 0:  # nothing in common to compare (e.g. two panels)
        return out

    rng = np.random.default_rng(C.SEED if seed is None else seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["a"], out["b"] = float(statistic(a[None])[0]), float(statistic(b[None])[0])
        diff = statistic(a[idx]) - statistic(b[idx])
    diff = diff[np.isfinite(diff)]
    if not np.isfinite(out["a"] - out["b"]) or len(diff) == 0:
        return out

    lo, hi = np.percentile(diff, [2.5, 97.5])
    p_value = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    out.update(diff=out["a"] - out["b"], ci_lo=float(lo), ci_hi=float(hi),
               p_value=float(p_value))
    return out
