import numpy as np

from .. import config as C

RESAMPLES = 10_000


def paired_bootstrap(statistic, a, b):
    idx = np.random.default_rng(C.SEED).integers(0, len(a), (RESAMPLES, len(a)))
    diff = statistic(a[idx]) - statistic(b[idx])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    stat_a, stat_b = float(statistic(a[None])[0]), float(statistic(b[None])[0])
    return {"a": stat_a, "b": stat_b, "diff": stat_a - stat_b, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))), "n": len(a)}
