import numpy as np

from .. import config as C

N_RESAMPLES = 10_000


def paired_bootstrap(statistic, a, b):
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, len(a), (N_RESAMPLES, len(a)))
    diff = statistic(a[idx]) - statistic(b[idx])
    lo, hi = np.percentile(diff, [2.5, 97.5])

    stat_a = float(statistic(a[None])[0])
    stat_b = float(statistic(b[None])[0])

    share_below = (diff <= 0).mean()
    share_above = (diff >= 0).mean()
    p_value = min(1.0, 2 * min(share_below, share_above))

    return {
        "a": stat_a,
        "b": stat_b,
        "diff": stat_a - stat_b,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(p_value),
        "n": len(a),
    }
