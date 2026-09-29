import numpy as np
from scipy import stats

from .. import config as C

RESAMPLES = 10_000


def diebold_mariano(cost_a, cost_b):
    return float(stats.ttest_rel(cost_a, cost_b).pvalue)


def paired_bootstrap(cost_a, cost_b, demand):
    idx = np.random.default_rng(C.SEED).integers(0, len(demand), (RESAMPLES, len(demand)))
    diff = (cost_a[idx].sum(1) - cost_b[idx].sum(1)) / demand[idx].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"diff": float((cost_a.sum() - cost_b.sum()) / demand.sum()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))}
