from __future__ import annotations

import numpy as np


def mase(y, f, history, m):
    if len(history) <= m:
        return float("nan")
    scale = np.mean(np.abs(history[m:] - history[:-m]))
    scale = scale if scale > 0 else 1e-8
    return float(np.mean(np.abs(y - f)) / scale)


def crps(y, quantile_forecast):
    losses = []
    for q, fq in quantile_forecast.items():
        diff = y - fq
        losses.append(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return float(np.mean(losses))
