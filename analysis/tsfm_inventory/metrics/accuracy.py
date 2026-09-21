from __future__ import annotations

import numpy as np


def mase(y, f, history, m):
    if len(history) <= m:
        return float("nan")
    scale = np.mean(np.abs(history[m:] - history[:-m]))
    if scale <= 0:  # a series flat over every season has no naive error to scale by
        return float("nan")
    return float(np.mean(np.abs(y - f)) / scale)
