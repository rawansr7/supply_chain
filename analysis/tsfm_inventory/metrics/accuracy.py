"""Forecast-accuracy metrics kept for the thesis: MASE and CRPS.

  MASE  : mean absolute error scaled by a seasonal-naive baseline (comparable across
          products and datasets; 1.0 = as good as naive, < 1.0 = better).
  CRPS  : quality of the whole predictive range, via the average pinball loss over the
          quantile grid (the decision uses the range, not just one number).
"""
from __future__ import annotations

import numpy as np


def mase(y: np.ndarray, f: np.ndarray, history: np.ndarray, m: int) -> float:
    """Mean Absolute Scaled Error: MAE divided by the in-sample seasonal-naive MAE."""
    if len(history) <= m:
        return float("nan")
    scale = np.mean(np.abs(history[m:] - history[:-m]))
    scale = scale if scale > 0 else 1e-8
    return float(np.mean(np.abs(y - f)) / scale)


def crps(y: np.ndarray, quantile_forecast: dict[float, np.ndarray]) -> float:
    """CRPS approximated by the average pinball (quantile) loss over the grid.

    quantile_forecast maps quantile level q -> forecast array (same length as y).
    """
    total = 0.0
    for q, fq in quantile_forecast.items():
        diff = y - fq
        total += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return float(total / len(quantile_forecast))
