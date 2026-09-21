from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .base import Dataset


def make_panel(n_series, length, seasonality, seed):
    rng = np.random.default_rng(seed)
    frames = []
    t = np.arange(length)
    for s in range(n_series):
        base = rng.uniform(10, 50)
        amp = rng.uniform(0, base * 0.5)
        y = base + amp * np.sin(2 * np.pi * t / seasonality) + rng.normal(0, base * 0.1, length)
        frames.append(pd.DataFrame({"series_id": f"S{s:03d}", "t": t,
                                    "y": np.clip(y, 0, None).round()}))
    return pd.concat(frames, ignore_index=True)


def make_smoke():
    panel = make_panel(C.SMOKE_N_SERIES, C.SMOKE_LENGTH, C.SMOKE_SEASONALITY, C.SEED)
    return Dataset("smoke", panel, C.SMOKE_SEASONALITY, C.COSTS["smoke"])


def load(smoke=False):
    if smoke:
        return make_smoke()
    return Dataset("synthetic", make_panel(30, 120, 12, C.SEED), 12, C.COSTS["synthetic"])
