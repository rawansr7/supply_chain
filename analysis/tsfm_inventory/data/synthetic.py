"""Synthetic data generator.

Two uses:
  1. the `synthetic` dataset (medium size) for developing without any download;
  2. the smoke-mode stand-in: every real loader, when called with smoke=True,
     returns a tiny synthetic panel with that dataset's covariate columns, so the
     full code path is exercised with no files, network, GPU or cloud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .base import Dataset


def make_panel(n_series: int, length: int, seasonality: int,
               covariate_cols: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_series):
        base = rng.uniform(10, 50)
        amp = rng.uniform(0, base * 0.5)
        t = np.arange(length)
        seasonal = amp * np.sin(2 * np.pi * t / seasonality)
        noise = rng.normal(0, base * 0.1, length)
        y = np.clip(base + seasonal + noise, 0, None).round()
        df = pd.DataFrame({"series_id": f"S{s:03d}", "t": t, "y": y})
        for c in covariate_cols:                       # generic numeric covariates
            df[c] = rng.normal(0, 1, length).round(3)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_smoke(covariate_cols: list[str]) -> Dataset:
    """Tiny panel used by every loader in smoke mode."""
    panel = make_panel(C.SMOKE_N_SERIES, C.SMOKE_LENGTH, C.SMOKE_SEASONALITY,
                        covariate_cols, C.SEED)
    return Dataset("smoke", panel, seasonality=C.SMOKE_SEASONALITY,
                   covariate_cols=covariate_cols)


def load(smoke: bool = False) -> Dataset:
    """The `synthetic` dataset (no download needed)."""
    if smoke:
        return make_smoke(["price"])
    panel = make_panel(n_series=30, length=120, seasonality=12,
                       covariate_cols=["price"], seed=C.SEED)
    return Dataset("synthetic", panel, seasonality=12, covariate_cols=["price"])
