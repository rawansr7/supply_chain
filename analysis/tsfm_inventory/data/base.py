"""The standard dataset shape every loader returns, plus the train/test split.

A dataset is just a long-format panel with three required columns:

    series_id : str   product/store identifier
    t         : int   integer time index 0,1,2,... per series (weeks)
    y         : float demand in that week

plus any covariate columns listed in `covariate_cols` (e.g. price, promo flag).

Keeping time as an integer index avoids all calendar/date handling and keeps the
rest of the pipeline simple.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config as C


def cached_panel(name: str, build_fn):
    """Build the weekly panel once and cache it as parquet (dtype-safe across envs).

    The real loaders are expensive (Favorita reads ~5 GB); caching lets every model cell
    reuse the same prepared panel instead of rebuilding it. Delete cache/<name>_panel.parquet
    to force a rebuild.
    """
    cache = C.CACHE_DIR / f"{name}_panel.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    panel = build_fn()
    panel.to_parquet(cache, index=False)
    return panel


@dataclass
class Dataset:
    name: str
    panel: pd.DataFrame          # columns: series_id, t, y, [covariates...]
    seasonality: int             # season length m (for MASE and seasonal-naive); weekly -> 52
    covariate_cols: list[str]    # may be empty
    freq: str = "W"


def train_test_split(panel: pd.DataFrame, horizon: int):
    """Last `horizon` weeks of every series = test; everything before = history.

    A single hold-out split (no cross-validation) — simple and enough for the thesis.
    """
    panel = panel.sort_values(["series_id", "t"])
    test = panel.groupby("series_id").tail(horizon)
    train = panel.drop(test.index)
    return train, test


def filter_short(panel: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """Drop series too short to form a history + test split."""
    size = panel.groupby("series_id")["t"].transform("size")
    return panel[size >= min_len].copy()
