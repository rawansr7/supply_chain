from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config as C

COLUMNS = ["series_id", "t", "y"]


@dataclass
class Dataset:
    name: str
    panel: pd.DataFrame
    seasonality: int


def cached_panel(name, build_fn):
    cache = C.CACHE_DIR / f"{name}_panel.parquet"
    if not cache.exists():
        build_fn().to_parquet(cache, index=False)
    return pd.read_parquet(cache)[COLUMNS]


def train_test_split(panel, horizon):
    panel = panel.sort_values(["series_id", "t"])
    test = panel.groupby("series_id").tail(horizon)
    return panel.drop(test.index), test


def filter_short(panel, min_len):
    size = panel.groupby("series_id")["t"].transform("size")
    return panel[size >= min_len].copy()
