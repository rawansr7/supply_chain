from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C

COLUMNS = ["series_id", "t", "y"]


@dataclass
class Dataset:
    name: str
    panel: pd.DataFrame
    seasonality: int
    costs: C.Costs


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


def sample_series(ids, n=None, seed=None):
    """A uniform random sample of `n` series ids, reproducible from `seed`.

    Sorted first, so the sample depends only on the ids themselves and not on the order
    the loader happened to produce them in.
    """
    ids = np.sort(pd.unique(np.asarray(ids)))
    n = C.N_SERIES if n is None else n
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    return np.sort(rng.choice(ids, size=n, replace=False))


def full_weeks(week_start, first_date, last_date):
    """Mask of the weeks whose full seven days lie inside the data's date range.

    All three datasets start or end mid-week, so the first/last bucket sums only a few
    days and is systematically short. The last one lands inside the test window, which
    would hand every model an unforecastable drop, so those weeks are dropped outright.
    """
    week_start = pd.to_datetime(week_start)
    return (week_start >= first_date) & (week_start + pd.Timedelta(days=6) <= last_date)


def on_weekly_grid(weekly, week_col, weeks=None):
    """Put every series on one shared weekly calendar, `t` counting weeks along it.

    A (series, week) pair with no row means the series recorded no sales that week, so it
    is filled with zero demand. Ranking each series' own weeks instead would silently
    close those gaps and shift the series off the common calendar.

    Pass `weeks` — the dataset's whole calendar, taken before the series were sampled —
    so a week in which none of the sampled series happened to sell still gets a column.
    """
    weeks = np.sort(weekly[week_col].unique() if weeks is None else pd.unique(weeks))
    index = pd.MultiIndex.from_product(
        [np.sort(weekly["series_id"].unique()), weeks], names=["series_id", week_col])
    panel = (weekly.groupby(["series_id", week_col])["y"].sum()
             .reindex(index, fill_value=0.0).reset_index())
    panel["t"] = panel[week_col].map(dict(zip(weeks, range(len(weeks)))))
    return panel[COLUMNS]
