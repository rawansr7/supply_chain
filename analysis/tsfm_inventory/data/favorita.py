from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import (Dataset, cached_panel, filter_short, full_weeks, on_weekly_grid,
                   sample_series)
from .synthetic import make_smoke

SEASONALITY = 52
EPOCH = pd.Timestamp("2013-01-01")


def load(smoke=False):
    if smoke:
        return make_smoke()
    return Dataset("favorita", cached_panel("favorita", _build), SEASONALITY,
                   C.COSTS["favorita"])


def _build():
    # ~125M rows: aggregate chunk by chunk, and build the string series_id only afterwards.
    parts, first_date, last_date = [], None, None
    reader = pd.read_csv(
        C.RAW_DIR / "favorita" / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"},
        chunksize=10_000_000)
    for chunk in reader:
        first_date = min(chunk["date"].min(), first_date or chunk["date"].min())
        last_date = max(chunk["date"].max(), last_date or chunk["date"].max())
        chunk["y"] = chunk["unit_sales"].clip(lower=0)
        chunk["week"] = ((chunk["date"] - EPOCH).dt.days // 7).astype("int32")
        parts.append(chunk.groupby(["store_nbr", "item_nbr", "week"], as_index=False)["y"].sum())

    weekly = (pd.concat(parts, ignore_index=True)
              .groupby(["store_nbr", "item_nbr", "week"], as_index=False)["y"].sum())
    # The file ends on 2017-08-15, a Tuesday, so its final bucket holds one day of seven.
    weekly = weekly[full_weeks(EPOCH + pd.to_timedelta(weekly["week"] * 7, "D"),
                               first_date, last_date)]
    weekly["series_id"] = weekly["store_nbr"].astype(str) + "_" + weekly["item_nbr"].astype(str)
    calendar = weekly["week"].unique()

    weekly = weekly[weekly["series_id"].isin(sample_series(weekly["series_id"]))]
    # A store-item pair only has a row in weeks it sold something, so the sampled series
    # are re-laid on the shared calendar with the silent weeks set to zero demand.
    panel = on_weekly_grid(weekly, "week", calendar)
    return filter_short(panel, SEASONALITY + C.HORIZON)
