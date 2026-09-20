from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel, filter_short
from .synthetic import make_smoke

SEASONALITY = 52
EPOCH = pd.Timestamp("2013-01-01")


def load(smoke=False):
    if smoke:
        return make_smoke()
    return Dataset("favorita", cached_panel("favorita", _build), SEASONALITY)


def _build():
    # ~125M rows: aggregate chunk by chunk, and build the string series_id only afterwards.
    parts = []
    reader = pd.read_csv(
        C.RAW_DIR / "favorita" / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"},
        chunksize=10_000_000)
    for chunk in reader:
        chunk["y"] = chunk["unit_sales"].clip(lower=0)
        chunk["week"] = ((chunk["date"] - EPOCH).dt.days // 7).astype("int32")
        parts.append(chunk.groupby(["store_nbr", "item_nbr", "week"], as_index=False)["y"].sum())

    weekly = (pd.concat(parts, ignore_index=True)
              .groupby(["store_nbr", "item_nbr", "week"], as_index=False)["y"].sum())
    weekly["series_id"] = weekly["store_nbr"].astype(str) + "_" + weekly["item_nbr"].astype(str)
    weekly["t"] = weekly.groupby("series_id")["week"].rank("dense").astype(int) - 1

    top = weekly.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    panel = weekly[weekly["series_id"].isin(top)][["series_id", "t", "y"]]
    return filter_short(panel, SEASONALITY + C.HORIZON)
