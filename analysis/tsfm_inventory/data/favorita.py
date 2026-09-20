"""Corporacion Favorita loader -> weekly store-item demand panel.

Download (Kaggle: favorita-grocery-sales-forecasting) and place:
    raw/favorita/train.csv     (columns: id, date, store_nbr, item_nbr, unit_sales, onpromotion)

We aggregate daily unit_sales to weekly per (store, item), keep the top-N series by
volume, and attach the weekly promotion count as a covariate. Negative unit_sales
(returns) are clipped to zero.
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel, filter_short
from .synthetic import make_smoke

COVARIATES = ["onpromotion"]
SEASONALITY = 52


def load(smoke: bool = False) -> Dataset:
    if smoke:
        return make_smoke(COVARIATES)
    panel = cached_panel("favorita", _build)
    return Dataset("favorita", panel, seasonality=SEASONALITY, covariate_cols=COVARIATES)


def _build():
    f = C.RAW_DIR / "favorita" / "train.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"Favorita train.csv not found at {f}. Download "
            "'favorita-grocery-sales-forecasting' from Kaggle and place train.csv there."
        )

    # 4.7 GB / ~125M rows: read in chunks and aggregate to weekly to bound memory.
    # Group by integer keys (store, item, week-number) and build the string series_id only
    # AFTER aggregation (the string id on 125M rows is what exhausts memory). The week
    # number is measured from a fixed epoch so it is consistent across chunks.
    epoch = pd.Timestamp("2013-01-01")
    parts = []
    reader = pd.read_csv(
        f, usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        parse_dates=["date"],
        # explicit dtype for EVERY non-date column: onpromotion is mixed True/False/empty,
        # and letting the chunked C-parser infer it triggers a pandas warning-path bug.
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32",
               "onpromotion": "str"},
        chunksize=10_000_000)
    for ch in reader:
        ch["unit_sales"] = ch["unit_sales"].clip(lower=0)
        ch["onpromotion"] = (ch["onpromotion"] == "True").astype("float32")
        ch["week"] = ((ch["date"] - epoch).dt.days // 7).astype("int32")
        parts.append(ch.groupby(["store_nbr", "item_nbr", "week"], as_index=False)
                     .agg(y=("unit_sales", "sum"), onpromotion=("onpromotion", "sum")))

    w = pd.concat(parts, ignore_index=True)
    w = w.groupby(["store_nbr", "item_nbr", "week"], as_index=False).agg(
        y=("y", "sum"), onpromotion=("onpromotion", "sum"))

    w["series_id"] = w["store_nbr"].astype(str) + "_" + w["item_nbr"].astype(str)
    w["t"] = w.groupby("series_id")["week"].rank("dense").astype(int) - 1

    top = w.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    w = w[w["series_id"].isin(top)]

    panel = w[["series_id", "t", "y"] + COVARIATES]
    return filter_short(panel, min_len=SEASONALITY + C.HORIZON)
