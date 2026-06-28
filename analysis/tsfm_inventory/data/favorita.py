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
from .base import Dataset, filter_short
from .synthetic import make_smoke

COVARIATES = ["onpromotion"]
SEASONALITY = 52


def load(smoke: bool = False) -> Dataset:
    if smoke:
        return make_smoke(COVARIATES)

    f = C.RAW_DIR / "favorita" / "train.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"Favorita train.csv not found at {f}. Download "
            "'favorita-grocery-sales-forecasting' from Kaggle and place train.csv there."
        )

    df = pd.read_csv(f, parse_dates=["date"],
                     usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"])
    df["unit_sales"] = df["unit_sales"].clip(lower=0)
    df["onpromotion"] = df["onpromotion"].fillna(False).astype(float)
    df["series_id"] = df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    weekly = (df.groupby(["series_id", "week"], as_index=False)
              .agg(y=("unit_sales", "sum"), onpromotion=("onpromotion", "sum")))
    weekly["t"] = weekly.groupby("series_id")["week"].rank("dense").astype(int) - 1

    top = weekly.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    weekly = weekly[weekly["series_id"].isin(top)]

    panel = weekly[["series_id", "t", "y"] + COVARIATES]
    panel = filter_short(panel, min_len=SEASONALITY + C.HORIZON)
    return Dataset("favorita", panel, seasonality=SEASONALITY, covariate_cols=COVARIATES)
