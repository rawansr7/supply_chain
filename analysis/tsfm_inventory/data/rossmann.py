"""Rossmann loader -> weekly per-store sales panel.

Download (Kaggle: rossmann-store-sales) and place:
    raw/rossmann/train.csv     (columns: Store, Date, Sales, Promo, Open, ...)

NOTE: Rossmann is STORE-level, not SKU-level, so the (s,S)/newsvendor decision is a
store-revenue proxy rather than true per-SKU replenishment. Used as a robustness
dataset only; flag this caveat in the write-up.

We aggregate daily Sales to weekly per store, with the weekly promo-day count as a
covariate.
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, filter_short
from .synthetic import make_smoke

COVARIATES = ["promo"]
SEASONALITY = 52


def load(smoke: bool = False) -> Dataset:
    if smoke:
        return make_smoke(COVARIATES)

    f = C.RAW_DIR / "rossmann" / "train.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"Rossmann train.csv not found at {f}. Download 'rossmann-store-sales' "
            "from Kaggle and place train.csv there."
        )

    df = pd.read_csv(f, parse_dates=["Date"], low_memory=False)
    df["series_id"] = df["Store"].astype(str)
    df["week"] = df["Date"].dt.to_period("W").dt.start_time

    weekly = (df.groupby(["series_id", "week"], as_index=False)
              .agg(y=("Sales", "sum"), promo=("Promo", "sum")))
    weekly["t"] = weekly.groupby("series_id")["week"].rank("dense").astype(int) - 1

    top = weekly.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    weekly = weekly[weekly["series_id"].isin(top)]

    panel = weekly[["series_id", "t", "y"] + COVARIATES]
    panel = filter_short(panel, min_len=SEASONALITY + C.HORIZON)
    return Dataset("rossmann", panel, seasonality=SEASONALITY, covariate_cols=COVARIATES)
