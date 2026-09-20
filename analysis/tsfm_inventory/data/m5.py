"""M5 (Walmart) loader -> weekly item-store demand panel.

Download (Kaggle: m5-forecasting-accuracy) and place the CSVs in:
    raw/m5/sales_train_evaluation.csv
    raw/m5/calendar.csv
    raw/m5/sell_prices.csv

We aggregate the daily series to weekly using M5's own `wm_yr_wk` week id, keep the
top-N series by total volume, and attach the weekly mean selling price as a covariate.
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel, filter_short
from .synthetic import make_smoke

COVARIATES = ["sell_price"]
SEASONALITY = 52


def load(smoke: bool = False) -> Dataset:
    if smoke:
        return make_smoke(COVARIATES)
    panel = cached_panel("m5", _build)
    return Dataset("m5", panel, seasonality=SEASONALITY, covariate_cols=COVARIATES)


def _build():
    path = C.RAW_DIR / "m5"
    f = path / "sales_train_evaluation.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"M5 files not found in {path}. Download 'm5-forecasting-accuracy' from "
            "Kaggle and place sales_train_evaluation.csv, calendar.csv, sell_prices.csv there."
        )

    sales = pd.read_csv(f)
    calendar = pd.read_csv(path / "calendar.csv")[["d", "wm_yr_wk"]]
    prices = pd.read_csv(path / "sell_prices.csv")

    id_cols = ["id", "item_id", "store_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="y")
    long = long.merge(calendar, on="d", how="left")

    # weekly demand per series
    weekly = (long.groupby(["id", "item_id", "store_id", "wm_yr_wk"], as_index=False)
              ["y"].sum())
    # weekly price covariate
    weekly = weekly.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    weekly["sell_price"] = weekly.groupby("id")["sell_price"].ffill().bfill().fillna(0.0)

    weekly = weekly.rename(columns={"id": "series_id"})
    weekly["t"] = weekly.groupby("series_id")["wm_yr_wk"].rank("dense").astype(int) - 1

    # keep the top-N series by total volume
    top = weekly.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    weekly = weekly[weekly["series_id"].isin(top)]

    panel = weekly[["series_id", "t", "y"] + COVARIATES]
    return filter_short(panel, min_len=SEASONALITY + C.HORIZON)
