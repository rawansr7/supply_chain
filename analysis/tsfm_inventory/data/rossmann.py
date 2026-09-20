from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel, filter_short
from .synthetic import make_smoke

SEASONALITY = 52


def load(smoke=False):
    if smoke:
        return make_smoke()
    return Dataset("rossmann", cached_panel("rossmann", _build), SEASONALITY)


def _build():
    df = pd.read_csv(C.RAW_DIR / "rossmann" / "train.csv",
                     usecols=["Store", "Date", "Sales"], parse_dates=["Date"])
    df["series_id"] = df["Store"].astype(str)
    df["week"] = df["Date"].dt.to_period("W").dt.start_time

    weekly = (df.groupby(["series_id", "week"], as_index=False)["Sales"].sum()
              .rename(columns={"Sales": "y"}))
    weekly["t"] = weekly.groupby("series_id")["week"].rank("dense").astype(int) - 1

    top = weekly.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    panel = weekly[weekly["series_id"].isin(top)][["series_id", "t", "y"]]
    return filter_short(panel, SEASONALITY + C.HORIZON)
