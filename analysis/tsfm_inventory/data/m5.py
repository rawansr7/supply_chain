from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel, filter_short
from .synthetic import make_smoke

SEASONALITY = 52


def load(smoke=False):
    if smoke:
        return make_smoke()
    return Dataset("m5", cached_panel("m5", _build), SEASONALITY)


def _build():
    path = C.RAW_DIR / "m5"
    sales = pd.read_csv(path / "sales_train_evaluation.csv")
    calendar = pd.read_csv(path / "calendar.csv")[["d", "wm_yr_wk"]]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales[["id"] + day_cols].melt(id_vars="id", var_name="d", value_name="y")
    weekly = (long.merge(calendar, on="d", how="left")
              .groupby(["id", "wm_yr_wk"], as_index=False)["y"].sum()
              .rename(columns={"id": "series_id"}))
    weekly["t"] = weekly.groupby("series_id")["wm_yr_wk"].rank("dense").astype(int) - 1

    top = weekly.groupby("series_id")["y"].sum().nlargest(C.TOP_N_SERIES).index
    panel = weekly[weekly["series_id"].isin(top)][["series_id", "t", "y"]]
    return filter_short(panel, SEASONALITY + C.HORIZON)
