from __future__ import annotations

import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel, filter_short, on_weekly_grid, sample_series
from .synthetic import make_smoke

SEASONALITY = 52


def load(smoke=False):
    if smoke:
        return make_smoke()
    return Dataset("m5", cached_panel("m5", _build), SEASONALITY, C.COSTS["m5"])


def _build():
    path = C.RAW_DIR / "m5"
    sales = pd.read_csv(path / "sales_train_evaluation.csv")
    calendar = pd.read_csv(path / "calendar.csv")[["d", "wm_yr_wk", "date"]]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    calendar = calendar[calendar["d"].isin(day_cols)]
    # M5 stops on d_1941, a Sunday: 277 whole Sat-Fri weeks plus a 2-day tail. Keep only
    # the weeks the data covers in full.
    days_per_week = calendar.groupby("wm_yr_wk")["d"].size()
    calendar = calendar[calendar["wm_yr_wk"].isin(days_per_week.index[days_per_week == 7])]

    # Sample the series before reshaping: melting all 30k of them costs 59M rows for
    # nothing, and the sample is uniform over the ids either way.
    sales = sales[sales["id"].isin(sample_series(sales["id"]))]

    long = (sales[["id"] + day_cols].melt(id_vars="id", var_name="d", value_name="y")
            .merge(calendar[["d", "wm_yr_wk"]], on="d", how="inner")
            .rename(columns={"id": "series_id"}))
    weekly = long.groupby(["series_id", "wm_yr_wk"], as_index=False)["y"].sum()
    # wm_yr_wk sorts chronologically (YYWW), so it can index the shared calendar directly.
    panel = on_weekly_grid(weekly, "wm_yr_wk", calendar["wm_yr_wk"])
    return filter_short(panel, SEASONALITY + C.HORIZON)
