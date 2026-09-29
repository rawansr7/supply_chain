from datetime import timedelta

import pandas as pd

from analysis.tsfm_inventory import config as C
from analysis.tsfm_inventory.models.chronos2 import Chronos2

HORIZON = 28


def _daily_panel(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    all_dates = pd.date_range(df["date"].min(), df["date"].max())
    panel = df.pivot_table(index=["item_id", "store_id"], columns="date", values="sold", aggfunc="sum")
    return panel.reindex(columns=all_dates).fillna(0)


def forecast_next_month(df):
    panel = _daily_panel(df)
    end_date = panel.columns.max()
    quantile_levels = [C.DEFAULT_COSTS.critical_ratio]
    model = Chronos2("zero_shot", HORIZON, quantile_levels)
    model.fit(panel)
    quantiles = model.predict_quantiles(panel)
    orders = quantiles[..., 0]

    forecast_results = []
    for (item_id, store_id), order in zip(panel.index, orders):
        for day, quantity in enumerate(order, start=1):
            forecast_results.append(
                {
                    "store_id": store_id,
                    "item_id": item_id,
                    "forecasted_sold": round(float(quantity)),
                    "forecasted_date": end_date + timedelta(days=day),
                }
            )
    return pd.DataFrame.from_records(forecast_results)
