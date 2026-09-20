"""Next-month stocking forecast for the web app.

Reuses the thesis benchmark (`analysis.tsfm_inventory`): Chronos-2 zero-shot for the
predictive distribution, then the newsvendor rule to turn it into an order quantity.
"""

from datetime import timedelta

import pandas as pd

from analysis.tsfm_inventory import config as C
from analysis.tsfm_inventory.metrics import inventory as inv
from analysis.tsfm_inventory.models.chronos2 import Chronos2

HORIZON = 28  # days ahead
SEASONALITY = 7  # daily data, weekly season


def _daily_panel(df):
    """One dense daily series per (item, store); days without sales count as zero."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    all_dates = pd.date_range(df["date"].min(), df["date"].max())
    panel = df.pivot_table(index=["item_id", "store_id"], columns="date", values="sold", aggfunc="sum")
    return panel.reindex(columns=all_dates).fillna(0)


def forecast_next_month(df):
    panel = _daily_panel(df)
    end_date = panel.columns.max()

    model = Chronos2(
        regime="zero_shot",
        horizon=HORIZON,
        quantile_levels=C.QUANTILE_LEVELS,
        seasonality=SEASONALITY,
    )
    model.fit()

    forecast_results = []
    for (item_id, store_id), history in panel.iterrows():
        quantiles = model.predict_quantiles(history.to_numpy(dtype=float))
        order = inv.order_from_quantiles(quantiles)
        for day, quantity in enumerate(order, start=1):
            forecast_results.append(
                {
                    "store_id": store_id,
                    "item_id": item_id,
                    "forecasted_sold": max(round(float(quantity)), 0),
                    "forecasted_date": end_date + timedelta(days=day),
                }
            )
    return pd.DataFrame.from_records(forecast_results)
