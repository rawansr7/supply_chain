import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel

EPOCH = pd.Timestamp("2013-01-01")
WEEKS = 241


def _build():
    reader = pd.read_csv(
        C.RAW_DIR / "favorita" / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"},
        chunksize=10_000_000,
    )
    parts = []
    for chunk in reader:
        chunk["week"] = (chunk["date"] - EPOCH).dt.days // 7
        chunk["y"] = chunk["unit_sales"].clip(lower=0)
        part = chunk.groupby(["store_nbr", "item_nbr", "week"])["y"].sum()
        parts.append(part)

    totals = pd.concat(parts)
    totals = totals.groupby(level=[0, 1, 2]).sum()
    weekly = totals.unstack(fill_value=0)
    weekly = weekly.iloc[:, :WEEKS]

    weekly.index = [f"{store}_{item}" for store, item in weekly.index]
    weekly.columns = pd.date_range(EPOCH, periods=WEEKS, freq="7D")
    return weekly.astype(float)


def load():
    return Dataset("favorita", cached_panel("favorita", _build), C.COSTS["favorita"])
