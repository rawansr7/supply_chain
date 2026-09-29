import pandas as pd

from .. import config as C
from .base import Dataset, cached

START = pd.Timestamp("2013-01-01")
WEEKS = 241


def build():
    parts = []
    for chunk in pd.read_csv(C.RAW_DIR / "favorita" / "train.csv",
                             usecols=["date", "store_nbr", "item_nbr", "unit_sales"], parse_dates=["date"],
                             dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"},
                             chunksize=10_000_000):
        chunk["week"] = (chunk["date"] - START).dt.days // 7
        chunk["y"] = chunk["unit_sales"].clip(lower=0)
        parts.append(chunk.groupby(["store_nbr", "item_nbr", "week"])["y"].sum())
    weekly = pd.concat(parts).groupby(level=[0, 1, 2]).sum().unstack(fill_value=0).iloc[:, :WEEKS]
    weekly.index = [f"{store}_{item}" for store, item in weekly.index]
    weekly.columns = pd.date_range(START, periods=WEEKS, freq="7D")
    return weekly.astype(float)


def load():
    return Dataset("favorita", cached("favorita", build), C.COSTS["favorita"])
