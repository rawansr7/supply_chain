import pandas as pd

from .. import config as C
from .base import Dataset, cached_panel

START = "2011-01-29"


def _build():
    path = C.RAW_DIR / "m5" / "sales_train_evaluation.csv"
    sales = pd.read_csv(path, index_col="id")
    sales = sales.loc[:, "d_1":]

    weeks = sales.shape[1] // 7
    days = sales.to_numpy()[:, :7 * weeks]
    days = days.reshape(len(sales), weeks, 7)
    weekly = days.sum(axis=2)

    columns = pd.date_range(START, periods=weeks, freq="7D")
    return pd.DataFrame(weekly, index=sales.index, columns=columns, dtype=float)


def load():
    return Dataset("m5", cached_panel("m5", _build), C.COSTS["m5"])
