import pandas as pd

from .. import config as C
from .base import Dataset, cached

START = "2011-01-29"


def build():
    sales = pd.read_csv(C.RAW_DIR / "m5" / "sales_train_evaluation.csv", index_col="id").loc[:, "d_1":]
    weeks = sales.shape[1] // 7
    weekly = sales.to_numpy()[:, :7 * weeks].reshape(len(sales), weeks, 7).sum(2)
    return pd.DataFrame(weekly, index=sales.index, dtype=float,
                        columns=pd.date_range(START, periods=weeks, freq="7D"))


def load():
    return Dataset("m5", cached("m5", build), C.COSTS["m5"])
