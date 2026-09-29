from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class Dataset:
    name: str
    Y: pd.DataFrame
    costs: C.Costs

    @property
    def levels(self):
        return [0.5, self.costs.critical_ratio]


def sample(weekly):
    listed = weekly.cumsum(axis=1) > 0
    established = listed.iloc[:, -(C.HORIZON + C.MIN_HISTORY)]
    tuning_start = (C.FOLDS + 1) * C.HORIZON
    active = weekly.iloc[:, -(tuning_start + C.ACTIVE):-tuning_start].sum(axis=1) > 0
    eligible = weekly.index[(established & active).to_numpy()]
    ids = np.sort(np.random.default_rng(C.SEED).choice(np.sort(eligible), C.N_SERIES, replace=False))
    return weekly.loc[ids].where(listed.loc[ids])


def cached(name, build):
    path = C.CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        C.CACHE_DIR.mkdir(exist_ok=True)
        sample(build()).rename(columns=str).to_parquet(path)
    Y = pd.read_parquet(path)
    Y.columns = pd.to_datetime(Y.columns)
    return Y
