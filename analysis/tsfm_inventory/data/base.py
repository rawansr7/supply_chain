from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class Dataset:
    name: str
    panel: pd.DataFrame
    costs: C.Costs

    @property
    def quantile_levels(self):
        return [0.5, self.costs.critical_ratio]


def sample_series(weekly):
    listed = weekly.cumsum(axis=1) > 0
    established = listed.iloc[:, -(C.HORIZON + C.MIN_HISTORY)]

    tuning_start = (C.FOLDS + 1) * C.HORIZON
    recent = weekly.iloc[:, -(tuning_start + C.ACTIVE):-tuning_start]
    active = recent.sum(axis=1) > 0

    keep = (established & active).to_numpy()
    eligible = np.sort(weekly.index[keep])
    rng = np.random.default_rng(C.SEED)
    ids = rng.choice(eligible, C.N_SERIES, replace=False)
    ids = np.sort(ids)

    chosen = weekly.loc[ids]
    return chosen.where(listed.loc[ids])


def cached_panel(name, build_fn):
    path = C.CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        C.CACHE_DIR.mkdir(exist_ok=True)
        weekly = build_fn()
        panel = sample_series(weekly)
        panel = panel.rename(columns=str)
        panel.to_parquet(path)
    panel = pd.read_parquet(path)
    panel.columns = pd.to_datetime(panel.columns)
    return panel
