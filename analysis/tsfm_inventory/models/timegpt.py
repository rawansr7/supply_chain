import os

import numpy as np
from nixtla import NixtlaClient

from .base import Forecaster


class TimeGPT(Forecaster):
    grids = {"zero_shot": [{}],
             "fine_tune": [{"finetune_steps": s, "finetune_depth": d}
                           for s in (10, 30, 100, 300, 1000) for d in (1, 2, 3, 4, 5)]}

    def predict(self, Y):
        df = Y.rename_axis(index="unique_id", columns="ds").stack(future_stack=True).dropna().rename("y").reset_index()
        out = NixtlaClient(api_key=os.environ["NIXTLA_API_KEY"], timeout=900, max_wait_time=1800).forecast(
            df=df, h=self.horizon, quantiles=self.levels, **self.params)
        columns = [f"TimeGPT-q-{int(q * 100)}" for q in self.levels]
        Q = out.sort_values(["unique_id", "ds"]).set_index("unique_id").loc[Y.index, columns].to_numpy()
        return np.clip(Q.reshape(len(Y), self.horizon, -1), 0, None)
