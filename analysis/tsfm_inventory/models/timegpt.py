import os

import numpy as np
from nixtla import NixtlaClient

from .base import Forecaster


class TimeGPT(Forecaster):
    grids = {"zero_shot": [{}],
             "fine_tune": [{"finetune_steps": s, "finetune_depth": d}
                           for s in (10, 30, 100, 300, 1000)
                           for d in (1, 2, 3, 4, 5)]}

    def predict_quantiles(self, history):
        long = history.rename_axis(index="unique_id", columns="ds")
        long = long.stack(future_stack=True)
        long = long.dropna()
        df = long.rename("y").reset_index()

        client = NixtlaClient(api_key=os.environ["NIXTLA_API_KEY"], timeout=900, max_wait_time=1800)
        res = client.forecast(df=df, h=self.horizon, quantiles=self.quantile_levels, **self.params)

        columns = [f"TimeGPT-q-{int(q * 100)}" for q in self.quantile_levels]
        res = res.sort_values(["unique_id", "ds"])
        res = res.set_index("unique_id")
        quantiles = res.loc[history.index, columns].to_numpy()
        quantiles = quantiles.reshape(len(history), self.horizon, -1)
        return np.clip(quantiles, 0, None)
