from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .base import Forecaster

RATE_LIMIT_SLEEP = 0.35


class TimeGPT(Forecaster):
    name = "timegpt"
    supported_regimes = ["zero_shot"]
    _client = None

    def fit(self, train_panel=None):
        if TimeGPT._client is None:
            from nixtla import NixtlaClient
            TimeGPT._client = NixtlaClient(api_key=os.environ.get("NIXTLA_API_KEY"))
        return self

    def predict_quantiles(self, history):
        df = pd.DataFrame({
            "unique_id": "s",
            "ds": pd.date_range("2000-01-02", periods=len(history), freq="W"),
            "y": np.asarray(history, "float32")})
        res = TimeGPT._client.forecast(df=df, h=self.horizon, quantiles=self.quantile_levels)
        time.sleep(RATE_LIMIT_SLEEP)
        # The API is asked for the exact levels and returns them, but names each column
        # by TRUNCATING the percentage (nixtla_client.py: f"TimeGPT-q-{int(q * 100)}"),
        # so Favorita's critical ratio 0.667 comes back as `TimeGPT-q-66`. Rounding the
        # name to 67 looks for a column that is never there. Unlike TimesFM's decile
        # grid, the values themselves are the levels we asked for.
        return {q: np.clip(res[f"TimeGPT-q-{int(q * 100)}"].to_numpy()[:self.horizon], 0, None)
                for q in self.quantile_levels}
