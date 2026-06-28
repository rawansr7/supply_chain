"""TimeGPT (Nixtla) adapter — the commercial / paid-API reference model.

Install: pip install nixtla ; set env NIXTLA_API_KEY.
This is the "buy a service" option and the data-governance case study (your demand
data is sent to an external API). Two regimes:
  zero_shot  : forecast from history only
  fine_tune  : forecast with finetune_steps > 0

NOTE: this calls the API once per series (simple but not cheap); batching all series
into one call is a sensible later optimisation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import Forecaster

FINETUNE_STEPS = 10


class TimeGPT(Forecaster):
    name = "timegpt"
    supported_regimes = ["zero_shot", "fine_tune"]
    needs_gpu = False
    _client = None

    def fit(self, train_panel, dataset):
        if TimeGPT._client is None:
            from nixtla import NixtlaClient
            TimeGPT._client = NixtlaClient(api_key=os.environ.get("NIXTLA_API_KEY"))
        return self

    def predict_quantiles(self, history):
        n = len(history)
        dates = pd.date_range("2000-01-02", periods=n, freq="W")
        df = pd.DataFrame({"unique_id": "s", "ds": dates, "y": np.asarray(history, "float32")})

        ft = FINETUNE_STEPS if self.regime == "fine_tune" else 0
        res = TimeGPT._client.forecast(
            df=df, h=self.horizon, quantiles=self.quantile_levels, finetune_steps=ft)

        out = {}
        for lvl in self.quantile_levels:
            col = f"TimeGPT-q-{int(round(lvl * 100))}"
            out[lvl] = np.clip(res[col].to_numpy()[:self.horizon], 0, None)
        return out
