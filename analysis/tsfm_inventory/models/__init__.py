"""Model registry: name -> class. Add a model by writing one file and listing it here.

Baselines run with no extra deps. TSFM adapters import their library lazily (only
when actually used), so the baselines / smoke test run without any of them installed.
"""
from __future__ import annotations

from .chronos2 import Chronos2
from .lag_llama import LagLlama
from .lightgbm_global import LightGBMGlobal
from .moving_average import MovingAverage
from .seasonal_naive import SeasonalNaive
from .timegpt import TimeGPT
from .timesfm import TimesFM

_CLASSES = [SeasonalNaive, MovingAverage, LightGBMGlobal,
            Chronos2, TimesFM, LagLlama, TimeGPT]
MODELS = {cls.name: cls for cls in _CLASSES}

BASELINES = ["seasonal_naive", "moving_average", "lightgbm_global"]
TSFMS = ["chronos2", "timesfm", "lag_llama", "timegpt"]


def get_model(name):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name]
