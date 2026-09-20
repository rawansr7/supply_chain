from __future__ import annotations

from .chronos2 import Chronos2
from .lag_llama import LagLlama
from .lightgbm_global import LightGBMGlobal
from .moving_average import MovingAverage
from .seasonal_naive import SeasonalNaive
from .timegpt import TimeGPT
from .timesfm import TimesFM

MODELS = {cls.name: cls for cls in [SeasonalNaive, MovingAverage, LightGBMGlobal,
                                    Chronos2, TimesFM, LagLlama, TimeGPT]}


def get_model(name):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name]
