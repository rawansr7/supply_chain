from __future__ import annotations

from . import favorita, m5, rossmann, synthetic
from .base import train_test_split

LOADERS = {"synthetic": synthetic.load, "m5": m5.load,
           "favorita": favorita.load, "rossmann": rossmann.load}
THESIS_DATASETS = ["m5", "favorita", "rossmann"]


def load_dataset(name, smoke=False):
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(LOADERS)}")
    return LOADERS[name](smoke=smoke)
