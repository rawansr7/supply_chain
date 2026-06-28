"""Dataset registry: name -> loader. Add a new dataset by writing one file here."""
from __future__ import annotations

from . import favorita, m5, rossmann, synthetic
from .base import Dataset, train_test_split

LOADERS = {
    "synthetic": synthetic.load,   # no download needed; for development
    "m5": m5.load,
    "favorita": favorita.load,
    "rossmann": rossmann.load,
}

# datasets that make up the full thesis run (synthetic excluded on purpose)
THESIS_DATASETS = ["m5", "favorita", "rossmann"]


def load_dataset(name: str, smoke: bool = False) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(LOADERS)}")
    return LOADERS[name](smoke=smoke)
