from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
LAG_LLAMA_CKPT = ROOT.parents[1] / "lag-llama.ckpt"

SEED = 51
HORIZON = 4
FOLDS = 2
SEASON = 52
N_SERIES = 1000
MIN_HISTORY = 104
ACTIVE = 13


@dataclass(frozen=True)
class Costs:
    holding: float
    stockout: float

    @property
    def critical_ratio(self):
        return self.stockout / (self.stockout + self.holding)


COSTS = {"m5": Costs(1.0, 4.0), "favorita": Costs(1.0, 2.0)}
DEFAULT_COSTS = COSTS["m5"]
