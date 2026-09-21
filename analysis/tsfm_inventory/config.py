from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
for _d in (RAW_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

SEED = 51
HORIZON = 4

# Each dataset contributes a uniform random sample of this many series, drawn with SEED.
# Deliberately NOT the top-N by volume: that keeps only the fast movers and throws away
# the intermittency these catalogues are actually made of.
N_SERIES = 300

# Every model returns a full predictive distribution on this grid. The deciles are the
# common reference grid; each dataset's critical ratio is added below so the order
# quantity is a quantile the model really produced rather than the nearest decile to it.
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass(frozen=True)
class Costs:
    """Cost of one unit of overage (holding) vs. one unit of underage (stockout)."""
    holding: float
    stockout: float
    rationale: str

    @property
    def ratio(self):
        return f"{self.stockout:g}:{self.holding:g}"

    @property
    def critical_ratio(self):
        return self.stockout / (self.stockout + self.holding)


# The cost asymmetry is a property of the goods, not a knob to sweep: what a retailer
# loses on an unsold unit versus a missed sale is fixed by what it sells. So each
# dataset gets the one ratio that fits its catalogue. Running several ratios on the same
# dataset would only re-answer "does a higher critical ratio order more?", which needs
# no experiment.
COSTS = {
    "m5": Costs(1.0, 4.0,
                "Walmart FOODS/HOUSEHOLD/HOBBIES SKUs: shelf-stable packaged goods, so "
                "holding is cheap and a stockout costs the lost margin plus the risk the "
                "shopper switches store. 4:1 orders at the 80th percentile."),
    "favorita": Costs(1.0, 2.0,
                      "Ecuadorian grocery dominated by fresh and perishable lines (unit_sales "
                      "is fractional for goods sold by weight). Unsold stock is written off "
                      "within days rather than carried, so overage costs nearly a full unit "
                      "and the policy should be the most cautious of the three."),
    "synthetic": Costs(1.0, 4.0, "Development data — mirrors the M5 setting."),
    "smoke": Costs(1.0, 4.0, "Smoke test — never a result."),
}

QUANTILE_LEVELS = sorted(set(DECILES) | {c.critical_ratio for c in COSTS.values()})

# What the web app stocks against: it takes any retailer's upload and knows nothing about
# what they sell, so it uses the mid-range general-retail setting.
DEFAULT_COSTS = COSTS["m5"]

SMOKE_N_SERIES = 6
SMOKE_LENGTH = 40
SMOKE_SEASONALITY = 4
SMOKE_HORIZON = 4
