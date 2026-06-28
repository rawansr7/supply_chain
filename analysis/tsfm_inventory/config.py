"""Central configuration. Everything the experiments need to be reproducible.

Keep this the single source of truth for paths, costs, horizon and quantiles.
"""
from __future__ import annotations

from pathlib import Path

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"            # put downloaded datasets here (raw/m5, raw/favorita, raw/rossmann)
RESULTS_DIR = ROOT / "results"    # one json per (dataset, model, regime) cell
CACHE_DIR = ROOT / "cache"
for _d in (RAW_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

# --- reproducibility -----------------------------------------------------
SEED = 51

# --- forecast setup ------------------------------------------------------
HORIZON = 4                                   # forecast the next 4 weeks
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOP_N_SERIES = 300                            # cap series per real dataset for tractability

# --- inventory / decision-centric evaluation -----------------------------
HOLDING_COST = 1.0                            # Co: cost of one unit left unsold
STOCKOUT_COST = 4.0                           # Cu: cost of one unit short (lost sale)
CRITICAL_RATIO = STOCKOUT_COST / (STOCKOUT_COST + HOLDING_COST)   # 0.80

# --- smoke mode (tiny, runs in seconds, no downloads/GPU/cloud) -----------
SMOKE_N_SERIES = 6
SMOKE_LENGTH = 40
SMOKE_SEASONALITY = 4
SMOKE_HORIZON = 4
