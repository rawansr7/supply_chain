"""Central configuration for the Online Retail II description-embedding study.

Everything that controls the experiment lives here so runs are reproducible:
fixed paths, fixed seeds, the forecast horizon, and the model/LLM identifiers
that must be recorded for the thesis.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(exist_ok=True)

RAW_XLSX = DATA_DIR / "online_retail_II.xlsx"
PANEL_PARQUET = DATA_DIR / "weekly_panel.parquet"           # cleaned weekly demand panel
PRODUCTS_CSV = DATA_DIR / "products.csv"                    # one row per StockCode
DESCRIPTIONS_NAME = DATA_DIR / "descriptions_name.csv"     # arm 'name': raw product name (no LLM)
DESCRIPTIONS_LLM = DATA_DIR / "descriptions_llm.csv"       # arm 'llm': LLM enrichment from name+price
EMBEDDINGS_DIR = CACHE_DIR                                  # embeddings_{arm}.parquet live here

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 51

# ---------------------------------------------------------------------------
# Demand panel construction
# ---------------------------------------------------------------------------
FREQ = "W"            # weekly aggregation (smooths the heavy daily intermittency)
HORIZON = 4           # forecast the next 4 weeks
# Products must have at least this many non-zero weeks to enter the "warm" pool.
MIN_NONZERO_WEEKS = 8
# Keep only the top-N products by total revenue to keep runs tractable & meaningful.
TOP_N_PRODUCTS = 800

# ---------------------------------------------------------------------------
# Ablation arms (everything held constant across arms except this feature block):
#   none        -> no text features (history/calendar/price only)
#   name        -> embedding of the raw terse product NAME (cheap text, no LLM)
#   llm         -> embedding of the LLM's enriched description (world knowledge)
#   llm_struct  -> the LLM's structured attributes as one-hot features (interpretable)
# The decisive comparison is llm vs name: that gap is the LLM world-knowledge gain.
# ---------------------------------------------------------------------------
ARMS = ("none", "name", "llm", "llm_struct")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Provider-agnostic. "hashing" is a dependency-free deterministic fallback so the
# whole pipeline runs offline; "openai" / "sentence-transformers" are the real ones.
EMBEDDING_BACKEND = "auto"      # auto -> sentence-transformers if available else hashing
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 256             # truncate/﻿project to this many dims

# ---------------------------------------------------------------------------
# LLM description generation (record exact identifiers for the thesis)
# ---------------------------------------------------------------------------
LLM_PROVIDER = "openai"         # openai | anthropic | offline_template
LLM_MODEL = "gpt-4o-mini"       # cheap + supports structured outputs; switch to gpt-4o / gpt-4.1 for higher quality
LLM_TEMPERATURE = 0.2           # low: we want grounded, repeatable descriptions

# ---------------------------------------------------------------------------
# Inventory / decision-centric evaluation
# ---------------------------------------------------------------------------
# Newsvendor critical ratio = Cu / (Cu + Co). Holding (overage) and stockout
# (underage) costs per unit per period. Service-level target derives from these.
HOLDING_COST = 1.0              # Co: cost of one unit left unsold per week
STOCKOUT_COST = 4.0            # Cu: cost of one unit of unmet demand (lost sale)
# => critical ratio 4/5 = 0.80 target service level.
