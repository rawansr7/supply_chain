"""Map an ablation arm name -> its product-level feature frame [StockCode, cols...].

  none        -> None (no text features)
  name        -> emb_*    : embedding of the raw product name
  llm         -> emb_*    : embedding of the LLM enriched description
  llm_struct  -> struct_* : one-hot of the LLM's categorical attributes (interpretable)

Keeping this dispatch in one place lets evaluate.py / coldstart.py stay arm-agnostic.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .descriptions import make_llm_descriptions, make_name_descriptions
from .embeddings import embed_descriptions

# Low-cardinality LLM attributes worth one-hot encoding (interpretable arm).
_STRUCT_FIELDS = ["seasonality", "gifting", "perishability", "price_tier", "impulse"]


def struct_features(force: bool = False) -> pd.DataFrame:
    """One-hot encode the LLM's categorical attributes -> struct_* columns."""
    llm = make_llm_descriptions()
    cols = [c for c in _STRUCT_FIELDS if c in llm.columns]
    dummies = pd.get_dummies(llm[cols].astype(str), prefix=[f"struct_{c}" for c in cols])
    dummies = dummies.astype("float32")
    out = pd.concat([llm[["StockCode"]].astype({"StockCode": str}), dummies], axis=1)
    return out


def arm_feature_frame(arm: str, backend: str | None = None, force: bool = False):
    if arm == "none":
        return None
    if arm == "name":
        return embed_descriptions(make_name_descriptions(force=force), "name", backend, force)
    if arm == "llm":
        return embed_descriptions(make_llm_descriptions(), "llm", backend, force)
    if arm == "llm_struct":
        return struct_features(force=force)
    raise ValueError(f"unknown arm: {arm}")
