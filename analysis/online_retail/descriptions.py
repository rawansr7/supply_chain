"""Generate the three description arms used in the ablation.

  arm "none" : no text (handled downstream — no file needed).
  arm "real" : the product's real description from the dataset (lightly normalised).
  arm "llm"  : LLM-generated description from the product NAME + price only.

The key experimental contrast: arm "llm" is produced *as if the real description
did not exist* — the generator is given only the terse stored name and the unit
price. This simulates the sparse-metadata regime and lets us measure how much of
arm "real"'s benefit arm "llm" recovers.

Providers:
  - anthropic / openai : real generation (needs API key in env).
  - offline_template    : deterministic, dependency-free heuristic so the whole
                          pipeline runs without network/keys (clearly labelled as
                          NOT the thesis result — for plumbing/CI only).

Results are cached to CSV so generation runs once.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from . import config as C
from . import prompts
from .data import load_or_build


# ---------------------------------------------------------------------------
# Name descriptions (arm 'name'): the raw terse product name, lightly normalised.
# This is the no-LLM text control. The decisive comparison is llm vs name.
# ---------------------------------------------------------------------------
def make_name_descriptions(force: bool = False) -> pd.DataFrame:
    if C.DESCRIPTIONS_NAME.exists() and not force:
        return pd.read_csv(C.DESCRIPTIONS_NAME)
    _, products = load_or_build()
    out = products[["StockCode", "real_description"]].copy()
    # Light normalisation: title-case the ALL-CAPS dataset text, collapse spaces.
    out["text"] = (out["real_description"].str.title()
                   .str.replace(r"\s+", " ", regex=True).str.strip())
    out = out[["StockCode", "text"]]
    out.to_csv(C.DESCRIPTIONS_NAME, index=False)
    return out


# ---------------------------------------------------------------------------
# LLM descriptions (arm C)
# ---------------------------------------------------------------------------
def _generate_anthropic(name: str, price: float) -> dict:
    import anthropic  # lazy import
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    # Opus 4.8/4.7 reject temperature/top_p/top_k (HTTP 400); we steer for grounded,
    # low-variance output via the prompt instead. Structured outputs guarantee a
    # parseable JSON object and constrain the categorical fields to valid enums.
    msg = client.messages.create(
        model=C.LLM_MODEL,
        max_tokens=600,
        system=prompts.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompts.build_user_prompt(name, price)}],
        output_config={"format": {"type": "json_schema", "schema": prompts.OUTPUT_SCHEMA}},
    )
    text = next(b.text for b in msg.content if b.type == "text")
    return prompts.parse_response(text)


def _generate_openai(name: str, price: float) -> dict:
    from openai import OpenAI  # lazy import (>=1.0 SDK)
    client = OpenAI()  # reads OPENAI_API_KEY from env
    # Structured Outputs (strict json_schema) guarantee a valid JSON object and
    # constrain the categorical fields to valid enums — exactly what llm_struct needs.
    resp = client.chat.completions.create(
        model=C.LLM_MODEL,
        temperature=C.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.build_user_prompt(name, price)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "product_attributes", "strict": True,
                            "schema": prompts.OUTPUT_SCHEMA},
        },
    )
    return prompts.parse_response(resp.choices[0].message.content)


def _generate_offline(name: str, price: float) -> dict:
    """Deterministic heuristic stand-in (NOT a thesis result).

    Lets the full pipeline run end-to-end with no API. It mimics the *shape* of a
    real generation by deriving coarse attributes from keywords in the name and
    the price, so downstream code (embedding, features, eval) is exercised.
    """
    n = name.lower()
    season = "none"
    for kw, s in [("christmas", "christmas"), ("xmas", "christmas"), ("advent", "christmas"),
                  ("heart", "valentine"), ("love", "valentine"),
                  ("easter", "easter"), ("egg", "easter"),
                  ("halloween", "halloween"), ("spooky", "halloween"),
                  ("garden", "spring_summer"), ("beach", "spring_summer")]:
        if kw in n:
            season = s
            break
    gifting = "high" if any(k in n for k in ["gift", "set", "box", "bag"]) else "medium"
    tier = "premium" if price >= 8 else "budget" if price < 2 else "mid"
    impulse = "high" if price < 2 else "low" if price > 10 else "medium"
    return {
        "description": f"A gift-shop item described as {name.title()}.",
        "category": name.split()[-1].lower() if name.split() else "giftware",
        "seasonality": season,
        "gifting": gifting,
        "perishability": "durable",
        "price_tier": tier,
        "impulse": impulse,
        "substitute_group": "_".join(name.lower().split()[-2:]) if name else "misc",
    }


_GENERATORS = {
    "anthropic": _generate_anthropic,
    "openai": _generate_openai,
    "offline_template": _generate_offline,
}


def make_llm_descriptions(provider: str | None = None, force: bool = False,
                          limit: int | None = None) -> pd.DataFrame:
    """Generate (and cache) LLM descriptions for every product.

    Caches incrementally so an interrupted/long run can resume. The cached CSV
    keeps the structured attributes too, for interpretability analysis.
    """
    provider = provider or C.LLM_PROVIDER
    gen = _GENERATORS[provider]
    _, products = load_or_build()

    done = {}
    if C.DESCRIPTIONS_LLM.exists() and not force:
        prev = pd.read_csv(C.DESCRIPTIONS_LLM)
        done = dict(zip(prev["StockCode"].astype(str), prev.to_dict("records")))

    rows = []
    todo = products if limit is None else products.head(limit)
    for i, (_, p) in enumerate(todo.iterrows()):
        code = str(p["StockCode"])
        if code in done:
            rows.append(done[code])
            continue
        attrs = gen(p["real_description"] if provider == "offline_template" else p["real_description"],
                    float(p["median_price"]))
        # NOTE: the generator is intentionally given only NAME+price. We pass the
        # stored name (real_description doubles as the terse name here) but NOT as
        # a "description to embed" — arm C embeds the LLM's *own* output below.
        row = {"StockCode": code, "text": prompts.attributes_to_text(attrs), **attrs}
        rows.append(row)
        if provider in ("anthropic", "openai"):
            time.sleep(0.05)  # gentle rate limiting
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(C.DESCRIPTIONS_LLM, index=False)
            print(f"  generated {i + 1}/{len(todo)}")

    out = pd.DataFrame(rows)
    out.to_csv(C.DESCRIPTIONS_LLM, index=False)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=C.LLM_PROVIDER,
                    choices=list(_GENERATORS), help="generation backend")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name = make_name_descriptions(force=args.force)
    print(f"name descriptions: {len(name)}")
    llm = make_llm_descriptions(provider=args.provider, force=args.force, limit=args.limit)
    print(f"llm descriptions:  {len(llm)} (provider={args.provider})")
    print(llm.head(5).to_string(index=False))
