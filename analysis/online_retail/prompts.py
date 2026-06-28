"""The LLM description-generation prompt (thesis artifact — keep under version control).

Design principles (these are defensible in the viva):

1. LEAKAGE-FREE. The model sees only *static* product attributes (name, price tier).
   It never sees sales, dates, or anything from the forecast horizon.

2. GROUNDED, NOT HALLUCINATED. It must reason only from the given fields plus
   general world knowledge about what such a product is. It is told not to invent
   specific unverifiable facts (brand claims, materials it cannot know).

3. STRUCTURED + PROSE. We ask for a short prose description AND a set of latent
   attributes (perishability, seasonality, gifting, price tier, usage occasion,
   substitute category). These attributes are the *mechanism* by which the LLM
   injects world knowledge the forecaster cannot learn from a sparse SKU id —
   and they are interpretable, so we can show what the LLM actually contributes.

The output JSON schema is fixed so generation is parseable and reproducible.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a retail product analyst. You are given the limited information a "
    "warehouse system stores about a gift-shop product. Using only that information "
    "and general world knowledge about such products, produce a concise, factual "
    "characterisation useful for demand planning. Do not invent specific facts you "
    "cannot infer (exact brand, exact materials, country of origin). If unsure, "
    "describe the product category generically. Never mention sales, popularity, "
    "dates, or demand."
)

# The latent attributes the LLM must fill — chosen because they plausibly drive
# demand dynamics yet are absent from the raw transactional data.
ATTRIBUTE_SCHEMA = {
    "description": "1-2 sentence plain description of what the product is and its use.",
    "category": "Best-guess product category (e.g. 'kitchenware', 'stationery', 'seasonal decoration').",
    "seasonality": "One of: none | christmas | spring_summer | valentine | halloween | easter.",
    "gifting": "One of: low | medium | high  — how gift-oriented the item is.",
    "perishability": "One of: durable | semi_durable  — gift-shop goods are non-food, so judge fragility/consumability.",
    "price_tier": "One of: budget | mid | premium  — relative to typical gift-shop items, informed by the unit price.",
    "impulse": "One of: low | medium | high  — likelihood of being an unplanned impulse purchase.",
    "substitute_group": "A short tag naming interchangeable items (e.g. 'tea_light_holder', 'gift_bag').",
}


#: JSON-schema for structured outputs (output_config.format). Guarantees a
#: parseable object AND constrains the categorical fields to valid enum values,
#: which is exactly what the llm_struct one-hot arm consumes.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "category": {"type": "string"},
        "seasonality": {"type": "string",
                        "enum": ["none", "christmas", "spring_summer", "valentine", "halloween", "easter"]},
        "gifting": {"type": "string", "enum": ["low", "medium", "high"]},
        "perishability": {"type": "string", "enum": ["durable", "semi_durable"]},
        "price_tier": {"type": "string", "enum": ["budget", "mid", "premium"]},
        "impulse": {"type": "string", "enum": ["low", "medium", "high"]},
        "substitute_group": {"type": "string"},
    },
    "required": ["description", "category", "seasonality", "gifting",
                 "perishability", "price_tier", "impulse", "substitute_group"],
    "additionalProperties": False,
}


def build_user_prompt(name: str, median_price: float) -> str:
    """Render the per-product user message."""
    schema_lines = "\n".join(f'  "{k}": <{v}>' for k, v in ATTRIBUTE_SCHEMA.items())
    return (
        f"Product name (as stored, may be terse/ALL CAPS): {name!r}\n"
        f"Typical unit price (GBP): {median_price:.2f}\n\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        "{\n" + schema_lines + "\n}\n"
    )


def parse_response(text: str) -> dict:
    """Extract the JSON object from a model response, tolerant of code fences."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def attributes_to_text(attrs: dict) -> str:
    """Flatten the structured attributes into one string to embed (arm C)."""
    return (
        f"{attrs.get('description', '')} "
        f"Category: {attrs.get('category', '')}. "
        f"Seasonality: {attrs.get('seasonality', 'none')}. "
        f"Gifting: {attrs.get('gifting', '')}. "
        f"Price tier: {attrs.get('price_tier', '')}. "
        f"Impulse: {attrs.get('impulse', '')}. "
        f"Substitute group: {attrs.get('substitute_group', '')}."
    ).strip()
