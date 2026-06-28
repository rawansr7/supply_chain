# Online Retail II — do product-description embeddings improve demand forecasting?

Reproducible study behind the thesis. We isolate the marginal contribution of
**product-description text** to weekly demand-forecast accuracy on a fully public
dataset, and ask whether **LLM-generated** descriptions can stand in for real ones
when real text is sparse/absent — including for **cold-start (new) products** — and
whether any gain survives into an **inventory-cost** evaluation.

## The three ablation arms
Everything is held constant across arms except the text-embedding feature block:

| arm    | text source fed to the embedder                                  |
|--------|------------------------------------------------------------------|
| `none` | no text (pure history/calendar/price baseline)                   |
| `real` | the product's real description from the dataset                  |
| `llm`  | an LLM description generated from the product **name + price only** (leakage-free; simulates sparse metadata) |

The headline question: **how much of `real`'s benefit does `llm` recover?**

## Pipeline / modules
- `config.py`        — all paths, seeds, horizon, costs, model/LLM identifiers.
- `data.py`          — clean Online Retail II → weekly demand panel + products table.
- `prompts.py`       — the LLM generation prompt (system + JSON schema + parser).
- `descriptions.py`  — arm B (real) and arm C (LLM) descriptions, cached to CSV.
- `embeddings.py`    — text → vectors (sentence-transformers / OpenAI / hashing fallback).
- `features.py`      — causal lag/rolling/calendar/price features + embedding block.
- `models.py`        — global LightGBM (point + critical-ratio quantile), seasonal-naive.
- `metrics.py`       — MAE/RMSE/RMSSE, newsvendor inventory costs, Diebold-Mariano.
- `evaluate.py`      — **(a)** rolling-origin backtest of the arms + significance.
- `coldstart.py`     — **(b)** new-product experiment (held-out products, embedding-KNN).
- `run_experiments.py` — one command to run it all.

## Quick start
```bash
conda activate supply
# offline smoke run — no API keys, deterministic (NOT a thesis result):
python -m analysis.online_retail.run_experiments --provider offline_template

# real run:
export ANTHROPIC_API_KEY=...        # for arm 'llm' generation
python -m analysis.online_retail.run_experiments \
    --provider anthropic --embedding-backend sentence-transformers
```
Data downloads once to `data/online_retail_II.xlsx` (UCI, CC BY 4.0). Results land
in `results/` (JSON summaries + per-arm prediction parquets).

## Evaluation
- **Accuracy**: MAE, RMSE, and RMSSE (M5-style per-product scaled error) vs a
  seasonal-naive floor.
- **Decision-centric**: each forecast → a newsvendor order at critical ratio
  `Cu/(Cu+Co)`; we report cost per unit demand, achieved service level, fill rate.
- **Significance**: Diebold-Mariano (paired) of each text arm vs `none`.

## Important caveats (read before citing numbers)
- `offline_template` descriptions and the `hashing` embedding backend are
  **plumbing stand-ins** so the pipeline runs without network/keys. Real results
  require a real LLM provider + a real embedding model.
- Record the exact `LLM_MODEL`, `EMBEDDING_MODEL`, and `SEED` from `config.py`
  with every reported result.
