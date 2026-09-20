# tsfm_inventory — decision-centric TSFM benchmark for inventory

Benchmarks off-the-shelf time-series foundation models (TSFMs) against classical/ML
baselines, scoring them on **both** forecast accuracy **and** the inventory decisions
their forecasts drive. Companion to `../../plan.md` and `../../NOVELTY.md`.

## Install

```bash
# baselines need only numpy/pandas/pyarrow/scipy/lightgbm.
# TSFMs are imported lazily — install only the ones you run (one env per model is easiest):
pip install chronos-forecasting        # chronos2 (fine-tune also needs: pip install peft)
pip install "timesfm[torch]==1.3.0"    # timesfm
pip install nixtla                     # timegpt (also: export NIXTLA_API_KEY=...)
# lag_llama: not on PyPI —
#   git clone https://github.com/time-series-foundation-models/lag-llama
#   pip install -r lag-llama/requirements.txt
#   huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir .
#   run with that clone on PYTHONPATH and lag-llama.ckpt in the working dir
```

## Datasets

Put raw Kaggle files under `raw/`:

```
raw/m5/sales_train_evaluation.csv, calendar.csv
raw/favorita/train.csv
raw/rossmann/train.csv
```

Each loader builds a weekly panel (`series_id, t, y`) of the top-300 series by volume
and caches it as `cache/<name>_panel.parquet`; delete that file to force a rebuild.
You need no download to develop: the `synthetic` dataset and `--smoke` generate data
in memory.

## Running

```bash
# FULL thesis run — every model x dataset x supported regime (24 cells)
python -m analysis.tsfm_inventory.run --full

# SELECTED
python -m analysis.tsfm_inventory.run --run chronos2:zero_shot --datasets m5
python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --datasets m5 favorita

# SMOKE — tiny synthetic data, just checks the code runs (no GPU/network)
python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --smoke

# write elsewhere instead of overwriting the committed thesis results
python -m analysis.tsfm_inventory.run --full --results-dir /tmp/rerun

python -m analysis.tsfm_inventory.run --list
```

A failing cell (missing library, unsupported regime) is reported and skipped — it never
aborts the rest of the run. Each cell writes `<results-dir>/<dataset>__<model>__<regime>.json`.

## Regimes

- `statistical` — the only regime for the baselines (seasonal_naive, moving_average, lightgbm_global).
- `zero_shot` — foundation model used out of the box, on the series' own history.
- `fine_tune` — **chronos2 only** (LoRA, the library's default config). The other three
  foundation models are evaluated off the shelf; see FINETUNING.md for why.

## Layout

```
config.py              paths, seed, horizon, quantiles, inventory costs, smoke sizes
data/<name>.py         one loader per dataset (m5, favorita, rossmann, synthetic)
data/base.py           panel columns, parquet cache, train/test split
models/<name>.py       one model per file; all share models/base.Forecaster
metrics/accuracy.py    MASE, CRPS (mean pinball loss)
metrics/inventory.py   newsvendor order, cost, fill rate
metrics/significance.py paired Diebold-Mariano test
experiment.py          run one cell: forecast -> decision -> metrics -> json
report.py              leaderboard + pairwise significance
run.py                 CLI
```

## Adding things

- **A dataset:** write `data/foo.py` with `load(smoke=False) -> Dataset`, register it in `data/__init__.py`.
- **A model:** write `models/foo.py` subclassing `Forecaster`, register it in `models/__init__.py`.

## Evaluation in one line

Forecast → order the **critical-ratio quantile** (Cu/(Cu+Co) = 0.80) → score by four
metrics: **MASE** and **CRPS** (accuracy), **cost per unit** and **fill rate**
(inventory), plus a paired significance test on per-series cost. See `RESULTS.md` for
the numbers and `../../NOVELTY.md` for the why.
