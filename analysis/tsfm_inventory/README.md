# tsfm_inventory — decision-centric TSFM benchmark for inventory

Benchmarks off-the-shelf time-series foundation models (TSFMs) against classical/ML
baselines, scoring them on **both** forecast accuracy **and** the inventory decisions
their forecasts drive. Companion to `../../plan.md` and `../../NOVELTY.md`.

## Install

```bash
conda activate supply
# baselines need only numpy/pandas/scipy/lightgbm (already in `supply`).
# TSFMs are optional and imported lazily — install only the ones you run:
pip install chronos-forecasting          # chronos2
pip install timesfm                       # timesfm
pip install nixtla                        # timegpt (also: export NIXTLA_API_KEY=...)
# lag_llama: follow its repo; needs lag-llama.ckpt in the working dir
```

## Datasets

Put raw Kaggle files under `raw/` (loaders tell you the exact filenames if missing):

```
raw/m5/sales_train_evaluation.csv, calendar.csv, sell_prices.csv
raw/favorita/train.csv
raw/rossmann/train.csv
```

You don't need any download to develop: the `synthetic` dataset and `--smoke` mode
generate tiny data in-memory.

## The three ways to run

```bash
# (1) FULL thesis run — every model x dataset x supported regime
python -m analysis.tsfm_inventory.run --full

# (2) SELECTED — pick models / regimes / datasets
python -m analysis.tsfm_inventory.run --run chronos2:zero_shot timegpt:fine_tune --datasets m5
python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --regimes zero_shot --datasets m5 favorita

# (3) SMOKE — same selection, tiny synthetic data, just checks the code runs (no GPU/network)
python -m analysis.tsfm_inventory.run --full --smoke
python -m analysis.tsfm_inventory.run --models seasonal_naive lightgbm_global --smoke

# list what's available
python -m analysis.tsfm_inventory.run --list
```

A failing cell (missing library, unimplemented regime) is reported and skipped — it
never aborts the rest of the run. Each cell writes `results/<dataset>__<model>__<regime>.json`.

## Regimes

- `statistical` — the only regime for the baselines (seasonal_naive, moving_average, lightgbm_global).
- `zero_shot` — foundation model used out of the box, on the series' own history.
- `fine_tune` — foundation model fine-tuned on the dataset first. Implemented for all
  foundation models: **chronos2, timesfm, lag_llama, timegpt**.

See **FINETUNING.md** for the per-model install/setup each fine-tune needs (verified against
each library's official recipe; not yet run — verify on first GPU run). Future work
(out of scope): feeding extra info like price/promotion, and a few-shot / new-product regime.

## Layout (one concern per file)

```
config.py              paths, seed, horizon, quantiles, inventory costs, smoke sizes
data/<name>.py         one loader per dataset (m5, favorita, rossmann, synthetic) -> standard panel
data/base.py           Dataset shape + train/test split
models/<name>.py       one model per file; all share models/base.Forecaster
metrics/accuracy.py    MAE, RMSE, sMAPE, MASE, mean-pinball (~CRPS)
metrics/inventory.py   newsvendor order, costs, service level, fill rate
metrics/significance.py paired Diebold-Mariano test
experiment.py          run one cell: forecast -> decision -> metrics -> json
report.py              leaderboard + pairwise significance
run.py                 CLI (the three modes above)
```

## Adding things

- **A dataset:** write `data/foo.py` with `load(smoke=False) -> Dataset`, register it in `data/__init__.py`.
- **A model:** write `models/foo.py` subclassing `Forecaster`, register it in `models/__init__.py`.

## Evaluation in one line

Forecast → order the **critical-ratio quantile** (Cu/(Cu+Co) = 0.80) → score by the
four kept metrics: **MASE** and **CRPS** (accuracy), **cost per unit** and **fill rate**
(inventory), plus a paired significance test on cost. See `../../NOVELTY.md` for the why.
