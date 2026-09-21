# tsfm_inventory — decision-centric TSFM benchmark for inventory

Benchmarks off-the-shelf time-series foundation models (TSFMs) against classical/ML
baselines, scoring them on **both** forecast accuracy **and** the inventory decisions
their forecasts drive. Companion to `../../plan.md` and `../../NOVELTY.md`.

## Install

```bash
# baselines need only numpy/pandas/pyarrow/scipy/lightgbm, plus torch for lstm_global.
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
```

Each loader builds a weekly panel (`series_id, t, y`) of **300 series drawn uniformly at
random** (`config.SEED`) and caches it as `cache/<name>_panel.parquet`; delete that file
to force a rebuild. You need no download to develop: the `synthetic` dataset and
`--smoke` generate data in memory.

Three properties of that panel are worth knowing before reading any result:

- **Random, not top-N.** Ranking by volume would keep only the fast movers and discard
  the intermittency these catalogues are made of — the median M5 and Favorita series
  record no sale in 40% and 47% of weeks. The sample is the catalogue, warts included.
- **Whole weeks only.** Both datasets end mid-week, and such a bucket sums a day or two
  rather than seven. It would land inside the four-week test window and hand every model
  an unforecastable drop, so partial weeks are dropped in the loader: M5 loses its 2-day
  tail and Favorita its 1-day tail.
- **One shared calendar.** `t` counts weeks along the dataset's own calendar, not along
  each series' own rows. A week in which a series sold nothing is zero demand, not a
  missing row to be closed up — so every series is aligned and the test window is the
  same four weeks for all of them.

## Running

```bash
# FULL thesis run — every model x dataset x supported regime (18 cells)
python -m analysis.tsfm_inventory.run --full

# SELECTED
python -m analysis.tsfm_inventory.run --run chronos2:zero_shot --datasets m5
python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --datasets m5 favorita

# SMOKE — tiny synthetic data, just checks the code runs (no GPU/network)
python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --smoke

# write elsewhere instead of overwriting the committed thesis results
python -m analysis.tsfm_inventory.run --full --results-dir /tmp/rerun

# rebuild the RESULTS.md tables from the saved cells — runs no model
python -m analysis.tsfm_inventory.run --report

python -m analysis.tsfm_inventory.run --list
```

A failing cell (missing library, unsupported regime) is reported and skipped — it never
aborts the rest of the run. Each cell writes `<results-dir>/<dataset>__<model>__<regime>.json`.

## Regimes

- `statistical` — the "make" side: a model you build and train on your own data
  (seasonal_naive, moving_average, lightgbm_global, lstm_global).
- `zero_shot` — foundation model used out of the box, on the series' own history.
- `fine_tune` — **chronos2 only** (LoRA, the library's default config). The other three
  foundation models are evaluated off the shelf; see FINETUNING.md for why.

## Cost asymmetry

The newsvendor orders the `Cu/(Cu+Co)` quantile, so the cost ratio *is* the decision.
It is a property of the goods rather than a knob to sweep, and each dataset gets the one
ratio that fits its catalogue (`config.COSTS` carries the reasoning):

| dataset | Cu:Co | critical ratio | why |
|---|---|---|---|
| M5 | 4:1 | 0.80 | shelf-stable packaged goods; cheap to hold, a stockout costs the margin |
| Favorita | 2:1 | 0.67 | heavy on fresh/perishable lines — unsold stock is written off, not carried |

Sweeping several ratios on one dataset would only re-answer "does a higher critical
ratio order more?", which needs no experiment.

## Layout

```
config.py              paths, seed, horizon, quantiles, per-dataset costs, smoke sizes
data/<name>.py         one loader per dataset (m5, favorita, synthetic)
data/base.py           panel columns, parquet cache, random sample, weekly grid, split
models/<name>.py       one model per file; all share models/base.Forecaster
metrics/accuracy.py    MASE
metrics/inventory.py   newsvendor order, cost, fill rate
metrics/significance.py paired Diebold-Mariano test, series-level bootstrap
experiment.py          run one cell: forecast -> decision -> metrics -> json
report.py              leaderboard + pairwise significance
run.py                 CLI
```

## Adding things

- **A dataset:** write `data/foo.py` with `load(smoke=False) -> Dataset`, register it in `data/__init__.py`.
- **A model:** write `models/foo.py` subclassing `Forecaster`, register it in `models/__init__.py`.

## Evaluation in one line

Forecast → order the **critical-ratio quantile** for that dataset → score by three
metrics: **MASE** (accuracy, reported as both mean and median), **cost per unit** and
**fill rate** (inventory). Significance comes from a paired Diebold-Mariano test on per-series cost
and a series-level bootstrap of the headline cost per unit — the bootstrap is the one to
quote, since a uniform sample of series spans four orders of magnitude of demand and a
paired test over raw costs is decided by the largest few. See `RESULTS.md` for the
numbers and `../../NOVELTY.md` for the why.
