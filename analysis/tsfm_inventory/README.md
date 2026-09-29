# tsfm_inventory

Decision-centric, make-vs-buy benchmark. Three off-the-shelf time-series foundation models
(Chronos-2, Lag-Llama, TimeGPT), each zero-shot and fine-tuned, against four models a
retailer could build (seasonal naive, moving average, global LightGBM, global LSTM). Every
model is scored on forecast accuracy and on the cost of the newsvendor orders its forecasts
drive. Numbers are in `RESULTS.md`, fine-tuning recipes in `FINETUNING.md`.

## Environments

| models | conda env | extra |
|---|---|---|
| all except `lag_llama` | `tsfm` | torch 2.12 (CUDA 13.0), chronos-forecasting 2.3.0, peft, lightgbm, nixtla |
| `lag_llama` | `tsfm_lag` | `external/lag-llama` on `PYTHONPATH`, `lag-llama.ckpt` in `supply_chain/` |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install chronos-forecasting peft lightgbm nixtla scipy pyarrow
git clone https://github.com/time-series-foundation-models/lag-llama external/lag-llama
pip install -r external/lag-llama/requirements.txt
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir .
```

Chronos-2, Lag-Llama and the LSTM train and predict on the GPU when one is visible (here an
NVIDIA L4, driver 595). LightGBM and the classical baselines run on CPU. TimeGPT runs on
Nixtla's servers and needs `NIXTLA_API_KEY`; all series of a dataset go in one request.

## Data

`raw/m5/sales_train_evaluation.csv` and `raw/favorita/train.csv` (Kaggle). Each loader:

- sums daily unit sales into whole weeks: 277 Walmart Saturday–Friday weeks for M5,
  241 Tuesday–Monday weeks from 2013-01-01 for Favorita. The partial final week is dropped;
- marks the weeks before a series' first sale as missing, not zero: the product was not yet
  on the shelf. Later weeks without a sale are zero demand;
- keeps established, active series: at least 104 weeks of history before the test window
  and at least one sale in the 13 weeks before the first validation window (88% of M5 and
  65% of Favorita store-item series);
- draws 1,000 of them uniformly at random (seed 51) and caches the panel as
  `cache/<dataset>.parquet`. Delete the file to rebuild it.

## Protocol

The horizon is 4 weeks. The last 4 weeks are the test window; the two 4-week windows before
it are validation windows. For every model and regime:

1. each configuration in the model's grid is fit on the history before each validation
   window and forecasts it; its score is the cost per unit pooled over both windows;
2. the best configuration is refit on all history before the test window;
3. it forecasts the test window once. The median gives MASE; the critical-ratio quantile
   is the newsvendor order.

| model | regime | grid |
|---|---|---|
| moving_average | statistical | window 2, 4, 6, 8, 13, 26, 52 |
| lightgbm_global | statistical | num_leaves 15, 63, 255 × learning_rate 0.02, 0.05, 0.1 (500 trees, quantile loss) |
| lstm_global | statistical | hidden 32, 64, 128, 256 × lr 3e-4, 1e-3, 3e-3, 1e-2 (5,000 steps, pinball loss) |
| chronos2 | fine_tune | LoRA lr 1e-5, 1e-4, 1e-3; full lr 1e-6, 1e-5, 3e-5 |
| lag_llama | zero_shot | context 32, 64, 128, 256 |
| lag_llama | fine_tune | context 32, 64, 128 × lr 1e-4, 5e-4 |
| timegpt | fine_tune | finetune_steps 10, 30, 100, 300, 1000 × finetune_depth 1–5 |

Seasonal naive, Chronos-2 zero-shot and TimeGPT zero-shot have nothing to tune: Chronos-2
reads the full history and TimeGPT chooses its own input window.

Costs are a property of the goods, one ratio per dataset: M5 (shelf-stable packaged goods)
stockout:holding 4:1, critical ratio 0.80; Favorita (fresh, perishable grocery) 2:1,
critical ratio 0.67.

## Running

```bash
cd supply_chain
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 NIXTLA_API_KEY=$(cat ../nixtla.key)
CELLS="seasonal_naive:statistical moving_average:statistical lightgbm_global:statistical lstm_global:statistical
       chronos2:zero_shot chronos2:fine_tune timegpt:zero_shot timegpt:fine_tune"
conda run -n tsfm python -m analysis.tsfm_inventory.run --run $CELLS
PYTHONPATH=$PWD:$PWD/external/lag-llama conda run -n tsfm_lag \
    python -m analysis.tsfm_inventory.run --run lag_llama:zero_shot lag_llama:fine_tune
python -m analysis.tsfm_inventory.run --run chronos2:fine_tune --datasets m5
python -m analysis.tsfm_inventory.run
```

`--full` runs every cell in one environment. With no cells, `run` only prints the report
tables from `results/`. Each cell writes `results/<dataset>__<model>__<regime>.json` with
the chosen parameters, every validation trial, the wall time and per-series cost, demand
and MASE. A failing cell is reported and skipped.

## Metrics

- MASE: seasonal (m = 52) scale over the series' own history, reported as mean and median.
- Cost per unit: pooled newsvendor cost divided by pooled demand.
- Fill rate: pooled units served from the order divided by pooled demand.
- Significance: series-level paired bootstrap of cost per unit (10,000 resamples) and a
  paired t-test (Diebold–Mariano) on per-series cost.

## Layout

```
config.py        paths, seed, horizon, sample rules, costs
data/            m5.py, favorita.py, base.py (sampling, cache)
models/          one file per model on models/base.Forecaster: fit(Y) and predict(Y) -> (series, horizon, levels)
metrics/         accuracy, inventory, significance
experiment.py    tune on validation, refit, evaluate on test, write json
report.py        markdown tables and pairwise tests
run.py           CLI
```
