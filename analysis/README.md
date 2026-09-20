# analysis

Two separate things.

**`forecast.py`** — what the web app calls behind *Run Forecast*: Chronos-2 zero-shot over
the uploaded daily history, then the newsvendor rule turns the predictive distribution
into a 28-day order quantity. Not run directly; it needs only `chronos-forecasting` from
the project's `requirements.txt`.

**`tsfm_inventory/`** — the thesis benchmark: off-the-shelf time-series foundation models
against classical baselines, scored on both forecast accuracy and the inventory decisions
those forecasts drive. It has its own requirements (one environment per model) and expects
raw Kaggle files under `tsfm_inventory/raw/`.

```bash
python -m analysis.tsfm_inventory.run --smoke   # tiny synthetic data, no download, no GPU
python -m analysis.tsfm_inventory.run --full    # every model x dataset x regime
```

See [`tsfm_inventory/README.md`](tsfm_inventory/README.md) for install, datasets and the
full CLI, and `tsfm_inventory/RESULTS.md` for the numbers.
