# analysis

Two separate things.

**`forecast.py`** — what the web app calls behind *Run Forecast*: Chronos-2 zero-shot over
the uploaded daily history, ordering the 0.8 quantile (the newsvendor critical ratio) for
each of the next 28 days. It needs only `chronos-forecasting` from the project's
`requirements.txt`.

**`tsfm_inventory/`** — the thesis benchmark: off-the-shelf time-series foundation models,
zero-shot and fine-tuned, against models a retailer could build, scored on forecast
accuracy and on the inventory decisions those forecasts drive. It has its own environments
and expects raw Kaggle files under `tsfm_inventory/raw/`.

```bash
python -m analysis.tsfm_inventory.run --full    # every model x dataset x regime
python -m analysis.tsfm_inventory.run           # rebuild the report tables from results/
```

See [`tsfm_inventory/README.md`](tsfm_inventory/README.md) for environments, data, protocol
and the full CLI, and `tsfm_inventory/RESULTS.md` for the numbers.
