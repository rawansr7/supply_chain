from __future__ import annotations

import json

import numpy as np

from . import config as C
from .data import load_dataset, train_test_split
from .metrics import accuracy, inventory
from .models import get_model


def run_cell(dataset_name, model_name, regime, smoke=False):
    ds = load_dataset(dataset_name, smoke=smoke)
    horizon = C.SMOKE_HORIZON if smoke else C.HORIZON
    train, test = train_test_split(ds.panel, horizon)

    model = get_model(model_name)(regime=regime, horizon=horizon,
                                  quantile_levels=C.QUANTILE_LEVELS,
                                  seasonality=ds.seasonality, smoke=smoke)
    model.fit(train)

    truths = {sid: g["y"].to_numpy(dtype=float) for sid, g in test.groupby("series_id")}
    per_series, mases, orders, demands = {}, [], [], []

    for sid, g in train.groupby("series_id"):
        if sid not in truths:
            continue
        history, truth = g["y"].to_numpy(dtype=float), truths[sid]
        quantiles = model.predict_quantiles(history)
        order = inventory.order_from_quantiles(quantiles, ds.costs)

        series_mase = accuracy.mase(truth, quantiles[0.5], history, ds.seasonality)
        # Everything is kept per series. A uniform sample spans several orders of
        # magnitude of demand, so the demand is needed to resample the headline
        # cost-per-unit rather than lean on a paired test the largest series would
        # decide — and the MASE of a barely-moving SKU divides by a near-zero naive
        # error, so the mean of the column needs a median beside it to be read safely.
        per_series[sid] = {
            "cost": float(inventory.cost(order, truth, ds.costs).sum()),
            "demand": float(truth.sum()),
            "MASE": series_mase,
        }
        mases.append(series_mase)
        orders.append(order)
        demands.append(truth)

    order, demand = np.concatenate(orders), np.concatenate(demands)
    result = {
        "dataset": dataset_name, "model": model_name, "regime": regime,
        "smoke": smoke, "horizon": horizon,
        "costs": {"holding": ds.costs.holding, "stockout": ds.costs.stockout,
                  "ratio": ds.costs.ratio, "critical_ratio": ds.costs.critical_ratio},
        "summary": {
            "MASE": float(np.nanmean(mases)),
            "MASE_median": float(np.nanmedian(mases)),
            "cost_per_unit": float(inventory.cost(order, demand, ds.costs).sum() / demand.sum()),
            "fill_rate": inventory.fill_rate(order, demand),
            "n_series": len(per_series),
        },
        "per_series": per_series,
    }

    tag = f"{dataset_name}__{model_name}__{regime}" + ("__smoke" if smoke else "")
    path = C.RESULTS_DIR / f"{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    return result, path
