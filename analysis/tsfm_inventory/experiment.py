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

    model_class = get_model(model_name)
    model = model_class(regime=regime, horizon=horizon,
                                  quantile_levels=C.QUANTILE_LEVELS,
                                  seasonality=ds.seasonality, smoke=smoke)
    model.fit(train)

    truths = {sid: g["y"].to_numpy(dtype=float) for sid, g in test.groupby("series_id")}
    per_series, mases, orders, demands = {}, [], [], []

    for sid, g in train.groupby("series_id"):
        if sid not in truths:
            continue
        # sid: bchamoun_apple
        # g: panel (series_id, y, t)*272
        history = g["y"].to_numpy(dtype=float)  # lista 272 value
        truth = truths[sid]  # lista 4 values
        quantiles = model.predict_quantiles(history)
        order = inventory.order_from_quantiles(quantiles, ds.costs)

        series_mase = accuracy.mase(truth, quantiles[0.5], history, ds.seasonality)
        series_cost = float(inventory.cost(order, truth, ds.costs).sum())
        series_demand = float(truth.sum())


        per_series[sid] = {
            "cost": series_cost,
            "demand": series_demand,
            "MASE": series_mase,
        }
        mases.append(series_mase)
        orders.append(order)
        demands.append(truth)

    # mases = [0.61, 1.2, 0.9, 0.2, ....] (300)
    # orders = [[120, 111, 135, 244], [532, 2345,224, 666], [6542, 9776, 234, 111], ...] (300)
    # demands = [[332, 653, 222, 654], [532, 2345,224, 666], [6542, 9776, 234, 111], ...] (300)

    order, demand = np.concatenate(orders), np.concatenate(demands)
    # order = [120, 111, 135, 244, 532, 2345,224, 666, 6542, 9776, 234, 111, ...]  (1200)
    # demand = [120, 111, 135, 244, 532, 2345,224, 666, 6542, 9776, 234, 111, ...]  (1200)
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
