import json
import time

import numpy as np

from . import config as C
from .data import LOADERS
from .metrics import accuracy, inventory
from .models import get_model


def forecast(cls, regime, quantile_levels, params, history):
    start = time.perf_counter()
    model = cls(regime, C.HORIZON, quantile_levels, **params)
    model.fit(history)
    quantiles = model.predict_quantiles(history)
    seconds = time.perf_counter() - start
    return quantiles, seconds


def tune(cls, regime, ds):
    trials = []
    for params in cls.grids[regime]:
        orders = []
        truths = []
        seconds = 0.0
        for fold in range(1, C.FOLDS + 1):
            window_start = -(fold + 1) * C.HORIZON
            window_end = -fold * C.HORIZON
            history = ds.panel.iloc[:, :window_start]
            truth = ds.panel.iloc[:, window_start:window_end].to_numpy()

            quantiles, fold_seconds = forecast(cls, regime, ds.quantile_levels, params, history)
            order = quantiles[..., 1]

            orders.append(order)
            truths.append(truth)
            seconds += fold_seconds

        order = np.hstack(orders)
        demand = np.hstack(truths)
        cost = inventory.cost_per_unit(order, demand, ds.costs)
        trials.append({"params": params, "cost_per_unit": cost, "seconds": seconds})
        print(f"    {params}: validation cost/unit {cost:.4f} ({seconds:.0f}s)", flush=True)
    return trials


def run_cell(dataset_name, model_name, regime):
    ds = LOADERS[dataset_name]()
    cls = get_model(model_name)

    trials = tune(cls, regime, ds)
    best = min(trials, key=lambda t: t["cost_per_unit"])
    params = best["params"]

    history = ds.panel.iloc[:, :-C.HORIZON]
    truth = ds.panel.iloc[:, -C.HORIZON:].to_numpy()
    quantiles, seconds = forecast(cls, regime, ds.quantile_levels, params, history)
    median = quantiles[..., 0]
    order = quantiles[..., 1]

    mases = accuracy.mase(truth, median, history.to_numpy(), C.SEASONALITY)
    series_costs = inventory.cost(order, truth, ds.costs).sum(axis=1)
    demands = truth.sum(axis=1)

    summary = {
        "MASE": float(mases.mean()),
        "MASE_median": float(np.median(mases)),
        "cost_per_unit": inventory.cost_per_unit(order, truth, ds.costs),
        "fill_rate": inventory.fill_rate(order, truth),
        "n_series": len(truth),
    }
    per_series = {}
    for sid, cost, demand, series_mase in zip(ds.panel.index, series_costs, demands, mases):
        per_series[sid] = {"cost": float(cost), "demand": float(demand), "MASE": float(series_mase)}

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "regime": regime,
        "params": params,
        "seconds": seconds,
        "tuning": trials,
        "summary": summary,
        "per_series": per_series,
    }
    C.RESULTS_DIR.mkdir(exist_ok=True)
    path = C.RESULTS_DIR / f"{dataset_name}__{model_name}__{regime}.json"
    path.write_text(json.dumps(result, indent=2))
    return result
