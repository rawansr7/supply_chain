import json
import time

import numpy as np

from . import config as C
from .data import LOADERS
from .metrics import accuracy, inventory
from .models import get_model


def forecast(cls, regime, levels, params, history):
    start = time.perf_counter()
    Q = cls(regime, C.HORIZON, levels, **params).fit(history).predict(history)
    return Q, time.perf_counter() - start


def tune(cls, regime, ds):
    trials = []
    for params in cls.grids[regime]:
        orders, truths, seconds = [], [], 0.0
        for fold in range(1, C.FOLDS + 1):
            Q, s = forecast(cls, regime, ds.levels, params, ds.Y.iloc[:, :-(fold + 1) * C.HORIZON])
            orders.append(Q[..., 1])
            truths.append(ds.Y.iloc[:, -(fold + 1) * C.HORIZON:-fold * C.HORIZON].to_numpy())
            seconds += s
        cost = inventory.cost_per_unit(np.hstack(orders), np.hstack(truths), ds.costs)
        trials.append({"params": params, "cost_per_unit": cost, "seconds": seconds})
        print(f"    {params}: validation cost/unit {cost:.4f} ({seconds:.0f}s)", flush=True)
    return trials


def run_cell(dataset, model, regime):
    ds, cls = LOADERS[dataset](), get_model(model)
    trials = tune(cls, regime, ds)
    params = min(trials, key=lambda t: t["cost_per_unit"])["params"]

    history, truth = ds.Y.iloc[:, :-C.HORIZON], ds.Y.iloc[:, -C.HORIZON:].to_numpy()
    Q, seconds = forecast(cls, regime, ds.levels, params, history)
    order = Q[..., 1]
    mase = accuracy.mase(truth, Q[..., 0], history.to_numpy(), C.SEASON)
    cost = inventory.cost(order, truth, ds.costs).sum(1)
    result = {
        "dataset": dataset, "model": model, "regime": regime, "params": params, "seconds": seconds,
        "tuning": trials,
        "summary": {"MASE": float(mase.mean()), "MASE_median": float(np.median(mase)),
                    "cost_per_unit": inventory.cost_per_unit(order, truth, ds.costs),
                    "fill_rate": inventory.fill_rate(order, truth), "n_series": len(truth)},
        "per_series": {sid: {"cost": float(c), "demand": float(d), "MASE": float(m)}
                       for sid, c, d, m in zip(ds.Y.index, cost, truth.sum(1), mase)},
    }
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / f"{dataset}__{model}__{regime}.json").write_text(json.dumps(result, indent=2))
    return result
