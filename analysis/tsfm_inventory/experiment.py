"""Run one (dataset, model, regime) cell: forecast -> stocking decision -> metrics.

Pipeline per cell:
  1. load dataset, split off the last `horizon` weeks of each series as the test
  2. fit the model on the history (no-op for zero-shot)
  3. for each series: forecast quantiles -> newsvendor order -> the four kept metrics
  4. save one results json (summary + per-series cost, for significance testing)

The four metrics: MASE, CRPS (accuracy) and cost-per-unit, fill-rate (inventory).
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .data import load_dataset, train_test_split
from .metrics import accuracy as acc
from .metrics import inventory as inv
from .models import get_model


def run_cell(dataset_name: str, model_name: str, regime: str, smoke: bool = False):
    ds = load_dataset(dataset_name, smoke=smoke)
    horizon = C.SMOKE_HORIZON if smoke else C.HORIZON
    train, test = train_test_split(ds.panel, horizon)

    model = get_model(model_name)(
        regime=regime, horizon=horizon,
        quantile_levels=C.QUANTILE_LEVELS, seasonality=ds.seasonality, smoke=smoke)
    model.fit(train, ds)

    test_groups = {sid: g.sort_values("t") for sid, g in test.groupby("series_id")}
    orders, demands = [], []
    per_series, mases, crpss = {}, [], []

    for sid, g in train.groupby("series_id"):
        if sid not in test_groups:
            continue
        g = g.sort_values("t")
        history = g["y"].to_numpy(dtype=float)
        truth = test_groups[sid]["y"].to_numpy(dtype=float)

        qf = model.predict_quantiles(history)
        point = qf[0.5]                                 # median, for MASE
        order = inv.order_from_quantiles(qf)
        cost, _, _ = inv.costs(order, truth)

        per_series[sid] = {"cost": float(cost.sum())}   # headline loss for significance
        mases.append(acc.mase(truth, point, history, ds.seasonality))
        crpss.append(acc.crps(truth, qf))
        orders.append(order); demands.append(truth)

    order = np.concatenate(orders); demand = np.concatenate(demands)
    cost, _, _ = inv.costs(order, demand)
    summary = {
        "MASE": float(np.nanmean(mases)),
        "CRPS": float(np.mean(crpss)),
        "cost_per_unit": float(cost.sum() / demand.sum()) if demand.sum() > 0 else float("nan"),
        "fill_rate": inv.fill_rate(order, demand),
        "n_series": len(per_series),
    }
    result = {"dataset": dataset_name, "model": model_name, "regime": regime,
              "smoke": smoke, "horizon": horizon, "summary": summary, "per_series": per_series}

    tag = f"{dataset_name}__{model_name}__{regime}" + ("__smoke" if smoke else "")
    path = C.RESULTS_DIR / f"{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    return result, path
