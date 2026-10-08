import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from .metrics.significance import paired_bootstrap

MAKE_MODELS = ["seasonal_naive", "moving_average", "lightgbm_global", "lstm_global"]
BUY_MODELS = ["chronos2", "lag_llama", "timegpt"]


def mean_mase(x):
    return x[..., 0].mean(axis=-1)


def median_mase(x):
    return np.median(x[..., 0], axis=-1)


def pooled_cost_per_unit(x):
    cost = x[..., 0].sum(axis=-1)
    demand = x[..., 1].sum(axis=-1)
    return cost / demand


METRICS = {
    "MASE": (["MASE"], mean_mase),
    "MASE_median": (["MASE"], median_mase),
    "cost_per_unit": (["cost", "demand"], pooled_cost_per_unit),
}


def load_results():
    paths = sorted(C.RESULTS_DIR.glob("*.json"))
    return [json.loads(p.read_text()) for p in paths]


def per_series_array(result, ids, fields):
    rows = []
    for sid in ids:
        series = result["per_series"][sid]
        rows.append([series[f] for f in fields])
    return np.array(rows)


def compare(result_a, result_b, metric):
    fields, statistic = METRICS[metric]
    ids = sorted(result_a["per_series"])
    a = per_series_array(result_a, ids, fields)
    b = per_series_array(result_b, ids, fields)
    return paired_bootstrap(statistic, a, b)


def label(result):
    return f"{result['model']} {result['regime']}"


def p_value(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def by_cost(result):
    return result["summary"]["cost_per_unit"]


def print_markdown(results):
    datasets = sorted({r["dataset"] for r in results})
    for dataset in datasets:
        rows = [r for r in results if r["dataset"] == dataset]
        rows = sorted(rows, key=by_cost)

        critical_ratio = C.COSTS[dataset].critical_ratio
        print(f"\n### {dataset} (critical ratio {critical_ratio:.3f})\n")
        print("| model | regime | params | MASE mean | MASE median | cost/unit | fill | seconds |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            s = r["summary"]
            print(f"| {r['model']} | {r['regime']} | {r['params']} "
                  f"| {s['MASE']:.3f} | {s['MASE_median']:.3f} "
                  f"| {s['cost_per_unit']:.3f} | {s['fill_rate']:.3f} | {r['seconds']:.0f} |")

        print("\n| model | regime | configs | validation cost/unit (best–worst) | tuning minutes |")
        print("|---|---|---|---|---|")
        for r in rows:
            costs = [t["cost_per_unit"] for t in r["tuning"]]
            minutes = sum(t["seconds"] for t in r["tuning"]) / 60
            print(f"| {r['model']} | {r['regime']} | {len(costs)} "
                  f"| {min(costs):.3f}–{max(costs):.3f} | {minutes:.1f} |")

        cells = {(r["model"], r["regime"]): r for r in rows}
        built = [r for r in rows if r["model"] in MAKE_MODELS]
        best_make = min(built, key=by_cost, default=None)

        print("\n| A | B | cost/unit A vs B | A − B | bootstrap p |")
        print("|---|---|---|---|---|")
        for model in BUY_MODELS:
            zero = cells.get((model, "zero_shot"))
            tuned = cells.get((model, "fine_tune"))
            for a, b in ((zero, best_make), (tuned, zero), (tuned, best_make)):
                if not (a and b):
                    continue
                boot = compare(a, b, "cost_per_unit")
                print(f"| {label(a)} | {label(b)} "
                      f"| {boot['a']:.3f} vs {boot['b']:.3f} "
                      f"| {boot['diff']:+.3f} "
                      f"| {p_value(boot['p_value'])} |")


def write_significance(results):
    rows = []
    datasets = sorted({r["dataset"] for r in results})
    for dataset in datasets:
        cells = [r for r in results if r["dataset"] == dataset]
        cells = sorted(cells, key=lambda r: (r["model"], r["regime"]))
        for a, b in itertools.combinations(cells, 2):
            for metric in METRICS:
                test = compare(a, b, metric)
                rows.append({
                    "dataset": dataset,
                    "model_a": a["model"],
                    "regime_a": a["regime"],
                    "model_b": b["model"],
                    "regime_b": b["regime"],
                    "metric": metric,
                    **test,
                })
    path = C.RESULTS_DIR / "significance.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, len(rows)
