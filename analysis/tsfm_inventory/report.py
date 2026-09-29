import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from .metrics.significance import paired_bootstrap

MAKE = ["seasonal_naive", "moving_average", "lightgbm_global", "lstm_global"]
BUY = ["chronos2", "lag_llama", "timegpt"]
METRICS = {
    "MASE": (["MASE"], lambda x: x[..., 0].mean(axis=-1)),
    "MASE_median": (["MASE"], lambda x: np.median(x[..., 0], axis=-1)),
    "cost_per_unit": (["cost", "demand"], lambda x: x[..., 0].sum(axis=-1) / x[..., 1].sum(axis=-1)),
}


def load_results():
    return [json.loads(p.read_text()) for p in sorted(C.RESULTS_DIR.glob("*.json"))]


def compare(result_a, result_b, metric):
    fields, statistic = METRICS[metric]
    ids = sorted(result_a["per_series"])
    a, b = (np.array([[r["per_series"][s][f] for f in fields] for s in ids]) for r in (result_a, result_b))
    return paired_bootstrap(statistic, a, b)


def label(result):
    return f"{result['model']} {result['regime']}"


def p_value(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_markdown(results):
    for dataset in sorted({r["dataset"] for r in results}):
        rows = sorted((r for r in results if r["dataset"] == dataset), key=lambda r: r["summary"]["cost_per_unit"])
        print(f"\n### {dataset} (critical ratio {C.COSTS[dataset].critical_ratio:.3f})\n")
        print("| model | regime | params | MASE mean | MASE median | cost/unit | fill | seconds |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            s = r["summary"]
            print(f"| {r['model']} | {r['regime']} | {r['params']} | {s['MASE']:.3f} | {s['MASE_median']:.3f} | "
                  f"{s['cost_per_unit']:.3f} | {s['fill_rate']:.3f} | {r['seconds']:.0f} |")

        print("\n| model | regime | configs | validation cost/unit (best–worst) | tuning minutes |")
        print("|---|---|---|---|---|")
        for r in rows:
            costs = [t["cost_per_unit"] for t in r["tuning"]]
            print(f"| {r['model']} | {r['regime']} | {len(costs)} | {min(costs):.3f}–{max(costs):.3f} | "
                  f"{sum(t['seconds'] for t in r['tuning']) / 60:.1f} |")

        cells = {(r["model"], r["regime"]): r for r in rows}
        make = min((r for r in rows if r["model"] in MAKE), key=lambda r: r["summary"]["cost_per_unit"], default=None)
        print("\n| A | B | cost/unit A vs B | A − B (95% CI) | bootstrap p |")
        print("|---|---|---|---|---|")
        for model in BUY:
            zero, tuned = cells.get((model, "zero_shot")), cells.get((model, "fine_tune"))
            for a, b in ((zero, make), (tuned, zero), (tuned, make)):
                if a and b:
                    boot = compare(a, b, "cost_per_unit")
                    print(f"| {label(a)} | {label(b)} | {boot['a']:.3f} vs {boot['b']:.3f} | {boot['diff']:+.3f} "
                          f"({boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}) | {p_value(boot['p_value'])} |")


def write_significance(results):
    rows = []
    for dataset in sorted({r["dataset"] for r in results}):
        cells = sorted((r for r in results if r["dataset"] == dataset), key=lambda r: (r["model"], r["regime"]))
        for a, b in itertools.combinations(cells, 2):
            for metric in METRICS:
                rows.append({"dataset": dataset, "model_a": a["model"], "regime_a": a["regime"],
                             "model_b": b["model"], "regime_b": b["regime"], "metric": metric,
                             **compare(a, b, metric)})
    path = C.RESULTS_DIR / "significance.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, len(rows)
