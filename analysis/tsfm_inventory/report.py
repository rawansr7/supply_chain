import json

import numpy as np

from . import config as C
from .metrics.significance import diebold_mariano, paired_bootstrap

MAKE = ["seasonal_naive", "moving_average", "lightgbm_global", "lstm_global"]
BUY = ["chronos2", "lag_llama", "timegpt"]


def load_results():
    return [json.loads(p.read_text()) for p in sorted(C.RESULTS_DIR.glob("*.json"))]


def label(result):
    return f"{result['model']} {result['regime']}"


def p_value(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def compare(a, b):
    ids = list(a["per_series"])
    cost_a, cost_b, demand = (np.array([r["per_series"][s][k] for s in ids])
                              for r, k in ((a, "cost"), (b, "cost"), (a, "demand")))
    boot = paired_bootstrap(cost_a, cost_b, demand)
    print(f"| {label(a)} | {label(b)} | {a['summary']['cost_per_unit']:.3f} vs {b['summary']['cost_per_unit']:.3f} | "
          f"{boot['diff']:+.3f} ({boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}) | {p_value(boot['p_value'])} | "
          f"{p_value(diebold_mariano(cost_a, cost_b))} |")


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
        print("\n| A | B | cost/unit A vs B | A − B (95% CI) | bootstrap p | DM p |")
        print("|---|---|---|---|---|---|")
        for model in BUY:
            zero, tuned = cells.get((model, "zero_shot")), cells.get((model, "fine_tune"))
            for a, b in ((zero, make), (tuned, zero), (tuned, make)):
                if a and b:
                    compare(a, b)
