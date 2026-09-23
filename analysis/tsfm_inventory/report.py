from __future__ import annotations

import json

from .metrics.significance import paired_bootstrap

_KEYS = [("model", 16), ("regime", 12), ("dataset", 10)]
_METRICS = [("MASE", "MASE", 8), ("med", "MASE_median", 7),
            ("cost/unit", "cost_per_unit", 10), ("fill", "fill_rate", 8)]
MAKE_MODELS = ["seasonal_naive", "moving_average", "lightgbm_global", "lstm_global"]


def print_leaderboard(results):
    if not results:
        print("(no results)")
        return
    header = "  ".join([f"{k:<{w}}" for k, w in _KEYS]
                       + [f"{title:>{w}}" for title, _, w in _METRICS])
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["summary"]["cost_per_unit"]):
        row = [f"{r[k]:<{w}}" for k, w in _KEYS]
        row += [f"{r['summary'][key]:>{w}.3f}" for _, key, w in _METRICS]
        print("  ".join(row))
    print()


def _aligned(result_a, result_b):
    a, b = result_a["per_series"], result_b["per_series"]
    common = sorted(set(a) & set(b))
    return ([a[s]["cost"] for s in common], [b[s]["cost"] for s in common],
            [a[s].get("demand", float("nan")) for s in common])


def compare_cost_per_unit(result_a, result_b):
    """Series-level bootstrap of the headline metric — the test the leaderboard needs."""
    return paired_bootstrap(*_aligned(result_a, result_b))


def load_results(results_dir):
    """Every non-smoke cell on disk, so the tables can be rebuilt without re-running."""
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))
            if not p.stem.endswith("__smoke")]


def _cell(results, dataset, model, regime):
    return next((r for r in results if (r["dataset"], r["model"], r["regime"])
                 == (dataset, model, regime)), None)


def print_markdown(results):
    """The RESULTS.md tables, straight from the saved cells."""
    for dataset in sorted({r["dataset"] for r in results}):
        rows = sorted((r for r in results if r["dataset"] == dataset),
                      key=lambda r: r["summary"]["cost_per_unit"])
        costs = rows[0].get("costs", {})
        print(f"\n### {dataset}  (Cu:Co = {costs.get('ratio', '?')}, "
              f"critical ratio {costs.get('critical_ratio', float('nan')):.2f})\n")
        print("| model | regime | MASE (mean) | MASE (median) | cost/unit | fill |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            s = r["summary"]
            print(f"| {r['model']} | {r['regime']} | {s['MASE']:.3f} | "
                  f"{s.get('MASE_median', float('nan')):.3f} | "
                  f"{s['cost_per_unit']:.3f} | {s['fill_rate']:.3f} |")

        best_make = min((r for r in rows if r["model"] in MAKE_MODELS),
                        key=lambda r: r["summary"]["cost_per_unit"], default=None)
        buy = _cell(results, dataset, "chronos2", "zero_shot")
        tuned = _cell(results, dataset, "chronos2", "fine_tune")
        for label, a, b in [("buy vs. make (chronos2 zero-shot vs. best built)", buy, best_make),
                            ("adapt vs. buy (chronos2 fine-tune vs. its zero-shot)", tuned, buy)]:
            if a is None or b is None or a is b:
                continue
            boot = compare_cost_per_unit(a, b)
            if boot["n"] == 0:
                continue
            print(f"\n{label} — vs `{b['model']}`: "
                  f"{a['summary']['cost_per_unit']:.3f} vs {b['summary']['cost_per_unit']:.3f}, "
                  f"diff {boot['diff']:+.3f} "
                  f"(95% CI {boot['ci_lo']:+.3f}..{boot['ci_hi']:+.3f}, "
                  f"bootstrap p={boot['p_value']:.3f})")
        print()
