from __future__ import annotations

from .metrics.significance import diebold_mariano

_KEYS = [("model", 16), ("regime", 12), ("dataset", 10)]
_METRICS = [("MASE", "MASE", 8), ("CRPS", "CRPS", 9),
            ("cost/unit", "cost_per_unit", 10), ("fill", "fill_rate", 8)]


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


def compare(result_a, result_b):
    a, b = result_a["per_series"], result_b["per_series"]
    common = sorted(set(a) & set(b))
    return diebold_mariano([a[s]["cost"] for s in common], [b[s]["cost"] for s in common])
