"""Turn cell results into a leaderboard, and run paired significance between two cells."""
from __future__ import annotations

from .metrics.significance import diebold_mariano

_TEXT = ("model", "regime", "dataset")
_COLS = [("model", "model", 16), ("regime", "regime", 12), ("dataset", "dataset", 10),
         ("MASE", "MASE", 8), ("CRPS", "CRPS", 9),
         ("cost/unit", "cost_per_unit", 10), ("fill", "fill_rate", 8)]


def print_leaderboard(results: list[dict]):
    if not results:
        print("(no results)")
        return
    rows = sorted(results, key=lambda r: r["summary"].get("cost_per_unit", 1e18))
    header = "  ".join(f"{h:<{w}}" if k in _TEXT else f"{h:>{w}}" for h, k, w in _COLS)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        s = r["summary"]
        cells = []
        for h, key, w in _COLS:
            v = r.get(key, s.get(key))
            if key in _TEXT:
                cells.append(f"{str(v):<{w}}")
            else:
                cells.append(f"{v:>{w}.3f}" if isinstance(v, (int, float)) else f"{'':>{w}}")
        print("  ".join(cells))
    print()


def compare(result_a: dict, result_b: dict):
    """Paired significance of two cells on the shared per-series stocking cost.

    Returns mean_diff (a - b; negative => a is cheaper/better), t_stat, p_value, n.
    """
    a, b = result_a["per_series"], result_b["per_series"]
    common = sorted(set(a) & set(b))
    return diebold_mariano([a[s]["cost"] for s in common],
                           [b[s]["cost"] for s in common])
