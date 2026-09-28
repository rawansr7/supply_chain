from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .metrics.significance import paired_bootstrap

METRICS = {
    "MASE": (["MASE"], lambda x: x[..., 0].mean(axis=-1)),
    "MASE_median": (["MASE"], lambda x: np.median(x[..., 0], axis=-1)),
    "cost_per_unit": (["cost", "demand"], lambda x: x[..., 0].sum(axis=-1) / x[..., 1].sum(axis=-1)),
}


def load_results(results_dir):
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))
            if not p.stem.endswith("__smoke")]


def compare(result_a, result_b, metric):
    fields, statistic = METRICS[metric]
    a, b = result_a["per_series"], result_b["per_series"]
    common = sorted(set(a) & set(b))

    def table(per_series):
        rows = [[per_series[s][f] for f in fields] for s in common]
        return np.array(rows, dtype=float).reshape(len(common), len(fields))

    a, b = table(a), table(b)
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return paired_bootstrap(statistic, a[keep], b[keep])


def compute_significance(results):
    rows = []
    for dataset in sorted({r["dataset"] for r in results}):
        cells = sorted((r for r in results if r["dataset"] == dataset),
                       key=lambda r: (r["model"], r["regime"]))
        for cell_a, cell_b in itertools.combinations(cells, 2):
            for metric in METRICS:
                rows.append({"dataset": dataset,
                             "model_a": cell_a["model"], "regime_a": cell_a["regime"],
                             "model_b": cell_b["model"], "regime_b": cell_b["regime"],
                             "metric": metric, **compare(cell_a, cell_b, metric)})
    return pd.DataFrame(rows)


def write_significance(results_dir):
    table = compute_significance(load_results(results_dir))
    path = results_dir / "significance.csv"
    table.to_csv(path, index=False)
    return path, len(table)
