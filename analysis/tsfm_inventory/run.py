"""Run experiment cells.

    python -m analysis.tsfm_inventory.run --full
    python -m analysis.tsfm_inventory.run --run chronos2:zero_shot --datasets m5
    python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --smoke
    python -m analysis.tsfm_inventory.run --list
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config as C
from . import report
from .data import LOADERS, THESIS_DATASETS
from .experiment import run_cell
from .models import MODELS, get_model


def build_cells(args):
    if args.full:
        return [(d, m, r) for d in THESIS_DATASETS for m in MODELS
                for r in get_model(m).supported_regimes]

    datasets = args.datasets or ["synthetic"]
    cells = []
    if args.run:
        for token in args.run:
            model, _, regime = token.partition(":")
            if not regime:
                raise SystemExit(f"--run expects model:regime tokens, got {token!r}")
            cells += [(d, model, regime) for d in datasets]
    elif args.models:
        for model in args.models:
            supported = get_model(model).supported_regimes
            for regime in (args.regimes or supported):
                if regime in supported:
                    cells += [(d, model, regime) for d in datasets]
                else:
                    print(f"  skip {model}:{regime} (supports {supported})")
    else:
        raise SystemExit("nothing selected — use --full, --run or --models (see --help)")
    return cells


def main():
    p = argparse.ArgumentParser(description="TSFM-for-inventory experiments")
    p.add_argument("--full", action="store_true", help="every model x dataset x regime")
    p.add_argument("--run", nargs="+", metavar="model:regime", help="explicit model:regime pairs")
    p.add_argument("--models", nargs="+", help="models to run")
    p.add_argument("--regimes", nargs="+", help="regimes to apply to --models")
    p.add_argument("--datasets", nargs="+", help=f"default synthetic; options {list(LOADERS)}")
    p.add_argument("--results-dir", type=Path, help=f"where results land (default {C.RESULTS_DIR})")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic data — just check it runs")
    p.add_argument("--list", action="store_true", help="list available models and datasets")
    args = p.parse_args()

    if args.list:
        print("datasets:", list(LOADERS), " thesis:", THESIS_DATASETS)
        for name, cls in MODELS.items():
            gpu = " [needs GPU]" if cls.needs_gpu else ""
            print(f"  {name:<16} regimes={cls.supported_regimes}{gpu}")
        return

    if args.results_dir:
        C.RESULTS_DIR = args.results_dir
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cells = build_cells(args)
    print(f"\n{len(cells)} cell(s){' [SMOKE]' if args.smoke else ''} -> {C.RESULTS_DIR}")

    results = []
    for dataset, model, regime in cells:
        try:
            res, _ = run_cell(dataset, model, regime, smoke=args.smoke)
            s = res["summary"]
            print(f"  OK  {dataset}/{model}/{regime}: cost/unit={s['cost_per_unit']:.3f} "
                  f"MASE={s['MASE']:.3f} fill={s['fill_rate']:.3f}")
            results.append(res)
        except Exception as e:
            print(f"  ERR {dataset}/{model}/{regime}: {type(e).__name__}: {e}")

    report.print_leaderboard(results)


if __name__ == "__main__":
    main()
