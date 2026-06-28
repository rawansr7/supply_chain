"""Command-line entry point.

Three ways to run (exactly what the thesis needs):

  (1) FULL thesis run — every model x dataset x supported regime:
        python -m analysis.tsfm_inventory.run --full

  (2) SELECTED combinations — choose models, regimes, datasets:
        # pick a regime per model explicitly:
        python -m analysis.tsfm_inventory.run --run chronos2:zero_shot timegpt:fine_tune --datasets m5
        # or cross models x regimes:
        python -m analysis.tsfm_inventory.run --models seasonal_naive chronos2 --regimes zero_shot --datasets m5 favorita

  (3) SMOKE test — same selection but tiny synthetic data, to check the code runs:
        python -m analysis.tsfm_inventory.run --full --smoke
        python -m analysis.tsfm_inventory.run --models seasonal_naive lightgbm_global --smoke

  List what's available:
        python -m analysis.tsfm_inventory.run --list
"""
from __future__ import annotations

import argparse

from . import report
from .data import LOADERS, THESIS_DATASETS
from .experiment import run_cell
from .models import MODELS, get_model


def build_cells(args):
    if args.full:
        return [(d, m, r) for d in THESIS_DATASETS
                for m in MODELS for r in get_model(m).supported_regimes]

    datasets = args.datasets or ["synthetic"]
    cells = []
    if args.run:
        for tok in args.run:
            model, _, regime = tok.partition(":")
            if not regime:
                raise SystemExit(f"--run expects model:regime tokens, got {tok!r}")
            cells += [(d, model, regime) for d in datasets]
    elif args.models:
        for m in args.models:
            supported = get_model(m).supported_regimes
            for r in (args.regimes or supported):
                if r not in supported:
                    print(f"  skip {m}:{r} (unsupported; {m} supports {supported})")
                    continue
                cells += [(d, m, r) for d in datasets]
    else:
        raise SystemExit("nothing selected — use --full, or --run, or --models (see --help)")
    return cells


def main():
    p = argparse.ArgumentParser(description="TSFM-for-inventory experiments")
    p.add_argument("--full", action="store_true", help="run every model x dataset x regime")
    p.add_argument("--run", nargs="+", metavar="model:regime", help="explicit model:regime pairs")
    p.add_argument("--models", nargs="+", help="models to run")
    p.add_argument("--regimes", nargs="+", help="regimes to apply to --models")
    p.add_argument("--datasets", nargs="+", help=f"datasets (default synthetic); options {list(LOADERS)}")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic data — just check it runs")
    p.add_argument("--list", action="store_true", help="list available models and datasets")
    args = p.parse_args()

    if args.list:
        print("datasets:", list(LOADERS), "  (thesis:", THESIS_DATASETS, ")")
        print("models:")
        for name, cls in MODELS.items():
            gpu = " [needs GPU]" if cls.needs_gpu else ""
            print(f"  {name:<16} regimes={cls.supported_regimes}{gpu}")
        return

    cells = build_cells(args)
    print(f"\nselected {len(cells)} cell(s){' [SMOKE]' if args.smoke else ''}:")
    for d, m, r in cells:
        gpu = " [needs GPU]" if get_model(m).needs_gpu and not args.smoke else ""
        print(f"  {d:<10} {m:<16} {r}{gpu}")
    print()

    results = []
    for d, m, r in cells:
        try:
            res, path = run_cell(d, m, r, smoke=args.smoke)
            s = res["summary"]
            print(f"  OK  {d}/{m}/{r}: cost/unit={s['cost_per_unit']:.3f} "
                  f"MASE={s['MASE']:.3f} fill={s['fill_rate']:.3f}")
            results.append(res)
        except Exception as e:                       # one bad cell never kills the run
            print(f"  ERR {d}/{m}/{r}: {type(e).__name__}: {e}")

    report.print_leaderboard(results)


if __name__ == "__main__":
    main()
