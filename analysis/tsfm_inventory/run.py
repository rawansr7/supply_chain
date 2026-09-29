import argparse

from . import report
from .data import LOADERS
from .experiment import run_cell
from .models import CELLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run", nargs="+", default=[], metavar="MODEL:REGIME")
    parser.add_argument("--datasets", nargs="+", default=list(LOADERS))
    parser.add_argument("--compute-significance", action="store_true")
    args = parser.parse_args()

    if args.compute_significance:
        path, n_tests = report.write_significance(report.load_results())
        print(f"{n_tests} test(s) -> {path}")
        return

    cells = CELLS if args.full else [tuple(token.split(":")) for token in args.run]
    for dataset in args.datasets:
        for model, regime in cells:
            print(f"{dataset}/{model}/{regime}", flush=True)
            try:
                s = run_cell(dataset, model, regime)["summary"]
                print(f"  cost/unit {s['cost_per_unit']:.3f}  MASE {s['MASE']:.3f}  fill {s['fill_rate']:.3f}", flush=True)
            except Exception as e:
                print(f"  ERROR {type(e).__name__}: {e}", flush=True)
    report.print_markdown(report.load_results())


if __name__ == "__main__":
    main()
