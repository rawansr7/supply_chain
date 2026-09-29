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

    if args.full:
        cells = CELLS
    else:
        cells = [tuple(token.split(":")) for token in args.run]

    for dataset in args.datasets:
        for model, regime in cells:
            print(f"{dataset}/{model}/{regime}", flush=True)
            try:
                result = run_cell(dataset, model, regime)
            except Exception as e:
                print(f"  ERROR {type(e).__name__}: {e}", flush=True)
                continue
            s = result["summary"]
            print(f"  cost/unit {s['cost_per_unit']:.3f}  MASE {s['MASE']:.3f}  fill {s['fill_rate']:.3f}",
                  flush=True)

    results = report.load_results()
    report.print_markdown(results)


if __name__ == "__main__":
    main()
