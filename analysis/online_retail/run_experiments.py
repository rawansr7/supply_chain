"""End-to-end driver: build data -> descriptions -> embeddings -> (a) backtest + (b) cold-start.

Examples
--------
  # Full offline smoke run (no API keys, deterministic):
  python -m analysis.online_retail.run_experiments --provider offline_template

  # Real thesis run (needs ANTHROPIC_API_KEY; sentence-transformers for embeddings):
  python -m analysis.online_retail.run_experiments --provider anthropic --embedding-backend sentence-transformers
"""
from __future__ import annotations

import argparse

from . import config as C
from . import coldstart, evaluate
from .arms import arm_feature_frame
from .data import load_or_build
from .descriptions import make_llm_descriptions, make_name_descriptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=C.LLM_PROVIDER,
                    choices=["anthropic", "openai", "offline_template"])
    ap.add_argument("--embedding-backend", default=C.EMBEDDING_BACKEND,
                    choices=["auto", "hashing", "sentence-transformers", "openai"])
    ap.add_argument("--n-test-weeks", type=int, default=12)
    ap.add_argument("--retrain-every", type=int, default=4)
    ap.add_argument("--cold-frac", type=float, default=0.2)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(">> 1/4 build/load demand panel")
    panel, products = load_or_build(force=args.force)
    print(f"   {len(products)} products x {panel['week'].nunique()} weeks")

    print(">> 2/4 descriptions (name + llm)")
    make_name_descriptions(force=args.force)
    make_llm_descriptions(provider=args.provider, force=args.force)

    print(">> 3/4 build per-arm feature blocks (embeddings / structured)")
    for arm in C.ARMS:
        arm_feature_frame(arm, backend=args.embedding_backend, force=args.force)

    print(">> 4/4 experiments")
    rep_a = evaluate.run(n_test_weeks=args.n_test_weeks, retrain_every=args.retrain_every)
    evaluate._print_report(rep_a)
    rep_b = coldstart.run(cold_frac=args.cold_frac, k=args.knn, n_test_weeks=args.n_test_weeks)
    coldstart._print_report(rep_b)

    print(f"\nResults written to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
