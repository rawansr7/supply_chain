"""(a) Rigorous evaluation harness: rolling-origin backtest of the 3 arms.

Protocol
--------
  - One-step-ahead forecasts over the last `n_test_weeks` weeks.
  - Expanding window: the model is retrained every `retrain_every` weeks on all
    data strictly before the current test week (no leakage).
  - The SAME features, hyperparameters and tuning are used for every arm; the ONLY
    thing that changes is the text-embedding block (none / real / llm). This is the
    clean ablation the thesis needs.

For each arm we report accuracy (MAE, RMSE, RMSSE), decision-centric inventory
metrics (newsvendor cost, service level, fill rate), and a Diebold-Mariano test
of each text arm against the 'none' baseline on per-product squared-error loss.

Run:
    python -m analysis.online_retail.evaluate
    python -m analysis.online_retail.evaluate --n-test-weeks 12 --retrain-every 4
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from . import metrics
from .arms import arm_feature_frame
from .data import load_or_build
from .features import build_features
from .models import GlobalLGBM, SeasonalNaive


def backtest_arm(arm: str, panel: pd.DataFrame, products: pd.DataFrame,
                 n_test_weeks: int, retrain_every: int) -> pd.DataFrame:
    """Return per (StockCode, week) rows: y_true, y_pred (point), q_order."""
    feat_df, feat_cols = build_features(panel, products, arm_feature_frame(arm))

    weeks = np.sort(feat_df["week"].unique())
    test_weeks = weeks[-n_test_weeks:]

    records = []
    model = None
    for i, wk in enumerate(test_weeks):
        if i % retrain_every == 0:
            train = feat_df[feat_df["week"] < wk]
            model = GlobalLGBM(feat_cols).fit(train)
        test = feat_df[feat_df["week"] == wk]
        if test.empty:
            continue
        rec = test[["StockCode", "week", "sold"]].rename(columns={"sold": "y_true"}).copy()
        rec["y_pred"] = model.predict_point(test)
        rec["q_order"] = model.predict_order(test)
        rec["arm"] = arm
        records.append(rec)
    return pd.concat(records, ignore_index=True)


def baseline_seasonal_naive(panel, products, n_test_weeks) -> pd.DataFrame:
    feat_df, _ = build_features(panel, products, None)
    weeks = np.sort(feat_df["week"].unique())
    test = feat_df[feat_df["week"].isin(weeks[-n_test_weeks:])]
    snaive = SeasonalNaive()
    rec = test[["StockCode", "week", "sold"]].rename(columns={"sold": "y_true"}).copy()
    rec["y_pred"] = snaive.predict_point(test)
    rec["q_order"] = rec["y_pred"]  # naive: order the point forecast
    rec["arm"] = "seasonal_naive"
    return rec


def summarise(preds: pd.DataFrame, train_panel: pd.DataFrame) -> dict:
    out = {
        "MAE": metrics.mae(preds["y_true"], preds["y_pred"]),
        "RMSE": metrics.rmse(preds["y_true"], preds["y_pred"]),
        "RMSSE": metrics.rmsse(preds[["StockCode", "y_true", "y_pred"]], train_panel),
    }
    out.update({f"inv_{k}": v for k, v in
                metrics.newsvendor_costs(preds["y_true"], preds["q_order"]).items()})
    return out


def per_product_sqerr(preds: pd.DataFrame) -> pd.Series:
    """Mean squared error per product (for the DM test)."""
    return (preds.assign(se=(preds["y_true"] - preds["y_pred"]) ** 2)
            .groupby("StockCode")["se"].mean())


def run(n_test_weeks: int = 12, retrain_every: int = 4, arms=C.ARMS) -> dict:
    panel, products = load_or_build()
    weeks = np.sort(panel["week"].dropna().unique())
    train_panel = panel[panel["week"] < weeks[-n_test_weeks]]

    all_preds, summary = {}, {}
    # Baseline first.
    snaive = baseline_seasonal_naive(panel, products, n_test_weeks)
    all_preds["seasonal_naive"] = snaive
    summary["seasonal_naive"] = summarise(snaive, train_panel)

    for arm in arms:
        preds = backtest_arm(arm, panel, products, n_test_weeks, retrain_every)
        all_preds[arm] = preds
        summary[arm] = summarise(preds, train_panel)

    # Significance: each text arm vs the 'none' baseline (lower SE = better).
    sig = {}
    if "none" in all_preds:
        base_se = per_product_sqerr(all_preds["none"])
        for arm in arms:
            if arm == "none":
                continue
            arm_se = per_product_sqerr(all_preds[arm]).reindex(base_se.index)
            sig[f"{arm}_vs_none"] = metrics.diebold_mariano(arm_se.values, base_se.values)

    report = {
        "config": {"n_test_weeks": n_test_weeks, "retrain_every": retrain_every,
                   "test_window": [str(weeks[-n_test_weeks]), str(weeks[-1])],
                   "n_products": int(products.shape[0]),
                   "critical_ratio": metrics.critical_ratio()},
        "summary": summary,
        "significance_vs_none": sig,
    }
    (C.RESULTS_DIR / "backtest_summary.json").write_text(json.dumps(report, indent=2, default=str))
    for arm, p in all_preds.items():
        p.to_parquet(C.RESULTS_DIR / f"preds_{arm}.parquet", index=False)
    return report


def _print_report(rep: dict):
    print(f"\n=== Backtest ({rep['config']['test_window'][0]} -> {rep['config']['test_window'][1]}, "
          f"{rep['config']['n_products']} products, cr={rep['config']['critical_ratio']:.2f}) ===")
    df = pd.DataFrame(rep["summary"]).T
    cols = ["MAE", "RMSE", "RMSSE", "inv_cost_per_unit_demand", "inv_service_level", "inv_fill_rate"]
    print(df[cols].round(4).to_string())
    print("\nSignificance vs 'none' (negative mean_diff => text arm better):")
    for k, v in rep["significance_vs_none"].items():
        star = "***" if v["p_value"] < 0.01 else "**" if v["p_value"] < 0.05 else "*" if v["p_value"] < 0.1 else ""
        print(f"  {k:16s} mean_diff={v['mean_diff']:+.4f}  t={v['t_stat']:+.2f}  p={v['p_value']:.4f} {star}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-weeks", type=int, default=12)
    ap.add_argument("--retrain-every", type=int, default=4)
    args = ap.parse_args()
    rep = run(n_test_weeks=args.n_test_weeks, retrain_every=args.retrain_every)
    _print_report(rep)
