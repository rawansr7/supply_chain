"""(b) Cold-start / new-product experiment with text-vs-temporal ablation.

See COLDSTART.md for the full rationale. Summary:

A "cold" product is forecast having observed NONE of its own sales. We train on
warm products (which have history) and transfer to cold products via static +
calendar + text features only (NO autoregressive lags — a new product has none).

Feature groups (toggleable, to attribute the gain):
  S = static non-text  (log price)
  K = calendar/seasonal (week-of-year sin/cos, month, is_q4)
  T = text             (name / llm embedding, or llm_struct one-hot)

Methods:
  GlobalMean              catalogue-mean level (no signal)            -> floor
  CategoryMean            mean within the product's (LLM) category    -> coarse-text baseline
  LGBM[S]                 price only
  LGBM[S+K]               + seasonality            -> TEMPORAL-ONLY
  LGBM[S+T:arm]           + text                   -> TEXT-ONLY
  LGBM[S+K+T:arm]         text + seasonality       -> FULL
  EmbeddingKNN[arm]       borrow demand level from k nearest text-neighbours

The decisive ablation: FULL must beat both TEMPORAL-ONLY and TEXT-ONLY for the
"text AND temporal" claim to hold. Repeated over several random warm/cold splits;
we report mean +/- std and paired significance on key contrasts.

Run:
    python -m analysis.online_retail.coldstart --n-seeds 5 --cold-frac 0.2 --knn 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from . import config as C
from . import metrics
from .arms import arm_feature_frame
from .data import load_or_build
from .descriptions import make_llm_descriptions
from .features import build_features

# Feature groups available to a cold (history-free) product.
S_COLS = ["log_price"]                                  # static, non-text
K_COLS = ["woy_sin", "woy_cos", "month", "is_q4"]       # calendar / seasonal
TEXT_ARMS = ("name", "llm", "llm_struct")               # arms that supply a T block
EMBED_ARMS = ("name", "llm")                            # arms with a metric embedding (for KNN)

_LGBM = dict(objective="regression", n_estimators=300, learning_rate=0.05,
             num_leaves=31, n_jobs=-1, verbosity=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text_cols(feat_df: pd.DataFrame) -> list[str]:
    from .features import ARM_FEATURE_PREFIXES
    return [c for c in feat_df.columns if c.startswith(ARM_FEATURE_PREFIXES)]


def split_cold(codes, cold_frac, seed):
    rng = np.random.default_rng(seed)
    codes = np.asarray(sorted(codes))
    cold = set(rng.choice(codes, size=int(len(codes) * cold_frac), replace=False))
    warm = set(codes) - cold
    return warm, cold


def _instance_metrics(truth: pd.DataFrame) -> dict:
    """truth has columns y_true, y_pred (order qty = y_pred)."""
    m = {"MAE": metrics.mae(truth["y_true"], truth["y_pred"]),
         "RMSE": metrics.rmse(truth["y_true"], truth["y_pred"])}
    m.update({f"inv_{k}": v for k, v in
              metrics.newsvendor_costs(truth["y_true"], truth["y_pred"]).items()})
    return m


def _per_product_se(truth: pd.DataFrame) -> pd.Series:
    return (truth.assign(se=(truth["y_true"] - truth["y_pred"]) ** 2)
            .groupby("StockCode")["se"].mean())


def embedding_knn_levels(emb: pd.DataFrame, warm_mean: pd.Series, cold, k: int) -> dict:
    emb = emb.copy()
    emb["StockCode"] = emb["StockCode"].astype(str)
    emb = emb.set_index("StockCode")
    cols = [c for c in emb.columns if c.startswith("emb_")]
    warm_codes = [c for c in warm_mean.index if c in emb.index]
    W = emb.loc[warm_codes, cols].values.astype(np.float32)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    vals = warm_mean.loc[warm_codes].values
    preds = {}
    for c in cold:
        if c not in emb.index:
            continue
        v = emb.loc[c, cols].values.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = Wn @ v
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        preds[c] = float(np.mean(vals[idx]))
    return preds


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------
def run_one_seed(seed, panel, feat_by_arm, base_feat, cat_map,
                 cold_frac, k, test_weeks):
    """Return {method_name: per-instance DataFrame[StockCode, week, y_true, y_pred]}."""
    codes = panel["StockCode"].unique()
    warm, cold = split_cold(codes, cold_frac, seed)

    train_mask = (~panel["week"].isin(test_weeks)) & (panel["StockCode"].isin(warm))
    warm_mean = panel[train_mask].groupby("StockCode")["sold"].mean()
    global_mean = float(warm_mean.mean())

    # Cold ground truth (drop pre-launch NaN weeks).
    truth = (panel[panel["StockCode"].isin(cold) & panel["week"].isin(test_weeks)]
             .dropna(subset=["sold"])[["StockCode", "week", "sold"]]
             .rename(columns={"sold": "y_true"}))

    out = {}

    def level_pred(level_map, label):
        t = truth.copy()
        t["y_pred"] = t["StockCode"].map(level_map).fillna(global_mean)
        out[label] = t

    # --- level baselines ---
    level_pred({c: global_mean for c in cold}, "GlobalMean")

    warm_cat = (warm_mean.rename("m").reset_index()
                .assign(cat=lambda d: d["StockCode"].map(cat_map))
                .groupby("cat")["m"].mean())
    cat_level = {c: warm_cat.get(cat_map.get(c), global_mean) for c in cold}
    level_pred(cat_level, "CategoryMean")

    # --- LGBM ablation grid ---
    def lgbm_pred(fdf, cols, label):
        tr_rows = fdf[fdf["StockCode"].isin(warm) & (~fdf["week"].isin(test_weeks))]
        te_rows = fdf[fdf["StockCode"].isin(cold) & (fdf["week"].isin(test_weeks))].copy()
        if te_rows.empty:
            return
        mdl = LGBMRegressor(random_state=seed, **_LGBM).fit(tr_rows[cols], tr_rows["sold"])
        te_rows["y_true"] = te_rows["sold"]
        te_rows["y_pred"] = np.clip(mdl.predict(te_rows[cols]), 0, None)
        out[label] = te_rows[["StockCode", "week", "y_true", "y_pred"]]

    # text-free models (base_feat carries S + K; identical across arms)
    lgbm_pred(base_feat, S_COLS, "LGBM[S]")
    lgbm_pred(base_feat, S_COLS + K_COLS, "LGBM[S+K]")

    # text models, per arm
    for arm in TEXT_ARMS:
        fdf = feat_by_arm[arm]
        tcols = _text_cols(fdf)
        lgbm_pred(fdf, S_COLS + tcols, f"LGBM[S+T:{arm}]")
        lgbm_pred(fdf, S_COLS + K_COLS + tcols, f"LGBM[S+K+T:{arm}]")

    # --- embedding KNN (level transfer) ---
    for arm in EMBED_ARMS:
        emb = arm_feature_frame(arm)
        level_pred(embedding_knn_levels(emb, warm_mean, cold, k), f"EmbeddingKNN[{arm}]")

    return out


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------
#: key contrasts tested for significance (a "better" if mean_diff < 0)
CONTRASTS = [
    ("LGBM[S+K+T:llm]", "LGBM[S+K]"),       # text on top of seasonality
    ("LGBM[S+K+T:llm]", "LGBM[S+T:llm]"),   # seasonality on top of text
    ("LGBM[S+K+T:llm]", "LGBM[S+K+T:name]"),  # llm vs raw name (world knowledge)
    ("EmbeddingKNN[llm]", "EmbeddingKNN[name]"),
    ("LGBM[S+K+T:llm]", "CategoryMean"),    # embedding vs coarse category
]


def run(n_seeds=5, cold_frac=0.2, k=10, n_test_weeks=12, base_seed=C.SEED):
    panel, products = load_or_build()
    panel["StockCode"] = panel["StockCode"].astype(str)
    products["StockCode"] = products["StockCode"].astype(str)
    weeks = np.sort(panel["week"].dropna().unique())
    test_weeks = set(weeks[-n_test_weeks:])

    # Precompute feature frames once (independent of the warm/cold split).
    feat_by_arm = {arm: build_features(panel, products, arm_feature_frame(arm))[0]
                   for arm in TEXT_ARMS}
    base_feat = feat_by_arm["name"]  # carries S + K (identical across arms)

    llm = make_llm_descriptions()
    cat_map = dict(zip(llm["StockCode"].astype(str), llm["category"].astype(str)))

    # Collect per-seed metrics and per-(seed,product) squared errors.
    seed_metrics: dict[str, list[dict]] = {}
    se_by_method: dict[str, dict] = {}   # method -> {(seed, code): mse}
    for s in range(n_seeds):
        seed = base_seed + s
        preds = run_one_seed(seed, panel, feat_by_arm, base_feat, cat_map,
                             cold_frac, k, test_weeks)
        for name, t in preds.items():
            seed_metrics.setdefault(name, []).append(_instance_metrics(t))
            se = _per_product_se(t)
            d = se_by_method.setdefault(name, {})
            for code, v in se.items():
                d[(seed, code)] = v

    # Aggregate: mean +/- std across seeds.
    summary = {}
    for name, rows in seed_metrics.items():
        df = pd.DataFrame(rows)
        summary[name] = {f"{c}_mean": float(df[c].mean()) for c in df.columns}
        summary[name].update({f"{c}_std": float(df[c].std(ddof=0)) for c in df.columns})

    # Significance on key contrasts (paired on common (seed, code) instances).
    sig = {}
    for a, b in CONTRASTS:
        if a not in se_by_method or b not in se_by_method:
            continue
        keys = sorted(set(se_by_method[a]) & set(se_by_method[b]))
        la = np.array([se_by_method[a][kk] for kk in keys])
        lb = np.array([se_by_method[b][kk] for kk in keys])
        sig[f"{a} vs {b}"] = metrics.diebold_mariano(la, lb)

    n_cold = int(len(panel["StockCode"].unique()) * cold_frac)
    report = {
        "config": {"n_seeds": n_seeds, "cold_frac": cold_frac, "k": k,
                   "n_test_weeks": n_test_weeks, "base_seed": base_seed,
                   "n_cold_per_seed": n_cold,
                   "feature_groups": {"S": S_COLS, "K": K_COLS}},
        "summary": summary,
        "significance": sig,
    }
    (C.RESULTS_DIR / "coldstart_ablation.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _print_report(rep):
    c = rep["config"]
    print(f"\n=== Cold-start ablation ({c['n_seeds']} seeds, {c['n_cold_per_seed']} cold/seed, "
          f"cold_frac={c['cold_frac']}, k={c['k']}, last {c['n_test_weeks']} wk) ===")
    df = pd.DataFrame(rep["summary"]).T
    show = [col for col in ["MAE_mean", "MAE_std", "RMSE_mean",
                            "inv_cost_per_unit_demand_mean", "inv_fill_rate_mean"]
            if col in df.columns]
    # stable, readable method order
    order = ["GlobalMean", "CategoryMean", "LGBM[S]", "LGBM[S+K]",
             "LGBM[S+T:name]", "LGBM[S+K+T:name]",
             "LGBM[S+T:llm]", "LGBM[S+K+T:llm]",
             "LGBM[S+T:llm_struct]", "LGBM[S+K+T:llm_struct]",
             "EmbeddingKNN[name]", "EmbeddingKNN[llm]"]
    df = df.reindex([m for m in order if m in df.index])
    print(df[show].round(4).to_string())
    print("\nSignificance (mean_diff<0 => first method better; paired on (seed,product)):")
    for kk, v in rep["significance"].items():
        star = "***" if v["p_value"] < 0.01 else "**" if v["p_value"] < 0.05 else "*" if v["p_value"] < 0.1 else ""
        print(f"  {kk:42s} mean_diff={v['mean_diff']:+.3f}  p={v['p_value']:.4f} {star}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--cold-frac", type=float, default=0.2)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--n-test-weeks", type=int, default=12)
    args = ap.parse_args()
    rep = run(n_seeds=args.n_seeds, cold_frac=args.cold_frac, k=args.knn,
              n_test_weeks=args.n_test_weeks)
    _print_report(rep)
