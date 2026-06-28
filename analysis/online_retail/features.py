"""Build the supervised feature matrix from the weekly panel.

Features (all strictly causal — computed from data at or before week t):
  - calendar:   week-of-year (sin/cos), month, is_q4 (holiday season)
  - lags:       sold_lag_{1,2,3,4,8,12,52}
  - rolling:    trailing mean/std over 4 and 12 weeks (shifted by 1 to avoid leak)
  - product:    log median price, weeks-since-launch
  - text block: emb_0..emb_{dim-1} from the chosen arm (or omitted for arm 'none')

The target is next-step weekly demand (sold at week t). For multi-step horizons
we forecast recursively / per-step in the evaluation harness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LAGS = [1, 2, 3, 4, 8, 12, 52]
ROLL_WINDOWS = [4, 12]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    wk = df["week"].dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * wk / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * wk / 52.0)
    df["month"] = df["week"].dt.month.astype("int8")
    df["is_q4"] = df["week"].dt.quarter.eq(4).astype("int8")  # Christmas build-up
    return df


def add_lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product causal lag and rolling features."""
    df = df.sort_values(["StockCode", "week"])
    g = df.groupby("StockCode")["sold"]
    for lag in LAGS:
        df[f"sold_lag_{lag}"] = g.shift(lag).astype("float32")
    for w in ROLL_WINDOWS:
        shifted = g.shift(1)  # ensure window ends at t-1, never sees t
        df[f"roll_mean_{w}"] = (shifted.groupby(df["StockCode"])
                                .transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32"))
        df[f"roll_std_{w}"] = (shifted.groupby(df["StockCode"])
                               .transform(lambda s: s.rolling(w, min_periods=1).std()).astype("float32"))
    return df


def add_product_features(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    p = products[["StockCode", "median_price"]].copy()
    p["log_price"] = np.log1p(p["median_price"]).astype("float32")
    df = df.merge(p[["StockCode", "log_price"]], on="StockCode", how="left")
    launch = df.groupby("StockCode")["week"].transform("min")
    df["weeks_since_launch"] = ((df["week"] - launch).dt.days // 7).astype("int16")
    return df


#: feature blocks supplied per-arm carry these prefixes (embeddings or structured)
ARM_FEATURE_PREFIXES = ("emb_", "struct_")


def attach_arm_features(df: pd.DataFrame, arm_feats: pd.DataFrame | None) -> pd.DataFrame:
    """Left-merge an arm's product-level feature block; absent codes get zeros.

    Zero-fill is what makes cold-start safe: a product unseen in training still
    gets a valid (if generic) row instead of NaN.
    """
    if arm_feats is None:
        return df
    arm_feats = arm_feats.copy()
    arm_feats["StockCode"] = arm_feats["StockCode"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df = df.merge(arm_feats, on="StockCode", how="left")
    cols = [c for c in arm_feats.columns
            if c.startswith(ARM_FEATURE_PREFIXES)]
    df[cols] = df[cols].fillna(0.0).astype("float32")
    return df


def build_features(panel: pd.DataFrame, products: pd.DataFrame,
                   arm_feats: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature_frame, feature_column_names).

    Rows with NaN target or missing core lags (warm-up period) are dropped.
    """
    df = panel.copy()
    df = df[df["sold"].notna()]                       # in-life weeks only
    df = add_calendar(df)
    df = add_lags_and_rolls(df)
    df = add_product_features(df, products)
    df = attach_arm_features(df, arm_feats)

    base_cols = (["woy_sin", "woy_cos", "month", "is_q4", "log_price", "weeks_since_launch"]
                 + [f"sold_lag_{l}" for l in LAGS]
                 + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
                 + [f"roll_std_{w}" for w in ROLL_WINDOWS])
    arm_cols = [c for c in df.columns if c.startswith(ARM_FEATURE_PREFIXES)]
    feat_cols = base_cols + arm_cols

    # Require the short lags to be present (drops each product's first 4 weeks).
    df = df.dropna(subset=[f"sold_lag_{l}" for l in [1, 2, 3, 4]])
    df[base_cols] = df[base_cols].fillna(0.0)
    return df, feat_cols
