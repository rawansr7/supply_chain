"""Load and clean Online Retail II into a weekly demand panel.

Pipeline:
  raw transactions (xlsx, 2 sheets)
    -> clean (drop cancellations, returns, non-product codes, bad rows)
    -> products table (one canonical real description + attributes per StockCode)
    -> weekly demand panel (StockCode x ISO-week, zero-filled over active span)

Run directly to build and cache everything:
    python -m analysis.online_retail.data
"""
from __future__ import annotations

import re
import zipfile

import numpy as np
import pandas as pd

from . import config as C

# StockCodes that are not real sellable products (postage, fees, adjustments...).
_NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "BANK CHARGES", "AMAZONFEE", "S", "CRUK",
    "PADS", "B", "GIFT", "TEST",
}
# A real product code is mostly digits, optionally with a 1-letter suffix (e.g. 85123A).
_PRODUCT_CODE_RE = re.compile(r"^\d{4,6}[A-Za-z]{0,2}$")


def _read_raw() -> pd.DataFrame:
    """Read both sheets of the Online Retail II workbook (zip or xlsx)."""
    path = C.RAW_XLSX
    # The UCI download is a zip; transparently extract the xlsx if needed.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
            with zf.open(name) as fh:
                xls = pd.ExcelFile(fh, engine="openpyxl")
                frames = [xls.parse(s) for s in xls.sheet_names]
    else:
        xls = pd.ExcelFile(path, engine="openpyxl")
        frames = [xls.parse(s) for s in xls.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    # Normalise column names across the two sheets.
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "CustomerID": "Customer ID"})
    return df


def _is_product_code(code: str) -> bool:
    code = str(code).strip()
    if code in _NON_PRODUCT_CODES:
        return False
    return bool(_PRODUCT_CODE_RE.match(code))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancellations, returns, non-products and malformed rows."""
    df = df.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["Description"] = df["Description"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df = df[~df["InvoiceNo"].str.startswith("C")]          # cancellations
    df = df[df["Quantity"] > 0]                             # returns / corrections
    df = df[df["UnitPrice"] > 0]                            # freebies / adjustments
    df = df[df["StockCode"].map(_is_product_code)]          # real products only
    df = df[df["Description"].notna() & (df["Description"] != "nan") & (df["Description"] != "")]
    return df


def build_products(clean_df: pd.DataFrame) -> pd.DataFrame:
    """One row per StockCode: canonical real description + simple attributes."""
    g = clean_df.groupby("StockCode")
    # Canonical description = the most frequent (mode) description for that code.
    desc = g["Description"].agg(lambda s: s.value_counts().idxmax())
    products = pd.DataFrame({
        "StockCode": desc.index,
        "real_description": desc.values,
        "median_price": g["UnitPrice"].median().values,
        "total_qty": g["Quantity"].sum().values,
        "total_revenue": (clean_df.assign(rev=clean_df["Quantity"] * clean_df["UnitPrice"])
                          .groupby("StockCode")["rev"].sum().values),
        "n_invoices": g["InvoiceNo"].nunique().values,
        "first_seen": g["InvoiceDate"].min().values,
        "last_seen": g["InvoiceDate"].max().values,
    }).reset_index(drop=True)
    return products


def build_weekly_panel(clean_df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Zero-filled weekly demand per product over the global active span.

    Each product's series spans from the global first week to the global last
    week, with weeks before its own first sale left as NaN (not yet launched) and
    in-life gaps filled with 0 (no sale that week).
    """
    df = clean_df.copy()
    # ISO week start (Monday) as the period key.
    df["week"] = df["InvoiceDate"].dt.to_period(C.FREQ).dt.start_time
    weekly = (df.groupby(["StockCode", "week"])["Quantity"].sum()
              .rename("sold").reset_index())

    all_weeks = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq=C.FREQ + "-MON")
    codes = products["StockCode"].unique()
    full_index = pd.MultiIndex.from_product([codes, all_weeks], names=["StockCode", "week"])
    panel = weekly.set_index(["StockCode", "week"]).reindex(full_index)

    # Mark each product's launch week; weeks before launch stay NaN, in-life -> 0.
    first_week = (weekly.groupby("StockCode")["week"].min().reindex(codes))
    panel = panel.reset_index()
    panel = panel.merge(first_week.rename("launch_week"), on="StockCode")
    in_life = panel["week"] >= panel["launch_week"]
    panel.loc[in_life, "sold"] = panel.loc[in_life, "sold"].fillna(0)
    panel["sold"] = panel["sold"].astype("float32")
    panel = panel.drop(columns="launch_week")
    return panel


def load_or_build(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly_panel, products), building & caching them on first call."""
    if (not force) and C.PANEL_PARQUET.exists() and C.PRODUCTS_CSV.exists():
        return pd.read_parquet(C.PANEL_PARQUET), pd.read_csv(C.PRODUCTS_CSV)

    raw = _read_raw()
    clean_df = clean(raw)
    products = build_products(clean_df)

    # Keep the top-N products by revenue (tractable, and the long tail is mostly noise).
    products = products.sort_values("total_revenue", ascending=False).head(C.TOP_N_PRODUCTS)
    clean_df = clean_df[clean_df["StockCode"].isin(products["StockCode"])]

    panel = build_weekly_panel(clean_df, products)

    # Drop products without enough signal to forecast at all.
    nonzero = panel.assign(nz=panel["sold"] > 0).groupby("StockCode")["nz"].sum()
    keep = nonzero[nonzero >= C.MIN_NONZERO_WEEKS].index
    panel = panel[panel["StockCode"].isin(keep)]
    products = products[products["StockCode"].isin(keep)].reset_index(drop=True)

    panel.to_parquet(C.PANEL_PARQUET, index=False)
    products.to_csv(C.PRODUCTS_CSV, index=False)
    return panel, products


if __name__ == "__main__":
    panel, products = load_or_build(force=True)
    n_weeks = panel["week"].nunique()
    print(f"Products: {len(products):,}")
    print(f"Weeks:    {n_weeks} ({panel['week'].min().date()} -> {panel['week'].max().date()})")
    print(f"Panel rows: {len(panel):,}")
    nz = (panel["sold"] > 0).mean()
    print(f"Non-zero weekly cells: {nz:.1%}")
    print(f"Median weekly demand (in-life, non-zero): "
          f"{panel.loc[panel['sold'] > 0, 'sold'].median():.0f}")
    print("\nExample real descriptions:")
    print(products[["StockCode", "real_description", "median_price", "total_qty"]].head(8).to_string(index=False))
