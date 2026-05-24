"""
labeler.py
==========
Merges fraud + bankruptcy labels onto the feature dataset to produce the
final ML-ready training table.

Label logic (avoiding look-ahead bias)
---------------------------------------
A company-year (cik, year) gets label = 1 (FRAUD) if:
  • The company appears in the AAER dataset  AND
  • year < fraud_date.year                   (filing was BEFORE discovery)
    OR year == fraud_date.year - 1           (the fraud year itself, often the last
                                              year where manipulated data appears)

A company-year gets label = 1 (BANKRUPT) if:
  • The company stopped filing 10-Ks within 3 years of this filing AND
  • Financial indicators suggest distress (not just voluntary delistings)

Label = 0 (CLEAN) for all other company-years.

Output schema
-------------
cik          : str
ticker       : str
company_name : str
year         : int
label        : int      0 = clean, 1 = fraud, 2 = bankruptcy
label_type   : str      'fraud' | 'bankruptcy' | 'clean'
is_fraud     : int      binary fraud indicator
is_bankrupt  : int      binary bankruptcy indicator
[+ all f_* feature columns]
"""

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "data" / "labels"
PROC_DIR   = ROOT / "data" / "processed"


# ── Fraud labels ──────────────────────────────────────────────────────────────
def attach_fraud_labels(feat_df: pd.DataFrame,
                        fraud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark company-years as fraud using AAER dates.

    fraud_df columns: [cik, fraud_date]
    feat_df columns:  [cik, year, ...]

    A (cik, year) is fraud if:
      fraud_date.year - 3 <= year <= fraud_date.year
    (We allow 3 years because SEC investigations often discover fraud
     that was happening for 2-3 years before the enforcement action.)
    """
    fraud_df = fraud_df[["cik", "fraud_date"]].dropna().copy()
    fraud_df["fraud_year"] = pd.to_datetime(fraud_df["fraud_date"]).dt.year
    # Zero-pad to 10 digits so they match feat_df's format
    fraud_df["cik"]        = fraud_df["cik"].astype(str).str.zfill(10)

    feat_df = feat_df.copy()
    feat_df["cik"] = feat_df["cik"].astype(str).str.zfill(10)

    # Create a year range for each fraud company
    fraud_ranges = []
    for _, row in fraud_df.iterrows():
        fy = int(row["fraud_year"])
        for yr in range(fy - 3, fy + 1):   # 3 years before + year of discovery
            fraud_ranges.append({"cik": row["cik"], "year": yr, "is_fraud": 1})

    if not fraud_ranges:
        feat_df["is_fraud"] = 0
        return feat_df

    fraud_year_df = pd.DataFrame(fraud_ranges).drop_duplicates()
    feat_df = feat_df.merge(fraud_year_df, on=["cik", "year"], how="left")
    feat_df["is_fraud"] = feat_df["is_fraud"].fillna(0).astype(int)

    n_fraud = feat_df["is_fraud"].sum()
    logger.info(f"Fraud labels attached: {n_fraud:,} company-years flagged as fraud")
    return feat_df


# ── Bankruptcy labels ──────────────────────────────────────────────────────────
def attach_bankruptcy_labels(feat_df: pd.DataFrame,
                             bk_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Mark company-years as pre-bankruptcy distress using Chapter 11 filing dates.

    If bk_df is provided (from get_bankruptcy_labels()), it is used directly.
    Falls back to a filing-gap heuristic if bk_df is None or empty.

    bk_df columns: [cik, bankruptcy_date]

    A (cik, year) is bankrupt if:
      bankruptcy_year - 2 <= year <= bankruptcy_year - 1
    (The 2 fiscal years BEFORE the Ch.11 filing show clear distress signals
     while the company was still filing annual reports.)
    """
    feat_df = feat_df.copy()
    feat_df["cik"] = feat_df["cik"].astype(str).str.zfill(10)

    # ── Path A: real labels from KNOWN_BANKRUPTCY_SEED ───────────────────────
    if bk_df is not None and not bk_df.empty:
        bk_df = bk_df[["cik", "bankruptcy_date"]].dropna().copy()
        bk_df["bk_year"] = pd.to_datetime(bk_df["bankruptcy_date"]).dt.year
        bk_df["cik"]     = bk_df["cik"].astype(str).str.zfill(10)

        bk_ranges = []
        for _, row in bk_df.iterrows():
            by = int(row["bk_year"])
            for yr in range(by - 2, by):      # 2 years before filing
                bk_ranges.append({"cik": row["cik"], "year": yr, "is_bankrupt": 1})

        if bk_ranges:
            bk_year_df = pd.DataFrame(bk_ranges).drop_duplicates()
            feat_df = feat_df.merge(bk_year_df, on=["cik", "year"], how="left")
            feat_df["is_bankrupt"] = feat_df["is_bankrupt"].fillna(0).astype(int)
            n_bk = feat_df["is_bankrupt"].sum()
            logger.info(f"Bankruptcy labels (real): {n_bk:,} company-years flagged")
            return feat_df

    # ── Path B: heuristic fallback (gone-dark + distress signals) ────────────
    logger.warning("No real bankruptcy labels — falling back to filing-gap heuristic")
    overall_max_year = feat_df["year"].max()
    max_year_per_co  = feat_df.groupby("cik")["year"].max().reset_index()
    max_year_per_co.columns = ["cik", "last_filing_year"]

    gone_dark = max_year_per_co[
        max_year_per_co["last_filing_year"] <= overall_max_year - 3
    ]["cik"].tolist()
    logger.info(f"Companies that stopped filing: {len(gone_dark)}")

    distressed_ciks = set()
    for cik in gone_dark:
        co_data   = feat_df[feat_df["cik"] == cik].sort_values("year")
        last_rows = co_data.tail(2)
        signals   = 0
        for col, thresh, direction in [
            ("f_current_ratio", 1.0,  "below"),
            ("f_debt_assets",   0.8,  "above"),
            ("f_net_margin",    0.0,  "below"),
            ("f_cfo_ni",        0.0,  "below"),
            ("f_z_x1",          0.0,  "below"),
        ]:
            if col in last_rows.columns:
                hit = (last_rows[col] < thresh if direction == "below"
                       else last_rows[col] > thresh)
                if hit.any():
                    signals += 1
        if signals >= 2:
            distressed_ciks.add(cik)

    logger.info(f"Distressed + dark companies: {len(distressed_ciks)}")
    bankrupt_rows = []
    for cik in distressed_ciks:
        co_data      = feat_df[feat_df["cik"] == cik]
        last_2_years = co_data["year"].nlargest(2).tolist()
        for yr in last_2_years:
            bankrupt_rows.append({"cik": cik, "year": yr, "is_bankrupt": 1})

    if not bankrupt_rows:
        feat_df["is_bankrupt"] = 0
        return feat_df

    bk_year_df = pd.DataFrame(bankrupt_rows).drop_duplicates()
    feat_df = feat_df.merge(bk_year_df, on=["cik", "year"], how="left")
    feat_df["is_bankrupt"] = feat_df["is_bankrupt"].fillna(0).astype(int)
    logger.info(f"Bankruptcy labels (heuristic): {feat_df['is_bankrupt'].sum():,} company-years flagged")
    return feat_df


# ── Supplemental: load pre-built academic labels ──────────────────────────────
def try_load_academic_labels() -> pd.DataFrame | None:
    """
    Attempt to load the Dechow et al. (2011) AAER firm-year dataset.
    This is the gold standard used in most fraud detection papers.
    Returns DataFrame[cik, year, is_fraud] or None if unavailable.
    """
    try:
        url = ("https://raw.githubusercontent.com/JarFraud/"
               "FraudDetection/master/data/AAER_firm_year.csv")
        df  = pd.read_csv(url)
        df.columns = df.columns.str.lower().str.strip()

        # Standardise
        col_map = {}
        for col in df.columns:
            if "cik"   in col: col_map[col] = "cik"
            if "fyear" in col or "year" in col: col_map[col] = "year"
            if "p_aaer" in col or "misstate" in col: col_map[col] = "is_fraud"
        df = df.rename(columns=col_map)

        needed = {"cik", "year", "is_fraud"}
        if not needed.issubset(df.columns):
            return None

        df["cik"]      = df["cik"].astype(str).str.zfill(10)
        df["is_fraud"] = (df["is_fraud"] > 0).astype(int)
        logger.info(f"Academic labels loaded: {df['is_fraud'].sum()} fraud firm-years")
        return df[["cik", "year", "is_fraud"]]
    except Exception as e:
        logger.warning(f"Could not load academic labels: {e}")
        return None


# ── Final label assembly ──────────────────────────────────────────────────────
def build_labeled_dataset(
    features_path: Path | None = None,
    fraud_labels_path: Path | None = None,
    use_academic_labels: bool = True,
) -> pd.DataFrame:
    """
    Master function: load features, attach all labels, save final dataset.

    Priority:
      1. Academic Dechow et al. labels (most reliable, if downloadable)
      2. Our scraped AAER labels
      3. Bankruptcy proxy labels

    Returns final DataFrame saved to data/processed/labeled_dataset.parquet
    """
    # Load features
    feat_path  = features_path or (PROC_DIR / "features.parquet")
    fraud_path = fraud_labels_path or (LABELS_DIR / "fraud_labels.csv")

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features not found at {feat_path}. "
            "Run pipeline/build_dataset.py first."
        )

    logger.info(f"Loading features from {feat_path}…")
    feat_df = pd.read_parquet(feat_path)
    logger.info(f"  {len(feat_df):,} company-years, {len(feat_df.columns)} columns")

    # ── Fraud labels ──────────────────────────────────────────────
    # Try academic first
    academic_labels = None
    if use_academic_labels:
        academic_labels = try_load_academic_labels()

    if academic_labels is not None:
        feat_df["cik"] = feat_df["cik"].astype(str).str.zfill(10)
        feat_df = feat_df.merge(
            academic_labels, on=["cik", "year"], how="left"
        )
        feat_df["is_fraud"] = feat_df["is_fraud"].fillna(0).astype(int)
        logger.info("Using academic labels (Dechow et al. 2011)")
    elif fraud_path.exists():
        fraud_df = pd.read_csv(fraud_path, parse_dates=["fraud_date"])
        feat_df  = attach_fraud_labels(feat_df, fraud_df)
        logger.info("Using scraped AAER labels")
    else:
        feat_df["is_fraud"] = 0
        logger.warning("No fraud labels found — all set to 0")

    # ── Bankruptcy labels ─────────────────────────────────────────
    # Try to load real Ch.11 labels from the bankruptcy seed list
    bk_df = None
    try:
        from pipeline.aaer_labels import get_bankruptcy_labels
        bk_cache = LABELS_DIR / "bankruptcy_labels.csv"
        if bk_cache.exists():
            bk_df = pd.read_csv(bk_cache, parse_dates=["bankruptcy_date"])
            logger.info(f"Loaded cached bankruptcy labels: {len(bk_df)} companies")
        else:
            bk_df = get_bankruptcy_labels()
            if not bk_df.empty:
                bk_df.to_csv(bk_cache, index=False)
                logger.info(f"Saved bankruptcy labels → {bk_cache}")
    except Exception as e:
        logger.warning(f"Could not load bankruptcy labels: {e}")

    feat_df = attach_bankruptcy_labels(feat_df, bk_df=bk_df)

    # ── Unified label ─────────────────────────────────────────────
    # 0 = clean, 1 = fraud, 2 = bankrupt
    # If both fraud + bankrupt → fraud takes priority
    conditions = [
        feat_df["is_fraud"]    == 1,
        feat_df["is_bankrupt"] == 1,
    ]
    choices = [1, 2]
    feat_df["label"]      = np.select(conditions, choices, default=0)
    feat_df["label_type"] = np.select(conditions,
                                       ["fraud", "bankruptcy"],
                                       default="clean")

    # ── Summary stats ─────────────────────────────────────────────
    dist = feat_df["label_type"].value_counts()
    logger.info(f"\nLabel distribution:\n{dist.to_string()}")
    logger.info(f"Fraud rate: {(feat_df['is_fraud'].mean()*100):.2f}%")
    logger.info(f"Bankruptcy rate: {(feat_df['is_bankrupt'].mean()*100):.2f}%")

    # ── Save ──────────────────────────────────────────────────────
    out = PROC_DIR / "labeled_dataset.parquet"
    feat_df.to_parquet(out, index=False)
    logger.info(f"Saved → {out}  ({len(feat_df):,} rows)")

    return feat_df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    df = build_labeled_dataset()
    print("\nSample:")
    print(df[["cik", "ticker", "year", "label", "label_type"]].head(30).to_string())
    print(f"\nFinal shape: {df.shape}")
