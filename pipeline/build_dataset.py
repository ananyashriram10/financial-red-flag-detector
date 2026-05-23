"""
build_dataset.py
================
Orchestrates the full data pipeline end-to-end.

Usage
-----
  python pipeline/build_dataset.py                      # S&P 1500, full run
  python pipeline/build_dataset.py --tickers AAPL MSFT  # specific tickers
  python pipeline/build_dataset.py --refresh            # ignore cache, re-fetch
  python pipeline/build_dataset.py --quick              # 50 tickers (dev/test)

Output
------
  data/processed/raw_financials.parquet     raw XBRL line items
  data/processed/features.parquet           46 engineered features
  data/labels/fraud_labels.csv              AAER fraud labels
  data/processed/labeled_dataset.parquet    FINAL: features + labels

Timeline (first run, S&P 1500)
-------------------------------
  EDGAR fetch     : ~25 min  (rate-limited, cached after first run)
  Feature eng.    : ~2 min
  Label scraping  : ~5 min
  Total first run : ~32 min

  Subsequent runs : < 1 min  (everything cached)
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import edgar_bulk, features as feat_eng, aaer_labels, labeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "pipeline_run.log"),
    ],
)
logger = logging.getLogger(__name__)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Financial Red Flag Detector — Data Pipeline              ║
║     Building training dataset from SEC EDGAR + AAER          ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_pipeline(
    tickers:       list[str] | None = None,
    force_refresh: bool = False,
    quick_mode:    bool = False,
) -> None:
    print_banner()
    t_total = time.time()

    # ── Step 1: EDGAR bulk fetch ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 1/4 — Fetching SEC EDGAR data")
    print("─" * 60)

    if quick_mode and tickers is None:
        # Use a small representative sample for dev/testing
        test_tickers = [
            "AAPL", "MSFT", "TSLA", "GE", "ENRN", "WBD", "NFLX",
            "SBUX", "JPM", "BAC", "XOM", "PFE", "JNJ", "AMZN", "META",
            "GOOGL", "WMT", "HD", "CVX", "ABBV", "MRK", "LLY", "TMO",
            "AVGO", "COST", "UNH", "ACN", "DHR", "NEE", "TXN",
            "QCOM", "HON", "PM", "UPS", "RTX", "SPGI", "SCHW", "IBM",
            "CAT", "GS", "AXP", "BA", "MMM", "ORCL", "AMAT", "LRCX",
            "MU", "INTC", "AMD", "NVDA",
        ]
        tickers = test_tickers
        logger.info(f"Quick mode: using {len(tickers)} tickers")

    t1 = time.time()
    raw_df = edgar_bulk.build_raw_table(
        tickers=tickers,
        force_refresh=force_refresh,
    )
    logger.info(f"EDGAR fetch done in {time.time()-t1:.0f}s  "
                f"({raw_df['cik'].nunique()} companies, {len(raw_df):,} rows)")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 2/4 — Engineering features (46 features)")
    print("─" * 60)

    t2 = time.time()
    feat_df = feat_eng.engineer_features(raw_df)
    out_feat = ROOT / "data" / "processed" / "features.parquet"
    feat_df.to_parquet(out_feat, index=False)
    logger.info(f"Features done in {time.time()-t2:.1f}s  "
                f"Shape: {feat_df.shape}  Saved → {out_feat}")

    # ── Step 3: Build fraud labels from AAER ─────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3/4 — Scraping SEC AAER fraud labels")
    print("─" * 60)

    t3 = time.time()
    use_cache = not force_refresh
    fraud_df  = aaer_labels.build_fraud_labels(use_cache=use_cache)
    logger.info(f"Fraud labels done in {time.time()-t3:.0f}s  "
                f"{len(fraud_df)} fraud companies found")

    # Try supplemental academic dataset too
    supp = aaer_labels.load_supplemental_aaer_dataset()
    if not supp.empty:
        logger.info(f"Supplemental AAER dataset: {len(supp)} rows loaded")

    # ── Step 4: Assemble labeled dataset ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 4/4 — Assembling labeled dataset")
    print("─" * 60)

    t4 = time.time()
    labeled_df = labeler.build_labeled_dataset(
        features_path=out_feat,
        fraud_labels_path=ROOT / "data" / "labels" / "fraud_labels.csv",
    )
    logger.info(f"Labeling done in {time.time()-t4:.1f}s  "
                f"Final shape: {labeled_df.shape}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    n_cos      = labeled_df["cik"].nunique()
    n_years    = labeled_df["year"].nunique()
    n_fraud    = labeled_df["is_fraud"].sum()
    n_bk       = labeled_df["is_bankrupt"].sum()
    n_clean    = (labeled_df["label"] == 0).sum()
    feat_cols  = [c for c in labeled_df.columns if c.startswith("f_")]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE  ({total_time:.0f}s)
╠══════════════════════════════════════════════════════════════╣
║  Companies     : {n_cos:>6,}
║  Year range    : {labeled_df['year'].min()} – {labeled_df['year'].max()}
║  Total rows    : {len(labeled_df):>6,}
║  Features      : {len(feat_cols):>6}
╠══════════════════════════════════════════════════════════════╣
║  LABEL DISTRIBUTION
║    Fraud       : {n_fraud:>6,}  ({n_fraud/len(labeled_df)*100:.1f}%)
║    Bankruptcy  : {n_bk:>6,}  ({n_bk/len(labeled_df)*100:.1f}%)
║    Clean       : {n_clean:>6,}  ({n_clean/len(labeled_df)*100:.1f}%)
╠══════════════════════════════════════════════════════════════╣
║  Outputs saved to data/processed/labeled_dataset.parquet
╚══════════════════════════════════════════════════════════════╝
""")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Specific tickers to fetch (default: S&P 1500)"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download even if cache exists"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 50-ticker subset for fast dev/testing"
    )
    args = parser.parse_args()

    run_pipeline(
        tickers=args.tickers,
        force_refresh=args.refresh,
        quick_mode=args.quick,
    )
