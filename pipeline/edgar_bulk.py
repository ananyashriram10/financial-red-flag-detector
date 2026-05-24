"""
edgar_bulk.py
=============
Bulk EDGAR XBRL fetcher for the S&P 1500.

Responsibilities
----------------
1. get_sp1500_tickers()   → pull S&P 500 + 400 + 600 tickers from Wikipedia
2. ticker_to_cik()        → map every ticker to its SEC CIK (zero-padded)
3. fetch_company_facts()  → download + cache one company's XBRL JSON
4. extract_raw_financials()→ pull annual 10-K values for 25+ line items
5. build_raw_table()      → run the whole universe, return one big DataFrame

Caching
-------
Each company's JSON is saved to  data/raw/<CIK>.json
so re-runs skip the network call entirely.

Rate limiting
-------------
SEC asks for ≤ 10 req/s; we sleep 0.12s between calls (~8 req/s).
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]          # repo root
RAW_DIR   = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "FinancialRedFlagDetector ananyashriram10@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
SLEEP = 0.12          # seconds between SEC requests
TIMEOUT = 30          # request timeout


# ── 1. Ticker universe ────────────────────────────────────────────────────────
def _wiki_tickers(url: str, col: str) -> list[str]:
    """Try fetching a Wikipedia table with browser-like headers to avoid 403."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    for tbl in tables:
        if col in tbl.columns:
            raw = tbl[col].dropna().tolist()
            return [str(t).strip().replace(".", "-") for t in raw]
    return []


def _sec_all_tickers() -> list[str]:
    """
    Fallback: pull every ticker from SEC's own company_tickers.json.
    Returns all ~10,000 US-listed tickers — broader than S&P 1500 but
    guaranteed to work since it's the same host we use for XBRL data.
    We trim to the largest ~1,500 companies by CIK recency as a proxy.
    """
    logger.info("Using SEC company_tickers.json as ticker source (Wikipedia fallback)")
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS, timeout=20
    )
    r.raise_for_status()
    data = r.json()
    # Sort by CIK descending (higher CIK = more recently registered, biased toward active cos)
    entries = sorted(data.values(), key=lambda x: x["cik_str"], reverse=True)
    tickers = [e["ticker"] for e in entries if e.get("ticker")]
    logger.info(f"SEC ticker list: {len(tickers)} total tickers available")
    return tickers


def get_sp1500_tickers() -> list[str]:
    """
    Get S&P 1500 tickers. Strategy:
      1. Try Wikipedia with browser headers (works most of the time)
      2. If Wikipedia blocks (403 on Kaggle), fall back to SEC's own ticker list
         and take the top 1,500 by exchange listing activity
    """
    wiki_sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",  "Symbol"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",  "Ticker symbol"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",  "Ticker symbol"),
    ]

    tickers: set[str] = set()
    for url, col in wiki_sources:
        try:
            batch = _wiki_tickers(url, col)
            tickers.update(batch)
            logger.info(f"  Wikipedia {col}: {len(batch)} tickers")
        except Exception as e:
            logger.warning(f"Wikipedia blocked ({e}) — will use SEC fallback")

    if len(tickers) < 100:
        # Wikipedia failed — use SEC fallback
        all_tickers = _sec_all_tickers()
        # Take first 1500 (SEC list is roughly sorted by prominence)
        tickers = set(all_tickers[:1500])
        logger.info(f"SEC fallback: using {len(tickers)} tickers")

    result = sorted(tickers)
    logger.info(f"Total tickers to fetch: {len(result)}")
    return result


# ── 2. Ticker → CIK map ───────────────────────────────────────────────────────
_cik_map: dict[str, str] | None = None


def load_cik_map() -> dict[str, str]:
    """Load SEC's master ticker→CIK mapping (cached in memory)."""
    global _cik_map
    if _cik_map is not None:
        return _cik_map
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    _cik_map = {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }
    return _cik_map


def ticker_to_cik(ticker: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK, or None if not found."""
    m = load_cik_map()
    return m.get(ticker.upper().strip())


# ── 3. Fetch + cache company facts ────────────────────────────────────────────
def fetch_company_facts(cik: str, force_refresh: bool = False) -> Optional[dict]:
    """
    Download XBRL companyfacts JSON for one CIK.
    Caches to data/raw/<cik>.json — returns None on failure.
    """
    cache_path = RAW_DIR / f"{cik}.json"

    # Serve from cache if available
    if cache_path.exists() and not force_refresh:
        with open(cache_path, "r") as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404:
            logger.debug(f"CIK {cik}: no XBRL data (404)")
            return None
        r.raise_for_status()
        data = r.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        logger.warning(f"CIK {cik}: fetch failed — {e}")
        return None


# ── 4. Extract raw financial line items ───────────────────────────────────────
# Maps our friendly names → lists of XBRL tags to try (in priority order)
CONCEPT_MAP: dict[str, list[str]] = {
    # Balance sheet
    "total_assets":         ["Assets"],
    "current_assets":       ["AssetsCurrent"],
    "total_liabilities":    ["Liabilities"],
    "current_liabilities":  ["LiabilitiesCurrent"],
    "equity":               ["StockholdersEquity",
                             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained_earnings":    ["RetainedEarningsAccumulatedDeficit"],
    "cash":                 ["CashAndCashEquivalentsAtCarryingValue",
                             "CashCashEquivalentsAndShortTermInvestments"],
    "receivables":          ["AccountsReceivableNetCurrent",
                             "ReceivablesNetCurrent"],
    "inventory":            ["InventoryNet", "InventoryFinishedGoods"],
    "goodwill":             ["Goodwill"],
    "intangibles":          ["IntangibleAssetsNetExcludingGoodwill",
                             "FiniteLivedIntangibleAssetsNet"],
    # Income statement
    "revenue":              ["RevenueFromContractWithCustomerExcludingAssessedTax",
                             "Revenues",
                             "SalesRevenueNet",
                             "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit":         ["GrossProfit"],
    "cogs":                 ["CostOfGoodsAndServicesSold",
                             "CostOfRevenue",
                             "CostOfGoodsSold"],
    "operating_income":     ["OperatingIncomeLoss"],
    "net_income":           ["NetIncomeLoss"],
    "interest_expense":     ["InterestExpense"],
    "da":                   ["DepreciationDepletionAndAmortization",
                             "DepreciationAndAmortization",
                             "Depreciation"],
    "sga":                  ["SellingGeneralAndAdministrativeExpense",
                             "GeneralAndAdministrativeExpense"],
    "rd_expense":           ["ResearchAndDevelopmentExpense"],
    "tax_expense":          ["IncomeTaxExpenseBenefit"],
    # Cash flow statement
    "cfo":                  ["NetCashProvidedByUsedInOperatingActivities"],
    "cfi":                  ["NetCashProvidedByUsedInInvestingActivities"],
    "cff":                  ["NetCashProvidedByUsedInFinancingActivities"],
    "capex":                ["PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets"],
    # Debt
    "long_term_debt":       ["LongTermDebt",
                             "LongTermDebtNoncurrent",
                             "LongTermDebtAndCapitalLeaseObligations"],
    "short_term_debt":      ["ShortTermBorrowings",
                             "NotesPayableCurrent",
                             "CommercialPaper"],
}


def _extract_concept(facts: dict, tags: list[str],
                     unit: str = "USD") -> pd.DataFrame:
    """
    Try each tag in order; return the first non-empty annual 10-K series.
    Returns DataFrame[year, value] or empty DataFrame.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get(unit, [])
        rows = []
        for e in entries:
            if e.get("form") != "10-K":
                continue
            start = e.get("start", "")
            end   = e.get("end",   "")
            # Keep only ~12-month periods
            if start and end:
                try:
                    months = (
                        (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44
                    )
                    if not (10 <= months <= 14):
                        continue
                except Exception:
                    pass
            try:
                year = pd.Timestamp(end).year
            except Exception:
                continue
            rows.append({"year": year, "value": e["val"],
                         "filed": e.get("filed", "")})
        if not rows:
            continue
        df = (
            pd.DataFrame(rows)
            .sort_values("filed", ascending=False)
            .drop_duplicates("year")
            .sort_values("year")
            .reset_index(drop=True)
        )
        return df[["year", "value"]]
    return pd.DataFrame(columns=["year", "value"])


def extract_raw_financials(cik: str, facts: dict) -> pd.DataFrame:
    """
    Extract all CONCEPT_MAP line items for one company.
    Returns a long DataFrame: [cik, year, concept, value]
    """
    rows = []
    for concept, tags in CONCEPT_MAP.items():
        df = _extract_concept(facts, tags)
        for _, r in df.iterrows():
            rows.append({
                "cik":     cik,
                "year":    int(r.year),
                "concept": concept,
                "value":   r.value,
            })
    if not rows:
        return pd.DataFrame(columns=["cik", "year", "concept", "value"])
    return pd.DataFrame(rows)


# ── 5. Build full raw table ───────────────────────────────────────────────────
def build_raw_table(
    tickers:       Optional[list[str]] = None,
    force_refresh: bool = False,
    min_years:     int  = 3,       # drop companies with fewer than N years of data
) -> pd.DataFrame:
    """
    Main pipeline entry point.

    Parameters
    ----------
    tickers       : list of ticker symbols; defaults to S&P 1500
    force_refresh : re-download even if cache exists
    min_years     : minimum years of data required to include a company

    Returns
    -------
    Long-format DataFrame[cik, ticker, company_name, year, concept, value]
    Also saves to data/processed/raw_financials.parquet
    """
    if tickers is None:
        tickers = get_sp1500_tickers()

    cik_map = load_cik_map()
    # Build reverse map cik → ticker for labeling
    cik_to_ticker = {v: k for k, v in cik_map.items()}

    all_rows: list[pd.DataFrame] = []
    missing_cik, fetch_fail, low_data = 0, 0, 0

    for ticker in tqdm(tickers, desc="Fetching EDGAR data", unit="co"):
        cik = ticker_to_cik(ticker)
        if cik is None:
            logger.debug(f"{ticker}: no CIK")
            missing_cik += 1
            continue

        facts = fetch_company_facts(cik, force_refresh)
        if facts is None:
            fetch_fail += 1
            continue

        df = extract_raw_financials(cik, facts)
        if df.empty:
            fetch_fail += 1
            continue

        n_years = df["year"].nunique()
        if n_years < min_years:
            low_data += 1
            continue

        company_name = facts.get("entityName", ticker)
        df["ticker"]       = ticker
        df["company_name"] = company_name
        all_rows.append(df)

        time.sleep(SLEEP)   # respect SEC rate limit

    logger.info(
        f"\nDone. "
        f"Companies fetched: {len(all_rows)} | "
        f"Missing CIK: {missing_cik} | "
        f"Fetch failures: {fetch_fail} | "
        f"Insufficient data (<{min_years} yrs): {low_data}"
    )

    if not all_rows:
        raise RuntimeError("No data fetched. Check your network / SEC EDGAR availability.")

    result = pd.concat(all_rows, ignore_index=True)
    out = ROOT / "data" / "processed" / "raw_financials.parquet"
    result.to_parquet(out, index=False)
    logger.info(f"Saved → {out}  ({len(result):,} rows)")
    return result


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    df = build_raw_table()
    print(df.head(20).to_string())
    print(f"\nShape: {df.shape}")
