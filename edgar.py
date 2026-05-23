"""
edgar.py — SEC EDGAR XBRL data fetcher.
Maps ticker → CIK, pulls companyfacts JSON, extracts annual 10-K values.
"""

from __future__ import annotations
import time
import functools
import requests
import pandas as pd
from typing import Optional

# SEC requires a real User-Agent with contact info
HEADERS = {
    "User-Agent": "FinancialRedFlagDetector ananyashriram10@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

_TICKER_MAP: dict[str, str] | None = None   # cached ticker→CIK


def _load_ticker_map() -> dict[str, str]:
    global _TICKER_MAP
    if _TICKER_MAP is not None:
        return _TICKER_MAP
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    _TICKER_MAP = {v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                   for v in data.values()}
    return _TICKER_MAP


def get_cik(ticker: str) -> str:
    """Return zero-padded 10-digit CIK for a ticker, or raise ValueError."""
    m = _load_ticker_map()
    t = ticker.upper().strip()
    if t not in m:
        raise ValueError(f"Ticker '{t}' not found in SEC EDGAR database.")
    return m[t]


@functools.lru_cache(maxsize=32)
def get_company_facts(cik: str) -> dict:
    """Fetch the full XBRL companyfacts JSON (cached)."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return r.json()


def _extract_concept(facts: dict, concept: str, unit: str = "USD") -> pd.DataFrame:
    """
    Pull annual (10-K) values for one XBRL concept.
    Returns DataFrame with columns: year, value, filed.
    """
    try:
        entries = (
            facts["facts"]["us-gaap"][concept]["units"][unit]
        )
    except KeyError:
        return pd.DataFrame(columns=["year", "value", "filed"])

    rows = []
    for e in entries:
        # Only 10-K filings, require exactly 12-month periods
        if e.get("form") != "10-K":
            continue
        start = e.get("start", "")
        end   = e.get("end", "")
        if start and end:
            # Approximate period length check
            try:
                s = pd.Timestamp(start)
                en = pd.Timestamp(end)
                months = (en.year - s.year) * 12 + (en.month - s.month)
                if not (10 <= months <= 14):
                    continue
            except Exception:
                pass
        year = pd.Timestamp(end).year if end else None
        if year is None:
            continue
        rows.append({"year": year, "value": e["val"], "filed": e.get("filed", "")})

    if not rows:
        return pd.DataFrame(columns=["year", "value", "filed"])

    df = pd.DataFrame(rows)
    # Keep most recently filed value per fiscal year
    df = (
        df.sort_values("filed", ascending=False)
          .drop_duplicates("year")
          .sort_values("year")
          .reset_index(drop=True)
    )
    return df[["year", "value"]]


# --- Revenue has many possible XBRL tags across eras ---
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
    "RevenueFromRelatedParties",
]

_COGS_TAGS = [
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
    "CostOfServices",
]

_LTD_TAGS = [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
]

_STD_TAGS = [
    "ShortTermBorrowings",
    "NotesPayableCurrent",
    "CommercialPaper",
    "ShortTermNonBankLoansAndNotesPayable",
]

_SGA_TAGS = [
    "SellingGeneralAndAdministrativeExpense",
    "GeneralAndAdministrativeExpense",
]

_DA_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
]

_CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CapitalExpendituresIncurredButNotYetPaid",
    "PaymentsToAcquireProductiveAssets",
]


def _best_concept(facts: dict, tags: list[str], unit: str = "USD") -> pd.DataFrame:
    """Try each tag in order, return the first non-empty result."""
    for tag in tags:
        df = _extract_concept(facts, tag, unit)
        if not df.empty:
            return df
    return pd.DataFrame(columns=["year", "value"])


def get_financial_data(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Main entry point.  Returns a dict of DataFrames keyed by metric name,
    each with columns [year, value].  Also injects company name.
    """
    cik   = get_cik(ticker)
    facts = get_company_facts(cik)
    name  = facts.get("entityName", ticker.upper())

    def c(concept, unit="USD"):
        return _extract_concept(facts, concept, unit)

    data: dict[str, pd.DataFrame] = {
        "company_name": name,
        "cik":          cik,
        # Balance sheet
        "total_assets":       c("Assets"),
        "current_assets":     c("AssetsCurrent"),
        "total_liabilities":  c("Liabilities"),
        "current_liabilities":c("LiabilitiesCurrent"),
        "equity":             c("StockholdersEquity"),
        "retained_earnings":  c("RetainedEarningsAccumulatedDeficit"),
        "cash":               c("CashAndCashEquivalentsAtCarryingValue"),
        "receivables":        c("AccountsReceivableNetCurrent"),
        "inventory":          c("InventoryNet"),
        # Income statement
        "revenue":            _best_concept(facts, _REVENUE_TAGS),
        "gross_profit":       c("GrossProfit"),
        "cogs":               _best_concept(facts, _COGS_TAGS),
        "operating_income":   c("OperatingIncomeLoss"),
        "net_income":         c("NetIncomeLoss"),
        "interest_expense":   c("InterestExpense"),
        "sga":                _best_concept(facts, _SGA_TAGS),
        "da":                 _best_concept(facts, _DA_TAGS),
        # Cash flow
        "cfo":                c("NetCashProvidedByUsedInOperatingActivities"),
        "cfi":                c("NetCashProvidedByUsedInInvestingActivities"),
        "cff":                c("NetCashProvidedByUsedInFinancingActivities"),
        "capex":              _best_concept(facts, _CAPEX_TAGS),
        # Debt
        "ltd":                _best_concept(facts, _LTD_TAGS),
        "std":                _best_concept(facts, _STD_TAGS),
    }
    return data
