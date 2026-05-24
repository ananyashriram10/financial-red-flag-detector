"""
aaer_labels.py
==============
Scrapes SEC Accounting & Auditing Enforcement Releases (AAERs) to build
a ground-truth fraud label dataset.

What are AAERs?
---------------
When the SEC catches a company committing accounting fraud, it issues an
Accounting and Auditing Enforcement Release. These are public records going
back to 1982. ~1,200 companies have been cited.

Pipeline
--------
1. scrape_aaer_index()     → download the AAER index HTML pages
2. parse_aaer_entries()    → extract company name + date from each entry
3. match_to_edgar()        → fuzzy-match company name → EDGAR CIK
4. build_fraud_labels()    → final DataFrame[cik, company_name, fraud_date, aaer_no]
5. save_labels()           → persist to data/labels/fraud_labels.csv

Fuzzy matching note
-------------------
AAER company names ("Enron Corp.") rarely match EDGAR names exactly
("ENRON CORP"). We use RapidFuzz with a threshold of 85 to handle:
  - Punctuation differences
  - Inc./Corp./Ltd. variations
  - Case differences
  - Abbreviations

Output schema
-------------
cik          : str   zero-padded 10-digit EDGAR CIK
company_name : str   as it appears in EDGAR
aaer_no      : int   AAER release number
fraud_date   : date  date of the AAER (= when fraud was publicly discovered)
match_score  : float fuzzy match confidence (0-100)
"""

from __future__ import annotations
import re
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "FinancialRedFlagDetector ananyashriram10@gmail.com",
}

AAER_BASE  = "https://www.sec.gov"
# SEC reorganised their site — try both URLs
AAER_URLS = [
    "https://www.sec.gov/litigation/aaers.htm",
    "https://www.sec.gov/divisions/enforce/enforcea.htm",   # old URL, kept as fallback
]

SLEEP = 0.15   # between SEC requests


# ── 1. Scrape the AAER index page ─────────────────────────────────────────────
def scrape_aaer_index() -> list[dict]:
    """
    Download the main AAER index and parse all entries.
    Tries multiple URLs in case SEC has reorganised the site.
    Returns a list of dicts: {aaer_no, date_str, raw_text, url}
    """
    r = None
    for url in AAER_URLS:
        try:
            logger.info(f"Fetching AAER index: {url}")
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                break
            logger.warning(f"  {url} → {r.status_code}")
        except Exception as e:
            logger.warning(f"  {url} → {e}")

    if r is None or r.status_code != 200:
        raise RuntimeError("All AAER URLs failed — SEC site may be restructured")

    soup = BeautifulSoup(r.text, "html.parser")

    entries = []
    # AAER entries look like:
    #   AAER No. 3900  |  January 15, 2020  |  In the Matter of XYZ Corp...
    # They appear as links or table rows depending on the year
    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True)
        # Match patterns like "AAER-3900" or links to /litigation/aaers/
        if "aaer" in href.lower() or "AAER" in text:
            entries.append({"url": AAER_BASE + href if href.startswith("/") else href,
                            "raw_text": text})

    # Also try to parse structured tables
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 3:
            text = " ".join(c.get_text(strip=True) for c in cells)
            if re.search(r"AAER.?\d{3,4}", text, re.I):
                link_tag = row.find("a", href=True)
                url = (AAER_BASE + link_tag["href"]
                       if link_tag and link_tag["href"].startswith("/")
                       else (link_tag["href"] if link_tag else ""))
                entries.append({"url": url, "raw_text": text})

    logger.info(f"Found {len(entries)} raw AAER entries on index page")
    return entries


# ── 2. Fetch individual AAER release pages ────────────────────────────────────
def fetch_aaer_detail(url: str) -> Optional[str]:
    """Download one AAER release page and return its text content."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return None


# ── 3. Parse AAER entries into structured records ─────────────────────────────
_DATE_PATTERNS = [
    r"(\w+ \d{1,2},?\s*\d{4})",            # January 15, 2020
    r"(\d{1,2}/\d{1,2}/\d{2,4})",          # 1/15/2020
    r"(\d{4}-\d{2}-\d{2})",                # 2020-01-15
]

_AAER_NO_RE = re.compile(r"AAER.?(?:No\.?)?\s*(\d{3,4})", re.I)

# Patterns for extracting the RESPONDENT company name from AAER text
_RESPONDENT_RE = [
    re.compile(r"In the Matter of\s+(.+?)(?:,|\.|;|\n)", re.I),
    re.compile(r"Respondent[s]?[:\s]+(.+?)(?:,|\.|;|\n)", re.I),
    re.compile(r"against\s+(.+?)(?:,|\.|;|\n)", re.I),
]


def _parse_date(text: str) -> datetime | None:
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            ds = m.group(1).replace(",", "").strip()
            for fmt in ["%B %d %Y", "%b %d %Y", "%m/%d/%Y", "%m/%d/%y",
                        "%Y-%m-%d"]:
                try:
                    return datetime.strptime(ds, fmt)
                except ValueError:
                    continue
    return None


def _parse_company(text: str) -> str | None:
    for pattern in _RESPONDENT_RE:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip()
            # Remove common noise
            noise = ["Inc", "Corp", "LLC", "Ltd", "Co", "Corporation",
                     "Company", "Holdings"]
            # Normalize to upper for matching later
            return name.strip(" .,;")
    return None


def parse_aaer_text(raw_text: str, url: str = "") -> dict | None:
    """Extract structured fields from one AAER text blob."""
    m = _AAER_NO_RE.search(raw_text)
    aaer_no = int(m.group(1)) if m else None

    date    = _parse_date(raw_text)
    company = _parse_company(raw_text)

    if not company or not date:
        return None

    return {
        "aaer_no":    aaer_no,
        "fraud_date": date.date(),
        "raw_name":   company,
        "url":        url,
    }


# ── 4. Match raw names → EDGAR CIKs ──────────────────────────────────────────
def match_to_edgar(
    fraud_entries: list[dict],
    threshold: float = 82.0,
) -> pd.DataFrame:
    """
    Fuzzy-match AAER company names to EDGAR company names.
    Uses RapidFuzz (fast C++ implementation of Levenshtein/token_sort_ratio).

    threshold: minimum match score to accept (0-100). 82 is conservative
               — reduces false positives (wrong CIK) at cost of coverage.
    """
    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError:
        raise ImportError("Run: pip install rapidfuzz")

    # Build name → CIK lookup from SEC master list
    logger.info("Loading SEC ticker/CIK map for fuzzy matching…")
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    master = r.json()

    # We match against COMPANY NAMES not tickers
    # Build: {normalised_name: cik, company_name}
    edgar_names: dict[str, dict] = {}
    for entry in master.values():
        raw = entry.get("title", "")
        norm = _normalise(raw)
        if norm:
            edgar_names[norm] = {
                "cik":          str(entry["cik_str"]).zfill(10),
                "company_name": raw,
                "ticker":       entry.get("ticker", ""),
            }

    choices = list(edgar_names.keys())
    matched = []

    for fe in fraud_entries:
        raw = fe["raw_name"]
        norm = _normalise(raw)
        if not norm:
            continue

        # rapidfuzz returns (match, score, index)
        result = rfp.extractOne(norm, choices,
                                scorer=fuzz.token_sort_ratio,
                                score_cutoff=threshold)
        if result is None:
            logger.debug(f"No match for: {raw!r}")
            continue

        best_name, score, _ = result
        info = edgar_names[best_name]

        matched.append({
            "cik":          info["cik"],
            "ticker":       info["ticker"],
            "company_name": info["company_name"],
            "aaer_no":      fe["aaer_no"],
            "fraud_date":   fe["fraud_date"],
            "raw_aaer_name":fe["raw_name"],
            "match_score":  round(score, 1),
        })
        logger.debug(f"  {raw!r} → {info['company_name']!r}  [{score:.0f}]")

    df = pd.DataFrame(matched)
    if not df.empty:
        df = df.drop_duplicates("cik")   # keep best match per company
    return df


def _normalise(name: str) -> str:
    """Lowercase, strip punctuation and common legal suffixes."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    for suffix in ["inc", "corp", "llc", "ltd", "co", "corporation",
                   "company", "holdings", "group", "international",
                   "enterprises", "partners"]:
        name = re.sub(rf"\b{suffix}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ── 5. Build the final fraud label set ────────────────────────────────────────
def build_fraud_labels(use_cache: bool = True) -> pd.DataFrame:
    """
    Full pipeline: scrape → parse → match → save.

    Returns DataFrame[cik, ticker, company_name, aaer_no, fraud_date, match_score]

    If use_cache=True and data/labels/fraud_labels.csv exists, load from disk.
    """
    cache_path = LABELS_DIR / "fraud_labels.csv"

    if use_cache and cache_path.exists():
        logger.info(f"Loading cached fraud labels from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["fraud_date"])
        logger.info(f"  {len(df)} fraud companies loaded")
        return df

    logger.info("=== Building fraud label dataset ===")

    # ── Priority 1: Academic dataset (Dechow et al. 2011) ────────────────────
    # Cleaner labels, peer-reviewed, hosted on GitHub — more reliable than scraping
    academic = load_supplemental_aaer_dataset()
    if not academic.empty and "cik" in academic.columns and "is_fraud" in academic.columns:
        logger.info(f"Academic dataset loaded: {len(academic)} rows, "
                    f"{academic.get('is_fraud', pd.Series()).sum()} fraud firm-years")

        # Convert to the same schema as our scraped labels
        fraud_rows = academic[academic.get("is_fraud", 0) == 1].copy()
        if "year" in fraud_rows.columns:
            # Create a fraud_date from year (use year-end as proxy)
            fraud_rows["fraud_date"] = pd.to_datetime(
                fraud_rows["year"].astype(str) + "-12-31"
            )
        fraud_rows["cik"] = fraud_rows["cik"].astype(str).str.zfill(10)
        fraud_rows["match_score"] = 100.0
        fraud_rows["aaer_no"]     = fraud_rows.get("aaer_no", pd.Series(dtype=int))
        fraud_rows["ticker"]       = fraud_rows.get("ticker", "")
        fraud_rows["company_name"] = fraud_rows.get("company_name", "")

        keep = ["cik", "ticker", "company_name", "aaer_no", "fraud_date", "match_score"]
        available = [c for c in keep if c in fraud_rows.columns]
        result_df = fraud_rows[available].drop_duplicates("cik").reset_index(drop=True)
        result_df.to_csv(cache_path, index=False)
        logger.info(f"Saved {len(result_df)} fraud companies → {cache_path}")
        return result_df

    # ── Priority 2: Scrape SEC AAER pages ────────────────────────────────────
    logger.info("Academic dataset unavailable — scraping SEC AAER pages…")
    try:
        index_entries = scrape_aaer_index()
    except RuntimeError as e:
        logger.error(f"AAER scraping failed: {e}")
        logger.warning("Returning empty fraud labels — model will train on distress only")
        return pd.DataFrame(columns=["cik", "ticker", "company_name",
                                     "aaer_no", "fraud_date", "match_score"])

    parsed: list[dict] = []
    seen_aaer_nos: set = set()

    for entry in tqdm(index_entries, desc="Parsing AAER entries"):
        result = parse_aaer_text(entry["raw_text"], entry.get("url", ""))
        if result is None and entry.get("url"):
            time.sleep(SLEEP)
            detail_text = fetch_aaer_detail(entry["url"])
            if detail_text:
                result = parse_aaer_text(detail_text, entry["url"])

        if result and result.get("aaer_no") not in seen_aaer_nos:
            parsed.append(result)
            if result.get("aaer_no"):
                seen_aaer_nos.add(result["aaer_no"])

    logger.info(f"Parsed {len(parsed)} AAER entries")

    if not parsed:
        logger.warning("No AAER entries parsed — returning empty fraud labels")
        return pd.DataFrame(columns=["cik", "ticker", "company_name",
                                     "aaer_no", "fraud_date", "match_score"])

    matched_df = match_to_edgar(parsed)
    logger.info(f"Matched {len(matched_df)} companies to EDGAR CIKs")
    matched_df.to_csv(cache_path, index=False)
    logger.info(f"Saved → {cache_path}")
    return matched_df


# ── Supplemental: academic/public fraud datasets ──────────────────────────────
def load_supplemental_aaer_dataset() -> pd.DataFrame:
    """
    Load the Beneish (1999) + Dechow et al. (2011) curated AAER datasets
    that researchers have made publicly available as CSVs.

    These are cleaner than scraping but cover only 1982-2008.
    We merge them with our scraped data for maximum coverage.

    URLs point to replications hosted on GitHub (academic open-source).
    """
    SOURCES = [
        # Dechow et al. (2011) "Predicting Material Accounting Misstatements"
        # Published data from their JAE paper
        "https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data/AAER_firm_year.csv",
    ]

    frames = []
    for url in SOURCES:
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.lower().str.strip()
            # Standardise column names
            rename = {
                "gvkey":    "gvkey",
                "fyear":    "year",
                "p_aaer":   "is_fraud",
                "cik":      "cik",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            frames.append(df)
            logger.info(f"Loaded supplemental dataset: {len(df)} rows from {url}")
        except Exception as e:
            logger.warning(f"Could not load supplemental dataset {url}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    df = build_fraud_labels(use_cache=False)
    print(df.head(30).to_string())
    print(f"\nTotal fraud companies: {len(df)}")
    print(f"Date range: {df['fraud_date'].min()} → {df['fraud_date'].max()}")
