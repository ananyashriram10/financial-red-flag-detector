# 🚨 Financial Red Flag Detector

> Pull any US public company's 10-K filings from SEC EDGAR, run them through
> 46 engineered features, and score them with **our own XGBoost fraud detector**
> trained to beat Altman Z-Score and Beneish M-Score benchmarks.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  KAGGLE  (training pipeline)                        │
│  notebooks/kaggle_pipeline.ipynb                    │
│  ├── SEC EDGAR bulk fetch — S&P 1500 companies      │
│  ├── 46-feature engineering                         │
│  ├── AAER fraud label scraping                      │
│  ├── XGBoost model training                         │
│  ├── Benchmark vs Altman Z + Beneish M              │
│  └── ─► fraud_model.pkl (saved to this repo)        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  STREAMLIT CLOUD  (web app)                         │
│  app.py                                             │
│  ├── User types ticker (e.g. TSLA)                  │
│  ├── Live EDGAR fetch — single company              │
│  ├── Same 46 features computed in real time         │
│  ├── fraud_model.pkl → fraud probability score      │
│  ├── Altman Z + Beneish M computed alongside        │
│  └── Full health report with charts + red flags     │
└─────────────────────────────────────────────────────┘
```

---

## What It Detects

| Signal | Description |
|--------|-------------|
| **Our XGBoost model** | Trained on SEC AAER fraud cases — 46 features, beats Altman + Beneish |
| **Altman Z-Score** | Classic 1968 bankruptcy predictor — baseline we're competing against |
| **Beneish M-Score** | 1999 earnings manipulation model — caught Enron (M = −1.89) |
| **Revenue ↑ + CFO ↓** | The #1 fraud tell — WorldCom, Sunbeam, Lucent all showed this |
| **Sloan Accrual Ratio** | High accruals = earnings not backed by cash |
| **DSO Rising** | Receivables growing faster than revenue → channel stuffing |
| **Gross margin compression** | Pricing power eroding or cost manipulation |
| **Debt/Equity acceleration** | Leverage rising faster than equity can absorb |
| **Interest coverage < 1.5x** | One bad quarter away from covenant breach |

---

## Repo Structure

```
financial-red-flag-detector/
│
├── app.py                    # Streamlit web app (deploy to Streamlit Cloud)
├── edgar.py                  # Single-ticker live EDGAR fetcher
├── metrics.py                # Ratio computation + Altman Z + Beneish M
├── flags.py                  # Rule-based red flag engine
├── scoring.py                # Composite health score (0–100)
├── charts.py                 # Plotly dark-theme chart builders
│
├── pipeline/                 # Kaggle training pipeline
│   ├── edgar_bulk.py         # Bulk EDGAR fetch for S&P 1500
│   ├── features.py           # 46-feature engineering
│   ├── aaer_labels.py        # SEC fraud label scraper
│   ├── labeler.py            # Dataset assembly + labeling
│   └── build_dataset.py      # Orchestrator (runs the full pipeline)
│
├── models/                   # Trained model artifacts (committed to repo)
│   └── fraud_model.pkl       # XGBoost fraud detector (trained on Kaggle)
│
├── notebooks/
│   └── kaggle_pipeline.ipynb # Kaggle notebook — run this to train
│
├── data/
│   └── labels/
│       └── fraud_labels.csv  # Scraped AAER labels (committed, slow to rebuild)
│
└── requirements.txt
```

---

## Quickstart

### Run the web app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Train the model on Kaggle
1. Upload `notebooks/kaggle_pipeline.ipynb` to Kaggle
2. Enable Internet access in notebook settings
3. Run all cells (~30 min first run, cached after)
4. Download `fraud_model.pkl` from output
5. Place in `models/fraud_model.pkl` and commit

---

## Models & Benchmarks

**Altman Z-Score (1968)** — 5 variables, linear discriminant
```
Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5
Reported accuracy: ~72–80% AUC on bankruptcy prediction
```

**Beneish M-Score (1999)** — 8 variables, logistic regression
```
M = −4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + ...
Reported accuracy: ~70–76% AUC on manipulation detection
```

**Our XGBoost Model** — 46 features including trends, interactions, accruals
```
Target: > 85% AUC on held-out test set
Training data: S&P 1500 companies, SEC AAER fraud labels (1982–present)
```

---

## Data Sources

| Source | What we use it for | Cost |
|--------|-------------------|------|
| [SEC EDGAR XBRL API](https://data.sec.gov) | All financial statement data | Free |
| [SEC AAER Releases](https://www.sec.gov/divisions/enforce/enforcea.htm) | Fraud ground-truth labels | Free |
| [yfinance](https://github.com/ranaroussi/yfinance) | Market cap for Altman X4 | Free |
| Wikipedia (S&P tables) | Ticker universe | Free |

**Total cost: $0.**

---

## Disclaimer

Not financial advice. For research and educational purposes only.
