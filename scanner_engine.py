import io
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Exclude only pure non-cyclical industries so Sugar, Paper, Polyfilms, Cement, Metals, Graphite all pass
NON_CYCLICAL_INDUSTRIES = {
    "Household & Personal Products",
    "Tobacco",
    "Beverages - Non-Alcoholic",
    "Grocery Stores",
    "Discount Stores",
    "Drug Manufacturers - General",
    "Drug Manufacturers - Specialty & Generic",
    "Biotechnology",
    "Medical Care Facilities",
    "Diagnostics & Research",
    "Software - Application",
    "Software - Infrastructure",
    "Information Technology Services",
    "Telecom Services",
    "Utilities - Regulated Electric",
    "Utilities - Regulated Gas",
    "Utilities - Regulated Water",
}

# Official whole-market NSE CSV sources (All NSE Equities first so small-cap cyclicals are included)
NSE_UNIVERSE_SOURCES = [
    (
        "All NSE Listed Equities (Whole Market)",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "SYMBOL",
    ),
    (
        "NSE Nifty Total Market (750 Stocks)",
        "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
        "Symbol",
    ),
    (
        "NSE Nifty 500",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "Symbol",
    ),
]


def get_whole_market_symbols():
    """Fetch whole-market NSE symbols from official NSE CSVs."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for label, url, col_name in NSE_UNIVERSE_SOURCES:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip()
            if col_name in df.columns:
                if "SERIES" in df.columns:
                    df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
                symbols = df[col_name].dropna().astype(str).str.strip().unique().tolist()
                symbols = [s for s in symbols if s and s.lower() != "nan"]
                if len(symbols) >= 100:
                    return [f"{s}.NS" for s in symbols], f"{label} ({len(symbols)} stocks)"
        except Exception:
            continue

    return [], "NSE blocked the cloud server request. Please upload a Screener.in CSV below."


def parse_screener_csv(uploaded_file):
    """Backup loader if NSE blocks the cloud server IP."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    col = next((c for c in ["NSE Code", "Symbol", "SYMBOL", "Ticker", "symbol", "NSE"] if c in df.columns), None)
    if not col:
        return [], "Error: CSV must have an
