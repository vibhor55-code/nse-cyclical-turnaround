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
        return [], "Error: CSV must have an 'NSE Code' or 'Symbol' column."
    syms = df[col].dropna().astype(str).str.strip().unique().tolist()
    tickers = [s if s.endswith(".NS") else f"{s}.NS" for s in syms if s and s.lower() != "nan"]
    return tickers, f"Uploaded CSV ({len(tickers)} stocks)"


def _extract_ticker_df(data, ticker, batch_len):
    if data is None or data.empty:
        return pd.DataFrame()
    if batch_len == 1:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.get_level_values(0):
        return data[ticker].copy()
    return pd.DataFrame()


def _get_statement_row(df, candidates):
    """Safely extract a row from yfinance financial statements across naming variations."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            vals = df.loc[name].dropna()
            if not vals.empty:
                return vals
    return None


def _score_technical_setup(drawdown_pct, base_range_pct, vol_ratio, dist_ema40_pct, ema40_slope_pct):
    """Institutional 0-65 technical score (35 more points added in Stage 2 deep forensic audit)."""
    depth_pts = min(15.0, max(0.0, (drawdown_pct - 15) * 0.35))
    tight_pts = min(15.0, max(0.0, (40 - base_range_pct) * 0.55))
    vol_pts = min(15.0, max(0.0, (vol_ratio - 0.6) * 8.5))

    if 0 <= dist_ema40_pct <= 8 and ema40_slope_pct >= 0:
        ema_pts = 20.0
    elif 0 <= dist_ema40_pct <= 14:
        ema_pts = 15.0
