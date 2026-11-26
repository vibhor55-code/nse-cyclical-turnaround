import io
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Pure defensive sectors excluded automatically
DEFENSIVE_SECTORS = {
    "Consumer Defensive",
    "Healthcare",
    "Technology",
    "Communication Services",
    "Utilities",
}

# Official whole-market NSE CSV sources (zero hardcoded stocks)
NSE_UNIVERSE_SOURCES = [
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
    (
        "All NSE Listed Equities",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "SYMBOL",
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


def _score_technical_setup(drawdown_pct, base_range_pct, vol_ratio, dist_ema40_pct, ema40_slope_pct):
    """Institutional 0-80 technical score (20 more points added in Stage 2 forensic audit)."""
    depth_pts = min(20.0, max(0.0, (drawdown_pct - 25) * 0.5))
    tight_pts = min(20.0, max(0.0, (38 - base_range_pct) * 0.8))
    vol_pts = min(20.0, max(0.0, (vol_ratio - 1.0) * 12.0))

    if 0 <= dist_ema40_pct <= 8 and ema40_slope_pct >= 0:
        ema_pts = 20.0
    elif 0 <= dist_ema40_pct <= 12:
        ema_pts = 16.0
    elif -6 <= dist_ema40_pct < 0 and ema40_slope_pct >= -1.0:
        ema_pts = 14.0
    else:
        ema_pts = 8.0

    return round(depth_pts + tight_pts + vol_pts + ema_pts, 1)


def run_cycle_search_engine(tickers, progress_callback=None):
    """
    Fully automated 2-Stage Institutional Cycle Investing Search Engine.
    Identifies realistic Entry Zones, Stop-Loss Zones, and Sector Clusters.
    """
    stage1 = []
    batch_size = 50
    total_batches = max(1, (len(tickers) + batch_size - 1) // batch_size)

    # --- STAGE 1: CAPITAL-CYCLE TROUGH, BASE FORMATION, 40W EMA SLOPE & VOLUME ---
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        if progress_callback:
            progress_callback(
                (batch_num / total_batches) * 0.65,
                f"Stage 1/2: Scanning whole market for cyclical bases (Batch {batch_num}/{total_batches})...",
            )

        try:
            data = yf.download(
                batch,
                period="3y",
                interval="1wk",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            continue

        for ticker in batch:
            try:
                df = _extract_ticker_df(data, ticker, len(batch))
                if df.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                    continue

                df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                if len(df) < 55:
                    continue

                high = df["High"].squeeze()
                low = df["Low"].squeeze()
                close = df["Close"].squeeze()
                vol = df["Volume"].squeeze()

                cmp = float(close.iloc[-1])
                high_3y = float(high.max())
                low_52w = float(low.tail(52).min())

                if high_3y <= 0 or low_52w <= 0 or cmp <= 0:
                    continue

                # Liquidity Gate: >= ₹2 Cr average weekly traded value
                avg_vol_20w = float(vol.tail(20).mean())
                if (avg_vol_20w * cmp) < 2e7:
                    continue

                # Pillar 1: Deep Capital-Cycle Trough (30% to 78% off 3Y high, stabilized above 52W low)
                drawdown_pct = ((high_3y - cmp) / high_3y) * 100
                if not (30 <= drawdown_pct <= 78):
                    continue

                above_52w_low_pct = ((cmp - low_52w) / low_52w) * 100
                if not (3 <= above_52w_low_pct <= 50):
                    continue

                # Pillar 2: 40-Week EMA Proximity & Slope (Rejects steep falling-knife bull traps)
                ema40_series = close.ewm(span=40, adjust=False).mean()
                ema40_now = float(ema40_series.iloc[-1])
                ema40_
