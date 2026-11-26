import io
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

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

    return [], "NSE blocked the cloud server request. Please switch to 'Upload Screener.in CSV' in the sidebar."


def parse_screener_csv(uploaded_file):
    """Extract NSE tickers from a Screener.in or NSE CSV export."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    col = None
    for candidate in ["NSE Code", "Symbol", "SYMBOL", "Ticker", "symbol", "NSE"]:
        if candidate in df.columns:
            col = candidate
            break
    if not col:
        return [], "Error: CSV must have an 'NSE Code' or 'Symbol' column."
    syms = df[col].dropna().astype(str).str.strip().unique().tolist()
    tickers = [s if s.endswith(".NS") else f"{s}.NS" for s in syms if s and s.lower() != "nan"]
    return tickers, f"Uploaded CSV ({len(tickers)} stocks)"


def score_setup(drawdown_pct, base_range_pct, vol_ratio, dist_ema40_pct):
    """Calculate a 0-100 Cyclical Turnaround Quality Score."""
    depth_pts = min(25.0, max(0.0, (drawdown_pct - 20) * 0.6))
    tight_pts = min(25.0, max(0.0, (40 - base_range_pct) * 0.85))
    vol_pts = min(25.0, max(0.0, (vol_ratio - 0.8) * 12.5))

    if 0 <= dist_ema40_pct <= 8:
        ema_pts = 25.0
    elif -8 <= dist_ema40_pct < 0:
        ema_pts = 18.0
    elif 8 < dist_ema40_pct <= 15:
        ema_pts = 15.0
    else:
        ema_pts = 5.0

    return round(depth_pts + tight_pts + vol_pts + ema_pts, 1)


def _extract_ticker_df(data, ticker, batch_len):
    """Safely extract single-ticker OHLCV DataFrame across all yfinance versions."""
    if data is None or data.empty:
        return pd.DataFrame()

    if batch_len == 1:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if ticker in lvl0:
            return data[ticker].copy()
    return pd.DataFrame()


def scan_price_bases(
    tickers,
    min_drawdown=30,
    max_above_52w_low=45,
    min_vol_ratio=1.2,
    ema_mode="Reclaiming / Near 40W EMA (-10% to +15%)",
    progress_callback=None,
):
    """Stage 1: Fast batch download of 3Y weekly data to find cyclical bases."""
    results = []
    batch_size = 50
    total_batches = max(1, (len(tickers) + batch_size - 1) // batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        if progress_callback:
            progress_callback(batch_num / total_batches, f"Stage 1: Scanning prices (Batch {batch_num}/{total_batches})...")

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

                open_p = df["Open"].squeeze()
                high = df["High"].squeeze()
                low = df["Low"].squeeze()
                close = df["Close"].squeeze()
                vol = df["Volume"].squeeze()

                cmp = float(close.iloc[-1])
                high_3y = float(high.max())
                low_52w = float(low.tail(52).min())

                if high_3y <= 0 or low_52w <= 0 or cmp <= 0:
                    continue

                # Liquidity filter: skip illiquid stocks (< ₹2 Cr avg weekly turnover)
                avg_vol_20w = float(vol.tail(20).mean())
                if (avg_vol_20w * cmp) < 2e7:
                    continue

                # 1. Cyclical Drawdown
                drawdown_pct = ((high_3y - cmp) / high_3y) * 100
                if drawdown_pct < min_drawdown:
                    continue

                # 2. Base Zone
                above_52w_low_pct = ((cmp - low_52w) / low_52w) * 100
                if above_52w_low_pct < 2 or above_52w_low_pct > max_above_52w_low:
                    continue

                # 3. 40-Week EMA
                ema40 = float(close.ewm(span=40, adjust=False).mean().iloc[-1])
                dist_ema40_pct = ((cmp - ema40) / ema40) * 100

                if ema_mode == "Above 40W EMA Only" and cmp < ema40:
                    continue
                elif ema_mode == "Reclaiming / Near 40W EMA (-10% to +15%)" and not (-10 <= dist_ema40_pct <= 15):
                    continue

                # 4. Bullish Volume Accumulation (last 2 weeks: must be green or close in upper half)
                recent_df = df.tail(2)
                best_bull_vol = 0.0
                for _, wk in recent_df.iterrows():
                    wk_h, wk_l, wk_c, wk_o, wk_v = float(wk["High"]), float(wk["Low"]), float(wk["Close"]), float(wk["Open"]), float(wk["Volume"])
                    mid_range = (wk_h + wk_l) / 2.0
                    if wk_c >= wk_o or wk_c >= mid_range:
                        best_bull_vol = max(best_bull_vol, wk_v)

                vol_ratio = best_bull_vol / avg_vol_20w if avg_vol_20w > 0 else 0
                if vol_ratio < min_vol_ratio:
                    continue

                # 5. 12-Week Base Tightness & Stop-Loss
                high_12w = float(high.tail(12).max())
                low_12w = float(low.tail(12).min())
                base_range_pct = ((high_12w - low_12w) / low_12w) * 100

                sl_candidate = min(ema40, low_12w) if ema40 < cmp else low_12w
                if sl_candidate >= cmp:
                    sl_candidate = cmp * 0.92
                sl_price = round(sl_candidate, 2)
                risk_pct = round(((cmp - sl_price) / cmp) * 100, 1)

                score = score_setup(drawdown_pct, base_range_pct, vol_ratio, dist_ema40_pct)
                sym = ticker.replace(".NS", "")

                results.append({
                    "Symbol": sym,
                    "Ticker": ticker,
                    "Score": score,
                    "CMP (₹)": round(cmp, 2),
                    "Drawdown %": round(drawdown_pct, 1),
                    "40W EMA (₹)": round(ema40, 2),
                    "vs 40W EMA %": round(dist_ema40_pct, 1),
                    "12W Base %": round(base_range_pct, 1),
                    "Vol Surge (x)": round(vol_ratio, 2),
                    "Stop-Loss (₹)": sl_price,
                    "Risk %": risk_pct,
                    "Screener": f"https://www.screener.in/company/{sym}/consolidated/",
                    "TradingView": f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym}",
                })
            except Exception:
                continue

    return pd.DataFrame(results)


def enrich_fundamentals(df, progress_callback=None):
    """Stage 2: Fetch Sector, Market Cap, P/B, and Debt/Equity for Stage 1 survivors."""
    if df.empty:
        return df

    sectors, pbs, des, mcaps = [], [], [], []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if progress_callback:
            progress_callback((idx + 1) / total, f"Stage 2: Fundamentals for {row['Symbol']} ({idx + 1}/{total})...")
        try:
            info = yf.Ticker(row["Ticker"]).info
            sector = info.get("sector", "Unknown")
            pb = info.get("priceToBook", None)
            de = info.get("debtToEquity", None)
            de_ratio = round(de / 100.0, 2) if de is not None else None
            mcap = info.get("marketCap", None)
            mcap_cr = round(mcap / 1e7, 0) if mcap else None
        except Exception:
            sector, pb, de_ratio, mcap_cr = "Unknown", None, None, None

        sectors.append(sector)
        pbs.append(round(pb, 2) if pb else None)
        des.append(de_ratio)
        mcaps.append(mcap_cr)
        time.sleep(0.15)

    out = df.copy()
    out["Sector"] = sectors
    out["Mkt Cap (Cr)"] = mcaps
    out["P/B"] = pbs
    out["D/E"] = des
    return out
