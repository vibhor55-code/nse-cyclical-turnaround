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
                ema40_8w_ago = float(ema40_series.iloc[-9])
                if ema40_now <= 0 or ema40_8w_ago <= 0:
                    continue

                dist_ema40_pct = ((cmp - ema40_now) / ema40_now) * 100
                ema40_slope_pct = ((ema40_now - ema40_8w_ago) / ema40_8w_ago) * 100

                if not (-10 <= dist_ema40_pct <= 15):
                    continue
                if ema40_slope_pct < -2.0:
                    continue

                ema_trend = "Curling Up" if ema40_slope_pct >= 0.3 else "Flattening"

                # Base Tightness & Pivot (Last 12 weeks)
                high_12w = float(high.tail(12).max())
                low_12w = float(low.tail(12).min())
                base_range_pct = ((high_12w - low_12w) / low_12w) * 100
                if base_range_pct > 38:
                    continue

                # Pillar 3: Institutional Bullish Volume Accumulation (green or upper-half close)
                best_bull_vol = 0.0
                for _, wk in df.tail(2).iterrows():
                    wk_h, wk_l, wk_c, wk_o, wk_v = (
                        float(wk["High"]),
                        float(wk["Low"]),
                        float(wk["Close"]),
                        float(wk["Open"]),
                        float(wk["Volume"]),
                    )
                    if wk_c >= wk_o or wk_c >= (wk_h + wk_l) / 2.0:
                        best_bull_vol = max(best_bull_vol, wk_v)

                vol_ratio = best_bull_vol / avg_vol_20w if avg_vol_20w > 0 else 0
                if vol_ratio < 1.2:
                    continue

                # Realistic Entry Zone Classification
                dist_to_pivot_pct = ((high_12w - cmp) / cmp) * 100
                if cmp >= ema40_now and dist_to_pivot_pct <= 4.0:
                    entry_zone = "Active Entry Zone"
                elif cmp >= ema40_now * 0.96:
                    entry_zone = f"Near Pivot (₹{round(high_12w, 1)})"
                else:
                    entry_zone = "Early Base Building"

                # Realistic Stop-Loss Zone (lower of 40W EMA or 12W Base Low)
                sl_candidate = min(ema40_now, low_12w) if ema40_now < cmp else low_12w
                if sl_candidate >= cmp:
                    sl_candidate = cmp * 0.92
                sl_price = round(sl_candidate, 2)
                risk_pct = round(((cmp - sl_price) / cmp) * 100, 1)

                tech_score = _score_technical_setup(
                    drawdown_pct, base_range_pct, vol_ratio, dist_ema40_pct, ema40_slope_pct
                )
                sym = ticker.replace(".NS", "")

                stage1.append({
                    "Symbol": sym,
                    "Ticker": ticker,
                    "TechScore": tech_score,
                    "Entry Zone": entry_zone,
                    "CMP (₹)": round(cmp, 2),
                    "Base Pivot (₹)": round(high_12w, 2),
                    "Drawdown %": round(drawdown_pct, 1),
                    "40W EMA (₹)": round(ema40_now, 2),
                    "vs 40W EMA %": round(dist_ema40_pct, 1),
                    "40W Slope": ema_trend,
                    "12W Base %": round(base_range_pct, 1),
                    "Vol Surge (x)": round(vol_ratio, 2),
                    "Stop-Loss Zone (₹)": sl_price,
                    "Risk %": risk_pct,
                    "Screener": f"https://www.screener.in/company/{sym}/consolidated/",
                    "TradingView": f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym}",
                })
            except Exception:
                continue

    if not stage1:
        return pd.DataFrame()

    # --- STAGE 2: FORENSIC AUDIT (BANK/NBFC AWARE + HONEST DATA FLAGS) ---
    s1_df = pd.DataFrame(stage1)
    final_rows = []
    total = len(s1_df)

    for idx, (_, row) in enumerate(s1_df.iterrows()):
        if progress_callback:
            progress_callback(
                0.65 + ((idx + 1) / total) * 0.35,
                f"Stage 2/2: Forensic & sector audit on {row['Symbol']} ({idx + 1}/{total})...",
            )

        try:
            info = yf.Ticker(row["Ticker"]).info
            sector = info.get("sector", "Unknown")
            pb = info.get("priceToBook", None)
            de = info.get("debtToEquity", None)
            de_ratio = round(de / 100.0, 2) if de is not None else None
            cfo = info.get("operatingCashflow", None)
            mcap = info.get("marketCap", None)
            mcap_cr = round(mcap / 1e7, 0) if mcap else None
        except Exception:
            sector, pb, de_ratio, cfo, mcap_cr = "Unknown", None, None, None, None

        time.sleep(0.15)

        if sector in DEFENSIVE_SECTORS:
            continue

        is_financial = sector == "Financial Services"
        flags = []
        fund_pts = 0.0

        if is_financial:
            fund_pts += 8.0
            de_display = "N/A (Fin)"
            if pb is not None and pb <= 2.2:
                fund_pts += 8.0
            elif pb is not None and pb <= 3.5:
                fund_pts += 4.0
            else:
                flags.append("Check P/B & GNPA")
            cfo_status = "N/A (Fin)"
        else:
            de_display = str(de_ratio) if de_ratio is not None else "N/A"
            if de_ratio is not None:
                if de_ratio > 2.5:
                    continue
                elif de_ratio > 1.5:
                    flags.append("Elevated Debt")
                    fund_pts += 2.0
                elif de_ratio <= 0.8:
                    fund_pts += 8.0
                else:
                    fund_pts += 5.0
            else:
                flags.append("Verify Debt")
                fund_pts += 3.0

            if cfo is not None and cfo > 0:
                fund_pts += 7.0
                cfo_status = "Positive"
            elif cfo is not None and cfo <= 0:
                cfo_status = "Negative"
                flags.append("Negative CFO")
            else:
                cfo_status = "N/A"
                flags.append("Verify CFO")
                fund_pts += 2.0

            if pb is not None and pb <= 3.0:
                fund_pts += 5.0
            elif pb is None:
                fund_pts += 2.0

        forensic_note = "Clean" if not flags else " | ".join(flags)
        total_score = round(min(100.0, row["TechScore"] + fund_pts), 1)

        if total_score >= 70 and row["Entry Zone"] == "Active Entry Zone":
            tier = "Tier 1: Actionable Entry Zone"
        else:
            tier = "Tier 2: Base Watchlist"

        rec = row.to_dict()
        rec["Score"] = total_score
        rec["Tier"] = tier
        rec["Sector"] = sector
        rec["Mkt Cap (Cr)"] = mcap_cr
        rec["P/B"] = round(pb, 2) if pb else None
        rec["D/E"] = de_display
        rec["CFO"] = cfo_status
        rec["Forensic Check"] = forensic_note
        final_rows.append(rec)

    if not final_rows:
        return pd.DataFrame()

    out = pd.DataFrame(final_rows)
    return out.sort_values(by=["Score", "Vol Surge (x)"], ascending=[False, False]).reset_index(drop=True)


def evaluate_holdings_exit_zones(symbol_list):
    """
    Evaluates existing cyclical holdings against the 3-Layer Realistic Exit Zone:
    1. EXIT ZONE (Trend Broken): Weekly close below 40-Week EMA.
    2. TRIM / EUPHORIA ZONE: Stretched > 45% above 40W EMA (or > 35% with high-vol red week).
    3. CAUTION ZONE: Within 0% to +4% of 40W EMA.
    4. HOLD ZONE: Riding cleanly above 40W EMA.
    """
    clean_syms = [s.strip().upper().replace(".NS", "") for s in symbol_list if s.strip()]
    if not clean_syms:
        return pd.DataFrame()

    tickers = [f"{s}.NS" for s in clean_syms]
    try:
        data = yf.download(
            tickers,
            period="3y",
            interval="1wk",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for sym, ticker in zip(clean_syms, tickers):
        try:
            df = _extract_ticker_df(data, ticker, len(tickers))
            if df.empty or len(df) < 42:
                continue
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            close = df["Close"].squeeze()
            open_p = df["Open"].squeeze()
            vol = df["Volume"].squeeze()

            cmp = float(close.iloc[-1])
            ema40_series = close.ewm(span=40, adjust=False).mean()
            ema40 = float(ema40_series.iloc[-1])
            dist_ema40_pct = ((cmp - ema40) / ema40) * 100

            avg_vol_20w = float(vol.tail(20).mean())
            last_vol_ratio = float(vol.iloc[-1]) / avg_vol_20w if avg_vol_20w > 0 else 1.0
            is_red_week = float(close.iloc[-1]) < float(open_p.iloc[-1])

            if cmp < ema40:
                status = "🔴 EXIT ZONE (Below 40W EMA)"
                action = "Exit / Protect Capital — Upcycle Trend Broken"
            elif dist_ema40_pct >= 45 or (dist_ema40_pct >= 35 and is_red_week and last_vol_ratio >= 1.5):
                status = "🟠 TRIM / EUPHORIA ZONE"
                action = "Book Partial Profits (25%-50%) — Stretched Above 40W EMA"
            elif dist_ema40_pct <= 4.0:
                status = "🟡 CAUTION (Testing 40W EMA)"
                action = "Watch Weekly Close Carefully"
            else:
                status = "🟢 HOLD (Upcycle Intact)"
                action = f"Ride Trend — Trail Stop at ₹{round(ema40, 1)}"

            rows.append({
                "Symbol": sym,
                "Exit Zone Status": status,
                "CMP (₹)": round(cmp, 2),
                "40W EMA Trail (₹)": round(ema40, 2),
                "vs 40W EMA %": round(dist_ema40_pct, 1),
                "Weekly Vol (x)": round(last_vol_ratio, 2),
                "Suggested Action": action,
                "TradingView": f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym}",
            })
        except Exception:
            continue

    return pd.DataFrame(rows)
