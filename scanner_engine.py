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
    elif -8 <= dist_ema40_pct < 0 and ema40_slope_pct >= -1.2:
        ema_pts = 12.0
    else:
        ema_pts = 6.0

    return round(depth_pts + tight_pts + vol_pts + ema_pts, 1)


def run_cycle_search_engine(tickers, progress_callback=None):
    """
    Fully automated 2-Stage Institutional Cycle Investing Search Engine.
    Stage 1: Whole-market cyclical base, 40W EMA slope, and bullish volume scan.
    Stage 2: Deep multi-year CFO/PAT, 3Y P/B Band, Capex Cycle, and Bank/NBFC-aware audit.
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
                (batch_num / total_batches) * 0.55,
                f"Stage 1/2: Scanning whole NSE market for cyclical bases (Batch {batch_num}/{total_batches})...",
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
                low_3y = float(low.min())
                low_52w = float(low.tail(52).min())

                if high_3y <= 0 or low_52w <= 0 or cmp <= 0:
                    continue

                # Liquidity Gate: >= ₹2 Cr average weekly traded value
                avg_vol_20w = float(vol.tail(20).mean())
                if (avg_vol_20w * cmp) < 2e7:
                    continue

                # Pillar 1: Cyclical Reset (20% to 78% off 3Y high, stabilized above 52W low)
                drawdown_pct = ((high_3y - cmp) / high_3y) * 100
                if not (20 <= drawdown_pct <= 78):
                    continue

                above_52w_low_pct = ((cmp - low_52w) / low_52w) * 100
                if not (2 <= above_52w_low_pct <= 55):
                    continue

                # Pillar 2: 40-Week EMA Proximity & Slope
                ema40_series = close.ewm(span=40, adjust=False).mean()
                ema40_now = float(ema40_series.iloc[-1])
                ema40_8w_ago = float(ema40_series.iloc[-9])
                if ema40_now <= 0 or ema40_8w_ago <= 0:
                    continue

                dist_ema40_pct = ((cmp - ema40_now) / ema40_now) * 100
                ema40_slope_pct = ((ema40_now - ema40_8w_ago) / ema40_8w_ago) * 100

                if not (-12 <= dist_ema40_pct <= 15):
                    continue
                if ema40_slope_pct < -2.5:
                    continue

                ema_trend = "Curling Up" if ema40_slope_pct >= 0.3 else "Flattening"

                # Base Tightness & Pivot (Last 12 weeks)
                high_12w = float(high.tail(12).max())
                low_12w = float(low.tail(12).min())
                base_range_pct = ((high_12w - low_12w) / low_12w) * 100
                if base_range_pct > 40:
                    continue

                # Pillar 3: Bullish Volume Accumulation (checks last 4 weeks)
                best_bull_vol = 0.0
                for _, wk in df.tail(4).iterrows():
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

                # 3-Year Price/Valuation Percentile (used for 3Y P/B Band)
                pb_band_pct = round(((cmp - low_3y) / (high_3y - low_3y)) * 100, 0) if high_3y > low_3y else 50.0

                # Realistic Entry Zone Classification
                dist_to_pivot_pct = ((high_12w - cmp) / cmp) * 100
                if cmp >= ema40_now and dist_to_pivot_pct <= 4.5 and vol_ratio >= 1.2:
                    entry_zone = "Active Entry Zone"
                elif cmp >= ema40_now * 0.96:
                    entry_zone = f"Near Pivot (₹{round(high_12w, 1)})"
                else:
                    entry_zone = "Quiet Base Building"

                # Realistic Stop-Loss Zone
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
                    "PB_Band_Pct": pb_band_pct,
                    "Stop-Loss Zone (₹)": sl_price,
                    "Risk %": risk_pct,
                    "Screener": f"https://www.screener.in/company/{sym}/consolidated/",
                    "TradingView": f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym}",
                })
            except Exception:
                continue

    if not stage1:
        return pd.DataFrame()

    # Keep Top 45 technical setups for deep multi-year financial statement audit
    s1_df = pd.DataFrame(stage1).sort_values(by="TechScore", ascending=False).head(45)

    # --- STAGE 2: DEEP FORENSIC & CAPITAL-CYCLE AUDIT ---
    final_rows = []
    total = len(s1_df)

    for idx, (_, row) in enumerate(s1_df.iterrows()):
        if progress_callback:
            progress_callback(
                0.55 + ((idx + 1) / total) * 0.45,
                f"Stage 2/2: Deep CFO/PAT & Capex audit on {row['Symbol']} ({idx + 1}/{total})...",
            )

        try:
            t_obj = yf.Ticker(row["Ticker"])
            info = t_obj.info
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            pb = info.get("priceToBook", None)
            de = info.get("debtToEquity", None)
            de_ratio = round(de / 100.0, 2) if de is not None else None
            insider_raw = info.get("heldPercentInsiders", None)
            insider_pct = f"{round(insider_raw * 100, 1)}%" if insider_raw is not None else "N/A"
            mcap = info.get("marketCap", None)
            mcap_cr = round(mcap / 1e7, 0) if mcap else None

            # Multi-year Cash Flow & Income Statement (CFO/PAT + Capex Cycle)
            cf = t_obj.cashflow
            fin = t_obj.financials
        except Exception:
            sector, industry, pb, de_ratio, insider_pct, mcap_cr = "Unknown", "Unknown", None, None, "N/A", None
            cf, fin = None, None

        time.sleep(0.15)

        if industry in NON_CYCLICAL_INDUSTRIES:
            continue

        sector_label = f"{sector} — {industry}" if industry != "Unknown" else sector
        is_financial = sector == "Financial Services"
        flags = []
        fund_pts = 0.0

        # 1. 3-Year P/B Band Score (max 10 pts)
        pb_band_pct = row["PB_Band_Pct"]
        pb_band_label = f"Bottom {int(pb_band_pct)}%"
        if pb_band_pct <= 25:
            fund_pts += 10.0
        elif pb_band_pct <= 45:
            fund_pts += 6.0
        else:
            fund_pts += 2.0

        # 2. Multi-Year CFO/PAT & Capex Cycle
        cfo_pat_display = "N/A"
        capex_status = "N/A"

        if is_financial:
            de_display = "N/A (Fin)"
            cfo_pat_display = "N/A (Fin)"
            capex_status = "N/A (Fin)"
            fund_pts += 15.0
            if pb is not None and pb <= 2.2:
                fund_pts += 10.0
            elif pb is not None and pb <= 3.5:
                fund_pts += 5.0
            else:
                flags.append("Check P/B & GNPA")
        else:
            # Debt/Equity Survival Check (max 8 pts)
            de_display = str(de_ratio) if de_ratio is not None else "N/A"
            if de_ratio is not None:
                if de_ratio > 2.5:
                    continue  # Reject extreme debt traps
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

            # Multi-Year Cumulative CFO / PAT Conversion (max 10 pts)
            cfo_vals = _get_statement_row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            pat_vals = _get_statement_row(fin, ["Net Income", "Net Income Common Stockholders"])

            if cfo_vals is not None and pat_vals is not None:
                cum_cfo = float(cfo_vals.sum())
                cum_pat = float(pat_vals.sum())
                if cum_pat > 0:
                    ratio = cum_cfo / cum_pat
                    cfo_pat_display = f"{round(ratio, 2)}x"
                    if ratio >= 0.75:
                        fund_pts += 10.0
                    elif ratio >= 0.45:
                        fund_pts += 5.0
                        flags.append("Weak CFO/PAT")
                    else:
                        flags.append("Poor CFO/PAT")
                elif cum_cfo > 0:
                    cfo_pat_display = "CFO+ / Loss"
                    fund_pts += 6.0  # Classic cyclical trough: accounting loss but cash-flow positive
                else:
                    cfo_pat_display = "Negative"
                    flags.append("Cash Burn")
            else:
                flags.append("Verify CFO/PAT")
                fund_pts += 3.0

            # Capital-Cycle Capex Trend (max 7 pts)
            capex_vals = _get_statement_row(cf, ["Capital Expenditure", "Capital Expenditures"])
            if capex_vals is not None and len(capex_vals) >= 2:
                # Capex is usually negative in cash flow statements, so use abs()
                latest_capex = abs(float(capex_vals.iloc[0]))
                prior_avg_capex = abs(float(capex_vals.iloc[1:].mean()))
                if prior_avg_capex > 0:
                    if latest_capex <= prior_avg_capex * 0.85:
                        capex_status = "Capex Freeze (Supply Tight)"
                        fund_pts += 7.0
                    elif latest_capex <= prior_avg_capex * 1.15:
                        capex_status = "Disciplined"
                        fund_pts += 5.0
                    else:
                        capex_status = "Expanding"
                        fund_pts += 3.0

        forensic_note = "Clean" if not flags else " | ".join(flags)
        total_score = round(min(100.0, row["TechScore"] + fund_pts), 1)

        if total_score >= 68 and row["Entry Zone"] == "Active Entry Zone":
            tier = "Tier 1: Actionable Entry Zone"
        else:
            tier = "Tier 2: Base Watchlist"

        rec = row.to_dict()
        rec["Score"] = total_score
        rec["Tier"] = tier
        rec["Sector"] = sector_label
        rec["Mkt Cap (Cr)"] = mcap_cr
        rec["P/B"] = round(pb, 2) if pb else None
        rec["3Y P/B Band"] = pb_band_label
        rec["D/E"] = de_display
        rec["CFO/PAT"] = cfo_pat_display
        rec["Capex Cycle"] = capex_status
        rec["Insider %"] = insider_pct
        rec["Forensic Check"] = forensic_note
        final_rows.append(rec)

    if not final_rows:
        return pd.DataFrame()

    out = pd.DataFrame(final_rows)
    return out.sort_values(by=["Score", "Vol Surge (x)"], ascending=[False, False]).reset_index(drop=True)


def evaluate_holdings_exit_zones(symbol_list):
    """
    Evaluates existing cyclical holdings against the 3-Layer Realistic Exit Zone.
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
