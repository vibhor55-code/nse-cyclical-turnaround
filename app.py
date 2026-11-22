import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Auto NSE Cyclical & Sector Scanner v7", layout="wide")

st.title("🔄 Auto NSE Cyclical Sector & Top 10 Workbench (v7)")
st.caption("Zero-Upload Nifty 500 Auto-Scan | Sector Breadth | RS (1–99) | Stage Duration | 3-Tranche Allocation")

# --- Sidebar Controls ---
st.sidebar.header("🌐 1. Auto Market Universe")
universe_mode = st.sidebar.radio(
    "Select Universe",
    ["Auto: Full Nifty 500 (All NSE Sectors)", "Auto: Nifty 200 (Faster Scan)"]
)

st.sidebar.header("💰 2. Capital & Risk Rules")
total_capital = st.sidebar.number_input("Total Portfolio Capital (₹)", min_value=50000, value=1000000, step=100000)
risk_per_trade = st.sidebar.slider("Max Risk per Trade (% of Capital)", 0.5, 3.0, 1.5, 0.25)
max_stock_pct = st.sidebar.slider("Max Allocation per Stock (%)", 5, 20, 10, 1)
max_sector_pct = st.sidebar.slider("Max Sector Cap (%) [Parmar Rule]", 15, 40, 25, 5)

st.sidebar.header("🛡️ 3. Kill-Switches & Filters")
min_turnover_cr = st.sidebar.slider("Liquidity Kill-Switch: Min Weekly Turnover (₹ Cr)", 0.0, 50.0, 10.0, 2.0)
max_ema40_ext = st.sidebar.slider("Overextended Warning: Max % Above 40W EMA", 15, 50, 25, 5)
min_drawdown = st.sidebar.slider("Min Drawdown from 3Y High (%) for Bottoming", 10, 70, 20, 5)
vol_threshold = st.sidebar.slider("Min Volume Surge (4W vs 20W Avg)", 1.0, 3.0, 1.2, 0.1)
min_sector_stocks = st.sidebar.slider("Min Stocks per Sector to Show Theme", 1, 10, 3, 1)

# Built-in fallback if NSE server blocks cloud IP
FALLBACK_UNIVERSE = [
    ("TATACHEM", "Chemicals"), ("DEEPAKNTR", "Chemicals"), ("GNFC", "Chemicals"), ("RAIN", "Chemicals"),
    ("SRF", "Chemicals"), ("ATUL", "Chemicals"), ("NAVINFLUOR", "Chemicals"), ("AARTIIND", "Chemicals"),
    ("TATASTEEL", "Metals & Mining"), ("SAIL", "Metals & Mining"), ("HINDALCO", "Metals & Mining"),
    ("JSWSTEEL", "Metals & Mining"), ("NMDC", "Metals & Mining"), ("VEDL", "Metals & Mining"),
    ("SBIN", "Financial Services"), ("HDFCBANK", "Financial Services"), ("ICICIBANK", "Financial Services"),
    ("CANBK", "Financial Services"), ("BANKBARODA", "Financial Services"), ("HDFCAMC", "Financial Services"),
    ("NAM-INDIA", "Financial Services"), ("CDSL", "Financial Services"), ("BSE", "Financial Services"),
    ("CHAMBLFERT", "Chemicals"), ("COROMANDEL", "Chemicals"), ("BALRAMCHIN", "FMCG"), ("TRIVENI", "FMCG"),
    ("HEG", "Capital Goods"), ("GRAPHITE", "Capital Goods"), ("LT", "Construction"), ("BHEL", "Capital Goods"),
    ("CUMMINSIND", "Capital Goods"), ("SIEMENS", "Capital Goods"), ("GPPL", "Services"), ("CONCOR", "Services"),
    ("DELHIVERY", "Services"), ("ONGC", "Oil Gas & Consumable Fuels"), ("BPCL", "Oil Gas & Consumable Fuels"),
    ("GAIL", "Oil Gas & Consumable Fuels"), ("TATAMOTORS", "Automobile"), ("M&M", "Automobile"),
    ("ASHOKLEY", "Automobile"), ("SUNPHARMA", "Healthcare"), ("CIPLA", "Healthcare"), ("DRREDDY", "Healthcare"),
    ("LAURUSLABS", "Healthcare"), ("TCS", "IT"), ("INFY", "IT"), ("WIPRO", "IT"), ("KPITTECH", "IT"),
    ("DLF", "Realty"), ("GODREJPROP", "Realty"), ("OBEROIRLTY", "Realty"), ("ULTRACEMCO", "Construction Materials"),
    ("AMBUJACEM", "Construction Materials"), ("ACC", "Construction Materials"), ("DALBHARAT", "Construction Materials")
]


@st.cache_data(ttl=86400)
def fetch_nse_universe(mode):
    urls = (
        [
            "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        ]
        if "500" in mode
        else [
            "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
            "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        ]
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and "Symbol" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                df = df[["Symbol", "Industry"]].dropna()
                df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
                df["Industry"] = df["Industry"].astype(str).str.strip()
                return list(df.itertuples(index=False, name=None)), "Live NSE Index"
        except Exception:
            continue
    return FALLBACK_UNIVERSE, "Built-in Multi-Sector Fallback (NSE busy)"


@st.cache_data(ttl=3600)
def get_nifty_weighted_return():
    try:
        nifty = yf.Ticker("^NSEI").history(period="2y", interval="1wk", auto_adjust=True)["Close"].dropna()
        if len(nifty) < 53:
            return 0.0
        r3 = (nifty.iloc[-1] / nifty.iloc[-13] - 1) * 100
        r6 = (nifty.iloc[-1] / nifty.iloc[-26] - 1) * 100
        r9 = (nifty.iloc[-1] / nifty.iloc[-39] - 1) * 100
        r12 = (nifty.iloc[-1] / nifty.iloc[-52] - 1) * 100
        return 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12
    except Exception:
        return 0.0


def count_stage_weeks(close, ema40):
    ema40_slope = (ema40 - ema40.shift(4)) / ema40.shift(4) * 100
    labels = []
    for i in range(len(close)):
        if i < 40 or pd.isna(ema40_slope.iloc[i]):
            labels.append("NA")
            continue
        p, e, sl = close.iloc[i], ema40.iloc[i], ema40_slope.iloc[i]
        if p > e and sl > 0.2:
            labels.append("Stage 2")
        elif p < e and sl < -0.5:
            labels.append("Stage 4")
        else:
            labels.append("Stage 1")

    curr_label = labels[-1]
    weeks = 0
    for lbl in reversed(labels):
        if lbl == curr_label:
            weeks += 1
        else:
            break
    return curr_label, weeks


def analyze_series(symbol, sector, close, vol, min_dd, vol_mult, min_turn_cr, max_ext, nifty_wret):
    close = close.dropna()
    vol = vol.dropna()
    if len(close) < 53 or len(vol) < 25:
        return None

    ema40 = close.ewm(span=40, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()

    curr_price = float(close.iloc[-1])
    curr_ema40 = float(ema40.iloc[-1])
    curr_ema10 = float(ema10.iloc[-1])
    ema40_4w_ago = float(ema40.iloc[-5])

    avg_wk_turnover_cr = round(float((close.iloc[-4:] * vol.iloc[-4:]).mean()) / 1e7, 2)

    high_3y = float(close.max())
    high_26w = float(close.iloc[-27:-1].max())
    drawdown_pct = round(((curr_price - high_3y) / high_3y) * 100, 1)
    dist_ema40_pct = round(((curr_price - curr_ema40) / curr_ema40) * 100, 1)

    avg_vol_4w = float(vol.iloc[-4:].mean())
    avg_vol_20w = float(vol.iloc[-24:-4].mean())
    vol_ratio = round(avg_vol_4w / avg_vol_20w, 2) if avg_vol_20w > 0 else 0.0

    r3 = (close.iloc[-1] / close.iloc[-13] - 1) * 100
    r6 = (close.iloc[-1] / close.iloc[-26] - 1) * 100
    r9 = (close.iloc[-1] / close.iloc[-39] - 1) * 100
    r12 = (close.iloc[-1] / close.iloc[-52] - 1) * 100
    raw_rs = round((0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12) - nifty_wret, 2)

    base_stage_lbl, stage_weeks = count_stage_weeks(close, ema40)

    above_ema40 = curr_price > curr_ema40
    ema40_rising = curr_ema40 > ema40_4w_ago
    ema40_slope_pct = ((curr_ema40 - ema40_4w_ago) / ema40_4w_ago) * 100
    breakout_26w = curr_price >= (high_26w * 0.98)

    if avg_wk_turnover_cr < min_turn_cr:
        stage = f"❌ KILL-SWITCH: Illiquid (<₹{min_turn_cr:.0f}Cr/wk)"
        rank = -1
    elif not above_ema40 and ema40_slope_pct < -0.5:
        stage = "🔴 Stage 4: Downturn (Kill-Switch)"
        rank = 0
    elif above_ema40 and breakout_26w and vol_ratio >= vol_mult and drawdown_pct <= -min_dd:
        stage = "🟢 Stage 1 Breakout (Bottom Turn)"
        rank = 4
    elif above_ema40 and dist_ema40_pct > max_ext:
        stage = f"⚠️ Extended Stage 2 (>+{max_ext}% vs 40W)"
        rank = 1
    elif above_ema40 and ema40_rising and curr_price > curr_ema10 and drawdown_pct > -15:
        stage = "🟠 Stage 3 / Near Peak (Pub Rule)"
        rank = 1
    elif above_ema40 and ema40_rising:
        stage = "🚀 Stage 2: Advancing Uptrend"
        rank = 3
    else:
        stage = "🟡 Stage 1: Basing / Watchlist"
        rank = 2

    return {
        "Theme / Sector": sector,
        "Symbol": symbol,
        "Stage": stage,
        "Stage Duration": f"{stage_weeks}w ({base_stage_lbl})",
        "Stage_Weeks_Num": stage_weeks,
        "Rank": rank,
        "Raw_RS": raw_rs,
        "Avg Wk Turnover (₹Cr)": avg_wk_turnover_cr,
        "CMP (₹)": round(curr_price, 2),
        "10W EMA (₹)": round(curr_ema10, 2),
        "40W EMA Stop (₹)": round(curr_ema40, 2),
        "3Y High (₹)": round(high_3y, 2),
        "26W High (₹)": round(high_26w, 2),
        "vs 40W EMA (%)": dist_ema40_pct,
        "Drawdown vs 3Y High (%)": drawdown_pct,
        "Vol Surge (4W/20W)": vol_ratio,
        "TradingView": f"https://www.tradingview.com/chart/?symbol=NSE:{symbol}",
        "Screener": f"https://www.screener.in/company/{symbol}/"
    }


def compute_expert_score(row):
    if row["Rank"] == -1:
        return 0.0, "Auto-Rejected: Illiquid"
    if row["Rank"] == 0:
        return 0.0, "Auto-Rejected: Stage 4 Downturn"
    if row["Rank"] == 1:
        return 18.0, "Caution: Overextended / Near Peak"

    score = 0.0
    reasons = []

    if row["Rank"] == 4:
        score += 35
        reasons.append("Fresh Stage 1 Breakout")
    elif row["Rank"] == 3:
        score += 28 if row["Stage_Weeks_Num"] <= 16 else 20
        reasons.append(f"Early Stage 2 ({row['Stage_Weeks_Num']}w)" if row["Stage_Weeks_Num"] <= 16 else f"Stage 2 ({row['Stage_Weeks_Num']}w)")
    else:
        score += 15
        reasons.append(f"Stage 1 Basing ({row['Stage_Weeks_Num']}w)")

    score += (row["RS Rating"] / 99.0) * 25.0
    if row["RS Rating"] >= 75:
        reasons.append(f"Strong RS ({row['RS Rating']})")

    score += min(row["Vol Surge (4W/20W)"] / 2.0, 1.0) * 15.0
    if row["Vol Surge (4W/20W)"] >= 1.3:
        reasons.append(f"Vol {row['Vol Surge (4W/20W)']}x")

    dd = abs(min(row["Drawdown vs 3Y High (%)"], 0))
    if 20 <= dd <= 65:
        score += 15
        reasons.append(f"Trough -{dd:.0f}%")
    elif 10 <= dd < 20:
        score += 8

    dist = row["vs 40W EMA (%)"]
    if 0 <= dist <= 12:
        score += 10
        reasons.append("Tight 40W Stop")
    elif 12 < dist <= 20:
        score += 5

    return round(score, 1), " | ".join(reasons)


def build_allocation_plan(top10_df, capital, risk_pct, max_stk_pct, max_sec_pct, max_ext):
    rows = []
    sector_used = {}
    total_deployed = 0.0
    max_stk_amt = capital * (max_stk_pct / 100.0)
    max_sec_amt = capital * (max_sec_pct / 100.0)
    risk_amt_per_trade = capital * (risk_pct / 100.0)

    for _, r in top10_df.iterrows():
        cmp = r["CMP (₹)"]
        ema40 = r["40W EMA Stop (₹)"]
        ema10 = r["10W EMA (₹)"]
        sec = r["Theme / Sector"]

        stop_loss = round(ema40, 2) if (ema40 < cmp * 0.97) else round(cmp * 0.90, 2)
        risk_per_sh = max(cmp - stop_loss, cmp * 0.03)
        derisk_2r = round(cmp + 2.0 * risk_per_sh, 2)

        raw_qty = int(risk_amt_per_trade / risk_per_sh)
        raw_alloc = raw_qty * cmp
        sec_remaining = max(0.0, max_sec_amt - sector_used.get(sec, 0.0))
        cap_remaining = max(0.0, capital - total_deployed)

        final_alloc = min(raw_alloc, max_stk_amt, sec_remaining, cap_remaining)
        final_qty = int(final_alloc / cmp) if cmp > 0 else 0
        final_alloc = round(final_qty * cmp, 0)

        sector_used[sec] = sector_used.get(sec, 0.0) + final_alloc
        total_deployed += final_alloc

        if r["vs 40W EMA (%)"] > max_ext:
            action = "⚠️ EXTENDED — Wait for Pullback"
            t1_zone = f"₹{ema10:.0f} (10W EMA)"
            t2_zone = f"₹{(ema10 + ema40)/2:.0f} (Mid)"
            t3_zone = f"₹{cmp * 1.05:.0f} (Resume)"
        elif r["Rank"] == 4:
            action = "🟢 BUY NOW (Stage 1 Breakout)"
            t1_zone = f"₹{cmp:.0f}"
            t2_zone = f"₹{min(ema10, cmp * 0.96):.0f} (Dip)"
            t3_zone = f"₹{cmp * 1.08:.0f} (+8%)"
        elif r["Rank"] == 3:
            action = "🚀 ACCUMULATE (Early Stg 2)"
            t1_zone = f"₹{cmp:.0f}"
            t2_zone = f"₹{min(ema10, cmp * 0.95):.0f} (10W EMA)"
            t3_zone = f"₹{cmp * 1.08:.0f} (Add)"
        else:
            action = "⚪ STAGGER / WATCH (Basing)"
            t1_zone = f"₹{r['26W High (₹)']:.0f} (Breakout)"
            t2_zone = f"₹{cmp:.0f} (Base)"
            t3_zone = f"₹{r['26W High (₹)'] * 1.08:.0f} (Confirm)"

        rows.append({
            "Top #": len(rows) + 1,
            "Symbol": r["Symbol"],
            "Theme / Sector": sec,
            "Action Plan": action,
            "Alloc (₹)": int(final_alloc),
            "Alloc (%)": round((final_alloc / capital) * 100, 1) if capital > 0 else 0.0,
            "Qty": final_qty,
            "CMP (₹)": cmp,
            "40W EMA Stop (₹)": stop_loss,
            "De-Risk Zone (2R)": derisk_2r,
            "Main Exit Plan": "Trail 40W EMA / Pub Rule",
            "Tranche 1 (40%)": t1_zone,
            "Tranche 2 (30%)": t2_zone,
            "Tranche 3 (30%)": t3_zone,
            "RS": r["RS Rating"],
            "Stage Duration": r["Stage Duration"],
            "TradingView": r["TradingView"]
        })

    return pd.DataFrame(rows), total_deployed


@st.cache_data(ttl=3600)
def fetch_deep_dive_data(symbol):
    tkr = yf.Ticker(f"{symbol}.NS")
    hist = tkr.history(period="3y", interval="1wk", auto_adjust=True)
    info = {}
    try:
        info = tkr.info or {}
    except Exception:
        pass
    return hist, info


# --- ONE-CLICK AUTO SCAN BUTTON ---
if st.button("🚀 Auto-Scan Whole Market (Find Turning Sectors & Top 10)", type="primary"):
    universe, source_label = fetch_nse_universe(universe_mode)
    st.info(f"Loaded **{len(universe)} stocks** from `{source_label}`. Batch-downloading weekly charts...")

    nifty_wret = get_nifty_weighted_return()
    sec_map = {sym: sec for sym, sec in universe}
    all_syms = [sym for sym, _ in universe]

    results = []
    batch_size = 100
    progress = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(all_syms), batch_size):
        batch = all_syms[i:i + batch_size]
        tickers = [f"{s}.NS" for s in batch]
        status_text.text(f"Scanning stocks {i+1} to {min(i+batch_size, len(all_syms))} of {len(all_syms)}...")
        try:
            data = yf.download(
                tickers, period="3y", interval="1wk",
                group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
            for sym in batch:
                t_key = f"{sym}.NS"
                try:
                    if len(batch) == 1:
                        sub = data
                    elif t_key in data.columns.get_level_values(0):
                        sub = data[t_key]
                    else:
                        continue
                    if "Close" in sub and "Volume" in sub:
                        res = analyze_series(
                            sym, sec_map[sym], sub["Close"], sub["Volume"],
                            min_drawdown, vol_threshold, min_turnover_cr, max_ema40_ext, nifty_wret
                        )
                        if res:
                            results.append(res)
                except Exception:
                    continue
        except Exception:
            continue
        progress.progress(min((i + batch_size) / len(all_syms), 1.0))

    status_text.empty()

    if results:
        res_df = pd.DataFrame(results)
        res_df["RS Rating"] = (res_df["Raw_RS"].rank(pct=True) * 98 + 1).round().astype(int)
        scores_and_reasons = res_df.apply(compute_expert_score, axis=1)
        res_df["Expert Score"] = [x[0] for x in scores_and_reasons]
        res_df["Why Chosen"] = [x[1] for x in scores_and_reasons]
        res_df = res_df.sort_values(by=["Expert Score", "RS Rating"], ascending=[False, False]).reset_index(drop=True)
        st.session_state["scan_results"] = res_df
    else:
        st.error("No data returned. Please try again in a minute.")

# --- Main Dashboard ---
if "scan_results" in st.session_state:
    res_df = st.session_state["scan_results"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stocks Scanned", len(res_df))
    c2.metric("🟢 Stage 1 Breakouts", int((res_df["Rank"] == 4).sum()))
    c3.metric("🚀 Stage 2 Advancing", int((res_df["Rank"] == 3).sum()))
    c4.metric("🔴 Stage 4 Downturn", int((res_df["Rank"] == 0).sum()))
    c5.metric("❌ Illiquid Rejected", int((res_df["Rank"] == -1).sum()))

    tab0, tab1, tab2, tab3 = st.tabs([
        "🧭 1. Turning Sectors (Auto-Detected)",
        "🏆 2. Top 10 Allocation & Accumulation Plan",
        "📊 3. All Stocks (RS & Stages)",
        "🔬 4. Deep-Dive (Chart + Porter/SOTP)"
    ])

    # --- TAB 0: SECTOR BREADTH FIRST ---
    with tab0:
        st.subheader("🧭 Auto-Detected Sectors Turning Up Across NSE")
        liquid_df = res_df[res_df["Rank"] >= 0]
        sec_df = liquid_df.groupby("Theme / Sector").agg(
            Total=("Symbol", "count"),
            Stage1_Breakout=("Rank", lambda x: (x == 4).sum()),
            Stage2_Uptrend=("Rank", lambda x: (x == 3).sum()),
            Stage1_Basing=("Rank", lambda x: (x == 2).sum()),
            Stage4_Downturn=("Rank", lambda x: (x == 0).sum()),
            Avg_RS=("RS Rating", "mean"),
            Avg_Drawdown=("Drawdown vs 3Y High (%)", "mean"),
            Leaders=("Symbol", lambda s: ", ".join(
                liquid_df.loc[s.index].query("Rank >= 3")["Symbol"].head(5).tolist()
            ))
        ).reset_index()

        sec_df["Turning Up (%)"] = round(
            ((sec_df["Stage1_Breakout"] + sec_df["Stage2_Uptrend"]) / sec_df["Total"]) * 100, 1
        )
        sec_df["Avg_RS"] = sec_df["Avg_RS"].round(0).astype(int)
        sec_df["Avg_Drawdown"] = sec_df["Avg_Drawdown"].round(1)
        sec_df = sec_df[sec_df["Total"] >= min_sector_stocks].sort_values(
            by=["Stage1_Breakout", "Avg_RS", "Turning Up (%)"], ascending=[False, False, False]
        )

        st.dataframe(
            sec_df[[
                "Theme / Sector", "Total", "Avg_RS", "Stage1_Breakout", "Stage2_Uptrend",
                "Stage1_Basing", "Stage4_Downturn", "Turning Up (%)", "Avg_Drawdown", "Leaders"
            ]],
            use_container_width=True,
            hide_index=True
        )

    # --- TAB 1: TOP 10 CAPITAL ALLOCATION ---
    with tab1:
        st.subheader("🏆 Expert Top 10: Capital Allocation, 40W EMA Stops & Accumulation Levels")
        st.caption(
            f"Sized for ₹{total_capital:,.0f} | {risk_per_trade}% risk/trade | {max_stock_pct}% max/stock | "
            f"{max_sector_pct}% sector cap | Min Liquidity: ₹{min_turnover_cr:.0f} Cr/wk."
        )

        top10_raw = res_df[res_df["Rank"] >= 2].head(10).copy()
        if not top10_raw.empty:
            plan_df, deployed = build_allocation_plan(
                top10_raw, total_capital, risk_per_trade, max_stock_pct, max_sector_pct, max_ema40_ext
            )
            cash_res = max(0.0, total_capital - deployed)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Capital Deployed", f"₹{deployed:,.0f}", f"{(deployed/total_capital)*100:.1f}% of portfolio")
            m2.metric("Cash Reserve Held", f"₹{cash_res:,.0f}", f"{(cash_res/total_capital)*100:.1f}% dry powder")
            m3.metric("Stocks Funded", int((plan_df["Qty"] > 0).sum()), f"Max {max_sector_pct}% per sector enforced")

            st.dataframe(
                plan_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "TradingView": st.column_config.LinkColumn("Chart", display_text="Open TV")
                }
            )
            st.download_button(
                "⬇️ Download Top 10 Allocation Plan CSV",
                plan_df.to_csv(index=False).encode("utf-8"),
                "auto_top10_capital_plan.csv",
                "text/csv"
            )
        else:
            st.warning("No stocks passed the current filters. Try lowering Min Weekly Turnover or Drawdown in the sidebar.")

    # --- TAB 2: ALL STOCKS SCREENER ---
    with tab2:
        col1, col2, col3 = st.columns(3)
        sel_sector = col1.selectbox("Filter by Theme / Sector", ["All Sectors"] + sorted(res_df["Theme / Sector"].unique().tolist()))
        sel_stages = col2.multiselect(
            "Filter by Stage",
            options=res_df["Stage"].unique().tolist(),
            default=[s for s in res_df["Stage"].unique().tolist() if "KILL-SWITCH" not in s]
        )
        min_rs = col3.slider("Minimum RS Rating", 1, 99, 1)

        filt = res_df.copy()
        if sel_sector != "All Sectors":
            filt = filt[filt["Theme / Sector"] == sel_sector]
        if sel_stages:
            filt = filt[filt["Stage"].isin(sel_stages)]
        filt = filt[filt["RS Rating"] >= min_rs]

        st.dataframe(
            filt[[
                "Symbol", "Theme / Sector", "Expert Score", "RS Rating", "Stage", "Stage Duration",
                "Avg Wk Turnover (₹Cr)", "CMP (₹)", "40W EMA Stop (₹)", "vs 40W EMA (%)",
                "Drawdown vs 3Y High (%)", "Vol Surge (4W/20W)", "TradingView", "Screener"
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "TradingView": st.column_config.LinkColumn("Chart", display_text="Open TV"),
                "Screener": st.column_config.LinkColumn("Fundamentals", display_text="Open Screener")
            }
        )

    # --- TAB 3: DEEP-DIVE WORKBENCH ---
    with tab3:
        st.subheader("Single-Stock Chart & Research Prompt")
        stock_choice = st.selectbox("Select a Stock to Audit", res_df["Symbol"].tolist())

        if stock_choice:
            row_data = res_df[res_df["Symbol"] == stock_choice].iloc[0]
            hist, info = fetch_deep_dive_data(stock_choice)

            st.markdown(
                f"### {stock_choice} — {row_data['Stage']} | **Duration:** `{row_data['Stage Duration']}` | "
                f"**RS:** `{row_data['RS Rating']}/99` | **Wk Turnover:** `₹{row_data['Avg Wk Turnover (₹Cr)']} Cr`"
            )

            if not hist.empty:
                hist["EMA40"] = hist["Close"].ewm(span=40, adjust=False).mean()
                hist["EMA10"] = hist["Close"].ewm(span=10, adjust=False).mean()
                high_26w = float(hist["Close"].iloc[-27:-1].max())

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist["Open"], high=hist["High"],
                    low=hist["Low"], close=hist["Close"], name="Weekly Price"
                ))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA40"], line=dict(color="orange", width=2), name="40W EMA (Trailing Stop)"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA10"], line=dict(color="cyan", width=1.2, dash="dot"), name="10W EMA (Accumulation)"))
                fig.add_hline(y=high_26w, line_dash="dash", line_color="green", annotation_text=f"26W Base High: ₹{high_26w:.1f}")
                fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📋 1-Click Gemini Deep-Dive Prompt (Porter's + SOTP + Kill-Switch)")
            prompt_text = (
                f"Perform a full cyclical research audit on NSE:{stock_choice} ({row_data['Theme / Sector']}). "
                f"Stage: {row_data['Stage']} | Duration: {row_data['Stage Duration']} | RS Rating: {row_data['RS Rating']}/99 | "
                f"CMP: ₹{row_data['CMP (₹)']} | 40W EMA Stop: ₹{row_data['40W EMA Stop (₹)']}. "
                f"Please run: (1) Porter's Five Forces & Chinese dumping/moat check, (2) Segment-wise SOTP Valuation, "
                f"(3) Bhaiya/Dharmawat/Parmar cyclical check (OCF/PAT, P/B, MCap/Sales, Pub Rule), and (4) Inversion Kill-Switch risks."
            )
            st.code(prompt_text, language="text")

            with st.expander("🛡️ Manual Porter's Five Forces & Inversion Kill-Switch Scratchpad"):
                pc1, pc2 = st.columns(2)
                with pc1:
                    p_rivalry = st.slider("Competitive Rivalry (Low dumping)", 1, 5, 3)
                    p_entrants = st.slider("Barriers to Entry", 1, 5, 3)
                    p_buyers = st.slider("Customer Pricing Power", 1, 5, 3)
                    p_suppliers = st.slider("Supplier Control", 1, 5, 3)
                    p_substitutes = st.slider("Low Threat of Substitutes", 1, 5, 3)
                    st.info(f"**Porter's Moat Score: {p_rivalry + p_entrants + p_buyers + p_suppliers + p_substitutes} / 25**")
                with pc2:
                    k1 = st.checkbox("Not a structural decline (temporary cycle trough)", value=True)
                    k2 = st.checkbox("Balance sheet survives downturn (Interest/PAT < 0.5)", value=True)
                    k3 = st.checkbox("Real cash flows (Multi-year OCF/PAT near 1.0)", value=True)
                    k4 = st.checkbox("Clean governance (No promoter pledge / serial dilution)", value=True)
                    st.success("✅ PASSED Kill-Switches") if all([k1, k2, k3, k4]) else st.error("❌ FAILED Kill-Switch")

            with st.expander("🧮 Manual Sum-of-the-Parts (SOTP) Valuation Calculator"):
                mcap_cr = round((info.get("marketCap", 0) or 0) / 1e7, 1)
                net_debt_cr = round(((info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)) / 1e7, 1)
                st.caption(f"Current Market Cap: **₹{mcap_cr:,.1f} Cr** | Net Debt: **₹{net_debt_cr:,.1f} Cr**")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    seg1_ebitda = st.number_input("Core Normalized EBITDA (₹ Cr)", value=100.0, step=10.0)
                    seg1_mult = st.number_input("Core EV/EBITDA Multiple", value=8.0, step=0.5)
                with sc2:
                    seg2_ebitda = st.number_input("Value-Add EBITDA (₹ Cr)", value=0.0, step=10.0)
                    seg2_mult = st.number_input("Value-Add Multiple", value=14.0, step=0.5)
                with sc3:
                    seg3_val = st.number_input("Non-Core / Land / Sub Value (₹ Cr)", value=0.0, step=10.0)
                    adj_net_debt = st.number_input("Net Debt to Subtract (₹ Cr)", value=float(net_debt_cr), step=10.0)
                sotp_eq = (seg1_ebitda * seg1_mult) + (seg2_ebitda * seg2_mult) + seg3_val - adj_net_debt
                upside = ((sotp_eq - mcap_cr) / mcap_cr * 100) if mcap_cr > 0 else 0.0
                r1, r2 = st.columns(2)
                r1.metric("Implied SOTP Equity Value", f"₹{sotp_eq:,.1f} Cr")
                r2.metric("SOTP Margin of Safety / Upside", f"{upside:+.1f}%")
