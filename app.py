import streamlit as st
import pandas as pd
from scanner_engine import (
    DEFENSIVE_SECTORS,
    get_whole_market_symbols,
    parse_screener_csv,
    scan_price_bases,
    enrich_fundamentals,
)

st.set_page_config(
    page_title="NSE Cyclical Turnaround Scanner",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 NSE Cyclical Turnaround & Base Scanner")
st.caption("Whole-market scanner for deep-cycle troughs, tight Stage 1/2 bases, bullish volume accumulation, and 40W EMA position sizing.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Stock Universe")
    universe_mode = st.radio("Universe", ["Whole Market (NSE Auto)", "Upload Screener.in CSV"])
    uploaded_file = None
    if universe_mode == "Upload Screener.in CSV":
        uploaded_file = st.file_uploader("Upload CSV ('Symbol' or 'NSE Code')", type=["csv"])

    st.header("2. Scan Filters (Requires Re-Scan)")
    min_drawdown = st.slider("Min Drawdown from 3Y High (%)", 20, 75, 35, step=5)
    max_above_52w = st.slider("Max % Above 52W Low (Base Zone)", 10, 100, 45, step=5)
    ema_mode = st.selectbox(
        "40-Week EMA Filter",
        [
            "Reclaiming / Near 40W EMA (-10% to +15%)",
            "Above 40W EMA Only",
            "Any (Show All Bases)",
        ],
    )
    min_vol_ratio = st.slider("Min Bullish Volume Surge (x 20W Avg)", 0.8, 3.0, 1.2, step=0.1)

    st.divider()
    st.header("3. Live Filters & Sizing (Instant)")
    pure_cyclicals = st.toggle("Pure Cyclicals Only (Exclude Defensives)", value=True)
    max_de = st.slider("Max Debt-to-Equity", 0.5, 5.0, 1.5, step=0.25)
    capital = st.number_input("Portfolio Capital (₹)", min_value=50000, value=1000000, step=50000)
    risk_per_trade = st.slider("Max Risk per Trade (%)", 0.5, 3.0, 1.5, step=0.25)
    max_alloc_pct = st.slider("Max Allocation per Stock (%)", 5, 30, 20, step=5)

# --- RUN SCAN BUTTON ---
if st.button("🚀 Run Cyclical Scan", type="primary", use_container_width=True):
    if universe_mode == "Upload Screener.in CSV":
        if not uploaded_file:
            st.warning("Please upload a Screener.in CSV file first.")
            st.stop()
        tickers, source_label = parse_screener_csv(uploaded_file)
    else:
        tickers, source_label = get_whole_market_symbols()

    if not tickers:
        st.error(source_label)
        st.stop()

    st.session_state["source_label"] = source_label
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(pct, msg):
        progress_bar.progress(min(max(pct, 0.0), 1.0))
        status_text.text(msg)

    stage1_df = scan_price_bases(
        tickers,
        min_drawdown=min_drawdown,
        max_above_52w_low=max_above_52w,
        min_vol_ratio=min_vol_ratio,
        ema_mode=ema_mode,
        progress_callback=update_progress,
    )

    if stage1_df.empty:
        progress_bar.empty()
        status_text.empty()
        st.session_state["raw_results"] = pd.DataFrame()
        st.warning("No stocks matched Stage 1 price/base filters. Try lowering Drawdown or Volume Surge.")
    else:
        enriched_df = enrich_fundamentals(stage1_df, progress_callback=update_progress)
        progress_bar.empty()
        status_text.empty()
        st.session_state["raw_results"] = enriched_df

# --- DISPLAY RESULTS (UPDATES INSTANTLY WHEN LIVE FILTERS CHANGE) ---
if "raw_results" in st.session_state and not st.session_state["raw_results"].empty:
    df = st.session_state["raw_results"].copy()
    st.info(f"**Universe Scanned:** {st.session_state.get('source_label', 'NSE')}")

    # Apply Instant Filters
    if pure_cyclicals:
        df = df[~df["Sector"].isin(DEFENSIVE_SECTORS)]
    df = df[(df["D/E"].isna()) | (df["D/E"] <= max_de)]

    if df.empty:
        st.warning("All scanned stocks were filtered out by Sector or Debt/Equity limits. Adjust Section 3 sliders in the sidebar.")
    else:
        # Position Sizing with Max Allocation Cap
        risk_rupees = capital * (risk_per_trade / 100.0)
        max_pos_rupees = capital * (max_alloc_pct / 100.0)

        risk_per_share = (df["CMP (₹)"] - df["Stop-Loss (₹)"]).clip(lower=0.5)
        qty_by_risk = (risk_rupees / risk_per_share).astype(int)
        qty_by_cap = (max_pos_rupees / df["CMP (₹)"]).astype(int)

        df["Qty to Buy"] = pd.concat([qty_by_risk, qty_by_cap], axis=1).min(axis=1)
        df["Position (₹)"] = (df["Qty to Buy"] * df["CMP (₹)"]).round(0).astype(int)
        df = df.sort_values(by=["Score", "Vol Surge (x)"], ascending=[False, False])

        # Top Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Cyclical Setups Found", len(df))
        c2.metric("Top Turnaround Score", f"{df['Score'].iloc[0]} / 100")
        c3.metric("Max Risk / Trade", f"₹{int(risk_rupees):,}")

        display_cols = [
            "Symbol", "Score", "Sector", "CMP (₹)", "Drawdown %", "vs 40W EMA %",
            "12W Base %", "Vol Surge (x)", "P/B", "D/E", "Stop-Loss (₹)",
            "Risk %", "Qty to Buy", "Position (₹)", "Screener", "TradingView",
        ]

        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "Screener": st.column_config.LinkColumn("Screener", display_text="Open"),
                "TradingView": st.column_config.LinkColumn("Chart", display_text="View"),
            },
        )

        csv = df.drop(columns=["Ticker"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Shortlist CSV",
            data=csv,
            file_name="nse_cyclical_turnaround_shortlist.csv",
            mime="text/csv",
            use_container_width=True,
        )
