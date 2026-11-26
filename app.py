import streamlit as st
import pandas as pd
import requests

def send_telegram_alert(df):
    token = st.secrets.get("TELEGRAM_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in Streamlit Secrets."
    
    top = df.head(10)
    lines = ["🔄 *Cycle Investing Alert — Top Setups*\n"]
    for _, r in top.iterrows():
        lines.append(
            f"• *{r['Symbol']}* (Score: {int(r['Score'])})\n"
            f"  CMP: ₹{r['CMP (₹)']} | SL: ₹{r['Stop-Loss Zone (₹)']} | {r['Entry Zone']}"
        )
    msg = "\n".join(lines)
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        res = requests.post(url, data={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        if res.status_code == 200:
            return True, "Sent to Telegram!"
        return False, res.text
    except Exception as e:
        return False, str(e)
from scanner_engine import (
    get_whole_market_symbols,
    parse_screener_csv,
    run_cycle_search_engine,
    evaluate_holdings_exit_zones,
)

st.set_page_config(
    page_title="Cycle Investing Search Engine",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Cycle Investing Search Engine")
st.caption(
    "Whole-market NSE engine for realistic Entry Zones (capital-cycle troughs + flattening/curling 40W EMA + bullish volume) "
    "and realistic Exit Zones (staggered euphoria trims + 40W EMA trailing stop)."
)

main_tab1, main_tab2 = st.tabs(["🔍 Whole-Market Entry Zone Scanner", "🛡️ My Holdings — Exit Zone Monitor"])

# ==========================================
# TAB 1: WHOLE-MARKET ENTRY ZONE SCANNER
# ==========================================
with main_tab1:
    if st.button("🚀 Run Whole-Market Cycle Search", type="primary", use_container_width=True):
        tickers, source_label = get_whole_market_symbols()

        if not tickers:
            st.error(source_label)
            st.session_state["nse_blocked"] = True
        else:
            st.session_state["nse_blocked"] = False
            st.session_state["source_label"] = source_label
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(min(max(pct, 0.0), 1.0))
                status_text.text(msg)

            results_df = run_cycle_search_engine(tickers, progress_callback=update_progress)
            progress_bar.empty()
            status_text.empty()
            st.session_state["results"] = results_df

    # Backup CSV loader (only shows if NSE blocks cloud server IP)
    if st.session_state.get("nse_blocked", False):
        uploaded = st.file_uploader("Upload NSE or Screener.in CSV to run the scan", type=["csv"])
        if uploaded and st.button("Scan Uploaded CSV", type="primary"):
            tickers, source_label = parse_screener_csv(uploaded)
            if not tickers:
                st.error(source_label)
            else:
                st.session_state["source_label"] = source_label
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_df = run_cycle_search_engine(
                    tickers,
                    progress_callback=lambda p, m: (progress_bar.progress(min(max(p, 0.0), 1.0)), status_text.text(m)),
                )
                progress_bar.empty()
                status_text.empty()
                st.session_state["results"] = results_df

    if "results" in st.session_state:
        df = st.session_state["results"]
        if df.empty:
            st.warning("No stocks passed all 4 institutional cyclical pillars today.")
        else:
            st.info(f"**Universe Scanned:** {st.session_state.get('source_label', 'NSE Whole Market')}")

            tier1 = df[df["Tier"] == "Tier 1: Actionable Entry Zone"]
            tier2 = df[df["Tier"] == "Tier 2: Base Watchlist"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Cyclical Survivors", len(df))
            c2.metric("Tier 1 (Active Entry Zones)", len(tier1))
            c3.metric("Tier 2 (Base Watchlist)", len(tier2))

            # Sector-Cycle Cluster Radar
            sector_counts = (
                df.groupby("Sector")
                .agg(Stocks=("Symbol", "count"), Avg_Score=("Score", "mean"))
                .reset_index()
                .sort_values(by=["Stocks", "Avg_Score"], ascending=[False, False])
            )
            sector_counts["Avg_Score"] = sector_counts["Avg_Score"].round(1)

            st.subheader("📡 Sector-Cycle Cluster Radar")
            st.caption("Sectors with 3+ stocks basing together signal an industry-wide capital-cycle turn.")
            st.dataframe(sector_counts, use_container_width=True, hide_index=True)

            display_cols = [
                "Symbol", "Score", "Entry Zone", "Sector", "CMP (₹)", "Base Pivot (₹)",
                "Drawdown %", "vs 40W EMA %", "40W Slope", "12W Base %", "Vol Surge (x)",
                "P/B", "D/E", "CFO", "Forensic Check", "Stop-Loss Zone (₹)", "Risk %",
                "Screener", "TradingView",
            ]

            col_cfg = {
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "Screener": st.column_config.LinkColumn("Screener", display_text="Open"),
                "TradingView": st.column_config.LinkColumn("Chart", display_text="View"),
            }

            sub1, sub2 = st.tabs([
                f"🔥 Tier 1: Active Entry Zones ({len(tier1)})",
                f"👀 Tier 2: Base Watchlist ({len(tier2)})",
            ])

            with sub1:
                if tier1.empty:
                    st.write("No Tier 1 active entries right now—check Tier 2 for bases building near the 40W EMA.")
                else:
                    st.dataframe(tier1[display_cols], use_container_width=True, hide_index=True, column_config=col_cfg)

            with sub2:
                if tier2.empty:
                    st.write("No Tier 2 watchlist setups right now.")
                else:
                    st.dataframe(tier2[display_cols], use_container_width=True, hide_index=True, column_config=col_cfg)

            csv = df.drop(columns=["Ticker", "TechScore"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Research Shortlist (CSV)",
                data=csv,
                file_name="cycle_investing_shortlist.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ==========================================
# TAB 2: MY HOLDINGS — EXIT ZONE MONITOR
# ==========================================
with main_tab2:
    st.subheader("🛡️ Cyclical Holdings Exit Zone Monitor")
    st.caption(
        "Enter your current cyclical holdings (NSE symbols separated by commas) to check whether to "
        "**HOLD**, **TRIM (Euphoria Zone)**, or **EXIT (40W EMA Broken)**."
    )

    holdings_input = st.text_area(
        "Enter NSE Symbols (comma-separated)",
        placeholder="Example: SBIN, TATASTEEL, GPPL, HEG, HDFCAMC",
        height=80,
    )

    if st.button("Check Exit Zones", type="primary"):
        symbols = [s.strip() for s in holdings_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Please enter at least one NSE symbol.")
        else:
            with st.spinner("Evaluating weekly trends & 40-Week EMA exit zones..."):
                exit_df = evaluate_holdings_exit_zones(symbols)

            if exit_df.empty:
                st.error("Could not fetch data for those symbols. Double-check the NSE spellings.")
            else:
                st.dataframe(
                    exit_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "TradingView": st.column_config.LinkColumn("Chart", display_text="View"),
                    },
                )
