import streamlit as st
import pandas as pd
import requests
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


def send_telegram_message(text):
    """Send a message via Telegram Bot using Streamlit Secrets."""
    token = st.secrets.get("TELEGRAM_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in Streamlit Secrets."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        res = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if res.status_code == 200:
            return True, "Telegram alert sent! Check your phone."
        return False, f"Telegram rejected it: {res.text}"
    except Exception as e:
        return False, f"Connection error: {e}"


def send_shortlist_to_telegram(df):
    """Format top 10 cyclical setups and send to Telegram."""
    top = df.head(10)
    lines = ["🔄 *Cycle Investing Alert — Top Setups*\n"]
    for _, r in top.iterrows():
        lines.append(
            f"• *{r['Symbol']}* (Score: {int(r['Score'])})\n"
            f"  CMP: ₹{r['CMP (₹)']} | SL: ₹{r['Stop-Loss Zone (₹)']} | {r['Entry Zone']}"
        )
    return send_telegram_message("\n".join(lines))


st.title("🔄 Cycle Investing Search Engine")
st.caption(
    "Whole-market NSE engine for realistic Entry Zones (capital-cycle troughs + flattening/curling 40W EMA + bullish volume) "
    "and realistic Exit Zones (staggered euphoria trims + 40W EMA trailing stop)."
)

# Quick 2-second Telegram Test Button
if st.button("🔔 Test Telegram Connection"):
    ok, msg = send_telegram_message("✅ *Test Successful!* Your Cycle Investing Search Engine is connected to Telegram.")
    if ok:
        st.success(msg)
    else:
        st.error(msg)

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

            if not results_df.empty:
                ok, tg_msg = send_shortlist_to_telegram(results_df)
                if ok:
                    st.success(f"📲 {tg_msg}")
                else:
                    st.warning(f"Scan finished, but Telegram failed: {tg_msg}")

    # Backup CSV loader (only shows if NSE blocks cloud server IP)
    if st.session_state.get("nse_blocked", False):
        uploaded = st.file_uploader("Upload NSE or Screener.in CSV to run the scan", type=["csv"])
        if uploaded and st
