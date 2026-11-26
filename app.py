import streamlit as st
import pandas as pd
import requests
from scanner_engine import (
    get_whole_market_symbols,
    parse_screener_csv,
    run_cycle_search_engine,
    generate_executive_sector_report,
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


def send_executive_report_to_telegram(df, sector_brief):
    """Send Executive Sector Cycle Brief + Top 8 Setups to Telegram."""
    lines = ["📋 *EXECUTIVE SECTOR CYCLE BRIEF*\n"]
    for item in sector_brief[:5]:
        lines.append(item["tg_bullet"])

    lines.append("\n🔥 *TOP CYCLICAL SETUPS*")
    for _, r in df.head(8).iterrows():
        lines.append(
            f"• *{r['Symbol']}* (Score {int(r['Score'])}) | ₹{r['CMP (₹)']} | SL ₹{r['Stop-Loss Zone (₹)']} | {r['Entry Zone']}"
        )

    return send_telegram_message("\n".join(lines))


st.title("🔄 Cycle Investing Search Engine")
st.caption(
    "Whole-market NSE engine combining Executive Sector Cycle Briefs, 3Y P/B Bands, Multi-Year CFO/PAT Forensics, "
    "Capex Supply Checks, 40-Week EMA Bases, and Realistic Exit Zones."
)

# Quick 2-second Telegram Test Button
if st.button("🔔 Test Telegram
