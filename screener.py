# screener.py - Improved version with Finviz scraping for custom sectors

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Optional
import requests  # For better Finviz fetching if needed


def render_screener() -> None:
    st.subheader("Stock Screener • Multi-Metric Filter")

    # ── Universe Selection ────────────────────────────────────────────────────
    with st.expander("1. Select Stock Universe", expanded=True):
        universe_option = st.radio(
            "Universe",
            ["Custom List", "S&P 500", "NASDAQ 100", "Custom Sector (via Finviz)"],
            horizontal=True
        )

        universe = []

        if universe_option == "Custom List":
            tickers_input = st.text_input(
                "Tickers (comma separated)",
                value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META",
                help="You can also add commodities/ETFs like GC=F, SPY, QQQ"
            )
            universe = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        elif universe_option == "S&P 500":
            with st.spinner("Fetching S&P 500 list from Wikipedia..."):
                try:
                    # Fetch with headers to mimic browser
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
                    tables = pd.read_html(response.text)
                    df = tables[0]  # Main table
                    universe = df["Symbol"].str.replace(".", "-", regex=False).tolist()
                    st.caption(f"Loaded {len(universe)} S&P 500 tickers")
                except Exception as e:
                    st.error(f"Failed to load S&P 500: {str(e)}. Use custom list.")

        elif universe_option == "NASDAQ 100":
            with st.spinner("Fetching NASDAQ 100 list from Wikipedia..."):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers)
                    tables = pd.read_html(response.text)
                    df = tables[4]  # Companies table
                    universe = df["Ticker"].tolist()
                    st.caption(f"Loaded {len(universe)} NASDAQ 100 tickers")
                except Exception as e:
                    st.error(f"Failed to load NASDAQ 100: {str(e)}. Use custom list.")

        elif universe_option == "Custom Sector (via Finviz)":
            sector = st.selectbox(
                "Select sector",
                ["Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary", "Consumer Staples", "Industrials", "Basic Materials", "Communication Services", "Utilities", "Real Estate"]
            )

            if sector:
                with st.spinner(f"Fetching all tickers in {sector} from Finviz (may take 30–60s for large sectors)..."):
                    try:
                        sector_map = {
                            "Technology": "sec_technology",
                            "Healthcare": "sec_healthcare",
                            "Financials": "sec_financial",
                            "Energy": "sec_energy",
                            "Consumer Discretionary": "sec_consumercyclical",
                            "Consumer Staples": "sec_consumernoncyclical",
                            "Industrials": "sec_industrials",
                            "Basic Materials": "sec_basicmaterials",
                            "Communication Services": "sec_communicationservices",
                            "Utilities": "sec_utilities",
                            "Real Estate": "sec_realestate"
                        }
                        sector_code = sector_map.get(sector, "")

                        all_tickers = []
                        page = 1
                        while True:
                            url = f"https://finviz.com/screener.ashx?v=111&f={sector_code}&r={page}"
                            headers = {'User-Agent': 'Mozilla/5.0'}
                            response = requests.get(url, headers=headers)
                            tables = pd.read_html(response.text)

                            if not tables or len(tables) < 8:  # Usually table index 7 is the screener table
                                break

                            df_page = tables[-1]  # Last table is the results
                            if df_page.empty or len(df_page) < 2:
                                break

                            # Extract tickers from column 1 (usually 'Ticker')
                            tickers_page = df_page.iloc[:, 1].dropna().unique().tolist()
                            if not tickers_page:
                                break

                            all_tickers.extend(tickers_page)
                            page += 20  # 20 per page on Finviz

                        universe = list(set(all_tickers))  # Remove duplicates
                        st.caption(f"Loaded {len(universe)} tickers from Finviz for {sector}")
                        if len(universe) < 10:
                            st.warning("Few tickers found — Finviz structure may have changed or temporary issue. Try custom list.")

                    except Exception as e:
                        st.error(f"Failed to fetch from Finviz: {str(e)}. Use custom list or try later.")

        max_stocks = st.slider("Max stocks to process (for speed)", 20, 500, min(150, len(universe)))
        universe = universe[:max_stocks]

    if not universe:
        st.info("No tickers selected. Enter or choose a universe.")
        return

    # ── Metric Filters ────────────────────────────────────────────────────────
    with st.form("filters_form"):
        st.markdown("### 2. Set Filters (blank = no filter)")

        metrics = {
            "Valuation": [
                ("Trailing P/E", "trailingPE"),
                ("Forward P/E", "forwardPE"),
                ("P/B Ratio", "priceToBook"),
                ("P/S Ratio", "priceToSalesTrailing12Months"),
                ("PEG Ratio", "pegRatio"),
                ("EV/EBITDA", "enterpriseToEbitda"),
            ],
            "Profitability": [
                ("ROE", "returnOnEquity"),
                ("ROA", "returnOnAssets"),
                ("Profit Margin", "profitMargins"),
                ("Operating Margin", "operatingMargins"),
            ],
            "Liquidity": [
                ("Current Ratio", "currentRatio"),
                ("Quick Ratio", "quickRatio"),
            ],
            "Leverage": [
                ("Debt/Equity", "debtToEquity"),
            ],
            "Growth": [
                ("Revenue Growth", "revenueGrowth"),
                ("Earnings Growth", "earningsGrowth"),
            ],
            "Dividends": [
                ("Dividend Yield", "dividendYield"),
                ("Payout Ratio", "payoutRatio"),
            ],
            "Market": [
                ("Market Cap (B)", "marketCap"),
                ("Beta", "beta"),
                ("Avg Volume", "averageVolume"),
            ],
            "Technical": [
                ("RSI (14-day)", None),
            ]
        }

        selected_categories = st.multiselect(
            "Select metric categories",
            options=list(metrics.keys()),
            default=["Valuation", "Profitability"]
        )

        filters = {}

        for cat in selected_categories:
            st.markdown(f"**{cat}**")
            for label, key in metrics[cat]:
                col_min, col_max = st.columns(2)
                with col_min:
                    min_val = st.number_input(
                        f"{label} min",
                        value=None,
                        step=0.01 if "Ratio" in label else 0.1
                    )
                with col_max:
                    max_val = st.number_input(
                        f"{label} max",
                        value=None,
                        step=0.01 if "Ratio" in label else 0.1
                    )
                if min_val is not None or max_val is not None:
                    filters[label] = (key, min_val, max_val)

        sort_by = st.selectbox(
            "Sort by",
            options=["Market Cap (B)", "Price", "Trailing P/E", "ROE", "Dividend Yield", "RSI (14-day)", "None"],
            index=0
        )
        sort_asc = st.checkbox("Sort ascending", value=False)

        submit = st.form_submit_button("Run Screener")

    if not submit:
        return

    # ── Fetch Data ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {len(universe)} tickers..."):
        data = {}

        for t in universe:
            try:
                info = yf.Ticker(t).info
                if not info or "symbol" not in info:
                    continue

                row = {
                    "Symbol": t,
                    "Name": info.get("longName", t),
                    "Sector": info.get("sector", ""),
                    "Price": info.get("currentPrice", np.nan),
                }

                for label, (key, _, _) in filters.items():
                    if key is None:
                        continue
                    val = info.get(key, np.nan)
                    if "Market Cap (B)" in label:
                        val /= 1e9
                    row[label] = val

                # Technical (require history)
                hist = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
                if not hist.empty:
                    rsi = ta.rsi(hist["Close"], length=14).iloc[-1]
                    row["RSI (14-day)"] = rsi if pd.notna(rsi) else np.nan

                data[t] = row

            except:
                continue

        if not data:
            st.error("No data fetched. Try different universe.")
            return

        df = pd.DataFrame.from_dict(data, orient="index")

    # ── Apply Filters ─────────────────────────────────────────────────────────
    passing = df.copy()

    for label, (key, min_v, max_v) in filters.items():
        if label not in passing.columns:
            continue
        col = passing[label]
        if min_v is not None:
            passing = passing[col >= min_v]
        if max_v is not None:
            passing = passing[col <= max_v]

    if passing.empty:
        st.warning("No stocks pass filters. Relax some criteria.")
        return

    # ── Sort & Display ────────────────────────────────────────────────────────
    if sort_by != "None" and sort_by in passing.columns:
        passing = passing.sort_values(sort_by, ascending=sort_asc)

    st.success(f"{len(passing)} stocks passed (from {len(df)} fetched)")

    st.dataframe(
        passing.style.format("{:.2f}"),
        use_container_width=True,
        height=500
    )

    csv = passing.to_csv()
    st.download_button("Download CSV", csv, "screener_results.csv", "text/csv")

# Standalone test
if __name__ == "__module__":
    st.set_page_config(page_title="Improved Screener", layout="wide")
    render_screener()
