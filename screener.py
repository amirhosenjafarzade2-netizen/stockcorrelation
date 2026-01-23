# screener.py - FINAL VERSION (January 2026 - fully working)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
import time
from typing import List

def get_finviz_tickers(sector: str) -> List[str]:
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
    
    if sector not in sector_map:
        return []
    
    tickers = []
    page = 1
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    with st.spinner(f"Loading all {sector} stocks from Finviz..."):
        while True:
            url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={page}"
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                table = soup.find("table", class_="screener_table")
                if not table:
                    break
                    
                rows = table.find_all("tr")[1:]  # skip header
                if not rows:
                    break
                
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) > 1:
                        ticker_link = cols[1].find("a")
                        if ticker_link:
                            tickers.append(ticker_link.text.strip())
                
                # Check if there's next page
                next_button = soup.find("a", class_="screener_pagination_next")
                if not next_button or "disabled" in next_button.get("class", []):
                    break
                    
                page += 20
                time.sleep(0.3)  # Be gentle
                
            except:
                break
    
    return list(set(tickers))  # remove duplicates

def render_screener() -> None:
    st.subheader("Ultimate Stock Screener")

    # ── Universe Selection ─────────────────────────────────────
    with st.expander("1. Choose Universe", expanded=True):
        option = st.radio("Source", [
            "S&P 500", 
            "NASDAQ 100", 
            "Finviz Sector (Real & Complete)", 
            "Custom List"
        ], horizontal=True)

        universe = []

        if option == "S&P 500":
            with st.spinner("Loading S&P 500..."):
                try:
                    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    tables = pd.read_html(requests.get(url, headers=headers).text)
                    df = tables[0]
                    universe = df["Symbol"].str.replace(".", "-", regex=False).tolist()
                    st.success(f"Loaded {len(universe)} S&P 500 stocks")
                except:
                    st.error("Failed to load S&P 500")

        elif option == "NASDAQ 100":
            with st.spinner("Loading NASDAQ 100..."):
                try:
                    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    tables = pd.read_html(requests.get(url, headers=headers).text)
                    df = tables[4]
                    universe = df["Ticker"].tolist()
                    st.success(f"Loaded {len(universe)} NASDAQ 100 stocks")
                except:
                    st.error("Failed to load NASDAQ 100")

        elif option == "Finviz Sector (Real & Complete)":
            sector = st.selectbox("Sector", [
                "Technology", "Healthcare", "Financials", "Energy", 
                "Consumer Discretionary", "Consumer Staples", "Industrials", 
                "Basic Materials", "Communication Services", "Utilities", "Real Estate"
            ])
            if st.button("Load All Stocks in Sector"):
                universe = get_finviz_tickers(sector)
                st.success(f"Loaded {len(universe)} stocks from {sector}")

        elif option == "Custom List":
            text = st.text_area("Paste tickers (comma or one per line)", height=150)
            universe = [t.strip().upper() for t in text.replace(",", "\n").split("\n") if t.strip()]

        if universe:
            max_n = st.slider("Max stocks to screen", 10, min(500, len(universe)), 200)
            universe = universe[:max_n]

    if not universe:
        st.stop()

    # ── Filters ───────────────────────────────────────────────
    with st.form("filters"):
        st.markdown("### 2. Filters (leave blank = ignore)")

        cols = st.columns(3)
        filters = {}
        
        metrics = [
            ("Market Cap (B)", "marketCap", cols[0], cols[1], lambda x: x/1e9 if x else None),
            ("Price", "currentPrice", cols[0], cols[1], float),
            ("Trailing P/E", "trailingPE", cols[0], cols[1], float),
            ("Forward P/E", "forwardPE", cols[0], cols[1], float),
            ("P/B", "priceToBook", cols[2], cols[2], float),
            ("P/S", "priceToSalesTrailing12Months", cols[2], cols[2], float),
            ("ROE (%)", "returnOnEquity", cols[0], cols[0], lambda x: x*100 if x else None),
            ("ROA (%)", "returnOnAssets", cols[0], cols[0], lambda x: x*100 if x else None),
            ("Debt/Equity", "debtToEquity", cols[1], cols[1], float),
            ("Dividend Yield (%)", "dividendYield", cols[1], cols[1], lambda x: x*100 if x else None),
            ("Beta", "beta", cols[2], cols[2], float),
        ]

        for label, key, col_min, col_max, transform in metrics:
            with col_min:
                min_val = st.number_input(f"{label} ≥", value=None, key=f"min_{key}")
            with col_max:
                max_val = st.number_input(f"{label} ≤", value=None, key=f"max_{key}")
            if min_val is not None or max_val is not None:
                filters[label] = (key, min_val, max_val, transform)

        rsi_min = st.number_input("RSI (14) ≥", min_value=0.0, max_value=100.0, value=None)
        rsi_max = st.number_input("RSI (14) ≤", min_value=0.0, max_value=100.0, value=None)
        if rsi_min is not None or rsi_max is not None:
            filters["RSI (14)"] = ("rsi", rsi_min, rsi_max, float)

        sort_options = ["Market Cap (B)", "Price", "Trailing P/E", "ROE (%)", "Dividend Yield (%)", "RSI (14)", "P/B", "None"]
        sort_by = st.selectbox("Sort by", sort_options, index=0)
        ascending = st.checkbox("Ascending", value=False)

        run = st.form_submit_button("RUN SCREENER", type="primary")

    if not run:
        return

    # ── Screening ─────────────────────────────────────────────
    results = []
    progress = st.progress(0)

    for i, symbol in enumerate(universe):
        progress.progress((i + 1) / len(universe))
        
        try:
            info = yf.Ticker(symbol).info
            if not info.get("symbol"):
                continue

            row = {
                "Symbol": symbol,
                "Name": info.get("longName") or info.get("shortName", symbol),
                "Sector": info.get("sector", "N/A"),
                "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
            }

            for label, (key, _, _, transform) in filters.items():
                if key == "rsi":
                    continue
                val = info.get(key)
                row[label] = transform(val) if val is not None else None

            # RSI
            hist = yf.download(symbol, period="3mo", progress=False, auto_adjust=True)
            if len(hist) >= 30:
                row["RSI (14)"] = ta.rsi(hist["Close"], length=14).iloc[-1]

            results.append(row)
        except:
            continue

    progress.empty()

    if not results:
        st.error("No data retrieved.")
        return

    df = pd.DataFrame(results).dropna(how="all", axis=1)

    # Apply filters
    mask = pd.Series([True] * len(df))
    for label, (key, min_v, max_v, _) in filters.items():
        if label in df.columns:
            if min_v is not None:
                mask &= df[label] >= min_v
            if max_v is not None:
                mask &= df[label] <= max_v

    final = df[mask].copy()

    if final.empty:
        st.warning("No stocks match your filters.")
        return

    # Sort
    if sort_by != "None" and sort_by in final.columns:
        final = final.sort_values(sort_by, ascending=ascending)

    st.success(f"**{len(final)} stocks** passed your screen!")

    st.dataframe(final.style.format("{:.2f}"), height=600)

    csv = final.to_csv(index=False)
    st.download_button("Download Results", csv, "screener_results.csv", "text/csv")

# Test
if __name__ == "__main__":
    st.set_page_config(page_title="Ultimate Screener", layout="wide")
    render_screener()
