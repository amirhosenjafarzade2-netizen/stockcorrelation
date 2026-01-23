# screener.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import date, timedelta
from typing import List, Dict, Optional


def render_screener() -> None:
    st.subheader("Stock Screener • Multi-Metric Filter (Improved)")

    # ── Universe Selection ────────────────────────────────────────────────────
    with st.expander("1. Select Stock Universe", expanded=True):
        universe_option = st.radio(
            "Universe",
            ["Custom List", "S&P 500", "NASDAQ 100", "Custom Sector"],
            horizontal=True
        )

        if universe_option == "Custom List":
            tickers_input = st.text_input(
                "Tickers (comma separated)",
                value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META",
                help="You can also add commodities/ETFs like GC=F, SPY, QQQ"
            )
            universe = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        elif universe_option == "S&P 500":
            try:
                df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
                universe = df["Symbol"].str.replace(".", "-", regex=False).tolist()
                st.caption(f"Loaded {len(universe)} S&P 500 tickers")
            except:
                st.error("Could not load S&P 500 list. Using empty universe.")
                universe = []

        elif universe_option == "NASDAQ 100":
            try:
                df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
                universe = df["Ticker"].tolist()
                st.caption(f"Loaded {len(universe)} NASDAQ 100 tickers")
            except:
                st.error("Could not load NASDAQ 100 list.")
                universe = []

        elif universe_option == "Custom Sector":
            sector = st.selectbox("Sector", ["Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary"])
            sector_tickers = {
                "Technology": ["AAPL","MSFT","NVDA","AMD","INTC","QCOM","AVGO","TXN","ADI","MU"],
                "Healthcare": ["LLY","JNJ","MRK","ABBV","PFE","AMGN","UNH","ABT","MDT","TMO"],
                "Financials": ["JPM","BAC","WFC","C","GS","MS","SCHW","BLK","USB","PNC"],
                "Energy": ["XOM","CVX","COP","SLB","OXY","EOG","VLO","MPC","PSX","HAL"],
                "Consumer Discretionary": ["AMZN","TSLA","HD","NKE","MCD","SBUX","TJX","LOW","BKNG","MELI"]
            }
            universe = sector_tickers.get(sector, [])

        max_stocks = st.slider("Max stocks to process (performance)", 20, 500, 150)
        universe = universe[:max_stocks]

    if not universe:
        st.stop()

    # ── Metric Filters ────────────────────────────────────────────────────────
    with st.form("screener_filters"):
        st.markdown("### 2. Set Filters (leave blank = no restriction)")

        # Grouped inputs – 2 columns layout for min/max
        metrics = {
            "Valuation": {
                "Trailing P/E":           ("trailingPE",          float),
                "Forward P/E":            ("forwardPE",           float),
                "P/B Ratio":              ("priceToBook",         float),
                "P/S Ratio (TTM)":        ("priceToSalesTrailing12Months", float),
                "PEG Ratio":              ("pegRatio",            float),
                "EV/EBITDA":              ("enterpriseToEbitda",  float),
            },
            "Profitability": {
                "ROE (%)":                ("returnOnEquity",     lambda x: x*100 if x is not None else None),
                "ROA (%)":                ("returnOnAssets",      lambda x: x*100 if x is not None else None),
                "Profit Margin (%)":      ("profitMargins",       lambda x: x*100 if x is not None else None),
                "Operating Margin (%)":   ("operatingMargins",    lambda x: x*100 if x is not None else None),
            },
            "Liquidity & Leverage": {
                "Current Ratio":          ("currentRatio",        float),
                "Quick Ratio":            ("quickRatio",          float),
                "Debt / Equity":          ("debtToEquity",        float),
            },
            "Growth": {
                "Revenue Growth (YoY)":   ("revenueGrowth",       lambda x: x*100 if x is not None else None),
                "Earnings Growth (YoY)":  ("earningsGrowth",      lambda x: x*100 if x is not None else None),
            },
            "Dividends": {
                "Dividend Yield (%)":     ("dividendYield",       lambda x: x*100 if x is not None else None),
                "Payout Ratio (%)":       ("payoutRatio",         lambda x: x*100 if x is not None else None),
            },
            "Market": {
                "Market Cap ($B)":        ("marketCap",           lambda x: x / 1e9 if x else None),
                "Beta (5Y)":              ("beta",                float),
                "Avg Volume (3M)":        ("averageVolume",       float),
            }
        }

        filters = {}

        for category, items in metrics.items():
            st.markdown(f"**{category}**")
            cols = st.columns(2)
            for i, (label, (key, dtype)) in enumerate(items.items()):
                with cols[i % 2]:
                    min_val = st.number_input(
                        f"{label}  min",
                        value=None,
                        format="%.4f" if "Ratio" in label or "Cap" in label else "%.2f",
                        key=f"min_{key}"
                    )
                    max_val = st.number_input(
                        f"{label}  max",
                        value=None,
                        format="%.4f" if "Ratio" in label or "Cap" in label else "%.2f",
                        key=f"max_{key}"
                    )
                    if min_val is not None or max_val is not None:
                        filters[label] = (key, min_val, max_val, dtype)

        # Technical filters (computed later)
        st.markdown("**Technical (last close)**")
        rsi_min = st.number_input("RSI (14) ≥", value=None, min_value=0.0, max_value=100.0, key="rsi_min")
        rsi_max = st.number_input("RSI (14) ≤", value=None, min_value=0.0, max_value=100.0, key="rsi_max")
        if rsi_min is not None or rsi_max is not None:
            filters["RSI (14)"] = ("rsi", rsi_min, rsi_max, float)

        golden_cross = st.checkbox("Golden Cross (50 SMA > 200 SMA)", value=False, key="golden_cross")
        if golden_cross:
            filters["Golden Cross"] = ("golden_cross", True, None, bool)

        sort_by = st.selectbox("Sort results by", ["Market Cap ($B)", "Trailing P/E", "ROE (%)", "Dividend Yield (%)", "None"], index=0)
        sort_asc = st.checkbox("Sort ascending", value=False)

        run_button = st.form_submit_button("Run Screener", type="primary", use_container_width=True)

    if not run_button:
        st.stop()

    # ── Fetch & Filter ────────────────────────────────────────────────────────
    with st.spinner(f"Screening {len(universe)} tickers..."):
        results = []

        for symbol in universe:
            try:
                t = yf.Ticker(symbol)
                info = t.info
                if not info or "symbol" not in info:
                    continue

                row = {
                    "Symbol": symbol,
                    "Name": info.get("longName", symbol),
                    "Sector": info.get("sector", "N/A"),
                    "Price": info.get("currentPrice", np.nan),
                    "Market Cap ($B)": info.get("marketCap", np.nan) / 1e9 if info.get("marketCap") else np.nan,
                }

                # Fundamental metrics
                for label, (key, min_v, max_v, dtype) in filters.items():
                    if label in ["RSI (14)", "Golden Cross"]:
                        continue
                    val = info.get(key, np.nan)
                    if callable(dtype):
                        val = dtype(val)
                    row[label] = val

                results.append(row)

            except Exception:
                continue

        if not results:
            st.error("No valid data retrieved. Try a smaller universe or different tickers.")
            st.stop()

        df = pd.DataFrame(results).set_index("Symbol")

    # ── Apply filters ─────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)

    for label, (key, min_v, max_v, _) in filters.items():
        if label not in df.columns:
            continue
        col = df[label]
        if min_v is not None:
            mask &= (col >= min_v)
        if max_v is not None:
            mask &= (col <= max_v)

    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No stocks match your current filters. Try loosening some conditions.")
        st.stop()

    # ── Technical indicators (after filtering – only on survivors) ───────────
    if any("RSI" in k or "Golden Cross" in k for k in filters):
        with st.spinner("Calculating technical indicators..."):
            for symbol in filtered.index:
                try:
                    hist = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
                    if hist.empty:
                        continue

                    # RSI
                    rsi = ta.rsi(hist["Close"], length=14).iloc[-1]
                    filtered.at[symbol, "RSI (14)"] = rsi

                    # Golden Cross
                    sma50  = ta.sma(hist["Close"], length=50).iloc[-1]
                    sma200 = ta.sma(hist["Close"], length=200).iloc[-1]
                    filtered.at[symbol, "Golden Cross"] = sma50 > sma200 if pd.notna(sma50) and pd.notna(sma200) else False

                except:
                    continue

        # Apply RSI / Golden Cross filters
        if "RSI (14)" in filters:
            min_rsi, max_rsi, _ = filters["RSI (14)"]
            if min_rsi is not None:
                filtered = filtered[filtered["RSI (14)"] >= min_rsi]
            if max_rsi is not None:
                filtered = filtered[filtered["RSI (14)"] <= max_rsi]

        if "Golden Cross" in filters:
            filtered = filtered[filtered["Golden Cross"] == True]

    # ── Sort & Display ────────────────────────────────────────────────────────
    if sort_by != "None" and sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=sort_asc)

    st.success(f"**{len(filtered)} stocks** passed all filters (from {len(universe)} screened)")

    # Format numbers nicely
    st.dataframe(
        filtered.style.format({
            col: "{:,.2f}" if "Cap" in col else "{:.2f}" if "%" in col or "Ratio" in col or "Yield" in col else "{:.4f}"
            for col in filtered.columns if filtered[col].dtype in [float, np.float64]
        }),
        use_container_width=True,
        height=600
    )

    # Export
    csv = filtered.to_csv()
    st.download_button(
        "Download results as CSV",
        csv,
        file_name="screener_results.csv",
        mime="text/csv",
        use_container_width=True
    )
