# screener.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import date, timedelta
from typing import List, Dict, Optional
import requests  # For fetching S&P list if needed


def render_screener() -> None:
    """
    Stock Screener module: Screen stocks based on user-adjusted metrics.
    Supports a wide range of fundamental, valuation, profitability, liquidity,
    leverage, growth, efficiency, dividend, market, and technical metrics.
    """
    st.subheader("Stock Screener • Multi-Metric Filter")

    # ── Universe Selection ────────────────────────────────────────────────────
    with st.expander("Select Stock Universe", expanded=True):
        universe_option = st.radio(
            "Choose stock universe",
            options=["Custom List", "S&P 500", "NASDAQ 100", "Custom Sector"],
            index=0
        )

        if universe_option == "Custom List":
            tickers_input = st.text_input(
                "Comma-separated tickers",
                value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META",
                help="e.g., AAPL,MSFT,GC=F (note: non-stocks may lack fundamentals)"
            )
            universe = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        elif universe_option == "S&P 500":
            with st.spinner("Fetching S&P 500 tickers..."):
                try:
                    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
                    universe = df["Symbol"].tolist()
                    universe = [t.replace(".", "-") for t in universe]  # Normalize tickers
                    st.info(f"Loaded {len(universe)} S&P 500 tickers.")
                except Exception as e:
                    st.error(f"Failed to fetch S&P list: {e}. Use custom list.")
                    return

        elif universe_option == "NASDAQ 100":
            with st.spinner("Fetching NASDAQ 100 tickers..."):
                try:
                    df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
                    universe = df["Ticker"].tolist()
                    st.info(f"Loaded {len(universe)} NASDAQ 100 tickers.")
                except Exception as e:
                    st.error(f"Failed to fetch NASDAQ list: {e}. Use custom list.")
                    return

        elif universe_option == "Custom Sector":
            sector = st.selectbox(
                "Select sector",
                options=["Technology", "Healthcare", "Finance", "Energy", "Consumer Discretionary"]
            )
            # Placeholder: In real app, fetch tickers by sector from API or list
            # For now, use small hardcoded examples
            sector_map = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC"],
                "Healthcare": ["PFE", "JNJ", "MRNA", "ABBV", "LLY"],
                "Finance": ["JPM", "BAC", "WFC", "C", "GS"],
                "Energy": ["XOM", "CVX", "COP", "SLB"],
                "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"]
            }
            universe = sector_map.get(sector, [])
            st.info(f"Loaded {len(universe)} tickers for {sector}.")

        max_stocks = st.slider("Max stocks to screen (for speed)", 10, 500, min(100, len(universe)))
        universe = universe[:max_stocks]  # Limit for performance

    if not universe:
        st.info("No tickers selected. Enter or choose a universe.")
        return

    # ── Metric Filters Setup ──────────────────────────────────────────────────
    # Group metrics into categories for better UI
    categories = {
        "Valuation": [
            ("Trailing P/E", "trailingPE", "min_max"),
            ("Forward P/E", "forwardPE", "min_max"),
            ("P/B Ratio", "priceToBook", "min_max"),
            ("P/S Ratio", "priceToSalesTrailing12Months", "min_max"),
            ("PEG Ratio", "pegRatio", "min_max"),
            ("EV/EBITDA", "enterpriseToEbitda", "min_max"),
            ("EV/Revenue", "enterpriseToRevenue", "min_max")
        ],
        "Profitability": [
            ("ROE", "returnOnEquity", "min"),
            ("ROA", "returnOnAssets", "min"),
            ("Profit Margin", "profitMargins", "min"),
            ("Operating Margin", "operatingMargins", "min"),
            ("Gross Margin", "grossMargins", "min"),
            ("EBITDA Margin", "ebitdaMargins", "min")
        ],
        "Liquidity": [
            ("Current Ratio", "currentRatio", "min"),
            ("Quick Ratio", "quickRatio", "min")
        ],
        "Leverage": [
            ("Debt/Equity", "debtToEquity", "max"),
            ("Interest Coverage", "interestCoverage", "min"),
            ("Net Debt/EBITDA", None, "max")  # Computed later
        ],
        "Growth": [
            ("Revenue Growth (YoY)", "revenueGrowth", "min"),
            ("Earnings Growth (YoY)", "earningsGrowth", "min"),
            ("EPS Growth (TTM)", "trailingEps", "min"),  # Approx
            ("FCF Growth", None, "min")  # From cashflow
        ],
        "Efficiency": [
            ("Asset Turnover", None, "min"),  # Revenue / Assets
            ("Inventory Turnover", "inventoryTurnover", "min")
        ],
        "Dividends": [
            ("Dividend Yield", "dividendYield", "min"),
            ("Payout Ratio", "payoutRatio", "max"),
            ("Dividend Growth", "fiveYearAvgDividendYield", "min")
        ],
        "Market": [
            ("Market Cap (B)", "marketCap", "min_max", 1e9),  # Divide by 1e9 for B
            ("Beta", "beta", "min_max"),
            ("Avg Volume", "averageVolume", "min"),
            ("52W High %", "fiftyTwoWeekHigh", "max", True),  # Computed as (price / 52high - 1)
            ("52W Low %", "fiftyTwoWeekLow", "min", True)   # (price / 52low - 1)
        ],
        "Technical": [
            ("RSI (14-day)", None, "min_max"),  # Computed
            ("MACD", None, "min"),  # Computed
            ("50D MA > 200D MA", None, "bool"),  # Golden cross
            ("Price > 50D MA", None, "bool")
        ]
    }

    # User selects categories and metrics
    selected_categories = st.multiselect(
        "Select metric categories",
        options=list(categories.keys()),
        default=["Valuation", "Profitability"]
    )

    filters = {}
    with st.form(key="filter_form"):
        for cat in selected_categories:
            st.markdown(f"**{cat} Metrics**")
            for label, key, filter_type, *extra in categories[cat]:
                if filter_type == "min_max":
                    min_val, max_val = st.slider(
                        f"{label} range",
                        min_value=0.0,
                        max_value=100.0 if "Ratio" in label else 1e6,
                        value=(0.0, 100.0 if "Ratio" in label else 1e6),
                        key=f"{key}_range"
                    )
                    filters[label] = (key, "min_max", min_val, max_val, extra[0] if extra else 1)
                elif filter_type == "min":
                    val = st.number_input(f"{label} min", value=0.0, key=f"{key}_min")
                    filters[label] = (key, "min", val)
                elif filter_type == "max":
                    val = st.number_input(f"{label} max", value=100.0, key=f"{key}_max")
                    filters[label] = (key, "max", val)
                elif filter_type == "bool":
                    val = st.checkbox(f"{label}", value=False, key=f"{key}_bool")
                    filters[label] = (key, "bool", val)

        sort_by = st.selectbox(
            "Sort results by",
            options=[m[0] for c in categories.values() for m in c],
            index=0
        )
        ascending = st.checkbox("Sort ascending", value=False)

        submit = st.form_submit_button("Run Screener")

    if not submit:
        return

    # ── Fetch Data ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {len(universe)} tickers..."):
        tickers_obj = yf.Tickers(universe)
        data = {}

        for t in universe:
            try:
                info = tickers_obj.tickers[t].info
                if not info or "symbol" not in info:
                    continue

                # Fundamentals from info
                metrics = {
                    "Symbol": t,
                    "Name": info.get("longName", t),
                    "Sector": info.get("sector", ""),
                    "Industry": info.get("industry", ""),
                    "Price": info.get("currentPrice", np.nan)
                }

                # Add all possible keys
                for cat in categories.values():
                    for _, key, _, *extra in cat:
                        if key:
                            val = info.get(key, np.nan)
                            if extra and isinstance(extra[0], bool):  # Computed %
                                if "fiftyTwoWeekHigh" == key:
                                    high = val
                                    val = (metrics["Price"] / high - 1) * 100 if high else np.nan
                                    key = "52W High %"
                                elif "fiftyTwoWeekLow" == key:
                                    low = val
                                    val = (metrics["Price"] / low - 1) * 100 if low else np.nan
                                    key = "52W Low %"
                            elif extra:
                                val /= extra[0]  # e.g., marketCap in B
                            metrics[key] = val

                # Computed metrics (need financials)
                income = tickers_obj.tickers[t].get_income_stmt(yearly=True)
                balance = tickers_obj.tickers[t].get_balance_sheet(yearly=True)
                cashflow = tickers_obj.tickers[t].get_cashflow(yearly=True)

                if not income.empty and not balance.empty:
                    # Asset Turnover: Revenue / Avg Assets
                    if "Total Revenue" in income.index and "Total Assets" in balance.index:
                        rev = income.loc["Total Revenue"].iloc[-1] if not income.loc["Total Revenue"].empty else np.nan
                        assets = balance.loc["Total Assets"].iloc[-1] if not balance.loc["Total Assets"].empty else np.nan
                        metrics["Asset Turnover"] = rev / assets if assets else np.nan

                    # Net Debt/EBITDA
                    if "EBITDA" in income.index:
                        ebitda = income.loc["EBITDA"].iloc[-1]
                        debt = balance.loc["Total Debt"].iloc[-1] if "Total Debt" in balance.index else 0
                        cash = balance.loc["Cash And Cash Equivalents"].iloc[-1] if "Cash And Cash Equivalents" in balance.index else 0
                        metrics["Net Debt/EBITDA"] = (debt - cash) / ebitda if ebitda else np.nan

                if not cashflow.empty:
                    # FCF Growth approx (last vs prev)
                    if "Free Cash Flow" in cashflow.index and len(cashflow.loc["Free Cash Flow"]) > 1:
                        fcf_latest = cashflow.loc["Free Cash Flow"].iloc[-1]
                        fcf_prev = cashflow.loc["Free Cash Flow"].iloc[-2]
                        metrics["FCF Growth"] = (fcf_latest - fcf_prev) / abs(fcf_prev) if fcf_prev else np.nan

                # Technical metrics (download history)
                hist = tickers_obj.tickers[t].history(period="1y", interval="1d", auto_adjust=True)
                if not hist.empty:
                    # RSI
                    rsi = ta.rsi(hist["Close"], length=14).iloc[-1]
                    metrics["RSI (14-day)"] = rsi

                    # MACD
                    macd = ta.macd(hist["Close"])
                    metrics["MACD"] = macd["MACD_12_26_9"].iloc[-1] if not macd.empty else np.nan

                    # MA cross
                    sma50 = ta.sma(hist["Close"], length=50).iloc[-1]
                    sma200 = ta.sma(hist["Close"], length=200).iloc[-1]
                    metrics["50D MA > 200D MA"] = 1 if sma50 > sma200 else 0
                    metrics["Price > 50D MA"] = 1 if metrics["Price"] > sma50 else 0

                data[t] = metrics

            except Exception as e:
                st.warning(f"Skipped {t}: {e}")

        if not data:
            st.error("No data fetched. Try different universe.")
            return

        # Build DataFrame
        screen_df = pd.DataFrame.from_dict(data, orient="index").dropna(how="all", axis=1)

    # ── Apply Filters ─────────────────────────────────────────────────────────
    passing = screen_df.copy()
    for label, (key, f_type, *vals) in filters.items():
        if key is None:
            key = label  # For computed

        if key not in passing.columns:
            continue

        col = passing[key]

        if f_type == "min_max":
            min_v, max_v = vals[0], vals[1]
            passing = passing[(col >= min_v) & (col <= max_v)]

        elif f_type == "min":
            passing = passing[col >= vals[0]]

        elif f_type == "max":
            passing = passing[col <= vals[0]]

        elif f_type == "bool":
            passing = passing[col == int(vals[0])]

    if passing.empty:
        st.warning("No stocks pass all filters. Relax criteria.")
        return

    # ── Display Results ───────────────────────────────────────────────────────
    st.success(f"{len(passing)} stocks pass the screen out of {len(screen_df)}.")

    # Sort
    if sort_by in passing.columns:
        passing = passing.sort_values(by=sort_by, ascending=ascending)

    # Style & show table
    st.dataframe(
        passing.style.format({
            col: "{:.2f}" for col in passing.columns if pd.api.types.is_numeric_dtype(passing[col])
        }),
        use_container_width=True,
        height=400 if len(passing) > 10 else None
    )

    # Download
    csv = passing.to_csv()
    st.download_button(
        "Download results CSV",
        csv,
        "screener_results.csv",
        "text/csv"
    )


# For quick local testing
if __name__ == "__main__":
    st.set_page_config(page_title="Screener Test", layout="wide")
    render_screener()
