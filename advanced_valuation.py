# advanced_valuation.py
# STABLE + FIXED VERSION
# Realistic screening logic
# Removed unstable valuation explosions
# Added sanity filters
# Better growth handling
# Better DCF math
# Better screener accuracy

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DEFAULT_MOS = 25.0

MAX_REASONABLE_UNDERVALUE = 200
MAX_INTRINSIC_RATIO = 3.0

# ─────────────────────────────────────────────────────────────
# FINVIZ
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_finviz_data(ticker):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    url = f"https://finviz.com/quote.ashx?t={ticker}"

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        table = soup.find(
            "table",
            {"class": "snapshot-table2"}
        )

        data = {}

        if table:

            rows = table.find_all("tr")

            for row in rows:

                cells = row.find_all("td")

                for i in range(0, len(cells), 2):

                    if i + 1 < len(cells):

                        key = cells[i].text.strip()

                        value = cells[i + 1].text.strip()

                        try:

                            clean = (
                                value
                                .replace("%", "")
                                .replace(",", "")
                                .replace("$", "")
                            )

                            data[key] = float(clean)

                        except:
                            data[key] = value

        return data

    except:
        return {}

# ─────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):

    try:

        ticker = ticker.replace(".", "-")

        stock = yf.Ticker(ticker)

        info = stock.info or {}

        growth = (
            info.get("earningsGrowth", 0.10)
        ) * 100

        # normalize unreliable yahoo growth
        growth = growth * 0.5

        growth = max(
            0,
            min(growth, 15)
        )

        data = {

            "ticker": ticker,

            "name": info.get(
                "longName",
                ticker
            ),

            "sector": info.get(
                "sector",
                "N/A"
            ),

            "current_price": info.get(
                "currentPrice",
                0
            ),

            "current_eps": info.get(
                "trailingEps",
                0
            ),

            "forward_eps": info.get(
                "forwardEps",
                info.get("trailingEps", 0)
            ),

            "book_value": info.get(
                "bookValue",
                0
            ),

            "dividend_per_share": info.get(
                "dividendRate",
                0
            ),

            "historical_pe": info.get(
                "trailingPE",
                15
            ),

            "beta": info.get(
                "beta",
                1
            ),

            "roe": (
                info.get(
                    "returnOnEquity",
                    0.15
                ) * 100
            ),

            "market_cap": info.get(
                "marketCap",
                0
            ),

            "shares_outstanding": info.get(
                "sharesOutstanding",
                1
            ),

            "fcf": info.get(
                "freeCashflow",
                0
            ),

            "net_debt": (
                info.get("totalDebt", 0)
                - info.get("totalCash", 0)
            ),

            "analyst_growth": growth,

            "wacc": 9.0,

            "stable_growth": 3.0,

            "desired_return": 10.0,

            "years_high_growth": 5
        }

        # finviz enhancement
        finviz = get_finviz_data(ticker)

        if finviz:

            if isinstance(
                finviz.get("P/E"),
                (float, int)
            ):

                data["historical_pe"] = min(
                    max(finviz["P/E"], 1),
                    35
                )

        # clamps
        data["historical_pe"] = min(
            max(data["historical_pe"], 1),
            35
        )

        data["roe"] = min(
            max(data["roe"], 0),
            40
        )

        return data

    except:
        return None

# ─────────────────────────────────────────────────────────────
# VALUATION MODELS
# ─────────────────────────────────────────────────────────────

def core_valuation(inputs):

    eps = max(inputs["forward_eps"], 0)

    pe = min(inputs["historical_pe"], 25)

    growth = min(
        inputs["analyst_growth"] / 100,
        0.12
    )

    years = inputs["years_high_growth"]

    r = inputs["desired_return"] / 100

    future_eps = eps * ((1 + growth) ** years)

    future_price = future_eps * pe

    intrinsic = (
        future_price
        / ((1 + r) ** years)
    )

    return {
        "intrinsic_value": max(intrinsic, 0)
    }

def comparable_company_analysis(inputs):

    eps = max(inputs["forward_eps"], 0)

    pe = min(
        inputs["historical_pe"],
        25
    )

    intrinsic = eps * pe

    return {
        "intrinsic_value": max(intrinsic, 0)
    }

def lynch_method(inputs):

    eps = max(inputs["forward_eps"], 0)

    growth = min(
        max(inputs["analyst_growth"], 0),
        15
    )

    justified_pe = min(
        growth * 1.5,
        25
    )

    intrinsic = eps * justified_pe

    return {
        "intrinsic_value": max(intrinsic, 0)
    }

def residual_income(inputs):

    bv = max(
        inputs["book_value"],
        0
    )

    roe = min(
        max(inputs["roe"] / 100, 0),
        0.30
    )

    r = inputs["desired_return"] / 100

    years = 10

    residual = bv * (roe - r)

    pv = 0

    for t in range(1, years + 1):

        fade = max(
            0.3,
            1 - t / years
        )

        ri_t = residual * fade

        pv += (
            ri_t
            / ((1 + r) ** t)
        )

    intrinsic = bv + pv

    return {
        "intrinsic_value": max(intrinsic, 0)
    }

def two_stage_dcf(inputs):

    fcf = inputs["fcf"]

    shares = inputs["shares_outstanding"]

    if fcf <= 0 or shares <= 0:
        return {"intrinsic_value": 0}

    growth = min(
        inputs["analyst_growth"] / 100,
        0.10
    )

    stable_growth = 0.03

    wacc = max(
        inputs["wacc"] / 100,
        0.08
    )

    years = inputs["years_high_growth"]

    pv_fcf = 0

    for t in range(1, years + 1):

        fcf_t = fcf * ((1 + growth) ** t)

        pv_fcf += (
            fcf_t
            / ((1 + wacc) ** t)
        )

    terminal_fcf = (
        fcf
        * ((1 + growth) ** years)
    )

    terminal_value = (
        terminal_fcf
        * (1 + stable_growth)
    ) / (wacc - stable_growth)

    pv_terminal = (
        terminal_value
        / ((1 + wacc) ** years)
    )

    enterprise_value = (
        pv_fcf
        + pv_terminal
    )

    equity_value = (
        enterprise_value
        - inputs["net_debt"]
    )

    intrinsic = equity_value / shares

    return {
        "intrinsic_value": max(intrinsic, 0)
    }

def ddm_valuation(inputs):

    dividend = inputs["dividend_per_share"]

    price = inputs["current_price"]

    if dividend <= 0 or price <= 0:
        return {"intrinsic_value": 0}

    dividend_yield = dividend / price

    # reject low-yield companies
    if dividend_yield < 0.01:
        return {"intrinsic_value": 0}

    growth = 0.03

    required = 0.09

    intrinsic = (
        dividend
        * (1 + growth)
    ) / (required - growth)

    return {
        "intrinsic_value": max(intrinsic, 0)
    }

# ─────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────

def calculate_valuation(inputs):

    model = inputs["model"]

    model_map = {

        "Core Valuation":
            core_valuation,

        "Comparable PE":
            comparable_company_analysis,

        "Lynch Method":
            lynch_method,

        "Residual Income":
            residual_income,

        "Two-Stage DCF":
            two_stage_dcf,

        "Dividend Discount Model":
            ddm_valuation
    }

    if model not in model_map:

        return {
            "intrinsic_value": 0
        }

    intrinsic = model_map[model](
        inputs
    )["intrinsic_value"]

    current = inputs["current_price"]

    undervaluation = (
        (
            intrinsic - current
        ) / current
    ) * 100 if current > 0 else 0

    mos = inputs.get(
        "core_mos",
        DEFAULT_MOS
    )

    safe_buy = (
        intrinsic
        * (1 - mos / 100)
    )

    verdict = (
        "Strong Buy"
        if undervaluation > 30
        else "Buy"
        if undervaluation > 10
        else "Hold"
        if undervaluation > -10
        else "Sell"
    )

    return {

        "intrinsic_value": intrinsic,

        "safe_buy_price": safe_buy,

        "undervaluation": undervaluation,

        "verdict": verdict
    }

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_inputs(inputs):

    if inputs["current_price"] <= 0:
        return False

    if inputs["shares_outstanding"] <= 0:
        return False

    return True

# ─────────────────────────────────────────────────────────────
# S&P 500
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_sp500_tickers():

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    try:

        df = pd.read_html(url)[0]

        df = df[
            ["Symbol", "Security"]
        ]

        df["Symbol"] = (
            df["Symbol"]
            .str.replace(".", "-", regex=False)
        )

        return df

    except:

        return pd.DataFrame({
            "Symbol": [
                "AAPL",
                "MSFT",
                "GOOGL"
            ]
        })

# ─────────────────────────────────────────────────────────────
# SCREENER
# ─────────────────────────────────────────────────────────────

def run_sp500_screener(
    model,
    min_undervaluation=10,
    max_stocks=100
):

    sp500 = get_sp500_tickers().head(
        max_stocks
    )

    results = []

    progress = st.progress(0)

    total = len(sp500)

    for i, row in sp500.iterrows():

        ticker = row["Symbol"]

        try:

            inputs = fetch_stock_data(
                ticker
            )

            if not inputs:
                continue

            if not validate_inputs(inputs):
                continue

            inputs["model"] = model

            val = calculate_valuation(
                inputs
            )

            intrinsic = val["intrinsic_value"]

            price = inputs["current_price"]

            underv = val["undervaluation"]

            # sanity filters
            if intrinsic <= 0:
                continue

            ratio = intrinsic / price

            if ratio > MAX_INTRINSIC_RATIO:
                continue

            if underv > MAX_REASONABLE_UNDERVALUE:
                continue

            if underv < min_undervaluation:
                continue

            results.append({

                "Ticker": ticker,

                "Name": inputs["name"],

                "Price": round(price, 2),

                "Intrinsic": round(
                    intrinsic,
                    2
                ),

                "Safe Buy": round(
                    val["safe_buy_price"],
                    2
                ),

                "Undervaluation %": round(
                    underv,
                    1
                ),

                "P/E": round(
                    inputs["historical_pe"],
                    1
                ),

                "Growth %": round(
                    inputs["analyst_growth"],
                    1
                ),

                "Verdict": val["verdict"]
            })

        except:
            pass

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    return df.sort_values(
        "Undervaluation %",
        ascending=False
    )

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

def render_advanced_valuation():

    st.set_page_config(
        page_title="Stable Valuation",
        layout="wide"
    )

    st.title(
        "💹 Stable Advanced Valuation"
    )

    ticker = st.text_input(
        "Ticker",
        "AAPL"
    ).upper()

    model = st.selectbox(
        "Model",
        [
            "Core Valuation",
            "Comparable PE",
            "Lynch Method",
            "Residual Income",
            "Two-Stage DCF",
            "Dividend Discount Model"
        ]
    )

    mos = st.slider(
        "Margin of Safety %",
        0,
        50,
        25
    )

    if st.button("Analyze"):

        with st.spinner("Analyzing..."):

            inputs = fetch_stock_data(
                ticker
            )

            if not inputs:

                st.error(
                    "Could not fetch data."
                )

                return

            inputs["model"] = model
            inputs["core_mos"] = mos

            val = calculate_valuation(
                inputs
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Current Price",
                f"${inputs['current_price']:.2f}"
            )

            c2.metric(
                "Intrinsic Value",
                f"${val['intrinsic_value']:.2f}"
            )

            c3.metric(
                "Safe Buy Price",
                f"${val['safe_buy_price']:.2f}"
            )

            c4.metric(
                "Undervaluation",
                f"{val['undervaluation']:.1f}%"
            )

            st.markdown(
                f"## {val['verdict']}"
            )

    st.divider()

    st.subheader(
        "🏛️ S&P 500 Screener"
    )

    screener_model = st.selectbox(
        "Screener Model",
        [
            "Core Valuation",
            "Comparable PE",
            "Lynch Method",
            "Residual Income",
            "Two-Stage DCF"
        ],
        key="screen_model"
    )

    min_underv = st.slider(
        "Minimum Undervaluation %",
        0,
        100,
        15
    )

    max_stocks = st.slider(
        "Max Stocks",
        10,
        500,
        100
    )

    if st.button("Run Screener"):

        with st.spinner("Screening..."):

            df = run_sp500_screener(
                screener_model,
                min_underv,
                max_stocks
            )

            if df.empty:

                st.warning(
                    "No stocks found."
                )

            else:

                st.success(
                    f"Found {len(df)} stocks."
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                fig = px.bar(
                    df.head(10),
                    x="Ticker",
                    y="Undervaluation %",
                    hover_data=[
                        "Price",
                        "Intrinsic"
                    ]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    render_advanced_valuation()
