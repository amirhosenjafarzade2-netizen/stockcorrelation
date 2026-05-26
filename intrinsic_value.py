# intrinsic_value.py
# ============================================================
# Intrinsic Value Pro
# Fixed Version
#
# FIXES:
# - Added valuation method selector UI
# - Removed Finviz dependency (403 fix)
# - Removed duplicate st.set_page_config()
# - Added safer NaN handling
# - Improved screener stability
# - Better error handling
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

DCF_YEARS = 5

BASE_DISCOUNT = 0.10
BASE_TERMINAL = 0.03

MAX_GROWTH = 0.15
MIN_GROWTH = 0.03

OUTLIER_MULTIPLE = 10

MAX_THREADS = 10

# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:

        if value is None:
            return np.nan

        if isinstance(value, str):

            value = (
                value.replace("%", "")
                .replace(",", "")
                .replace("$", "")
                .strip()
            )

            if value in ["-", "", "N/A"]:
                return np.nan

        return float(value)

    except:
        return np.nan


def clamp_growth(growth):

    return np.clip(
        growth,
        MIN_GROWTH,
        MAX_GROWTH
    )


def remove_outliers(values, current_price):

    cleaned = []

    for v in values:

        if np.isnan(v):
            continue

        if v <= 0:
            continue

        if v > current_price * OUTLIER_MULTIPLE:
            continue

        cleaned.append(v)

    return cleaned

# ============================================================
# STOCK DATA
# ============================================================

@st.cache_data(ttl=1800)
def fetch_stock_data(ticker: str) -> Optional[Dict]:

    try:

        stock = yf.Ticker(ticker)

        info = stock.info

        if not info:
            return None

        price = safe_float(
            info.get("currentPrice")
        )

        if np.isnan(price) or price <= 0:
            return None

        market_cap = safe_float(
            info.get("marketCap")
        )

        shares = safe_float(
            info.get("sharesOutstanding")
        )

        total_debt = safe_float(
            info.get("totalDebt")
        )

        cash = safe_float(
            info.get("totalCash")
        )

        fcf = safe_float(
            info.get("freeCashflow")
        )

        roe = (
            safe_float(info.get("returnOnEquity")) * 100
            if info.get("returnOnEquity") is not None
            else np.nan
        )

        data = {

            "Symbol": ticker,

            "Price": price,

            "EPS": safe_float(
                info.get("trailingEps")
            ),

            "Book Value": safe_float(
                info.get("bookValue")
            ),

            "Forward EPS": safe_float(
                info.get("forwardEps")
            ),

            "Shares": shares,

            "Market Cap": market_cap,

            "Debt": total_debt,

            "Cash": cash,

            "FCF": fcf,

            "Sector": info.get(
                "sector",
                "N/A"
            ),

            "Industry": info.get(
                "industry",
                "N/A"
            ),

            "PE": safe_float(
                info.get("trailingPE")
            ),

            "PEG": safe_float(
                info.get("pegRatio")
            ),

            "Dividend Yield": (
                safe_float(
                    info.get("dividendYield")
                ) * 100
                if info.get("dividendYield")
                else 0
            ),

            "ROE": roe
        }

        # ====================================================
        # EPS CAGR
        # ====================================================

        growth = 0.05

        try:

            income = stock.get_income_stmt()

            eps_series = None

            if "DilutedEPS" in income.index:
                eps_series = income.loc["DilutedEPS"]

            elif "BasicEPS" in income.index:
                eps_series = income.loc["BasicEPS"]

            if eps_series is not None:

                eps_series = (
                    eps_series
                    .dropna()
                    [::-1]
                )

                if len(eps_series) >= 2:

                    start_eps = eps_series.iloc[0]
                    end_eps = eps_series.iloc[-1]

                    years = len(eps_series) - 1

                    if (
                        start_eps > 0
                        and end_eps > 0
                    ):

                        growth = (
                            (
                                end_eps / start_eps
                            ) ** (
                                1 / years
                            ) - 1
                        )

        except:
            pass

        growth = clamp_growth(growth)

        data["Growth"] = growth

        return data

    except Exception:
        return None

# ============================================================
# VALUATION METHODS
# ============================================================

def graham_number(data):

    eps = data["EPS"]
    bv = data["Book Value"]

    if (
        np.isnan(eps)
        or np.isnan(bv)
        or eps <= 0
        or bv <= 0
    ):
        return np.nan

    return np.sqrt(
        22.5 * eps * bv
    )


def lynch_value(data):

    eps = data["EPS"]

    growth = data["Growth"] * 100

    div_yield = data["Dividend Yield"]

    if (
        np.isnan(eps)
        or eps <= 0
    ):
        return np.nan

    fair_pe = np.clip(
        growth + div_yield,
        8,
        30
    )

    return eps * fair_pe


def dcf_value(
    data,
    discount_rate,
    terminal_growth
):

    fcf = data["FCF"]

    shares = data["Shares"]

    if (
        np.isnan(fcf)
        or np.isnan(shares)
        or fcf <= 0
        or shares <= 0
    ):
        return np.nan

    spread = (
        discount_rate
        - terminal_growth
    )

    if spread < 0.03:
        return np.nan

    growth = clamp_growth(
        data["Growth"]
    )

    projected = []

    current = fcf

    for year in range(1, DCF_YEARS + 1):

        current *= (1 + growth)

        pv = current / (
            (1 + discount_rate) ** year
        )

        projected.append(pv)

    terminal_fcf = (
        current
        * (1 + terminal_growth)
    )

    terminal = (
        terminal_fcf / spread
    )

    pv_terminal = terminal / (
        (1 + discount_rate)
        ** DCF_YEARS
    )

    enterprise_value = (
        sum(projected)
        + pv_terminal
    )

    debt = (
        data["Debt"]
        if not np.isnan(data["Debt"])
        else 0
    )

    cash = (
        data["Cash"]
        if not np.isnan(data["Cash"])
        else 0
    )

    equity_value = (
        enterprise_value
        - debt
        + cash
    )

    intrinsic = (
        equity_value / shares
    )

    return intrinsic

# ============================================================
# SCENARIO DCF
# ============================================================

def scenario_dcf(data):

    scenarios = {

        "Bear": {
            "discount": 0.12,
            "terminal": 0.02
        },

        "Base": {
            "discount": 0.10,
            "terminal": 0.03
        },

        "Bull": {
            "discount": 0.08,
            "terminal": 0.04
        }
    }

    results = {}

    for name, params in scenarios.items():

        results[name] = dcf_value(
            data,
            params["discount"],
            params["terminal"]
        )

    return results

# ============================================================
# COMBINED VALUE
# ============================================================

def intrinsic_value(data):

    methods = {}

    methods["Graham"] = graham_number(data)

    methods["Lynch"] = lynch_value(data)

    if (
        data["Sector"]
        not in [
            "Financial Services",
            "Banks"
        ]
    ):

        methods["DCF"] = dcf_value(
            data,
            BASE_DISCOUNT,
            BASE_TERMINAL
        )

    values = list(methods.values())

    values = remove_outliers(
        values,
        data["Price"]
    )

    if not values:
        intrinsic = np.nan
    else:
        intrinsic = np.median(values)

    return intrinsic, methods

# ============================================================
# VISUALS
# ============================================================

def valuation_chart(
    current_price,
    methods
):

    names = []
    vals = []
    colors = []

    for k, v in methods.items():

        if not np.isnan(v):

            names.append(k)

            vals.append(v)

            colors.append(
                "green"
                if v > current_price
                else "red"
            )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=names,
            y=vals,
            marker_color=colors,
            text=[
                f"${x:.2f}"
                for x in vals
            ],
            textposition="outside"
        )
    )

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="blue",
        annotation_text=(
            f"Price ${current_price:.2f}"
        )
    )

    fig.update_layout(
        title="Valuation Methods",
        height=450
    )

    return fig

# ============================================================
# UNIVERSES
# ============================================================

@st.cache_data(ttl=86400)
def get_sp500():

    try:

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )

        df = tables[0]

        return (
            df["Symbol"]
            .str.replace(".", "-", regex=False)
            .tolist()
        )

    except:
        return []


@st.cache_data(ttl=86400)
def get_nasdaq100():

    try:

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100"
        )

        for table in tables:

            if "Ticker" in table.columns:

                return (
                    table["Ticker"]
                    .tolist()
                )

    except:
        pass

    return []

# ============================================================
# SINGLE STOCK UI
# ============================================================

def single_stock_ui():

    st.header("Single Stock")

    ticker = st.text_input(
        "Ticker",
        "AAPL"
    ).upper().strip()

    valuation_mode = st.selectbox(
        "Valuation Method",
        [
            "Combined",
            "Graham",
            "Lynch",
            "DCF",
            "Scenario DCF"
        ]
    )

    if st.button(
        "Calculate",
        type="primary"
    ):

        with st.spinner(
            "Analyzing..."
        ):

            data = fetch_stock_data(
                ticker
            )

            if not data:

                st.error(
                    "Could not fetch data."
                )

                return

            all_intrinsic, methods = (
                intrinsic_value(data)
            )

            if valuation_mode == "Combined":

                intrinsic = all_intrinsic

            elif valuation_mode == "Graham":

                intrinsic = methods.get(
                    "Graham",
                    np.nan
                )

            elif valuation_mode == "Lynch":

                intrinsic = methods.get(
                    "Lynch",
                    np.nan
                )

            elif valuation_mode == "DCF":

                intrinsic = methods.get(
                    "DCF",
                    np.nan
                )

            elif valuation_mode == "Scenario DCF":

                scenarios = scenario_dcf(
                    data
                )

                valid_vals = [
                    v for v in scenarios.values()
                    if not np.isnan(v)
                ]

                intrinsic = (
                    np.mean(valid_vals)
                    if valid_vals
                    else np.nan
                )

                methods = scenarios

            price = data["Price"]

            if (
                np.isnan(intrinsic)
                or intrinsic <= 0
            ):
                premium = np.nan
            else:
                premium = (
                    (price - intrinsic)
                    / intrinsic
                ) * 100

            mos_price = (
                intrinsic * 0.7
                if not np.isnan(intrinsic)
                else np.nan
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Price",
                f"${price:.2f}"
            )

            c2.metric(
                "Intrinsic",
                (
                    f"${intrinsic:.2f}"
                    if not np.isnan(intrinsic)
                    else "N/A"
                )
            )

            c3.metric(
                "Premium",
                (
                    f"{premium:+.1f}%"
                    if not np.isnan(premium)
                    else "N/A"
                )
            )

            c4.metric(
                "Margin of Safety",
                (
                    f"${mos_price:.2f}"
                    if not np.isnan(mos_price)
                    else "N/A"
                )
            )

            fig = valuation_chart(
                price,
                methods
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "Scenario DCF"
            )

            scenarios = scenario_dcf(
                data
            )

            scenario_df = pd.DataFrame({
                "Scenario":
                list(scenarios.keys()),
                "DCF":
                list(scenarios.values())
            })

            st.dataframe(
                scenario_df.style.format({
                    "DCF": "${:.2f}"
                }),
                use_container_width=True
            )

            st.subheader(
                "Company Data"
            )

            raw = {
                k: (
                    round(v, 4)
                    if isinstance(v, float)
                    and not np.isnan(v)
                    else v
                )
                for k, v in data.items()
            }

            st.json(raw)

# ============================================================
# SCREENER
# ============================================================

def process_stock(symbol):

    try:

        data = fetch_stock_data(
            symbol
        )

        if not data:
            return None

        intrinsic, methods = (
            intrinsic_value(data)
        )

        if np.isnan(intrinsic):
            return None

        price = data["Price"]

        if intrinsic <= 0:
            return None

        premium = (
            (price - intrinsic)
            / intrinsic
        ) * 100

        return {

            "Symbol": symbol,

            "Price": price,

            "Intrinsic": intrinsic,

            "Premium %": premium,

            "Sector": data["Sector"],

            "PE": data["PE"],

            "Growth %": (
                data["Growth"] * 100
            ),

            "ROE": data["ROE"],
        }

    except:
        return None


def screener_ui():

    st.header("Multi-Stock Screener")

    universe = st.selectbox(
        "Universe",
        [
            "S&P 500",
            "NASDAQ 100",
            "Custom"
        ]
    )

    if universe == "S&P 500":

        tickers = get_sp500()

    elif universe == "NASDAQ 100":

        tickers = get_nasdaq100()

    else:

        custom = st.text_area(
            "Tickers",
            "AAPL,MSFT,GOOGL"
        )

        tickers = [
            x.strip().upper()
            for x in custom.split(",")
            if x.strip()
        ]

    if not tickers:

        st.warning(
            "Could not load universe."
        )

        return

    max_stocks = st.slider(
        "Max Stocks",
        10,
        min(300, len(tickers)),
        min(50, len(tickers))
    )

    tickers = tickers[:max_stocks]

    screen = st.selectbox(
        "Screen",
        [
            "Undervalued",
            "Overvalued",
            "All"
        ]
    )

    if st.button(
        "Run Screener",
        type="primary"
    ):

        results = []

        progress = st.progress(0)

        with ThreadPoolExecutor(
            max_workers=MAX_THREADS
        ) as executor:

            futures = {
                executor.submit(
                    process_stock,
                    t
                ): t
                for t in tickers
            }

            completed = 0

            for future in as_completed(
                futures
            ):

                completed += 1

                progress.progress(
                    completed / len(tickers)
                )

                try:

                    result = future.result()

                    if result:

                        premium = (
                            result["Premium %"]
                        )

                        include = False

                        if (
                            screen
                            == "Undervalued"
                            and premium < 0
                        ):
                            include = True

                        elif (
                            screen
                            == "Overvalued"
                            and premium > 0
                        ):
                            include = True

                        elif screen == "All":
                            include = True

                        if include:
                            results.append(
                                result
                            )

                except:
                    continue

        if not results:

            st.warning(
                "No stocks found."
            )

            return

        df = pd.DataFrame(results)

        df = df.sort_values(
            "Premium %",
            ascending=True
        )

        st.subheader(
            "Results"
        )

        st.dataframe(
            df.style.format({

                "Price": "${:.2f}",

                "Intrinsic": "${:.2f}",

                "Premium %": "{:.1f}%",

                "PE": "{:.1f}",

                "Growth %": "{:.1f}%",

                "ROE": "{:.1f}"
            }),
            use_container_width=True
        )

        fig = px.scatter(
            df,
            x="Growth %",
            y="Premium %",
            size="PE",
            hover_name="Symbol",
            color="Premium %"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            "Download CSV",
            csv,
            file_name=(
                f"screener_"
                f"{datetime.now().strftime('%Y%m%d')}.csv"
            ),
            mime="text/csv"
        )

# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "📈 Intrinsic Value Pro"
    )

    st.markdown("""
    Professional intrinsic value calculator
    using stabilized valuation models.

    Educational purposes only.
    """)

    mode = st.radio(
        "Mode",
        [
            "Single Stock",
            "Screener"
        ],
        horizontal=True
    )

    if mode == "Single Stock":

        single_stock_ui()

    else:

        screener_ui()

# ============================================================
# STREAMLIT MODULE ENTRY
# ============================================================

def render_intrinsic_value():
    main()

# ============================================================
# STANDALONE RUN
# ============================================================

if __name__ == "__main__":
    render_intrinsic_value()
