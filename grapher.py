"""
Financial Grapher Pro - FIXED VERSION
Fully corrected:
- MultiIndex-safe yfinance handling
- Proper datetime handling
- Fixed Plotly time-series rendering
- Safer Series conversion
- Better caching behavior
- Accurate year calculations
- Improved debugging
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Financial Grapher Pro",
    page_icon="📈",
    layout="wide"
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_series(data):
    """Safely convert DataFrame to Series."""
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def format_large_number(num: float) -> str:
    if pd.isna(num):
        return "N/A"

    abs_num = abs(num)

    if abs_num >= 1e12:
        return f"${num / 1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"${num / 1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"${num / 1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"${num / 1e3:.2f}K"

    return f"${num:.2f}"


def calculate_cagr(series: pd.Series) -> float:
    series = series.dropna()

    if len(series) < 2:
        return np.nan

    start_val = series.iloc[0]
    end_val = series.iloc[-1]

    if start_val <= 0 or end_val <= 0:
        return np.nan

    years = (series.index[-1] - series.index[0]).days / 365.25

    if years <= 0:
        return np.nan

    return (np.exp(np.log(end_val / start_val) / years) - 1) * 100


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_price_data(
    ticker: str,
    start_date: date,
    end_date: date,
) -> pd.Series:

    max_retries = 3

    for attempt in range(max_retries):

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )

            if df.empty:
                return pd.Series(dtype=float)

            # Handle MultiIndex columns safely
            if isinstance(df.columns, pd.MultiIndex):

                if ("Close", ticker) in df.columns:
                    close = df[("Close", ticker)]

                elif "Close" in df.columns.get_level_values(0):
                    close = df["Close"].iloc[:, 0]

                else:
                    close = df.iloc[:, 0]

            else:
                close = (
                    df["Close"]
                    if "Close" in df.columns
                    else df.iloc[:, 0]
                )

            close = ensure_series(close)

            close = close.dropna()

            # Force proper datetime index
            close.index = pd.to_datetime(close.index)

            # Remove timezone info if exists
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)

            return close.sort_index()

        except Exception as e:

            if attempt == max_retries - 1:
                st.error(f"Failed to fetch price data: {e}")
                return pd.Series(dtype=float)

            time.sleep(1)


@st.cache_data(ttl=3600)
def fetch_yfinance_fundamentals(ticker: str, frequency: str):

    try:
        ticker_obj = yf.Ticker(ticker)

        is_annual = frequency == "Annual"

        if is_annual:
            income = ticker_obj.get_income_stmt(pretty=True, freq="yearly")
            balance = ticker_obj.get_balance_sheet(pretty=True, freq="yearly")
            cashflow = ticker_obj.get_cash_flow(pretty=True, freq="yearly")
        else:
            income = ticker_obj.get_income_stmt(pretty=True, freq="quarterly")
            balance = ticker_obj.get_balance_sheet(pretty=True, freq="quarterly")
            cashflow = ticker_obj.get_cash_flow(pretty=True, freq="quarterly")

        return (
            income if income is not None else pd.DataFrame(),
            balance if balance is not None else pd.DataFrame(),
            cashflow if cashflow is not None else pd.DataFrame(),
            ticker_obj.info,
        )

    except Exception as e:
        st.error(f"Fundamental fetch failed: {e}")

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )


# ══════════════════════════════════════════════════════════════════════════════
# SAFE GET
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str):

    if df.empty:
        return pd.Series(dtype=float)

    if key in df.index:
        return ensure_series(df.loc[key]).sort_index()

    for idx in df.index:

        if (
            isinstance(idx, str)
            and key.lower() in idx.lower()
        ):
            return ensure_series(df.loc[idx]).sort_index()

    return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_line(
    data: pd.Series,
    title: str,
    yaxis: str = "Value",
    color: str = "#1f77b4",
    show_growth: bool = False,
):

    data = ensure_series(data)

    if data.empty or data.isna().all():
        st.info(f"No data for {title}")
        return

    data = data.dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(data.index),
            y=data.astype(float).values,
            mode="lines",
            name=title,
            line=dict(width=2, color=color),
        )
    )

    # CAGR
    cagr = calculate_cagr(data)

    if not np.isnan(cagr):
        fig.update_layout(
            title=f"{title}<br><sup>CAGR: {cagr:.2f}%</sup>"
        )
    else:
        fig.update_layout(title=title)

    # Optional growth overlay
    if show_growth:

        growth = data.pct_change() * 100
        growth = growth.dropna()

        if not growth.empty:

            fig.add_trace(
                go.Scatter(
                    x=growth.index,
                    y=growth.values,
                    mode="lines",
                    name="Growth %",
                    yaxis="y2",
                    line=dict(
                        dash="dot",
                        color="gray",
                    ),
                )
            )

            fig.update_layout(
                yaxis2=dict(
                    title="Growth %",
                    overlaying="y",
                    side="right",
                )
            )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        yaxis_title=yaxis,
        xaxis=dict(type="date"),
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_bar(
    data: pd.Series,
    title: str,
    yaxis: str = "Value",
    color: str = "#2ca02c",
):

    data = ensure_series(data)

    if data.empty or data.isna().all():
        st.info(f"No data for {title}")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data.index.astype(str),
            y=data.values,
            marker_color=color,
            name=title,
        )
    )

    cagr = calculate_cagr(data)

    if not np.isnan(cagr):
        title = f"{title}<br><sup>CAGR: {cagr:.2f}%</sup>"

    fig.update_layout(
        title=title,
        yaxis_title=yaxis,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_multi(
    df: pd.DataFrame,
    title: str,
    yaxis: str = "Value",
):

    if df.empty:
        st.info(f"No data for {title}")
        return

    fig = go.Figure()

    added = False

    for col in df.columns:

        series = ensure_series(df[col]).dropna()

        if series.empty:
            continue

        added = True

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(series.index),
                y=series.values,
                mode="lines+markers",
                name=col,
            )
        )

    if not added:
        st.info(f"No data for {title}")
        return

    fig.update_layout(
        title=title,
        yaxis_title=yaxis,
        height=500,
        hovermode="x unified",
        xaxis=dict(type="date"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def render_grapher():

    st.title("📈 Financial Grapher Pro")

    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
        ).strip().upper()

    with col2:
        frequency = st.selectbox(
            "Frequency",
            ["Quarterly", "Annual"],
        )

    with col3:
        price_years = st.slider(
            "Price History (Years)",
            1,
            30,
            10,
        )

    if not ticker:
        return

    end_date = date.today()

    # FIXED
    start_date = end_date - relativedelta(years=price_years)

    if st.button(
        "🚀 Load & Analyze",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("Fetching data..."):

            prices = fetch_price_data(
                ticker,
                start_date,
                end_date,
            )

            income, balance, cashflow, info = (
                fetch_yfinance_fundamentals(
                    ticker,
                    frequency,
                )
            )

            if prices.empty:
                st.error("No price data available")
                return

            # DEBUG
            with st.expander("🔍 Debug"):

                st.write("Rows:", len(prices))

                st.write(
                    "Start:",
                    prices.index.min(),
                )

                st.write(
                    "End:",
                    prices.index.max(),
                )

                st.write(prices.head())

                st.write(prices.tail())

            # COMPANY INFO
            if info:

                st.markdown("## 📋 Company Overview")

                cols = st.columns(4)

                cols[0].metric(
                    "Company",
                    info.get("longName", ticker),
                )

                cols[1].metric(
                    "Sector",
                    info.get("sector", "N/A"),
                )

                cols[2].metric(
                    "Industry",
                    info.get("industry", "N/A"),
                )

                market_cap = info.get("marketCap")

                cols[3].metric(
                    "Market Cap",
                    format_large_number(market_cap)
                    if market_cap
                    else "N/A",
                )

            st.markdown("---")

            # PRICE
            st.markdown("## 📈 Stock Price")

            plot_line(
                prices,
                f"{ticker} Stock Price",
                "Price (USD)",
            )

            cols = st.columns(4)

            cols[0].metric(
                "Current",
                f"${prices.iloc[-1]:.2f}",
            )

            cols[1].metric(
                "High",
                f"${prices.max():.2f}",
            )

            cols[2].metric(
                "Low",
                f"${prices.min():.2f}",
            )

            total_return = (
                prices.iloc[-1] / prices.iloc[0] - 1
            ) * 100

            cols[3].metric(
                "Return",
                f"{total_return:.2f}%",
            )

            # REVENUE
            revenue = safe_get(income, "Total Revenue")

            if not revenue.empty:

                st.markdown("---")
                st.markdown("## 💰 Revenue")

                plot_bar(
                    revenue,
                    "Total Revenue",
                    "USD",
                )

            # NET INCOME
            net_income = safe_get(
                income,
                "Net Income",
            )

            if not net_income.empty:

                st.markdown("---")
                st.markdown("## 💵 Net Income")

                plot_line(
                    net_income,
                    "Net Income",
                    "USD",
                    "#2ca02c",
                )

            # BALANCE SHEET
            assets = safe_get(
                balance,
                "Total Assets",
            )

            liabilities = safe_get(
                balance,
                "Total Liabilities",
            )

            if not assets.empty:

                st.markdown("---")
                st.markdown("## 📊 Balance Sheet")

                balance_df = pd.DataFrame({
                    "Assets": assets,
                    "Liabilities": liabilities,
                })

                plot_multi(
                    balance_df,
                    "Balance Sheet",
                    "USD",
                )

            st.success("✅ Analysis Complete")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    st.markdown(
        """
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_grapher()
