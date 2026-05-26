"""
Financial Grapher Pro - FULLY FIXED VERSION
Fixes:
✅ MultiIndex yfinance bug
✅ 1-year graph truncation bug
✅ Missing graph selections
✅ Plotly rendering failures
✅ Broken UI/tabs
✅ Timezone/date index issues
✅ DataFrame vs Series inconsistencies
✅ Duplicate timestamp problems
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_price_data(ticker: str, start_date: date, end_date: date) -> pd.Series:
    """Fetch clean historical price data"""

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False
        )

        if data is None or data.empty:
            return pd.Series(dtype=float)

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):

            if ("Close", ticker) in data.columns:
                series = data[("Close", ticker)]

            elif ("Adj Close", ticker) in data.columns:
                series = data[("Adj Close", ticker)]

            else:
                series = data.iloc[:, 0]

        else:

            if "Close" in data.columns:
                series = data["Close"]

            elif "Adj Close" in data.columns:
                series = data["Adj Close"]

            else:
                series = data.iloc[:, 0]

        # Force true Series
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        # Clean index
        series.index = pd.to_datetime(series.index).tz_localize(None)

        # Remove duplicates
        series = series[~series.index.duplicated(keep="first")]

        # Sort
        series = series.sort_index()

        # Numeric conversion
        series = pd.to_numeric(series, errors="coerce")

        # Drop NaNs
        series = series.dropna()

        return series.astype(float)

    except Exception as e:
        st.error(f"Price fetch error: {e}")
        return pd.Series(dtype=float)


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

        if income is None:
            income = pd.DataFrame()

        if balance is None:
            balance = pd.DataFrame()

        if cashflow is None:
            cashflow = pd.DataFrame()

        info = ticker_obj.info

        return income, balance, cashflow, info

    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None):

    if df.empty:
        return pd.Series(dtype=float)

    if key in df.index:
        s = df.loc[key]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.sort_index()

    for idx in df.index:
        if isinstance(idx, str) and key.lower() in idx.lower():
            s = df.loc[idx]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.sort_index()

    if default is not None:
        return pd.Series(default, index=df.columns)

    return pd.Series(dtype=float)


def calculate_cagr(series: pd.Series):

    clean = series.dropna()

    if len(clean) < 2:
        return np.nan

    start_val = clean.iloc[0]
    end_val = clean.iloc[-1]

    if start_val <= 0 or end_val <= 0:
        return np.nan

    years = (clean.index[-1] - clean.index[0]).days / 365.25

    if years <= 0:
        return np.nan

    return ((end_val / start_val) ** (1 / years) - 1) * 100


def format_large_number(num):

    if pd.isna(num):
        return "N/A"

    abs_num = abs(num)

    if abs_num >= 1e12:
        return f"${num/1e12:.2f}T"

    elif abs_num >= 1e9:
        return f"${num/1e9:.2f}B"

    elif abs_num >= 1e6:
        return f"${num/1e6:.2f}M"

    return f"${num:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_line(
    data: pd.Series,
    title: str,
    yaxis: str = "Value",
    color: str = None,
    show_growth: bool = False
):

    if data is None or len(data) == 0:
        st.info(f"📊 {title}: No data")
        return

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = data.copy()

    data.index = pd.to_datetime(data.index)

    data = pd.to_numeric(data, errors="coerce").dropna()

    if len(data) == 0:
        st.info(f"📊 {title}: No data")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="lines",
            name=title,
            line=dict(color=color) if color else None
        )
    )

    cagr = calculate_cagr(data)

    chart_title = title

    if not np.isnan(cagr):
        chart_title += f"<br><sup>CAGR: {cagr:.1f}%</sup>"

    if show_growth and len(data) > 1:

        growth = data.pct_change() * 100
        growth = growth.dropna()

        if len(growth) > 0:

            fig.add_trace(
                go.Scatter(
                    x=growth.index,
                    y=growth.values,
                    name="Growth %",
                    yaxis="y2",
                    line=dict(
                        dash="dot",
                        color="gray"
                    )
                )
            )

            fig.update_layout(
                yaxis2=dict(
                    title="Growth %",
                    overlaying="y",
                    side="right"
                )
            )

    fig.update_layout(
        title=chart_title,
        yaxis_title=yaxis,
        hovermode="x unified",
        height=450,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_bar(
    data: pd.Series,
    title: str,
    yaxis: str = "Value",
    color: str = None
):

    if data is None or len(data) == 0:
        st.info(f"📊 {title}: No data")
        return

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = data.copy()

    data.index = pd.to_datetime(data.index)

    data = pd.to_numeric(data, errors="coerce").dropna()

    if len(data) == 0:
        st.info(f"📊 {title}: No data")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data.values,
            marker_color=color
        )
    )

    cagr = calculate_cagr(data)

    chart_title = title

    if not np.isnan(cagr):
        chart_title += f"<br><sup>CAGR: {cagr:.1f}%</sup>"

    fig.update_layout(
        title=chart_title,
        yaxis_title=yaxis,
        height=450,
        xaxis=dict(type="date")
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_multi(
    df: pd.DataFrame,
    title: str,
    yaxis: str = "Value",
    colors: list = None
):

    if df is None or df.empty:
        st.info(f"📊 {title}: No data")
        return

    fig = go.Figure()

    plotted = False

    for i, col in enumerate(df.columns):

        series = df[col]

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = pd.to_numeric(series, errors="coerce").dropna()

        if len(series) == 0:
            continue

        plotted = True

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines+markers",
                name=str(col),
                line=dict(
                    color=colors[i]
                ) if colors and i < len(colors) else None
            )
        )

    if not plotted:
        st.info(f"📊 {title}: No data")
        return

    fig.update_layout(
        title=title,
        yaxis_title=yaxis,
        hovermode="x unified",
        height=450,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_grapher():

    st.title("📈 Financial Grapher Pro")

    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL"
        ).strip().upper()

    with col2:
        frequency = st.selectbox(
            "Frequency",
            ["Quarterly", "Annual"]
        )

    with col3:
        price_years = st.slider(
            "Price History (Years)",
            1,
            30,
            10
        )

    if not ticker:
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * price_years)

    if st.button("🚀 Load & Analyze", type="primary"):

        with st.spinner("Loading data..."):

            prices = fetch_price_data(
                ticker,
                start_date,
                end_date
            )

            income, balance, cashflow, info = fetch_yfinance_fundamentals(
                ticker,
                frequency
            )

            if prices.empty:
                st.error("No price data found")
                return

            # Company Info
            if info:

                st.markdown("### 📋 Company Overview")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Company",
                        info.get("longName", ticker)
                    )

                with c2:
                    st.metric(
                        "Sector",
                        info.get("sector", "N/A")
                    )

                with c3:
                    st.metric(
                        "Industry",
                        info.get("industry", "N/A")
                    )

                with c4:
                    market_cap = info.get("marketCap")
                    st.metric(
                        "Market Cap",
                        format_large_number(market_cap)
                        if market_cap else "N/A"
                    )

            st.markdown("---")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Price",
                "💰 Revenue",
                "📊 Profitability",
                "💵 Cash Flow",
                "📐 Balance Sheet"
            ])

            # PRICE TAB
            with tab1:

                plot_line(
                    prices,
                    f"{ticker} Price History",
                    "Price (USD)",
                    "#1f77b4"
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Current",
                        f"${prices.iloc[-1]:.2f}"
                    )

                with c2:
                    st.metric(
                        "High",
                        f"${prices.max():.2f}"
                    )

                with c3:
                    st.metric(
                        "Low",
                        f"${prices.min():.2f}"
                    )

                with c4:
                    total_return = (
                        (prices.iloc[-1] / prices.iloc[0]) - 1
                    ) * 100

                    st.metric(
                        "Return",
                        f"{total_return:.1f}%"
                    )

            # REVENUE TAB
            with tab2:

                revenue = safe_get(income, "Total Revenue")

                if not revenue.empty:
                    plot_bar(
                        revenue,
                        "Revenue",
                        "USD",
                        "#2ca02c"
                    )
                else:
                    st.warning("No revenue data")

            # PROFITABILITY TAB
            with tab3:

                gross_profit = safe_get(income, "Gross Profit")
                operating_income = safe_get(income, "Operating Income")
                net_income = safe_get(income, "Net Income")

                margin_df = pd.DataFrame()

                revenue = safe_get(income, "Total Revenue")

                if not gross_profit.empty and not revenue.empty:
                    margin_df["Gross Margin"] = (
                        gross_profit / revenue
                    ) * 100

                if not operating_income.empty and not revenue.empty:
                    margin_df["Operating Margin"] = (
                        operating_income / revenue
                    ) * 100

                if not net_income.empty and not revenue.empty:
                    margin_df["Net Margin"] = (
                        net_income / revenue
                    ) * 100

                if not margin_df.empty:
                    plot_multi(
                        margin_df,
                        "Margins (%)",
                        "Margin %",
                        ["#ff7f0e", "#d62728", "#9467bd"]
                    )
                else:
                    st.warning("No margin data")

            # CASH FLOW TAB
            with tab4:

                ocf = safe_get(cashflow, "Operating Cash Flow")
                capex = safe_get(cashflow, "Capital Expenditure", 0)

                if not ocf.empty:

                    plot_line(
                        ocf,
                        "Operating Cash Flow",
                        "USD",
                        "#17becf",
                        True
                    )

                    fcf = ocf + capex

                    if not fcf.empty:
                        plot_line(
                            fcf,
                            "Free Cash Flow",
                            "USD",
                            "#2ca02c",
                            True
                        )

                else:
                    st.warning("No cash flow data")

            # BALANCE SHEET TAB
            with tab5:

                assets = safe_get(balance, "Total Assets")
                liabilities = safe_get(balance, "Total Liabilities")
                equity = safe_get(balance, "Stockholders Equity")

                bs_df = pd.DataFrame()

                if not assets.empty:
                    bs_df["Assets"] = assets

                if not liabilities.empty:
                    bs_df["Liabilities"] = liabilities

                if not equity.empty:
                    bs_df["Equity"] = equity

                if not bs_df.empty:

                    plot_multi(
                        bs_df,
                        "Balance Sheet",
                        "USD",
                        ["#2ca02c", "#d62728", "#1f77b4"]
                    )

                else:
                    st.warning("No balance sheet data")


if __name__ == "__main__":

    st.set_page_config(
        page_title="Financial Grapher Pro",
        page_icon="📈",
        layout="wide"
    )

    render_grapher()
