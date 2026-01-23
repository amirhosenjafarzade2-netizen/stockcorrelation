# grapher.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np


def render_grapher() -> None:
    st.subheader("Grapher • Enhanced Fundamental Analysis")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    with col2:
        frequency = st.selectbox("Frequency", ["Quarterly", "Annual"], index=0)

    with col3:
        years_back = st.slider(
            "Price history lookback (years) – fundamentals show all available data",
            min_value=1,
            max_value=30,
            value=10
        )

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * (years_back + 1))

    if st.button("Load & Plot", type="primary", use_container_width=True):
        with st.spinner(f"Fetching {ticker} ({frequency.lower()})..."):
            try:
                ticker_obj = yf.Ticker(ticker)

                is_annual = (frequency == "Annual")

                # ── Financial statements ────────────────────────────────────────
                income   = ticker_obj.income_stmt    if is_annual else ticker_obj.quarterly_income_stmt
                balance  = ticker_obj.balance_sheet  if is_annual else ticker_obj.quarterly_balance_sheet
                cashflow = ticker_obj.cashflow       if is_annual else ticker_obj.quarterly_cashflow

                # Fallback
                if income.empty:   income   = ticker_obj.get_income_stmt(freq="yearly" if is_annual else "quarterly")
                if balance.empty:  balance  = ticker_obj.get_balance_sheet(freq="yearly" if is_annual else "quarterly")
                if cashflow.empty: cashflow = ticker_obj.get_cashflow(freq="yearly" if is_annual else "quarterly")

                # Price history — respects the slider
                prices = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )["Close"]

                if prices.empty and income.empty and balance.empty and cashflow.empty:
                    st.error(f"No data returned for {ticker}.")
                    return

                # ── Show actual available ranges (this explains a lot) ──────────
                if not prices.empty:
                    st.caption(f"**Price history:** {prices.index.min().date()} → {prices.index.max().date()}  ({len(prices)} points)")

                if not income.empty:
                    fin_start = income.columns.min().date()
                    fin_end   = income.columns.max().date()
                    fin_years = (income.columns.max() - income.columns.min()).days / 365.25
                    st.caption(f"**Fundamentals available:** {fin_start} → {fin_end}  (~{fin_years:.1f} years, {len(income.columns)} periods)")

                    requested_years = years_back
                    if fin_years < requested_years * 0.7:
                        st.warning(
                            f"Only ~{fin_years:.1f} years of fundamental data available "
                            f"(you requested {requested_years} years for price history). "
                            "This is a limitation of Yahoo Finance — many companies provide only recent statements."
                        )

                common_dates = income.columns.intersection(balance.columns).intersection(cashflow.columns)
                income   = income[common_dates]
                balance  = balance[common_dates]
                cashflow = cashflow[common_dates]

                # ── Plot helpers (robust against shape issues) ──────────────────
                def to_1d(data):
                    if isinstance(data, pd.DataFrame):
                        if data.shape[1] == 1:
                            return data.squeeze()
                        return data.iloc[:, 0]
                    if isinstance(data, pd.Series):
                        return data
                    if hasattr(data, 'to_numpy'):
                        return pd.Series(data.to_numpy().flatten(), index=getattr(data, 'index', None))
                    return pd.Series(data)

                def plot_line(data, title: str, yaxis: str = "Value", color=None, show_growth=False):
                    data = to_1d(data)
                    if data.empty:
                        return
                    fig = px.line(
                        x=data.index if hasattr(data, 'index') else range(len(data)),
                        y=data,
                        title=title,
                        markers=True,
                        color_discrete_sequence=[color] if color else None
                    )
                    if show_growth and len(data) > 4:
                        growth = data.pct_change().rolling(4).mean() * 100
                        fig.add_scatter(x=growth.index, y=growth, name="4-per Avg Growth %",
                                        yaxis="y2", line=dict(dash='dot', color='gray'))
                        fig.update_layout(yaxis2=dict(title="Growth %", overlaying="y", side="right"))
                    fig.update_layout(yaxis_title=yaxis, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                def plot_bar(data, title: str, yaxis: str = "Value", color=None):
                    data = to_1d(data)
                    if data.empty:
                        return
                    fig = px.bar(
                        x=data.index if hasattr(data, 'index') else range(len(data)),
                        y=data,
                        title=title,
                        color_discrete_sequence=[color] if color else None
                    )
                    fig.update_layout(yaxis_title=yaxis)
                    st.plotly_chart(fig, use_container_width=True)

                def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "Value", colors=None):
                    if df.empty:
                        return
                    fig = go.Figure()
                    for i, col in enumerate(df.columns):
                        s = to_1d(df[col])
                        if s.empty:
                            continue
                        fig.add_trace(go.Scatter(
                            x=s.index,
                            y=s,
                            mode="lines+markers",
                            name=col,
                            line=dict(color=colors[i] if colors and i < len(colors) else None)
                        ))
                    if fig.data:
                        fig.update_layout(title=title, yaxis_title=yaxis, hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

                # ── Graphs ───────────────────────────────────────────────────────
                st.markdown("### 1. Price History")
                plot_line(prices, f"{ticker} Adjusted Close Price", color="#1f77b4")

                if "Total Revenue" in income.index:
                    st.markdown("### 2. Revenue")
                    plot_bar(income.loc["Total Revenue"], "Total Revenue", color="#2ca02c")

                if all(k in income.index for k in ["Gross Profit", "Operating Income", "Net Income", "Total Revenue"]):
                    st.markdown("### 3. Margin Trends (%)")
                    margins = pd.DataFrame({
                        "Gross":     income.loc["Gross Profit"]     / income.loc["Total Revenue"],
                        "Operating": income.loc["Operating Income"] / income.loc["Total Revenue"],
                        "Net":       income.loc["Net Income"]       / income.loc["Total Revenue"]
                    }) * 100
                    plot_multi(margins, "Gross / Operating / Net Margin", yaxis="%", colors=["#ff7f0e","#d62728","#9467bd"])

                profit_df = pd.DataFrame({
                    "Operating Income": income.get("Operating Income", pd.Series()),
                    "Net Income":       income.get("Net Income", pd.Series())
                })
                if not profit_df.empty:
                    st.markdown("### 4. Profitability")
                    plot_multi(profit_df, "Operating & Net Income", colors=["#17becf", "#bcbd22"])

                if "Operating Cash Flow" in cashflow.index and "Net Income" in income.index:
                    st.markdown("### 5. Earnings Quality")
                    eq_df = pd.DataFrame({
                        "Op. Cash Flow": cashflow.loc["Operating Cash Flow"],
                        "Net Income":    income.loc["Net Income"]
                    })
                    plot_multi(eq_df, "Operating Cash Flow vs Net Income", colors=["#1f77b4", "#ff7f0e"])

                ocf   = cashflow.get("Operating Cash Flow", pd.Series())
                capex = cashflow.get("Capital Expenditure", pd.Series(0, index=ocf.index))
                fcf   = ocf + capex
                if not fcf.empty and fcf.abs().sum() > 0:
                    st.markdown("### 6. Free Cash Flow")
                    plot_line(fcf, f"Free Cash Flow", show_growth=True)

                sbc = cashflow.get("Stock Based Compensation", pd.Series())
                if not sbc.empty and sbc.abs().sum() > 0:
                    st.markdown("### 7. Stock-Based Compensation")
                    plot_bar(sbc, "Stock-Based Compensation (negative = expense)", color="#9467bd")

                shares_basic   = income.get("Basic Average Shares", pd.Series())
                shares_diluted = income.get("Diluted Average Shares", pd.Series())
                if not shares_basic.empty or not shares_diluted.empty:
                    st.markdown("### 8. Share Count & Dilution")
                    shares_df = pd.DataFrame({
                        "Basic Avg":   shares_basic,
                        "Diluted Avg": shares_diluted
                    })
                    plot_multi(shares_df, "Basic vs Diluted Shares", yaxis="Shares")

                    if not shares_diluted.empty:
                        chg_pct = shares_diluted.pct_change() * 100
                        st.markdown("### 9. YoY Share Change % (positive = buyback)")
                        plot_line(chg_pct.dropna(), "Share Count Change YoY %", yaxis="% Change", color="#e377c2")

                if all(k in d for k in ["Net Income", "Total Assets", "Total Debt", "Cash And Cash Equivalents"] for d in [income, balance]):
                    st.markdown("### 10. Return on Invested Capital (ROIC) – approx")
                    nopat = income.loc["Net Income"] + income.get("Tax Provision", pd.Series(0, index=common_dates))
                    inv_cap = balance.loc["Total Assets"] - balance.get("Cash And Cash Equivalents", pd.Series(0, index=common_dates)) - balance.get("Total Debt", pd.Series(0, index=common_dates))
                    roic = (nopat / inv_cap.shift(1)) * 100
                    plot_line(roic.dropna(), "ROIC (%)", yaxis="ROIC %", color="#7f7f7f")

                st.caption("Note: Fundamental charts use **all available historical periods** from Yahoo Finance — often much shorter than the price history range.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try:\n• Major US stocks (AAPL, META, TSLA, NVDA)\n• Switch Annual ↔ Quarterly\n• Check connection")


if __name__ == "__main__":
    st.set_page_config(page_title="Grapher", layout="wide")
    render_grapher()
