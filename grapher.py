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
        years_back = st.slider("Years back", 1, 30, 12)

    if not ticker:
        st.info("Enter a ticker symbol.")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * (years_back + 2))

    if st.button("Load Fundamentals", type="primary"):
        with st.spinner(f"Loading {ticker} data…"):
            try:
                ticker_obj = yf.Ticker(ticker)

                is_annual = (frequency == "Annual")

                # Modern yfinance access
                income   = ticker_obj.income_stmt    if is_annual else ticker_obj.quarterly_income_stmt
                balance  = ticker_obj.balance_sheet  if is_annual else ticker_obj.quarterly_balance_sheet
                cashflow = ticker_obj.cashflow       if is_annual else ticker_obj.quarterly_cashflow

                # Fallback
                if income.empty:   income   = ticker_obj.get_income_stmt(freq="yearly" if is_annual else "quarterly")
                if balance.empty:  balance  = ticker_obj.get_balance_sheet(freq="yearly" if is_annual else "quarterly")
                if cashflow.empty: cashflow = ticker_obj.get_cashflow(freq="yearly" if is_annual else "quarterly")

                prices = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )["Close"]

                if income.empty and balance.empty and cashflow.empty and prices.empty:
                    st.error("No meaningful data returned for this ticker.")
                    return

                common_dates = income.columns.intersection(balance.columns).intersection(cashflow.columns)
                income   = income[common_dates]
                balance  = balance[common_dates]
                cashflow = cashflow[common_dates]

                st.success(f"Loaded {len(common_dates)} periods • {len(prices)} price points")

                # ── Robust plotting helpers ──────────────────────────────────────
                def to_1d(data):
                    """Force input to 1-dimensional Series or array"""
                    if isinstance(data, pd.DataFrame):
                        if data.shape[1] == 1:
                            return data.squeeze()
                        elif data.shape[1] > 1:
                            return data.iloc[:, 0]  # take first column as fallback
                    if isinstance(data, pd.Series):
                        return data
                    if hasattr(data, 'to_numpy'):
                        return pd.Series(data.to_numpy().flatten(), index=getattr(data, 'index', None))
                    return pd.Series(data)

                def plot_line(data, title: str, yaxis: str = "$", color=None, show_growth=False):
                    data = to_1d(data)
                    fig = px.line(
                        x=data.index if hasattr(data, 'index') else range(len(data)),
                        y=data,
                        title=title,
                        markers=True,
                        color_discrete_sequence=[color] if color else None
                    )

                    if show_growth and len(data) > 4:
                        growth = data.pct_change().rolling(4).mean() * 100
                        fig.add_scatter(
                            x=growth.index,
                            y=growth,
                            name="4-per. Avg Growth %",
                            yaxis="y2",
                            line=dict(dash='dot', color='gray')
                        )
                        fig.update_layout(
                            yaxis2=dict(title="Growth %", overlaying="y", side="right")
                        )

                    fig.update_layout(yaxis_title=yaxis, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                def plot_bar(data, title: str, yaxis: str = "$", color=None):
                    data = to_1d(data)
                    fig = px.bar(
                        x=data.index if hasattr(data, 'index') else range(len(data)),
                        y=data,
                        title=title,
                        color_discrete_sequence=[color] if color else None
                    )
                    fig.update_layout(yaxis_title=yaxis)
                    st.plotly_chart(fig, use_container_width=True)

                def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "$", colors=None):
                    fig = go.Figure()
                    for i, col in enumerate(df.columns):
                        series = to_1d(df[col])
                        fig.add_trace(go.Scatter(
                            x=series.index,
                            y=series,
                            mode="lines+markers",
                            name=col,
                            line=dict(color=colors[i] if colors and i < len(colors) else None)
                        ))
                    fig.update_layout(title=title, yaxis_title=yaxis, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                # ── 1. Price History ─────────────────────────────────────────────
                st.markdown("### Price History")
                plot_line(prices, f"{ticker} Adjusted Close", color="#1f77b4")

                # ── 2. Revenue ───────────────────────────────────────────────────
                if "Total Revenue" in income.index:
                    st.markdown("### Revenue")
                    plot_bar(income.loc["Total Revenue"], "Total Revenue", color="#2ca02c")

                # ── 3. Margins ───────────────────────────────────────────────────
                if all(k in income.index for k in ["Gross Profit", "Operating Income", "Net Income", "Total Revenue"]):
                    st.markdown("### Margin Trends")
                    margins = pd.DataFrame({
                        "Gross Margin":   income.loc["Gross Profit"]     / income.loc["Total Revenue"],
                        "Operating Margin": income.loc["Operating Income"] / income.loc["Total Revenue"],
                        "Net Margin":     income.loc["Net Income"]       / income.loc["Total Revenue"]
                    }) * 100
                    plot_multi(margins, "Gross / Operating / Net Margin (%)", yaxis="Margin %",
                               colors=["#ff7f0e", "#d62728", "#9467bd"])

                # ── 4. Net Income & Operating Income ─────────────────────────────
                st.markdown("### Profitability")
                profit_df = pd.DataFrame({
                    "Operating Income": income.get("Operating Income", pd.Series()),
                    "Net Income":       income.get("Net Income", pd.Series())
                })
                if not profit_df.empty:
                    plot_multi(profit_df, "Operating & Net Income", colors=["#17becf", "#bcbd22"])

                # ── 5. Operating Cash Flow vs Net Income ─────────────────────────
                if "Operating Cash Flow" in cashflow.index and "Net Income" in income.index:
                    st.markdown("### Earnings Quality: Op. Cash Flow vs Net Income")
                    eq_df = pd.DataFrame({
                        "Operating Cash Flow": cashflow.loc["Operating Cash Flow"],
                        "Net Income": income.loc["Net Income"]
                    })
                    plot_multi(eq_df, "Op. Cash Flow vs Net Income", colors=["#1f77b4", "#ff7f0e"])

                # ── 6. Free Cash Flow ────────────────────────────────────────────
                ocf   = cashflow.get("Operating Cash Flow", pd.Series())
                capex = cashflow.get("Capital Expenditure", pd.Series(0, index=ocf.index))
                fcf   = ocf + capex

                if not fcf.empty and fcf.abs().sum() > 0:
                    st.markdown("### Free Cash Flow")
                    plot_line(fcf, f"{ticker} Free Cash Flow", show_growth=True)

                # ── 7. Stock-Based Compensation ──────────────────────────────────
                sbc = cashflow.get("Stock Based Compensation", pd.Series())
                if not sbc.empty and sbc.abs().sum() > 0:
                    st.markdown("### Stock-Based Compensation")
                    plot_bar(sbc, "Stock-Based Compensation (negative = expense)", color="#9467bd")

                # ── 8. Shares Outstanding & Dilution ─────────────────────────────
                shares_basic   = income.get("Basic Average Shares", pd.Series())
                shares_diluted = income.get("Diluted Average Shares", pd.Series())

                if not shares_basic.empty or not shares_diluted.empty:
                    st.markdown("### Share Count & Dilution Trend")
                    shares_df = pd.DataFrame({
                        "Basic Avg Shares": shares_basic,
                        "Diluted Avg Shares": shares_diluted
                    })
                    plot_multi(shares_df, "Basic vs Diluted Average Shares", yaxis="Shares (millions)")

                    # Buyback / Dilution intensity
                    if not shares_diluted.empty:
                        share_change_pct = shares_diluted.pct_change() * 100
                        st.markdown("### YoY Share Count Change (positive = buyback, negative = dilution)")
                        plot_line(share_change_pct.dropna(), "Share Count Change % (YoY)", yaxis="% Change", color="#e377c2")

                # ── 9. ROIC (simplified) ─────────────────────────────────────────
                if all(k in d for k in ["Net Income", "Total Assets", "Total Debt", "Cash And Cash Equivalents"] for d in [income, balance]):
                    st.markdown("### Return on Invested Capital (ROIC) – simplified")
                    nopat = income.loc["Net Income"] + income.get("Tax Provision", pd.Series(0, index=common_dates))
                    invested_capital = balance.loc["Total Assets"] - balance.get("Cash And Cash Equivalents", pd.Series(0, index=common_dates)) - balance.get("Total Debt", pd.Series(0, index=common_dates))
                    roic = (nopat / invested_capital.shift(1)) * 100
                    plot_line(roic.dropna(), "ROIC (%)", yaxis="ROIC %", color="#7f7f7f")

                st.markdown("---")
                st.caption("Note: Some metrics may be missing or approximate depending on data availability.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("• Try a major US stock (AAPL, MSFT, NVDA…)\n• Some tickers have incomplete fundamentals\n• Check your internet connection")


if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Grapher", layout="wide")
    render_grapher()
