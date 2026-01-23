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
            "Years of history",
            min_value=1,
            max_value=30,
            value=10,
            help="Years of history to display for all charts"
        )

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years_back)

    if st.button("Load & Plot", type="primary", use_container_width=True):
        with st.spinner(f"Fetching {ticker} ({frequency.lower()})..."):
            try:
                ticker_obj = yf.Ticker(ticker)
                is_annual = (frequency == "Annual")

                # ── Get financials - fetch all, then filter ──────────
                st.info("📊 Fetching fundamental data...")
                
                # Use the newer API methods
                if is_annual:
                    income = ticker_obj.get_income_stmt(freq="yearly")
                    balance = ticker_obj.get_balance_sheet(freq="yearly") 
                    cashflow = ticker_obj.get_cashflow(freq="yearly")
                else:
                    income = ticker_obj.get_income_stmt(freq="quarterly")
                    balance = ticker_obj.get_balance_sheet(freq="quarterly")
                    cashflow = ticker_obj.get_cashflow(freq="quarterly")

                # ── FILTER fundamental data by date range ──────────────────
                def filter_by_date(df, start_date):
                    """Filter DataFrame columns by date range"""
                    if df.empty:
                        return df
                    # Keep only columns (dates) that are >= start_date
                    valid_cols = [col for col in df.columns if col.date() >= start_date]
                    return df[valid_cols]
                
                income = filter_by_date(income, start_date)
                balance = filter_by_date(balance, start_date)
                cashflow = filter_by_date(cashflow, start_date)

                # Price history - respects slider
                prices = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                if isinstance(prices, pd.DataFrame) and "Close" in prices.columns:
                    prices = prices["Close"]
                elif isinstance(prices, pd.DataFrame) and not prices.empty:
                    prices = prices.iloc[:, 0]
                else:
                    prices = pd.Series(dtype=float)

                if prices.empty and income.empty and balance.empty and cashflow.empty:
                    st.error(f"No data returned for {ticker}. Check ticker symbol.")
                    return

                # ── Show what we actually got ───────────────────────────────────
                if not prices.empty:
                    st.success(f"✓ Price: {prices.index.min().date()} → {prices.index.max().date()} ({len(prices)} days)")

                if not income.empty:
                    dates = sorted(income.columns)
                    years_available = (dates[-1] - dates[0]).days / 365.25
                    st.success(f"✓ Fundamentals: {dates[0].date()} → {dates[-1].date()} ({len(dates)} periods, ~{years_available:.1f} years)")
                else:
                    st.warning("⚠️ No fundamental data available for this ticker in the selected time range")

                # Align dates across statements
                if not income.empty and not balance.empty and not cashflow.empty:
                    common_dates = income.columns.intersection(balance.columns).intersection(cashflow.columns)
                    income = income[common_dates]
                    balance = balance[common_dates]
                    cashflow = cashflow[common_dates]
                    st.caption(f"Using {len(common_dates)} periods where all statements available")

                # ── Helper functions ────────────────────────────────────────────
                def safe_get(df, key, default=None):
                    """Get row from DataFrame with fallback"""
                    if df.empty:
                        return pd.Series(dtype=float)
                    
                    if key in df.index:
                        return df.loc[key]
                    
                    # Try alternatives
                    alternatives = {
                        "Total Revenue": ["TotalRevenue", "Total Revenues"],
                        "Gross Profit": ["GrossProfit", "Gross Income"],
                        "Operating Income": ["OperatingIncome", "EBIT", "Operating Revenue"],
                        "Net Income": ["NetIncome", "Net Income Common Stockholders"],
                        "Operating Cash Flow": ["OperatingCashFlow", "Total Cash From Operating Activities"],
                        "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures"],
                        "Stock Based Compensation": ["StockBasedCompensation", "Stock Based Compensation"],
                        "Basic Average Shares": ["BasicAverageShares", "Ordinary Shares Number", "Share Issued"],
                        "Diluted Average Shares": ["DilutedAverageShares", "Diluted NI Available To Com Stockholders"],
                        "Total Assets": ["TotalAssets", "Total Assets"],
                        "Total Debt": ["TotalDebt", "Long Term Debt", "Total Debt"],
                        "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"],
                        "Tax Provision": ["TaxProvision", "Tax Effect Of Unusual Items", "Income Tax Expense"]
                    }
                    
                    for alt in alternatives.get(key, []):
                        if alt in df.index:
                            return df.loc[alt]
                    
                    if default is not None:
                        return pd.Series(default, index=df.columns if not df.empty else [])
                    return pd.Series(dtype=float)

                def plot_line(data, title: str, yaxis: str = "Value", color=None, show_growth=False):
                    if isinstance(data, pd.DataFrame):
                        data = data.squeeze()
                    if data.empty or data.isna().all():
                        st.info(f"📊 {title}: No data available")
                        return
                    
                    fig = px.line(x=data.index, y=data, title=title, markers=True,
                                  color_discrete_sequence=[color] if color else None)
                    
                    if show_growth and len(data.dropna()) > 2:
                        growth = data.pct_change() * 100
                        if not growth.dropna().empty:
                            fig.add_scatter(x=growth.index, y=growth, 
                                          name="Growth %", yaxis="y2",
                                          line=dict(dash='dot', color='gray'))
                            fig.update_layout(yaxis2=dict(title="Growth %", overlaying="y", side="right"))
                    
                    fig.update_layout(yaxis_title=yaxis, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                def plot_bar(data, title: str, yaxis: str = "Value", color=None):
                    if isinstance(data, pd.DataFrame):
                        data = data.squeeze()
                    if data.empty or data.isna().all():
                        st.info(f"📊 {title}: No data available")
                        return
                    
                    fig = px.bar(x=data.index, y=data, title=title,
                                color_discrete_sequence=[color] if color else None)
                    fig.update_layout(yaxis_title=yaxis)
                    st.plotly_chart(fig, use_container_width=True)

                def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "Value", colors=None):
                    if df.empty or df.isna().all().all():
                        st.info(f"📊 {title}: No data available")
                        return
                    
                    fig = go.Figure()
                    has_data = False
                    for i, col in enumerate(df.columns):
                        series = df[col]
                        if series.empty or series.isna().all():
                            continue
                        has_data = True
                        fig.add_trace(go.Scatter(
                            x=series.index, y=series, mode="lines+markers", name=col,
                            line=dict(color=colors[i] if colors and i < len(colors) else None)
                        ))
                    
                    if not has_data:
                        st.info(f"📊 {title}: No data available")
                        return
                    
                    fig.update_layout(title=title, yaxis_title=yaxis, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                # ── CHARTS ──────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 1. Price History")
                plot_line(prices, f"{ticker} Adjusted Close", color="#1f77b4")

                revenue = safe_get(income, "Total Revenue")
                if not revenue.empty:
                    st.markdown("### 2. Revenue")
                    plot_bar(revenue, "Total Revenue", color="#2ca02c")

                gross_profit = safe_get(income, "Gross Profit")
                operating_income = safe_get(income, "Operating Income")
                net_income = safe_get(income, "Net Income")
                
                if not revenue.empty and all(not x.empty for x in [gross_profit, operating_income, net_income]):
                    st.markdown("### 3. Margin Trends (%)")
                    margins = pd.DataFrame({
                        "Gross": (gross_profit / revenue) * 100,
                        "Operating": (operating_income / revenue) * 100,
                        "Net": (net_income / revenue) * 100
                    })
                    plot_multi(margins, "Margins Over Time", yaxis="%", 
                              colors=["#ff7f0e", "#d62728", "#9467bd"])

                if not operating_income.empty or not net_income.empty:
                    st.markdown("### 4. Profitability")
                    profit_df = pd.DataFrame({
                        "Operating Income": operating_income,
                        "Net Income": net_income
                    })
                    plot_multi(profit_df, "Operating vs Net Income", 
                              colors=["#17becf", "#bcbd22"])

                ocf = safe_get(cashflow, "Operating Cash Flow")
                if not ocf.empty and not net_income.empty:
                    st.markdown("### 5. Earnings Quality")
                    eq_df = pd.DataFrame({
                        "Operating Cash Flow": ocf,
                        "Net Income": net_income
                    })
                    plot_multi(eq_df, "Cash Flow vs Earnings", 
                              colors=["#1f77b4", "#ff7f0e"])

                capex = safe_get(cashflow, "Capital Expenditure", 0)
                if not ocf.empty:
                    fcf = ocf + capex  # CapEx usually negative
                    if fcf.abs().sum() > 0:
                        st.markdown("### 6. Free Cash Flow")
                        plot_line(fcf, "Free Cash Flow (OCF + CapEx)", 
                                 show_growth=True, color="#2ca02c")

                sbc = safe_get(cashflow, "Stock Based Compensation")
                if not sbc.empty and sbc.abs().sum() > 0:
                    st.markdown("### 7. Stock-Based Compensation")
                    plot_bar(sbc, "Stock-Based Compensation", color="#9467bd")

                shares_basic = safe_get(income, "Basic Average Shares")
                shares_diluted = safe_get(income, "Diluted Average Shares")
                
                if not shares_basic.empty or not shares_diluted.empty:
                    st.markdown("### 8. Share Count")
                    shares_df = pd.DataFrame({
                        "Basic": shares_basic,
                        "Diluted": shares_diluted
                    })
                    plot_multi(shares_df, "Outstanding Shares", yaxis="Shares")

                    if not shares_diluted.empty and len(shares_diluted.dropna()) > 1:
                        chg_pct = -shares_diluted.pct_change() * 100
                        if not chg_pct.dropna().empty:
                            st.markdown("### 9. Share Count Change %")
                            plot_line(chg_pct.dropna(), 
                                     "Share Change (+ = buyback, - = dilution)", 
                                     yaxis="% Change", color="#e377c2")

                # ── ROIC ────────────────────────────────────────────────────────
                total_assets = safe_get(balance, "Total Assets")
                total_debt = safe_get(balance, "Total Debt", 0)
                cash = safe_get(balance, "Cash And Cash Equivalents", 0)
                tax = safe_get(income, "Tax Provision", 0)
                
                if not net_income.empty and not total_assets.empty:
                    st.markdown("### 10. Return on Invested Capital (ROIC)")
                    nopat = net_income + tax.abs()
                    inv_cap = total_assets - cash - total_debt
                    inv_cap_lag = inv_cap.shift(1)
                    
                    roic = (nopat / inv_cap_lag) * 100
                    roic = roic.replace([np.inf, -np.inf], np.nan)
                    
                    if not roic.dropna().empty:
                        plot_line(roic.dropna(), "ROIC (approx)", 
                                 yaxis="ROIC %", color="#7f7f7f")

                st.success("✓ Analysis complete")

            except Exception as e:
                st.error(f"Error loading {ticker}: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    st.set_page_config(page_title="Grapher", layout="wide")
    render_grapher()
