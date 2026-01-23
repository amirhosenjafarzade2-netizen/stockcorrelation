# grapher.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import Optional


def render_grapher() -> None:
    """
    Grapher module inspired by Qualtrim: Fundamental analysis graphs.
    Fetches and plots interactive charts for Price, Revenue, EBITDA, FCF (vs SBC),
    EPS, key ratios, and a simple DCF projection graph.
    """
    st.subheader("Grapher • Fundamental Analysis (Inspired by Qualtrim)")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        ticker = st.text_input(
            "Ticker symbol",
            value="AAPL",
            help="Examples: AAPL, MSFT, AMZN, GC=F (limited fundamentals for non-stocks)"
        ).strip().upper()

    with col2:
        frequency = st.selectbox(
            "Data frequency",
            options=["Quarterly", "Annual"],
            index=0,
            help="Quarterly for more detail, Annual for longer history"
        )

    years_back = st.slider(
        "Years of history",
        min_value=1,
        max_value=30,
        value=10,
        help="Max available depends on ticker (e.g., 30+ for established stocks)"
    )

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    # Calculate date range (approximate, since financials are periodic)
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years_back)

    # ── Fetch Data ────────────────────────────────────────────────────────────
    if st.button("Generate Graphs", type="primary", use_container_width=True):
        with st.spinner(f"Fetching fundamentals for {ticker} ({frequency.lower()}) ..."):
            try:
                yf_ticker = yf.Ticker(ticker)

                # Financial statements
                yearly = (frequency == "Annual")
                income = yf_ticker.get_income_stmt(yearly=yearly)
                balance = yf_ticker.get_balance_sheet(yearly=yearly)
                cashflow = yf_ticker.get_cashflow(yearly=yearly)

                # Historical prices (for price chart and some ratios)
                prices = yf.download(ticker, start=start_date, end=end_date, progress=False)["Adj Close"]

                if income.empty or balance.empty or cashflow.empty:
                    st.error(f"Limited or no fundamental data for {ticker}. Try a stock ticker.")
                    return

                # Align dates (financials are as-of dates)
                common_dates = income.columns.intersection(balance.columns).intersection(cashflow.columns)
                income = income[common_dates]
                balance = balance[common_dates]
                cashflow = cashflow[common_dates]

                if len(common_dates) < 2:
                    st.warning("Too few periods for meaningful graphs.")
                    return

                st.success(f"Data loaded: {len(common_dates)} {frequency.lower()} periods")

                # ── Helper to create interactive Plotly line/bar ─────────────────
                def plot_metric(df: pd.DataFrame, metric: str, title: str, yaxis: str = "Value",
                                kind: str = "line", color: Optional[str] = None) -> None:
                    plot_df = df.T.reset_index()
                    plot_df.columns = ["Date"] + list(df.index)
                    fig = px.line(plot_df, x="Date", y=metric, title=title, markers=True) if kind == "line" else px.bar(plot_df, x="Date", y=metric, title=title, color=color)
                    fig.update_layout(yaxis_title=yaxis, xaxis_title="Period", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                def plot_multi(df: pd.DataFrame, metrics: list[str], title: str, yaxis: str = "Value",
                               colors: Optional[list[str]] = None) -> None:
                    fig = go.Figure()
                    for i, m in enumerate(metrics):
                        fig.add_trace(go.Scatter(x=df.columns, y=df.loc[m], mode="lines+markers",
                                                 name=m, line=dict(color=colors[i] if colors else None)))
                    fig.update_layout(title=title, yaxis_title=yaxis, xaxis_title="Period", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                # ── 1. Price Chart ───────────────────────────────────────────────
                st.markdown("**Price History** (Interactive Line Chart)")
                price_df = prices.reset_index()
                fig_price = px.line(price_df, x="Date", y="Adj Close", title=f"{ticker} Adjusted Close Price")
                fig_price.update_layout(yaxis_title="Price ($)", hovermode="x")
                st.plotly_chart(fig_price, use_container_width=True)

                # ── 2. Revenue Chart ─────────────────────────────────────────────
                revenue = income.loc["Total Revenue"] if "Total Revenue" in income.index else pd.Series()
                if not revenue.empty:
                    st.markdown("**Revenue Over Time** (Bar Chart)")
                    plot_metric(pd.DataFrame(revenue), "Total Revenue", f"{ticker} Revenue", "$", kind="bar")

                # ── 3. EBITDA Chart ──────────────────────────────────────────────
                ebitda = income.loc["EBITDA"] if "EBITDA" in income.index else pd.Series()
                if not ebitda.empty:
                    st.markdown("**EBITDA Over Time** (Line Chart)")
                    plot_metric(pd.DataFrame(ebitda), "EBITDA", f"{ticker} EBITDA", "$")

                # ── 4. Free Cash Flow vs SBC ─────────────────────────────────────
                ocf = cashflow.loc["Operating Cash Flow"] if "Operating Cash Flow" in cashflow.index else pd.Series()
                capex = cashflow.loc["Capital Expenditure"] if "Capital Expenditure" in cashflow.index else pd.Series(0, index=ocf.index)
                fcf = ocf + capex  # CapEx is negative
                sbc = cashflow.loc["Share Based Compensation"] if "Share Based Compensation" in cashflow.index else pd.Series(0, index=fcf.index)

                if not fcf.empty:
                    st.markdown("**Free Cash Flow vs Stock-Based Compensation** (Multi-Line Chart)")
                    fcf_df = pd.DataFrame({"Free Cash Flow": fcf, "Stock-Based Comp": sbc})
                    plot_multi(fcf_df, ["Free Cash Flow", "Stock-Based Comp"], f"{ticker} FCF vs SBC", "$",
                               colors=["blue", "red"])

                # ── 5. EPS Chart ─────────────────────────────────────────────────
                eps = income.loc["Basic EPS"] if "Basic EPS" in income.index else pd.Series()
                if not eps.empty:
                    st.markdown("**Earnings Per Share (EPS)** (Line Chart)")
                    plot_metric(pd.DataFrame(eps), "Basic EPS", f"{ticker} EPS", "$/Share")

                # ── 6. Key Financial Ratios ──────────────────────────────────────
                st.markdown("**Key Ratios** (Multi-Line Chart)")
                ratios = {}

                # ROE = Net Income / Shareholders Equity
                net_income = income.loc["Net Income"] if "Net Income" in income.index else pd.Series(0)
                equity = balance.loc["Stockholders Equity"] if "Stockholders Equity" in balance.index else pd.Series(1)
                ratios["ROE"] = net_income / equity

                # Debt/Equity = Total Debt / Shareholders Equity
                debt = balance.loc["Total Debt"] if "Total Debt" in balance.index else pd.Series(0)
                ratios["Debt/Equity"] = debt / equity

                # For P/E: Approximate historical P/E using period-end price (need to align dates)
                # Find closest price to financial date
                pe = {}
                for dt in common_dates:
                    closest_date = prices.index[prices.index.get_loc(dt, method="nearest")]
                    close_price = prices.loc[closest_date]
                    eps_val = eps.get(dt, 0)
                    pe[dt] = close_price / eps_val if eps_val != 0 else np.nan
                ratios["P/E"] = pd.Series(pe)

                ratios_df = pd.DataFrame(ratios).dropna(how="all")
                if not ratios_df.empty:
                    plot_multi(ratios_df.T, list(ratios_df.columns), f"{ticker} Key Ratios", "Ratio Value")

                # ── 7. Simple DCF Projection Graph ──────────────────────────────
                st.markdown("**Simple DCF Projection** (Customize Assumptions)")
                growth_rate = st.slider("Expected Annual Growth Rate (%)", -10, 50, 10) / 100
                discount_rate = st.slider("Discount Rate (%)", 5, 20, 10) / 100
                terminal_multiple = st.slider("Terminal P/E Multiple", 5, 50, 15)
                projection_years = st.slider("Projection Years", 1, 20, 5)

                if not fcf.empty:
                    last_fcf = fcf.iloc[-1]
                    last_eps = eps.iloc[-1] if not eps.empty else 0

                    # Project FCF
                    proj_dates = [common_dates[-1] + timedelta(days=365 * (i+1)) for i in range(projection_years)]
                    proj_fcf = [last_fcf * (1 + growth_rate)**(i+1) for i in range(projection_years)]

                    proj_df = pd.DataFrame({"Projected FCF": proj_fcf}, index=proj_dates)

                    # Terminal Value (using EPS * multiple for simplicity)
                    proj_eps = last_eps * (1 + growth_rate)**projection_years
                    terminal_value = proj_eps * terminal_multiple
                    pv_terminal = terminal_value / (1 + discount_rate)**projection_years

                    # Discounted FCF
                    disc_fcf = sum([fcf / (1 + discount_rate)**(i+1) for i, fcf in enumerate(proj_fcf)]) + pv_terminal

                    st.write(f"Estimated Fair Value (DCF): ${disc_fcf:.2f}")

                    fig_dcf = px.bar(proj_df, y="Projected FCF", title=f"{ticker} Projected Free Cash Flow")
                    fig_dcf.update_layout(yaxis_title="$")
                    st.plotly_chart(fig_dcf, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Note: Fundamentals best for US stocks. Commodities/forex may lack data.")


# For quick local testing
if __name__ == "__main__":
    st.set_page_config(page_title="Grapher Test", layout="wide")
    render_grapher()
