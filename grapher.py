import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
from functools import lru_cache
import time


# ══════════════════════════════════════════════════════════════════════════════
# CACHING & DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_price_data(ticker: str, start_date: date, end_date: date) -> pd.Series:
    """Fetch and cache price data with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prices = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if isinstance(prices, pd.DataFrame) and "Close" in prices.columns:
                return prices["Close"]
            elif isinstance(prices, pd.DataFrame) and not prices.empty:
                return prices.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Failed to fetch price data after {max_retries} attempts: {str(e)}")
                return pd.Series(dtype=float)
            time.sleep(1)  # Wait before retry


@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker: str, frequency: str):
    """
    Fetch fundamental data with improved error handling.
    Returns tuple of (income, balance, cashflow, info_dict)
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        is_annual = (frequency == "Annual")
        
        # Fetch financial statements
        if is_annual:
            income = ticker_obj.get_income_stmt(freq="yearly")
            balance = ticker_obj.get_balance_sheet(freq="yearly") 
            cashflow = ticker_obj.get_cashflow(freq="yearly")
        else:
            income = ticker_obj.get_income_stmt(freq="quarterly")
            balance = ticker_obj.get_balance_sheet(freq="quarterly")
            cashflow = ticker_obj.get_cashflow(freq="quarterly")
        
        # Get additional company info
        info = ticker_obj.info
        
        return income, balance, cashflow, info
        
    except Exception as e:
        st.warning(f"Error fetching fundamental data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """Get row from DataFrame with intelligent fallback to alternative field names"""
    if df.empty:
        return pd.Series(dtype=float)
    
    if key in df.index:
        return df.loc[key]
    
    # Comprehensive alternatives mapping
    alternatives = {
        "Total Revenue": ["TotalRevenue", "Total Revenues", "Revenue"],
        "Gross Profit": ["GrossProfit", "Gross Income"],
        "Operating Income": ["OperatingIncome", "EBIT", "Operating Revenue", "EBITDA"],
        "Net Income": ["NetIncome", "Net Income Common Stockholders", "Net Income Available To Common Stockholders"],
        "Operating Cash Flow": ["OperatingCashFlow", "Total Cash From Operating Activities", "Cash Flow From Operations"],
        "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures", "Purchase Of PPE"],
        "Stock Based Compensation": ["StockBasedCompensation", "Stock Based Compensation"],
        "Basic Average Shares": ["BasicAverageShares", "Ordinary Shares Number", "Share Issued", "Basic Shares Outstanding"],
        "Diluted Average Shares": ["DilutedAverageShares", "Diluted NI Available To Com Stockholders", "Diluted Shares Outstanding"],
        "Total Assets": ["TotalAssets", "Total Assets"],
        "Total Debt": ["TotalDebt", "Long Term Debt", "Total Debt", "Short Long Term Debt"],
        "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"],
        "Tax Provision": ["TaxProvision", "Tax Effect Of Unusual Items", "Income Tax Expense"],
        "Research Development": ["ResearchAndDevelopment", "Research Development", "R&D Expense"],
        "Selling General Administrative": ["SellingGeneralAndAdministrative", "SG&A Expense", "Selling General Administrative"],
        "Total Liabilities": ["TotalLiabilities", "Total Liabilities Net Minority Interest"],
        "Stockholders Equity": ["StockholdersEquity", "Total Equity Gross Minority Interest", "Stockholder Equity"],
        "Interest Expense": ["InterestExpense", "Interest Expense Non Operating", "Net Interest Income"],
    }
    
    for alt in alternatives.get(key, []):
        if alt in df.index:
            return df.loc[alt]
    
    if default is not None:
        return pd.Series(default, index=df.columns if not df.empty else [])
    return pd.Series(dtype=float)


def format_large_number(num: float) -> str:
    """Format large numbers for display (e.g., 1.5B, 234.5M)"""
    if pd.isna(num):
        return "N/A"
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def calculate_cagr(series: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate"""
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    
    start_val = clean.iloc[0]
    end_val = clean.iloc[-1]
    
    if start_val <= 0 or end_val <= 0:
        return np.nan
    
    # Calculate time period in years
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    
    if years <= 0:
        return np.nan
    
    cagr = (pow(end_val / start_val, 1 / years) - 1) * 100
    return cagr


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_line(data: pd.Series, title: str, yaxis: str = "Value", 
              color: str = None, show_growth: bool = False) -> None:
    """Enhanced line plot with optional growth overlay"""
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if data.empty or data.isna().all():
        st.info(f"📊 {title}: No data available")
        return
    
    fig = px.line(x=data.index, y=data, title=title, markers=True,
                  color_discrete_sequence=[color] if color else None)
    
    # Add CAGR to title if we have enough data
    cagr = calculate_cagr(data)
    if not np.isnan(cagr):
        fig.update_layout(title=f"{title}<br><sup>CAGR: {cagr:.1f}%</sup>")
    
    if show_growth and len(data.dropna()) > 1:
        growth = data.pct_change() * 100
        if not growth.dropna().empty:
            fig.add_scatter(x=growth.index, y=growth, 
                          name="Growth %", yaxis="y2",
                          line=dict(dash='dot', color='gray'))
            fig.update_layout(
                yaxis2=dict(title="Growth %", overlaying="y", side="right"),
                showlegend=True
            )
    
    fig.update_layout(yaxis_title=yaxis, hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_bar(data: pd.Series, title: str, yaxis: str = "Value", 
             color: str = None) -> None:
    """Enhanced bar plot"""
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if data.empty or data.isna().all():
        st.info(f"📊 {title}: No data available")
        return
    
    fig = px.bar(x=data.index, y=data, title=title,
                color_discrete_sequence=[color] if color else None)
    
    # Add CAGR to title
    cagr = calculate_cagr(data)
    if not np.isnan(cagr):
        fig.update_layout(title=f"{title}<br><sup>CAGR: {cagr:.1f}%</sup>")
    
    fig.update_layout(yaxis_title=yaxis, height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "Value", 
               colors: list = None) -> None:
    """Enhanced multi-line plot"""
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
    
    fig.update_layout(title=title, yaxis_title=yaxis, hovermode="x unified", 
                     height=450, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_waterfall(data: pd.Series, title: str) -> None:
    """Create a waterfall chart showing period-over-period changes"""
    if data.empty or len(data) < 2:
        st.info(f"📊 {title}: Insufficient data for waterfall chart")
        return
    
    changes = data.diff()
    
    fig = go.Figure(go.Waterfall(
        x=data.index,
        y=changes,
        measure=["relative"] * len(changes),
        text=[format_large_number(v) for v in changes],
        textposition="outside",
    ))
    
    fig.update_layout(title=title, yaxis_title="Change", height=450)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_grapher() -> None:
    st.subheader("📈 Grapher • Enhanced Fundamental Analysis")
    st.caption("Professional-grade financial analysis with comprehensive metrics")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", 
                              help="Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)").strip().upper()

    with col2:
        frequency = st.selectbox("Frequency", ["Quarterly", "Annual"], index=0,
                                help="Quarterly shows last 4 quarters, Annual shows last 4 years")

    with col3:
        years_back = st.slider(
            "Price History (Years)",
            min_value=1,
            max_value=30,
            value=10,
            help="Controls price chart only. Fundamentals limited by Yahoo Finance API."
        )
    
    with col4:
        chart_style = st.selectbox("Chart Style", ["Professional", "Colorful", "Minimal"], 
                                  help="Visual style for charts")

    if not ticker:
        st.info("👆 Enter a ticker symbol to begin analysis.")
        return

    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years_back)

    if st.button("🚀 Load & Analyze", type="primary", use_container_width=True):
        with st.spinner(f"Fetching comprehensive data for {ticker}..."):
            try:
                # ── Fetch all data ────────────────────────────────────────────
                prices = fetch_price_data(ticker, start_date, end_date)
                income, balance, cashflow, info = fetch_fundamental_data(ticker, frequency)

                if prices.empty and income.empty and balance.empty and cashflow.empty:
                    st.error(f"❌ No data returned for {ticker}. Please check the ticker symbol.")
                    return

                # ── Company Info Summary ──────────────────────────────────────
                if info:
                    st.markdown("### 📋 Company Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Company", info.get("longName", ticker))
                    with col2:
                        st.metric("Sector", info.get("sector", "N/A"))
                    with col3:
                        st.metric("Industry", info.get("industry", "N/A"))
                    with col4:
                        market_cap = info.get("marketCap", None)
                        if market_cap:
                            st.metric("Market Cap", format_large_number(market_cap))
                        else:
                            st.metric("Market Cap", "N/A")

                # ── Data Availability Info ────────────────────────────────────
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if not prices.empty:
                        st.success(f"✅ **Price Data**: {prices.index.min().date()} → {prices.index.max().date()} ({len(prices)} days)")
                    else:
                        st.warning("⚠️ No price data available")

                with col2:
                    if not income.empty:
                        dates = sorted(income.columns)
                        years_available = (dates[-1] - dates[0]).days / 365.25
                        st.success(f"✅ **Fundamentals**: {dates[0].date()} → {dates[-1].date()} ({len(dates)} periods)")
                    else:
                        st.warning("⚠️ No fundamental data available")

                # Align dates across statements
                if not income.empty and not balance.empty and not cashflow.empty:
                    common_dates = income.columns.intersection(balance.columns).intersection(cashflow.columns)
                    income = income[common_dates]
                    balance = balance[common_dates]
                    cashflow = cashflow[common_dates]

                # ── Extract key metrics ───────────────────────────────────────
                revenue = safe_get(income, "Total Revenue")
                gross_profit = safe_get(income, "Gross Profit")
                operating_income = safe_get(income, "Operating Income")
                net_income = safe_get(income, "Net Income")
                rd_expense = safe_get(income, "Research Development")
                sga_expense = safe_get(income, "Selling General Administrative")
                
                ocf = safe_get(cashflow, "Operating Cash Flow")
                capex = safe_get(cashflow, "Capital Expenditure", 0)
                sbc = safe_get(cashflow, "Stock Based Compensation")
                
                total_assets = safe_get(balance, "Total Assets")
                total_liabilities = safe_get(balance, "Total Liabilities")
                stockholders_equity = safe_get(balance, "Stockholders Equity")
                total_debt = safe_get(balance, "Total Debt", 0)
                cash = safe_get(balance, "Cash And Cash Equivalents", 0)
                
                shares_basic = safe_get(income, "Basic Average Shares")
                shares_diluted = safe_get(income, "Diluted Average Shares")
                tax = safe_get(income, "Tax Provision", 0)

                # ══════════════════════════════════════════════════════════════
                # CHARTS SECTION
                # ══════════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("## 📊 Financial Analysis")
                
                # Tabs for organized view
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Price & Revenue", 
                    "💰 Profitability", 
                    "💵 Cash Flow",
                    "📊 Balance Sheet",
                    "📐 Ratios & Metrics"
                ])

                # ── TAB 1: Price & Revenue ────────────────────────────────────
                with tab1:
                    st.markdown("### Stock Price History")
                    if not prices.empty:
                        plot_line(prices, f"{ticker} Adjusted Close Price", 
                                 yaxis="Price (USD)", color="#1f77b4", show_growth=False)
                        
                        # Price statistics - extract scalar values
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            current_price = float(prices.iloc[-1])
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            max_price = float(prices.max())
                            st.metric("52W High", f"${max_price:.2f}")
                        with col3:
                            min_price = float(prices.min())
                            st.metric("52W Low", f"${min_price:.2f}")
                        with col4:
                            ytd_return = ((float(prices.iloc[-1]) / float(prices.iloc[0])) - 1) * 100
                            st.metric("Period Return", f"{ytd_return:.1f}%")
                    
                    st.markdown("### Revenue Growth")
                    if not revenue.empty:
                        plot_bar(revenue, "Total Revenue", yaxis="Revenue (USD)", color="#2ca02c")
                        
                        # Revenue metrics - extract scalar values
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            latest_rev = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                            st.metric("Latest Revenue", format_large_number(latest_rev))
                        with col2:
                            if len(revenue) > 1:
                                rev_last = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                                rev_prev = float(revenue.iloc[-2]) if not pd.isna(revenue.iloc[-2]) else 0
                                if rev_prev > 0:
                                    rev_growth = ((rev_last / rev_prev) - 1) * 100
                                    st.metric("Last Period Growth", f"{rev_growth:.1f}%")
                        with col3:
                            cagr = calculate_cagr(revenue)
                            if not np.isnan(cagr):
                                st.metric("Revenue CAGR", f"{cagr:.1f}%")

                # ── TAB 2: Profitability ──────────────────────────────────────
                with tab2:
                    st.markdown("### Profit Margins")
                    if not revenue.empty and all(not x.empty for x in [gross_profit, operating_income, net_income]):
                        margins = pd.DataFrame({
                            "Gross Margin": (gross_profit / revenue) * 100,
                            "Operating Margin": (operating_income / revenue) * 100,
                            "Net Margin": (net_income / revenue) * 100
                        })
                        plot_multi(margins, "Profit Margins Over Time", yaxis="Margin (%)", 
                                  colors=["#ff7f0e", "#d62728", "#9467bd"])
                        
                        # Latest margins - extract scalar values
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            gross_margin = float(margins['Gross Margin'].iloc[-1]) if not pd.isna(margins['Gross Margin'].iloc[-1]) else 0
                            st.metric("Gross Margin", f"{gross_margin:.1f}%")
                        with col2:
                            op_margin = float(margins['Operating Margin'].iloc[-1]) if not pd.isna(margins['Operating Margin'].iloc[-1]) else 0
                            st.metric("Operating Margin", f"{op_margin:.1f}%")
                        with col3:
                            net_margin = float(margins['Net Margin'].iloc[-1]) if not pd.isna(margins['Net Margin'].iloc[-1]) else 0
                            st.metric("Net Margin", f"{net_margin:.1f}%")
                    
                    st.markdown("### Absolute Profitability")
                    if not operating_income.empty or not net_income.empty:
                        profit_df = pd.DataFrame({
                            "Operating Income": operating_income,
                            "Net Income": net_income
                        })
                        plot_multi(profit_df, "Operating vs Net Income", 
                                  yaxis="Income (USD)", colors=["#17becf", "#bcbd22"])
                    
                    st.markdown("### Operating Expenses")
                    if not rd_expense.empty or not sga_expense.empty:
                        opex_df = pd.DataFrame({
                            "R&D": rd_expense,
                            "SG&A": sga_expense
                        })
                        plot_multi(opex_df, "R&D and SG&A Expenses", 
                                  yaxis="Expense (USD)", colors=["#e377c2", "#7f7f7f"])

                # ── TAB 3: Cash Flow ──────────────────────────────────────────
                with tab3:
                    st.markdown("### Operating Cash Flow")
                    if not ocf.empty:
                        plot_line(ocf, "Operating Cash Flow", yaxis="Cash Flow (USD)", 
                                 color="#1f77b4", show_growth=True)
                    
                    st.markdown("### Free Cash Flow")
                    if not ocf.empty:
                        fcf = ocf + capex  # CapEx is typically negative
                        if fcf.abs().sum() > 0:
                            plot_line(fcf, "Free Cash Flow (OCF + CapEx)", 
                                     yaxis="FCF (USD)", color="#2ca02c", show_growth=True)
                            
                            # FCF metrics - extract scalar values
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                latest_fcf = float(fcf.iloc[-1]) if not pd.isna(fcf.iloc[-1]) else 0
                                st.metric("Latest FCF", format_large_number(latest_fcf))
                            with col2:
                                if not net_income.empty:
                                    fcf_val = float(fcf.iloc[-1]) if not pd.isna(fcf.iloc[-1]) else 0
                                    ni_val = float(net_income.iloc[-1]) if not pd.isna(net_income.iloc[-1]) else 0
                                    if ni_val != 0:
                                        fcf_conversion = (fcf_val / ni_val) * 100
                                        st.metric("FCF Conversion", f"{fcf_conversion:.0f}%")
                            with col3:
                                cagr = calculate_cagr(fcf)
                                if not np.isnan(cagr):
                                    st.metric("FCF CAGR", f"{cagr:.1f}%")
                    
                    st.markdown("### Earnings Quality: Cash vs Accrual")
                    if not ocf.empty and not net_income.empty:
                        eq_df = pd.DataFrame({
                            "Operating Cash Flow": ocf,
                            "Net Income": net_income
                        })
                        plot_multi(eq_df, "Cash Flow vs Net Income", 
                                  colors=["#1f77b4", "#ff7f0e"])
                        
                        st.info("💡 **Tip**: High-quality earnings show OCF consistently ≥ Net Income")
                    
                    st.markdown("### Stock-Based Compensation")
                    if not sbc.empty and sbc.abs().sum() > 0:
                        plot_bar(sbc, "Stock-Based Compensation", 
                                yaxis="SBC (USD)", color="#9467bd")
                        
                        if not revenue.empty:
                            sbc_val = float(sbc.iloc[-1]) if not pd.isna(sbc.iloc[-1]) else 0
                            rev_val = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                            if rev_val > 0:
                                sbc_pct = (sbc_val / rev_val) * 100
                                st.metric("SBC as % of Revenue", f"{sbc_pct:.1f}%")

                # ── TAB 4: Balance Sheet ──────────────────────────────────────
                with tab4:
                    st.markdown("### Assets & Liabilities")
                    if not total_assets.empty and not total_liabilities.empty:
                        bal_df = pd.DataFrame({
                            "Total Assets": total_assets,
                            "Total Liabilities": total_liabilities,
                            "Equity": stockholders_equity if not stockholders_equity.empty else total_assets - total_liabilities
                        })
                        plot_multi(bal_df, "Balance Sheet Components", 
                                  yaxis="Amount (USD)", colors=["#2ca02c", "#d62728", "#1f77b4"])
                    
                    st.markdown("### Debt & Cash Position")
                    if not total_debt.empty or not cash.empty:
                        debt_df = pd.DataFrame({
                            "Total Debt": total_debt,
                            "Cash": cash,
                            "Net Debt": total_debt - cash
                        })
                        plot_multi(debt_df, "Debt vs Cash", 
                                  yaxis="Amount (USD)", colors=["#d62728", "#2ca02c", "#ff7f0e"])
                        
                        # Debt metrics - extract scalar values
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            debt_val = float(total_debt.iloc[-1]) if not pd.isna(total_debt.iloc[-1]) else 0
                            st.metric("Total Debt", format_large_number(debt_val))
                        with col2:
                            cash_val = float(cash.iloc[-1]) if not pd.isna(cash.iloc[-1]) else 0
                            st.metric("Cash", format_large_number(cash_val))
                        with col3:
                            net_debt = debt_val - cash_val
                            st.metric("Net Debt", format_large_number(net_debt))
                    
                    st.markdown("### Share Count Evolution")
                    if not shares_basic.empty or not shares_diluted.empty:
                        shares_df = pd.DataFrame({
                            "Basic Shares": shares_basic,
                            "Diluted Shares": shares_diluted
                        })
                        plot_multi(shares_df, "Outstanding Shares", 
                                  yaxis="Shares", colors=["#1f77b4", "#ff7f0e"])
                        
                        if not shares_diluted.empty and len(shares_diluted.dropna()) > 1:
                            # Calculate buyback/dilution
                            share_change = shares_diluted.pct_change() * -100  # Negative for easier interpretation
                            if not share_change.dropna().empty:
                                st.markdown("### Share Count Change %")
                                plot_line(share_change.dropna(), 
                                         "Share Count Change (+ = Buyback, - = Dilution)", 
                                         yaxis="% Change", color="#e377c2")

                # ── TAB 5: Ratios & Metrics ───────────────────────────────────
                with tab5:
                    st.markdown("### Return on Invested Capital (ROIC)")
                    if not net_income.empty and not total_assets.empty:
                        # Simplified ROIC calculation
                        nopat = net_income + tax.abs()
                        invested_capital = total_assets - cash - total_debt
                        invested_capital_lagged = invested_capital.shift(1)
                        
                        roic = (nopat / invested_capital_lagged) * 100
                        roic = roic.replace([np.inf, -np.inf], np.nan)
                        
                        if not roic.dropna().empty:
                            plot_line(roic.dropna(), "Return on Invested Capital", 
                                     yaxis="ROIC (%)", color="#7f7f7f")
                            
                            latest_roic = float(roic.dropna().iloc[-1]) if len(roic.dropna()) > 0 else 0
                            st.metric("Latest ROIC", f"{latest_roic:.1f}%")
                            st.info("💡 **Benchmark**: ROIC > 15% is generally considered good, > 20% is excellent")
                    
                    st.markdown("### Return on Equity (ROE)")
                    if not net_income.empty and not stockholders_equity.empty:
                        equity_lagged = stockholders_equity.shift(1)
                        roe = (net_income / equity_lagged) * 100
                        roe = roe.replace([np.inf, -np.inf], np.nan)
                        
                        if not roe.dropna().empty:
                            plot_line(roe.dropna(), "Return on Equity", 
                                     yaxis="ROE (%)", color="#17becf")
                            
                            latest_roe = float(roe.dropna().iloc[-1]) if len(roe.dropna()) > 0 else 0
                            st.metric("Latest ROE", f"{latest_roe:.1f}%")
                    
                    st.markdown("### Debt-to-Equity Ratio")
                    if not total_debt.empty and not stockholders_equity.empty:
                        debt_to_equity = total_debt / stockholders_equity
                        debt_to_equity = debt_to_equity.replace([np.inf, -np.inf], np.nan)
                        
                        if not debt_to_equity.dropna().empty:
                            plot_line(debt_to_equity.dropna(), "Debt-to-Equity Ratio", 
                                     yaxis="Ratio", color="#d62728")
                            
                            latest_de = float(debt_to_equity.dropna().iloc[-1]) if len(debt_to_equity.dropna()) > 0 else 0
                            st.metric("Latest D/E", f"{latest_de:.2f}x")
                            st.info("💡 **Benchmark**: < 1.0 is generally conservative, varies by industry")
                    
                    st.markdown("### Per-Share Metrics")
                    if not net_income.empty and not shares_diluted.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            eps = net_income / shares_diluted
                            if not eps.dropna().empty:
                                plot_line(eps.dropna(), "Earnings Per Share (EPS)", 
                                         yaxis="EPS ($)", color="#2ca02c", show_growth=True)
                        
                        with col2:
                            if not ocf.empty:
                                cfps = ocf / shares_diluted
                                if not cfps.dropna().empty:
                                    plot_line(cfps.dropna(), "Cash Flow Per Share", 
                                             yaxis="CFPS ($)", color="#1f77b4", show_growth=True)

                st.markdown("---")
                st.success("✅ **Analysis Complete!** All available metrics have been calculated and displayed.")
                
                # Export option
                st.markdown("### 📥 Export Data")
                if st.button("Generate CSV Export"):
                    # Combine all data for export
                    export_data = pd.DataFrame({
                        'Revenue': revenue,
                        'Gross_Profit': gross_profit,
                        'Operating_Income': operating_income,
                        'Net_Income': net_income,
                        'Operating_Cash_Flow': ocf,
                        'Free_Cash_Flow': ocf + capex if not ocf.empty else pd.Series(dtype=float),
                        'Total_Assets': total_assets,
                        'Total_Debt': total_debt,
                        'Cash': cash,
                    })
                    
                    csv = export_data.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_financial_data.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
                st.info("💡 Try refreshing the page or checking if the ticker symbol is correct.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.set_page_config(
        page_title="Financial Grapher Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    render_grapher()
