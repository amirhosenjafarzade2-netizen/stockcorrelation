"""
Ultimate Financial Grapher
- SEC Edgar for 10+ years of historical data
- All comprehensive graphs and metrics
- Analyst forecasts and predictions
- Professional presentation in organized tabs
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
import time

# Try to import edgartools
try:
    from edgar import Company, set_identity
    from edgar.xbrl import XBRLS
    EDGAR_AVAILABLE = True
    set_identity("finance_user user@example.com")
except ImportError:
    EDGAR_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)
def fetch_edgar_data(ticker: str, num_years: int = 10):
    """Fetch from SEC Edgar"""
    if not EDGAR_AVAILABLE:
        return None, None, None, "not available"
    
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K").head(num_years)
        
        if len(filings) == 0:
            return None, None, None, "no filings"
        
        xbrls = XBRLS.from_filings(filings)
        statements = xbrls.statements
        
        income_df = statements.income_statement().to_dataframe() if statements.income_statement() else pd.DataFrame()
        balance_df = statements.balance_sheet().to_dataframe() if statements.balance_sheet() else pd.DataFrame()
        cashflow_df = statements.cashflow_statement().to_dataframe() if statements.cashflow_statement() else pd.DataFrame()
        
        return income_df, balance_df, cashflow_df, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data(ttl=7200)
def fetch_edgar_quarterly(ticker: str, num_quarters: int = 20):
    """Fetch quarterly from SEC Edgar"""
    if not EDGAR_AVAILABLE:
        return None, None, None, "not available"
    
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-Q").head(num_quarters)
        
        if len(filings) == 0:
            return None, None, None, "no filings"
        
        xbrls = XBRLS.from_filings(filings)
        statements = xbrls.statements
        
        income_df = statements.income_statement().to_dataframe() if statements.income_statement() else pd.DataFrame()
        balance_df = statements.balance_sheet().to_dataframe() if statements.balance_sheet() else pd.DataFrame()
        cashflow_df = statements.cashflow_statement().to_dataframe() if statements.cashflow_statement() else pd.DataFrame()
        
        return income_df, balance_df, cashflow_df, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data(ttl=3600)
def fetch_price_data(ticker: str, start_date: date, end_date: date) -> pd.Series:
    """Fetch price data"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prices = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if isinstance(prices, pd.DataFrame) and "Close" in prices.columns:
                return prices["Close"]
            elif isinstance(prices, pd.DataFrame) and not prices.empty:
                return prices.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        except Exception as e:
            if attempt == max_retries - 1:
                return pd.Series(dtype=float)
            time.sleep(1)


@st.cache_data(ttl=3600)
def fetch_yfinance_fundamentals(ticker: str, frequency: str):
    """Fetch from yfinance (4 periods max)"""
    try:
        ticker_obj = yf.Ticker(ticker)
        is_annual = (frequency == "Annual")
        
        if is_annual:
            income = ticker_obj.get_income_stmt(freq="yearly")
            balance = ticker_obj.get_balance_sheet(freq="yearly") 
            cashflow = ticker_obj.get_cashflow(freq="yearly")
        else:
            income = ticker_obj.get_income_stmt(freq="quarterly")
            balance = ticker_obj.get_balance_sheet(freq="quarterly")
            cashflow = ticker_obj.get_cashflow(freq="quarterly")
        
        info = ticker_obj.info
        return income, balance, cashflow, info
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


@st.cache_data(ttl=3600)
def fetch_analyst_data(ticker: str):
    """Fetch analyst forecasts and recommendations"""
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get various analyst data
        recommendations = ticker_obj.recommendations
        analyst_price_target = ticker_obj.analyst_price_targets if hasattr(ticker_obj, 'analyst_price_targets') else None
        earnings_forecasts = ticker_obj.earnings_forecasts if hasattr(ticker_obj, 'earnings_forecasts') else None
        revenue_forecasts = ticker_obj.revenue_estimate if hasattr(ticker_obj, 'revenue_estimate') else None
        earnings_trend = ticker_obj.earnings_trend if hasattr(ticker_obj, 'earnings_trend') else None
        upgrades_downgrades = ticker_obj.upgrades_downgrades if hasattr(ticker_obj, 'upgrades_downgrades') else None
        
        return {
            'recommendations': recommendations,
            'price_target': analyst_price_target,
            'earnings_forecasts': earnings_forecasts,
            'revenue_forecasts': revenue_forecasts,
            'earnings_trend': earnings_trend,
            'upgrades_downgrades': upgrades_downgrades
        }
    except Exception as e:
        return {}


def process_edgar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process Edgar dataframe"""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.T
    if not isinstance(df.columns, pd.DatetimeIndex):
        try:
            df.columns = pd.to_datetime(df.columns)
        except:
            pass
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """Smart field extraction"""
    if df.empty:
        return pd.Series(dtype=float)
    
    if key in df.index:
        return df.loc[key]
    
    # Case-insensitive partial
    for idx in df.index:
        if isinstance(idx, str) and key.lower() in idx.lower():
            return df.loc[idx]
    
    alternatives = {
        "Total Revenue": ["Revenue", "TotalRevenue", "Revenues", "Sales", "Net Sales"],
        "Gross Profit": ["GrossProfit", "Gross Income"],
        "Operating Income": ["OperatingIncome", "Operating Income Loss", "EBIT"],
        "Net Income": ["NetIncome", "Net Income Loss", "Net Income Available To Common Stockholders"],
        "Operating Cash Flow": ["OperatingCashFlow", "Net Cash From Operating Activities"],
        "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures", "Payments To Acquire PPE"],
        "Stock Based Compensation": ["StockBasedCompensation", "Share Based Compensation"],
        "Basic Average Shares": ["BasicAverageShares", "Weighted Average Shares Outstanding Basic"],
        "Diluted Average Shares": ["DilutedAverageShares", "Weighted Average Shares Outstanding Diluted"],
        "Total Assets": ["TotalAssets", "Assets"],
        "Total Debt": ["TotalDebt", "Long Term Debt", "Debt"],
        "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash"],
        "Tax Provision": ["TaxProvision", "Income Tax Expense"],
        "Research Development": ["ResearchAndDevelopment", "R&D"],
        "Selling General Administrative": ["SellingGeneralAndAdministrative", "SG&A"],
        "Total Liabilities": ["TotalLiabilities", "Liabilities"],
        "Stockholders Equity": ["StockholdersEquity", "Total Equity"],
    }
    
    for alt in alternatives.get(key, []):
        if alt in df.index:
            return df.loc[alt]
        for idx in df.index:
            if isinstance(idx, str) and alt.lower() in idx.lower():
                return df.loc[idx]
    
    if default is not None:
        return pd.Series(default, index=df.columns if not df.empty else [])
    return pd.Series(dtype=float)


def format_large_number(num: float) -> str:
    """Format numbers"""
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
    """Calculate CAGR"""
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    start_val, end_val = clean.iloc[0], clean.iloc[-1]
    if start_val <= 0 or end_val <= 0:
        return np.nan
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (pow(end_val / start_val, 1 / years) - 1) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_line(data: pd.Series, title: str, yaxis: str = "Value", color: str = None, show_growth: bool = False):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if data.empty or data.isna().all():
        st.info(f"📊 {title}: No data")
        return
    
    fig = px.line(x=data.index, y=data, title=title, markers=True,
                  color_discrete_sequence=[color] if color else None)
    
    cagr = calculate_cagr(data)
    if not np.isnan(cagr):
        fig.update_layout(title=f"{title}<br><sup>CAGR: {cagr:.1f}%</sup>")
    
    if show_growth and len(data.dropna()) > 1:
        growth = data.pct_change() * 100
        if not growth.dropna().empty:
            fig.add_scatter(x=growth.index, y=growth, name="Growth %", yaxis="y2",
                          line=dict(dash='dot', color='gray'))
            fig.update_layout(yaxis2=dict(title="Growth %", overlaying="y", side="right"), showlegend=True)
    
    fig.update_layout(yaxis_title=yaxis, hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_bar(data: pd.Series, title: str, yaxis: str = "Value", color: str = None):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if data.empty or data.isna().all():
        st.info(f"📊 {title}: No data")
        return
    
    fig = px.bar(x=data.index, y=data, title=title, color_discrete_sequence=[color] if color else None)
    cagr = calculate_cagr(data)
    if not np.isnan(cagr):
        fig.update_layout(title=f"{title}<br><sup>CAGR: {cagr:.1f}%</sup>")
    fig.update_layout(yaxis_title=yaxis, height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "Value", colors: list = None):
    if df.empty or df.isna().all().all():
        st.info(f"📊 {title}: No data")
        return
    
    fig = go.Figure()
    has_data = False
    for i, col in enumerate(df.columns):
        series = df[col]
        if series.empty or series.isna().all():
            continue
        has_data = True
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines+markers", name=col,
                                line=dict(color=colors[i] if colors and i < len(colors) else None)))
    
    if not has_data:
        st.info(f"📊 {title}: No data")
        return
    
    fig.update_layout(title=title, yaxis_title=yaxis, hovermode="x unified", height=450, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_grapher():
    st.title("📈 Ultimate Financial Grapher")
    
    if EDGAR_AVAILABLE:
        st.success("✅ **SEC Edgar Active** - 10+ years of data available!")
    else:
        st.warning("⚠️ Limited to Yahoo Finance (4 periods). Install `edgartools` for more: `pip install edgartools`")

    # INPUTS
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    with col2:
        frequency = st.selectbox("Frequency", ["Annual", "Quarterly"], index=0)

    with col3:
        if EDGAR_AVAILABLE and frequency == "Annual":
            num_periods = st.slider("Years of Data", 1, 15, 10)
        elif EDGAR_AVAILABLE and frequency == "Quarterly":
            num_periods = st.slider("Quarters", 4, 40, 20)
        else:
            num_periods = 4
            st.caption("Limited to 4 periods")
    
    with col4:
        price_years = st.slider("Price History (Years)", 1, 30, 10)

    if not ticker:
        st.info("👆 Enter a ticker")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * price_years)

    if st.button("🚀 Load & Analyze", type="primary", use_container_width=True):
        with st.spinner(f"Fetching comprehensive data for {ticker}..."):
            
            # TRY EDGAR FIRST
            use_edgar = False
            income, balance, cashflow = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            if EDGAR_AVAILABLE:
                st.info("🔍 Fetching from SEC Edgar...")
                if frequency == "Annual":
                    income, balance, cashflow, error = fetch_edgar_data(ticker, num_periods)
                else:
                    income, balance, cashflow, error = fetch_edgar_quarterly(ticker, num_periods)
                
                if not error:
                    use_edgar = True
                    income = process_edgar_df(income)
                    balance = process_edgar_df(balance)
                    cashflow = process_edgar_df(cashflow)
                    st.success(f"✅ Edgar: {len(income.columns) if not income.empty else 0} periods")
                else:
                    st.warning(f"Edgar failed: {error}. Using Yahoo Finance...")
            
            # FALLBACK TO YFINANCE
            if not use_edgar:
                st.info("📊 Using Yahoo Finance...")
                income, balance, cashflow, info = fetch_yfinance_fundamentals(ticker, frequency)
            else:
                try:
                    info = yf.Ticker(ticker).info
                except:
                    info = {}
            
            # PRICE DATA
            prices = fetch_price_data(ticker, start_date, end_date)
            
            # ANALYST DATA
            st.info("📊 Fetching analyst forecasts...")
            analyst_data = fetch_analyst_data(ticker)

            if prices.empty and income.empty:
                st.error("❌ No data available")
                return

            # COMPANY INFO
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
                    market_cap = info.get("marketCap")
                    st.metric("Market Cap", format_large_number(market_cap) if market_cap else "N/A")

            # DATA SOURCE INFO
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if not prices.empty:
                    st.success(f"✅ **Price**: {len(prices)} days")
            with col2:
                if not income.empty:
                    source = "SEC Edgar" if use_edgar else "Yahoo Finance"
                    st.success(f"✅ **Fundamentals ({source})**: {len(income.columns)} periods")

            # EXTRACT METRICS
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

            # ══════════════════════════════════════════════════════════════════
            # TABBED INTERFACE
            # ══════════════════════════════════════════════════════════════════
            
            st.markdown("---")
            st.markdown("## 📊 Comprehensive Analysis")
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Price & Revenue", 
                "💰 Profitability", 
                "💵 Cash Flow",
                "📊 Balance Sheet",
                "📐 Ratios & Metrics",
                "🔮 Analyst Forecasts"
            ])

            # TAB 1: PRICE & REVENUE
            with tab1:
                st.markdown("### Stock Price History")
                if not prices.empty:
                    plot_line(prices, f"{ticker} Adjusted Close Price", "Price (USD)", "#1f77b4", False)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        current_price = float(prices.iloc[-1])
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        max_price = float(prices.max())
                        st.metric("Period High", f"${max_price:.2f}")
                    with col3:
                        min_price = float(prices.min())
                        st.metric("Period Low", f"${min_price:.2f}")
                    with col4:
                        ret = ((float(prices.iloc[-1]) / float(prices.iloc[0])) - 1) * 100
                        st.metric("Period Return", f"{ret:.1f}%")
                
                st.markdown("### Revenue Growth")
                if not revenue.empty:
                    plot_bar(revenue, "Total Revenue", "Revenue (USD)", "#2ca02c")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        latest_rev = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                        st.metric("Latest Revenue", format_large_number(latest_rev))
                    with col2:
                        if len(revenue) > 1:
                            rev_last = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                            rev_prev = float(revenue.iloc[-2]) if not pd.isna(revenue.iloc[-2]) else 0
                            if rev_prev > 0:
                                growth = ((rev_last / rev_prev) - 1) * 100
                                st.metric("Period Growth", f"{growth:.1f}%")
                    with col3:
                        cagr = calculate_cagr(revenue)
                        if not np.isnan(cagr):
                            st.metric("Revenue CAGR", f"{cagr:.1f}%")

            # TAB 2: PROFITABILITY
            with tab2:
                st.markdown("### Profit Margins")
                # Check all series exist and have data
                can_calc_margins = (not revenue.empty and not gross_profit.empty and 
                                   not operating_income.empty and not net_income.empty)
                
                margins = None
                if can_calc_margins:
                    try:
                        margins = pd.DataFrame({
                            "Gross Margin": (gross_profit / revenue) * 100,
                            "Operating Margin": (operating_income / revenue) * 100,
                            "Net Margin": (net_income / revenue) * 100
                        })
                        plot_multi(margins, "Margins Over Time", "Margin (%)", ["#ff7f0e", "#d62728", "#9467bd"])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            gm = float(margins['Gross Margin'].iloc[-1]) if not pd.isna(margins['Gross Margin'].iloc[-1]) else 0
                            st.metric("Gross Margin", f"{gm:.1f}%")
                        with col2:
                            om = float(margins['Operating Margin'].iloc[-1]) if not pd.isna(margins['Operating Margin'].iloc[-1]) else 0
                            st.metric("Operating Margin", f"{om:.1f}%")
                        with col3:
                            nm = float(margins['Net Margin'].iloc[-1]) if not pd.isna(margins['Net Margin'].iloc[-1]) else 0
                            st.metric("Net Margin", f"{nm:.1f}%")
                    except Exception as e:
                        st.warning(f"Could not calculate margins: {str(e)}")
                else:
                    st.info("📊 Margin data not available - missing required financial metrics")
                
                st.markdown("### Absolute Profitability")
                if not operating_income.empty or not net_income.empty:
                    try:
                        profit_df = pd.DataFrame({"Operating Income": operating_income, "Net Income": net_income})
                        plot_multi(profit_df, "Operating vs Net Income", "Income (USD)", ["#17becf", "#bcbd22"])
                    except Exception as e:
                        st.warning(f"Could not plot profitability: {str(e)}")
                
                st.markdown("### Operating Expenses")
                if not rd_expense.empty or not sga_expense.empty:
                    try:
                        opex_df = pd.DataFrame({"R&D": rd_expense, "SG&A": sga_expense})
                        plot_multi(opex_df, "R&D and SG&A Expenses", "Expense (USD)", ["#e377c2", "#7f7f7f"])
                    except Exception as e:
                        st.warning(f"Could not plot expenses: {str(e)}")

            # TAB 3: CASH FLOW
            with tab3:
                st.markdown("### Operating Cash Flow")
                if not ocf.empty:
                    plot_line(ocf, "Operating Cash Flow", "Cash Flow (USD)", "#1f77b4", True)
                
                st.markdown("### Free Cash Flow")
                if not ocf.empty:
                    fcf = ocf + capex
                    if fcf.abs().sum() > 0:
                        plot_line(fcf, "Free Cash Flow (OCF + CapEx)", "FCF (USD)", "#2ca02c", True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            fcf_val = float(fcf.iloc[-1]) if not pd.isna(fcf.iloc[-1]) else 0
                            st.metric("Latest FCF", format_large_number(fcf_val))
                        with col2:
                            if not net_income.empty:
                                fcf_val = float(fcf.iloc[-1]) if not pd.isna(fcf.iloc[-1]) else 0
                                ni_val = float(net_income.iloc[-1]) if not pd.isna(net_income.iloc[-1]) else 0
                                if ni_val != 0:
                                    conv = (fcf_val / ni_val) * 100
                                    st.metric("FCF Conversion", f"{conv:.0f}%")
                        with col3:
                            cagr = calculate_cagr(fcf)
                            if not np.isnan(cagr):
                                st.metric("FCF CAGR", f"{cagr:.1f}%")
                
                st.markdown("### Earnings Quality")
                if not ocf.empty and not net_income.empty:
                    try:
                        eq_df = pd.DataFrame({"Operating Cash Flow": ocf, "Net Income": net_income})
                        plot_multi(eq_df, "Cash Flow vs Earnings", "Amount (USD)", ["#1f77b4", "#ff7f0e"])
                        st.info("💡 High-quality earnings: OCF ≥ Net Income")
                    except Exception as e:
                        st.warning(f"Could not plot earnings quality: {str(e)}")
                
                st.markdown("### Stock-Based Compensation")
                if not sbc.empty and sbc.abs().sum() > 0:
                    plot_bar(sbc, "Stock-Based Compensation", "SBC (USD)", "#9467bd")
                    if not revenue.empty:
                        sbc_val = float(sbc.iloc[-1]) if not pd.isna(sbc.iloc[-1]) else 0
                        rev_val = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                        if rev_val > 0:
                            pct = (sbc_val / rev_val) * 100
                            st.metric("SBC as % of Revenue", f"{pct:.1f}%")

            # TAB 4: BALANCE SHEET
            with tab4:
                st.markdown("### Assets & Liabilities")
                if not total_assets.empty and not total_liabilities.empty:
                    try:
                        bal_df = pd.DataFrame({
                            "Total Assets": total_assets,
                            "Total Liabilities": total_liabilities,
                            "Equity": stockholders_equity if not stockholders_equity.empty else total_assets - total_liabilities
                        })
                        plot_multi(bal_df, "Balance Sheet Components", "Amount (USD)", ["#2ca02c", "#d62728", "#1f77b4"])
                    except Exception as e:
                        st.warning(f"Could not plot balance sheet: {str(e)}")
                
                st.markdown("### Debt & Cash Position")
                if not total_debt.empty or not cash.empty:
                    try:
                        debt_df = pd.DataFrame({
                            "Total Debt": total_debt,
                            "Cash": cash,
                            "Net Debt": total_debt - cash
                        })
                        plot_multi(debt_df, "Debt vs Cash", "Amount (USD)", ["#d62728", "#2ca02c", "#ff7f0e"])
                    except Exception as e:
                        st.warning(f"Could not plot debt position: {str(e)}")
                    
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
                    try:
                        shares_df = pd.DataFrame({"Basic Shares": shares_basic, "Diluted Shares": shares_diluted})
                        plot_multi(shares_df, "Outstanding Shares", "Shares", ["#1f77b4", "#ff7f0e"])
                    except Exception as e:
                        st.warning(f"Could not plot shares: {str(e)}")
                    
                    if not shares_diluted.empty and len(shares_diluted.dropna()) > 1:
                        try:
                            share_change = shares_diluted.pct_change() * -100
                            if not share_change.dropna().empty:
                                st.markdown("### Share Count Change")
                                plot_line(share_change.dropna(), "Share Change (+ = Buyback, - = Dilution)", "% Change", "#e377c2")
                        except Exception as e:
                            st.warning(f"Could not calculate share change: {str(e)}")

            # TAB 5: RATIOS
            with tab5:
                st.markdown("### Return on Invested Capital (ROIC)")
                if not net_income.empty and not total_assets.empty:
                    try:
                        nopat = net_income + tax.abs()
                        inv_cap = total_assets - cash - total_debt
                        inv_cap_lag = inv_cap.shift(1)
                        roic = (nopat / inv_cap_lag) * 100
                        roic = roic.replace([np.inf, -np.inf], np.nan)
                        
                        if not roic.dropna().empty:
                            plot_line(roic.dropna(), "Return on Invested Capital", "ROIC (%)", "#7f7f7f")
                            latest = float(roic.dropna().iloc[-1]) if len(roic.dropna()) > 0 else 0
                            st.metric("Latest ROIC", f"{latest:.1f}%")
                            st.info("💡 Benchmark: ROIC > 15% good, > 20% excellent")
                    except Exception as e:
                        st.warning(f"Could not calculate ROIC: {str(e)}")
                
                st.markdown("### Return on Equity (ROE)")
                if not net_income.empty and not stockholders_equity.empty:
                    try:
                        equity_lag = stockholders_equity.shift(1)
                        roe = (net_income / equity_lag) * 100
                        roe = roe.replace([np.inf, -np.inf], np.nan)
                        
                        if not roe.dropna().empty:
                            plot_line(roe.dropna(), "Return on Equity", "ROE (%)", "#17becf")
                            latest = float(roe.dropna().iloc[-1]) if len(roe.dropna()) > 0 else 0
                            st.metric("Latest ROE", f"{latest:.1f}%")
                    except Exception as e:
                        st.warning(f"Could not calculate ROE: {str(e)}")
                
                st.markdown("### Debt-to-Equity Ratio")
                if not total_debt.empty and not stockholders_equity.empty:
                    try:
                        de = total_debt / stockholders_equity
                        de = de.replace([np.inf, -np.inf], np.nan)
                        
                        if not de.dropna().empty:
                            plot_line(de.dropna(), "Debt-to-Equity Ratio", "Ratio", "#d62728")
                            latest = float(de.dropna().iloc[-1]) if len(de.dropna()) > 0 else 0
                            st.metric("Latest D/E", f"{latest:.2f}x")
                            st.info("💡 Benchmark: < 1.0 conservative (varies by industry)")
                    except Exception as e:
                        st.warning(f"Could not calculate D/E ratio: {str(e)}")
                
                st.markdown("### Per-Share Metrics")
                if not net_income.empty and not shares_diluted.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        try:
                            eps = net_income / shares_diluted
                            if not eps.dropna().empty:
                                plot_line(eps.dropna(), "Earnings Per Share (EPS)", "EPS ($)", "#2ca02c", True)
                        except Exception as e:
                            st.warning(f"Could not calculate EPS: {str(e)}")
                    with col2:
                        if not ocf.empty:
                            try:
                                cfps = ocf / shares_diluted
                                if not cfps.dropna().empty:
                                    plot_line(cfps.dropna(), "Cash Flow Per Share", "CFPS ($)", "#1f77b4", True)
                            except Exception as e:
                                st.warning(f"Could not calculate CFPS: {str(e)}")

            # TAB 6: ANALYST FORECASTS
            with tab6:
                st.markdown("### 🔮 Analyst Forecasts & Predictions")
                
                has_analyst_data = False
                if analyst_data:
                    for key, value in analyst_data.items():
                        if value is not None:
                            if isinstance(value, pd.DataFrame):
                                if not value.empty:
                                    has_analyst_data = True
                                    break
                            elif isinstance(value, dict):
                                if value:
                                    has_analyst_data = True
                                    break
                            else:
                                has_analyst_data = True
                                break
                
                if has_analyst_data:
                    
                    # Price Targets
                    st.markdown("#### 🎯 Analyst Price Targets")
                    price_target = analyst_data.get('price_target')
                    
                    # Check if price_target is a valid dict (not a DataFrame)
                    is_valid_target = False
                    if price_target is not None:
                        if isinstance(price_target, dict) and not isinstance(price_target, pd.DataFrame):
                            is_valid_target = True
                    
                    if is_valid_target:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            current = price_target.get('current', 0)
                            st.metric("Current Price", f"${current:.2f}")
                        with col2:
                            mean_target = price_target.get('mean', 0)
                            st.metric("Mean Target", f"${mean_target:.2f}")
                            if current > 0:
                                upside = ((mean_target - current) / current) * 100
                                st.caption(f"Upside: {upside:.1f}%")
                        with col3:
                            low = price_target.get('low', 0)
                            st.metric("Low Target", f"${low:.2f}")
                        with col4:
                            high = price_target.get('high', 0)
                            st.metric("High Target", f"${high:.2f}")
                    
                    # Recommendations
                    st.markdown("#### 📊 Analyst Recommendations")
                    recommendations = analyst_data.get('recommendations')
                    if recommendations is not None and not recommendations.empty:
                        recent = recommendations.tail(20)
                        
                        # Count recommendation types
                        if 'To Grade' in recent.columns:
                            rec_counts = recent['To Grade'].value_counts()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                buy_count = rec_counts.get('Buy', 0) + rec_counts.get('Strong Buy', 0) + rec_counts.get('Outperform', 0)
                                st.metric("Buy Ratings", buy_count)
                            with col2:
                                hold_count = rec_counts.get('Hold', 0) + rec_counts.get('Neutral', 0)
                                st.metric("Hold Ratings", hold_count)
                            with col3:
                                sell_count = rec_counts.get('Sell', 0) + rec_counts.get('Strong Sell', 0) + rec_counts.get('Underperform', 0)
                                st.metric("Sell Ratings", sell_count)
                            
                            # Show recent upgrades/downgrades
                            st.markdown("##### Recent Analyst Actions")
                            st.dataframe(recent.tail(10), use_container_width=True)
                        else:
                            st.dataframe(recent.tail(10), use_container_width=True)
                    
                    # Earnings Trend
                    st.markdown("#### 📈 Earnings Estimates Trend")
                    earnings_trend = analyst_data.get('earnings_trend')
                    if earnings_trend is not None and not earnings_trend.empty:
                        st.dataframe(earnings_trend, use_container_width=True)
                        
                        # Try to visualize EPS estimates if available
                        if '7d' in earnings_trend.index or '30d' in earnings_trend.index:
                            st.info("📊 Earnings estimates are being revised. Check the trend table above for details.")
                    
                    # Revenue Forecasts
                    st.markdown("#### 💰 Revenue Forecasts")
                    revenue_forecast = analyst_data.get('revenue_forecasts')
                    if revenue_forecast is not None and not revenue_forecast.empty:
                        st.dataframe(revenue_forecast, use_container_width=True)
                    
                    # Upgrades/Downgrades Summary
                    st.markdown("#### ⬆️⬇️ Recent Upgrades & Downgrades")
                    upgrades = analyst_data.get('upgrades_downgrades')
                    if upgrades is not None and not upgrades.empty:
                        st.dataframe(upgrades.tail(15), use_container_width=True)
                    
                    # Add interpretation guide
                    st.markdown("---")
                    st.info("""
                    **💡 How to Interpret Analyst Forecasts:**
                    - **Price Targets**: Mean target shows average analyst expectation
                    - **Recommendations**: More Buy ratings = bullish sentiment
                    - **Upgrades/Downgrades**: Recent changes indicate shifting sentiment
                    - **Earnings Trend**: Upward revisions = improving outlook
                    - **Note**: Analysts can be wrong! Use as one data point among many.
                    """)
                    
                else:
                    st.warning("⚠️ No analyst forecast data available for this ticker")
                    st.info("Analyst data is typically available for larger, well-covered companies")

            # COMPLETION
            st.markdown("---")
            st.success(f"✅ **Analysis Complete!** {len(income.columns) if not income.empty else 0} periods analyzed")
            
            if use_edgar:
                st.info("💡 Data from SEC filings - most reliable source!")
            
            # Export
            st.markdown("### 📥 Export Data")
            if st.button("Generate CSV Export"):
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
                st.download_button("Download CSV", csv, f"{ticker}_data.csv", "text/csv")


if __name__ == "__main__":
    st.set_page_config(page_title="Ultimate Financial Grapher", page_icon="📈", layout="wide")
    
    st.markdown("""
        <style>
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
        </style>
    """, unsafe_allow_html=True)
    
    render_grapher()
