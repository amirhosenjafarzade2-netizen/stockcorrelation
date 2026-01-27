"""
Enhanced Financial Grapher with SEC Edgar Integration AND Analyst Predictions
Fetches 10+ years of financial data using edgartools (SEC filings)
Falls back to yfinance for non-US companies or when Edgar data unavailable
INCLUDES FULL ANALYST FORECASTS SECTION
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
from functools import lru_cache
import time

# Try to import edgartools (optional dependency)
try:
    from edgar import Company, set_identity
    from edgar.xbrl import XBRLS
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Set Edgar identity if available (required by SEC)
if EDGAR_AVAILABLE:
    set_identity("finance_research user@example.com")


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING - SEC EDGAR (10+ years)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)  # Cache for 2 hours
def fetch_edgar_data(ticker: str, num_years: int = 10):
    """
    Fetch financial statements from SEC Edgar filings
    Returns up to num_years of annual data
    """
    if not EDGAR_AVAILABLE:
        return None, None, None, "edgartools not installed"
    
    try:
        company = Company(ticker)
        
        # Get annual reports (10-K filings)
        filings = company.get_filings(form="10-K").head(num_years)
        
        if len(filings) == 0:
            return None, None, None, "No 10-K filings found"
        
        # Create multi-period financials
        xbrls = XBRLS.from_filings(filings)
        statements = xbrls.statements
        
        # Extract statements
        income_stmt = statements.income_statement()
        balance_sheet = statements.balance_sheet()
        cash_flow = statements.cashflow_statement()
        
        # Convert to pandas DataFrames with dates as columns
        income_df = income_stmt.to_dataframe() if income_stmt else pd.DataFrame()
        balance_df = balance_sheet.to_dataframe() if balance_sheet else pd.DataFrame()
        cashflow_df = cash_flow.to_dataframe() if cash_flow else pd.DataFrame()
        
        return income_df, balance_df, cashflow_df, None
        
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data(ttl=7200)
def fetch_edgar_quarterly_data(ticker: str, num_quarters: int = 20):
    """
    Fetch quarterly financial statements from SEC Edgar (10-Q filings)
    Returns up to num_quarters of quarterly data
    """
    if not EDGAR_AVAILABLE:
        return None, None, None, "edgartools not installed"
    
    try:
        company = Company(ticker)
        
        # Get quarterly reports (10-Q filings)
        filings = company.get_filings(form="10-Q").head(num_quarters)
        
        if len(filings) == 0:
            return None, None, None, "No 10-Q filings found"
        
        # Create multi-period financials
        xbrls = XBRLS.from_filings(filings)
        statements = xbrls.statements
        
        # Extract statements
        income_stmt = statements.income_statement()
        balance_sheet = statements.balance_sheet()
        cash_flow = statements.cashflow_statement()
        
        # Convert to DataFrames
        income_df = income_stmt.to_dataframe() if income_stmt else pd.DataFrame()
        balance_df = balance_sheet.to_dataframe() if balance_sheet else pd.DataFrame()
        cashflow_df = cash_flow.to_dataframe() if cash_flow else pd.DataFrame()
        
        return income_df, balance_df, cashflow_df, None
        
    except Exception as e:
        return None, None, None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING - YFINANCE (Fallback, 4 periods only)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
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
                return pd.Series(dtype=float)
            time.sleep(1)


@st.cache_data(ttl=3600)
def fetch_yfinance_fundamentals(ticker: str, frequency: str):
    """Fallback: Fetch fundamental data from yfinance (limited to 4 periods)"""
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
        
        return {
            'recommendations': ticker_obj.recommendations,
            'price_target': ticker_obj.analyst_price_targets if hasattr(ticker_obj, 'analyst_price_targets') else None,
            'earnings_forecasts': ticker_obj.earnings_forecasts if hasattr(ticker_obj, 'earnings_forecasts') else None,
            'revenue_forecasts': ticker_obj.revenue_estimate if hasattr(ticker_obj, 'revenue_estimate') else None,
            'earnings_trend': ticker_obj.earnings_trend if hasattr(ticker_obj, 'earnings_trend') else None,
            'upgrades_downgrades': ticker_obj.upgrades_downgrades if hasattr(ticker_obj, 'upgrades_downgrades') else None
        }
    except Exception as e:
        return {}


def process_edgar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Edgar DataFrame to have dates as columns (transpose if needed)
    """
    if df.empty:
        return df
    
    # Edgar dataframes often have dates as index, we want them as columns
    # and metrics as index (like yfinance format)
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to transpose
        df = df.T
    
    # Ensure columns are datetime
    if not isinstance(df.columns, pd.DatetimeIndex):
        try:
            df.columns = pd.to_datetime(df.columns)
        except:
            pass
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """Get row from DataFrame with intelligent fallback"""
    if df.empty:
        return pd.Series(dtype=float)
    
    # Try exact match
    if key in df.index:
        return df.loc[key]
    
    # Try case-insensitive partial match
    for idx in df.index:
        if isinstance(idx, str) and key.lower() in idx.lower():
            return df.loc[idx]
    
    # Comprehensive alternatives
    alternatives = {
        "Total Revenue": ["Revenue", "TotalRevenue", "Revenues", "Sales", "Net Sales"],
        "Gross Profit": ["GrossProfit", "Gross Income"],
        "Operating Income": ["OperatingIncome", "Operating Income Loss", "EBIT"],
        "Net Income": ["NetIncome", "Net Income Loss", "Net Income Available To Common Stockholders"],
        "Operating Cash Flow": ["OperatingCashFlow", "Net Cash From Operating Activities", "Cash From Operating Activities"],
        "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures", "Payments To Acquire PPE"],
        "Stock Based Compensation": ["StockBasedCompensation", "Share Based Compensation"],
        "Basic Average Shares": ["BasicAverageShares", "Weighted Average Shares Outstanding Basic"],
        "Diluted Average Shares": ["DilutedAverageShares", "Weighted Average Shares Outstanding Diluted"],
        "Total Assets": ["TotalAssets", "Assets"],
        "Total Debt": ["TotalDebt", "Long Term Debt", "Debt"],
        "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash", "Cash Equivalents"],
        "Tax Provision": ["TaxProvision", "Income Tax Expense"],
        "Research Development": ["ResearchAndDevelopment", "R&D"],
        "Total Liabilities": ["TotalLiabilities", "Liabilities"],
        "Stockholders Equity": ["StockholdersEquity", "Stockholders Equity", "Total Equity"],
    }
    
    for alt in alternatives.get(key, []):
        if alt in df.index:
            return df.loc[alt]
        # Try case-insensitive
        for idx in df.index:
            if isinstance(idx, str) and alt.lower() in idx.lower():
                return df.loc[idx]
    
    if default is not None:
        return pd.Series(default, index=df.columns if not df.empty else [])
    return pd.Series(dtype=float)


def format_large_number(num: float) -> str:
    """Format large numbers"""
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
    
    start_val = clean.iloc[0]
    end_val = clean.iloc[-1]
    
    if start_val <= 0 or end_val <= 0:
        return np.nan
    
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
    """Enhanced line plot"""
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if data.empty or data.isna().all():
        st.info(f"📊 {title}: No data available")
        return
    
    fig = px.line(x=data.index, y=data, title=title, markers=True,
                  color_discrete_sequence=[color] if color else None)
    
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


def plot_bar(data: pd.Series, title: str, yaxis: str = "Value", color: str = None) -> None:
    """Bar plot"""
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if data.empty or data.isna().all():
        st.info(f"📊 {title}: No data available")
        return
    
    fig = px.bar(x=data.index, y=data, title=title,
                color_discrete_sequence=[color] if color else None)
    
    cagr = calculate_cagr(data)
    if not np.isnan(cagr):
        fig.update_layout(title=f"{title}<br><sup>CAGR: {cagr:.1f}%</sup>")
    
    fig.update_layout(yaxis_title=yaxis, height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "Value", colors: list = None) -> None:
    """Multi-line plot"""
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_grapher() -> None:
    st.title("📈 Enhanced Financial Grapher with Analyst Forecasts")
    
    if EDGAR_AVAILABLE:
        st.success("✅ **SEC Edgar Integration Active** - Access to 10+ years of financial data!")
    else:
        st.warning("⚠️ **Using Yahoo Finance** - Limited to 4 periods. Install `edgartools` for more data: `pip install edgartools`")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    with col2:
        frequency = st.selectbox("Frequency", ["Annual", "Quarterly"], index=0)

    with col3:
        if EDGAR_AVAILABLE and frequency == "Annual":
            num_periods = st.slider("Years of Data", 1, 15, 10)
        elif EDGAR_AVAILABLE and frequency == "Quarterly":
            num_periods = st.slider("Quarters of Data", 4, 40, 20)
        else:
            num_periods = st.slider("Periods", 1, 10, 4)
            st.caption("yfinance limited to 4 periods")
    
    with col4:
        price_years = st.slider("Price History (Years)", 1, 30, 10)

    if not ticker:
        st.info("👆 Enter a ticker symbol")
        return

    # Date range for price data
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * price_years)

    if st.button("🚀 Load & Analyze", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            
            # Try Edgar first for US companies
            use_edgar = False
            income, balance, cashflow = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            if EDGAR_AVAILABLE:
                st.info("🔍 Attempting to fetch from SEC Edgar (10+ years available)...")
                
                if frequency == "Annual":
                    income, balance, cashflow, error = fetch_edgar_data(ticker, num_periods)
                else:
                    income, balance, cashflow, error = fetch_edgar_quarterly_data(ticker, num_periods)
                
                if error:
                    st.warning(f"Edgar fetch failed: {error}. Falling back to Yahoo Finance...")
                else:
                    use_edgar = True
                    # Process dataframes
                    income = process_edgar_dataframe(income)
                    balance = process_edgar_dataframe(balance)
                    cashflow = process_edgar_dataframe(cashflow)
                    
                    st.success(f"✅ **Edgar Data**: {len(income.columns) if not income.empty else 0} periods retrieved!")
            
            # Fallback to yfinance if Edgar failed or unavailable
            if not use_edgar:
                st.info("📊 Fetching from Yahoo Finance (limited to 4 periods)...")
                income, balance, cashflow, info = fetch_yfinance_fundamentals(ticker, frequency)
            else:
                # Still get info from yfinance
                try:
                    info = yf.Ticker(ticker).info
                except:
                    info = {}
            
            # Fetch price data
            prices = fetch_price_data(ticker, start_date, end_date)
            
            # Fetch analyst data
            st.info("📊 Fetching analyst forecasts...")
            analyst_data = fetch_analyst_data(ticker)

            # Check if we have any data
            if prices.empty and income.empty:
                st.error("❌ No data available for this ticker")
                return

            # ── Company Info ──────────────────────────────────────────────────
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

            # Show data source
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not prices.empty:
                    st.success(f"✅ **Price**: {len(prices)} days")
            
            with col2:
                if not income.empty:
                    periods = len(income.columns)
                    source = "SEC Edgar" if use_edgar else "Yahoo Finance"
                    st.success(f"✅ **Fundamentals ({source})**: {periods} periods")
            
            with col3:
                if analyst_data:
                    st.success("✅ **Analyst Data**: Available")

            # Extract metrics
            revenue = safe_get(income, "Total Revenue")
            gross_profit = safe_get(income, "Gross Profit")
            operating_income = safe_get(income, "Operating Income")
            net_income = safe_get(income, "Net Income")
            
            ocf = safe_get(cashflow, "Operating Cash Flow")
            capex = safe_get(cashflow, "Capital Expenditure", 0)
            sbc = safe_get(cashflow, "Stock Based Compensation")
            
            total_assets = safe_get(balance, "Total Assets")
            total_debt = safe_get(balance, "Total Debt", 0)
            cash = safe_get(balance, "Cash And Cash Equivalents", 0)
            stockholders_equity = safe_get(balance, "Stockholders Equity")

            # ══════════════════════════════════════════════════════════════════
            # TABBED INTERFACE
            # ══════════════════════════════════════════════════════════════════
            
            st.markdown("---")
            st.markdown("## 📊 Financial Analysis")
            
            tab1, tab2, tab3 = st.tabs(["📊 Fundamentals", "📈 Charts", "🔮 Analyst Forecasts"])
            
            with tab1:
                # Price Chart
                if not prices.empty:
                    st.markdown("### 💹 Stock Price")
                    plot_line(prices, f"{ticker} Price History", "Price (USD)", "#1f77b4")

                # Revenue
                if not revenue.empty:
                    st.markdown("### 💰 Revenue")
                    plot_bar(revenue, "Total Revenue", "Revenue (USD)", "#2ca02c")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        latest = float(revenue.iloc[-1]) if not pd.isna(revenue.iloc[-1]) else 0
                        st.metric("Latest Revenue", format_large_number(latest))
                    with col2:
                        cagr = calculate_cagr(revenue)
                        if not np.isnan(cagr):
                            st.metric("Revenue CAGR", f"{cagr:.1f}%")

            with tab2:
                # Margins
                if not revenue.empty and not net_income.empty:
                    st.markdown("### 📈 Profit Margins")
                    margins = pd.DataFrame({
                        "Gross Margin": (gross_profit / revenue) * 100 if not gross_profit.empty else pd.Series(),
                        "Operating Margin": (operating_income / revenue) * 100 if not operating_income.empty else pd.Series(),
                        "Net Margin": (net_income / revenue) * 100
                    })
                    plot_multi(margins, "Margins Over Time", "Margin (%)", 
                              ["#ff7f0e", "#d62728", "#9467bd"])

                # Profitability
                if not operating_income.empty or not net_income.empty:
                    st.markdown("### 💵 Profitability")
                    profit_df = pd.DataFrame({
                        "Operating Income": operating_income,
                        "Net Income": net_income
                    })
                    plot_multi(profit_df, "Operating vs Net Income", "Income (USD)", 
                              ["#17becf", "#bcbd22"])

                # Cash Flow
                if not ocf.empty:
                    st.markdown("### 💸 Cash Flow")
                    
                    fcf = ocf + capex
                    if fcf.abs().sum() > 0:
                        plot_line(fcf, "Free Cash Flow", "FCF (USD)", "#2ca02c", show_growth=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            latest_fcf = float(fcf.iloc[-1]) if not pd.isna(fcf.iloc[-1]) else 0
                            st.metric("Latest FCF", format_large_number(latest_fcf))
                        with col2:
                            fcf_cagr = calculate_cagr(fcf)
                            if not np.isnan(fcf_cagr):
                                st.metric("FCF CAGR", f"{fcf_cagr:.1f}%")

                # Balance Sheet
                if not total_assets.empty:
                    st.markdown("### 🏦 Balance Sheet")
                    debt_df = pd.DataFrame({
                        "Total Assets": total_assets,
                        "Total Debt": total_debt,
                        "Cash": cash
                    })
                    plot_multi(debt_df, "Assets, Debt & Cash", "Amount (USD)", 
                              ["#2ca02c", "#d62728", "#1f77b4"])
            
            with tab3:
                st.markdown("### 🔮 Analyst Forecasts & Predictions")
                
                has_data = any(
                    isinstance(v, pd.DataFrame) and not v.empty or 
                    isinstance(v, dict) and v 
                    for v in analyst_data.values() if v is not None
                )
                
                if has_data:
                    # Price Targets
                    price_target = analyst_data.get('price_target')
                    if isinstance(price_target, dict) and not isinstance(price_target, pd.DataFrame):
                        st.markdown("#### 🎯 Analyst Price Targets")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            current = price_target.get('current', 0)
                            st.metric("Current Price", f"${current:.2f}")
                        with col2:
                            mean = price_target.get('mean', 0)
                            st.metric("Mean Target", f"${mean:.2f}")
                            if current > 0:
                                upside = ((mean - current) / current) * 100
                                st.caption(f"Upside: {upside:.1f}%")
                        with col3:
                            st.metric("Low Target", f"${price_target.get('low', 0):.2f}")
                        with col4:
                            st.metric("High Target", f"${price_target.get('high', 0):.2f}")
                    
                    # Recommendations
                    recommendations = analyst_data.get('recommendations')
                    if recommendations is not None and not recommendations.empty:
                        st.markdown("#### 📊 Analyst Recommendations")
                        st.dataframe(recommendations.tail(10), use_container_width=True)
                    
                    # Earnings Trend
                    earnings_trend = analyst_data.get('earnings_trend')
                    if earnings_trend is not None and not earnings_trend.empty:
                        st.markdown("#### 📈 Earnings Estimates Trend")
                        st.dataframe(earnings_trend, use_container_width=True)
                    
                    # Revenue Forecasts
                    revenue_forecast = analyst_data.get('revenue_forecasts')
                    if revenue_forecast is not None and not revenue_forecast.empty:
                        st.markdown("#### 💰 Revenue Forecasts")
                        st.dataframe(revenue_forecast, use_container_width=True)
                    
                    # Earnings Forecasts
                    earnings_forecasts = analyst_data.get('earnings_forecasts')
                    if earnings_forecasts is not None and not earnings_forecasts.empty:
                        st.markdown("#### 📊 Earnings Forecasts")
                        st.dataframe(earnings_forecasts, use_container_width=True)
                    
                    # Upgrades/Downgrades
                    upgrades = analyst_data.get('upgrades_downgrades')
                    if upgrades is not None and not upgrades.empty:
                        st.markdown("#### ⬆️⬇️ Recent Upgrades & Downgrades")
                        st.dataframe(upgrades.tail(15), use_container_width=True)
                    
                    # Interpretation guide
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

            st.markdown("---")
            st.success(f"✅ **Analysis Complete!** Displayed {len(income.columns) if not income.empty else 0} periods of data")
            
            if use_edgar:
                st.info("💡 **Tip**: This data comes directly from SEC filings - the most reliable source available!")
            else:
                st.info("💡 **Tip**: Install `edgartools` to access 10+ years of historical data: `pip install edgartools`")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.set_page_config(
        page_title="Enhanced Financial Grapher",
        page_icon="📈",
        layout="wide"
    )
    
    render_grapher()
