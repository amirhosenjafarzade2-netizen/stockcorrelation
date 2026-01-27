"""
Financial Grapher Pro - Clean Working Version
Uses Yahoo Finance with full analyst predictions
No SEC Edgar complexity - just reliable, working code
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
    """Fetch price data with retry logic"""
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
    """Fetch fundamental data from yfinance"""
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """Get row from DataFrame with intelligent fallback"""
    if df.empty:
        return pd.Series(dtype=float)
    
    if key in df.index:
        return df.loc[key]
    
    # Case-insensitive partial match
    for idx in df.index:
        if isinstance(idx, str) and key.lower() in idx.lower():
            return df.loc[idx]
    
    # Alternatives
    alternatives = {
        "Total Revenue": ["TotalRevenue", "Total Revenues", "Revenue"],
        "Gross Profit": ["GrossProfit", "Gross Income"],
        "Operating Income": ["OperatingIncome", "EBIT", "Operating Revenue"],
        "Net Income": ["NetIncome", "Net Income Common Stockholders"],
        "Operating Cash Flow": ["OperatingCashFlow", "Total Cash From Operating Activities"],
        "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures", "Purchase Of PPE"],
        "Stock Based Compensation": ["StockBasedCompensation", "Stock Based Compensation"],
        "Basic Average Shares": ["BasicAverageShares", "Ordinary Shares Number"],
        "Diluted Average Shares": ["DilutedAverageShares"],
        "Total Assets": ["TotalAssets", "Total Assets"],
        "Total Debt": ["TotalDebt", "Long Term Debt"],
        "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments"],
        "Tax Provision": ["TaxProvision", "Tax Effect Of Unusual Items"],
        "Research Development": ["ResearchAndDevelopment", "Research Development"],
        "Selling General Administrative": ["SellingGeneralAndAdministrative", "Selling General Administrative"],
        "Total Liabilities": ["TotalLiabilities", "Total Liabilities Net Minority Interest"],
        "Stockholders Equity": ["StockholdersEquity", "Total Equity Gross Minority Interest"],
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
    start_val, end_val = clean.iloc[0], clean.iloc[-1]
    if start_val <= 0 or end_val <= 0:
        return np.nan
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (pow(end_val / start_val, 1 / years) - 1) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
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
    st.title("📈 Financial Grapher Pro")
    st.caption("Professional financial analysis with comprehensive metrics")

    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    with col2:
        frequency = st.selectbox("Frequency", ["Quarterly", "Annual"], index=0)

    with col3:
        price_years = st.slider("Price History (Years)", 1, 30, 10)

    if not ticker:
        st.info("👆 Enter a ticker symbol")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * price_years)

    if st.button("🚀 Load & Analyze", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            
            income, balance, cashflow, info = fetch_yfinance_fundamentals(ticker, frequency)
            prices = fetch_price_data(ticker, start_date, end_date)
            analyst_data = fetch_analyst_data(ticker)

            if prices.empty and income.empty:
                st.error(f"❌ No data available for {ticker}")
                return

            # Company Info
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

            # Data availability
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if not prices.empty:
                    st.success(f"✅ **Price**: {len(prices)} days")
            with col2:
                if not income.empty:
                    st.success(f"✅ **Fundamentals**: {len(income.columns)} periods")
            with col3:
                if analyst_data:
                    st.success("✅ **Analyst Data**: Available")

            # Extract metrics
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

            # Tabs
            st.markdown("---")
            st.markdown("## 📊 Comprehensive Analysis")
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Price & Revenue", "💰 Profitability", "💵 Cash Flow",
                "📊 Balance Sheet", "📐 Ratios", "🔮 Analyst Forecasts"
            ])

            with tab1:
                st.markdown("### Stock Price History")
                if not prices.empty:
                    plot_line(prices, f"{ticker} Price", "Price (USD)", "#1f77b4")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current", f"${float(prices.iloc[-1]):.2f}")
                    with col2:
                        st.metric("High", f"${float(prices.max()):.2f}")
                    with col3:
                        st.metric("Low", f"${float(prices.min()):.2f}")
                    with col4:
                        ret = ((float(prices.iloc[-1]) / float(prices.iloc[0])) - 1) * 100
                        st.metric("Return", f"{ret:.1f}%")
                
                st.markdown("### Revenue")
                if not revenue.empty:
                    plot_bar(revenue, "Total Revenue", "Revenue (USD)", "#2ca02c")

            with tab2:
                st.markdown("### Margins")
                if not revenue.empty and not net_income.empty:
                    margins = pd.DataFrame({
                        "Gross Margin": (gross_profit / revenue) * 100 if not gross_profit.empty else pd.Series(),
                        "Operating Margin": (operating_income / revenue) * 100 if not operating_income.empty else pd.Series(),
                        "Net Margin": (net_income / revenue) * 100
                    })
                    plot_multi(margins, "Margins", "Margin (%)", ["#ff7f0e", "#d62728", "#9467bd"])
                
                st.markdown("### Profitability")
                if not operating_income.empty or not net_income.empty:
                    profit_df = pd.DataFrame({"Operating Income": operating_income, "Net Income": net_income})
                    plot_multi(profit_df, "Income", "USD", ["#17becf", "#bcbd22"])

            with tab3:
                st.markdown("### Operating Cash Flow")
                if not ocf.empty:
                    plot_line(ocf, "OCF", "USD", "#1f77b4", True)
                
                st.markdown("### Free Cash Flow")
                if not ocf.empty:
                    fcf = ocf + capex
                    if fcf.abs().sum() > 0:
                        plot_line(fcf, "FCF", "USD", "#2ca02c", True)

            with tab4:
                st.markdown("### Balance Sheet")
                if not total_assets.empty:
                    bal_df = pd.DataFrame({
                        "Assets": total_assets,
                        "Liabilities": total_liabilities,
                        "Equity": stockholders_equity if not stockholders_equity.empty else total_assets - total_liabilities
                    })
                    plot_multi(bal_df, "Balance Sheet", "USD", ["#2ca02c", "#d62728", "#1f77b4"])

            with tab5:
                st.markdown("### Return on Equity")
                if not net_income.empty and not stockholders_equity.empty:
                    roe = (net_income / stockholders_equity.shift(1)) * 100
                    roe = roe.replace([np.inf, -np.inf], np.nan)
                    if not roe.dropna().empty:
                        plot_line(roe.dropna(), "ROE", "ROE (%)", "#17becf")

            with tab6:
                st.markdown("### 🔮 Analyst Forecasts")
                
                has_data = any(
                    isinstance(v, pd.DataFrame) and not v.empty or 
                    isinstance(v, dict) and v 
                    for v in analyst_data.values() if v is not None
                )
                
                if has_data:
                    price_target = analyst_data.get('price_target')
                    if isinstance(price_target, dict) and not isinstance(price_target, pd.DataFrame):
                        st.markdown("#### 🎯 Price Targets")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current", f"${price_target.get('current', 0):.2f}")
                        with col2:
                            mean = price_target.get('mean', 0)
                            st.metric("Mean Target", f"${mean:.2f}")
                        with col3:
                            st.metric("Low", f"${price_target.get('low', 0):.2f}")
                        with col4:
                            st.metric("High", f"${price_target.get('high', 0):.2f}")
                    
                    recommendations = analyst_data.get('recommendations')
                    if recommendations is not None and not recommendations.empty:
                        st.markdown("#### 📊 Recommendations")
                        st.dataframe(recommendations.tail(10), use_container_width=True)
                    
                    earnings_trend = analyst_data.get('earnings_trend')
                    if earnings_trend is not None and not earnings_trend.empty:
                        st.markdown("#### 📈 Earnings Trend")
                        st.dataframe(earnings_trend, use_container_width=True)
                    
                    revenue_forecast = analyst_data.get('revenue_forecasts')
                    if revenue_forecast is not None and not revenue_forecast.empty:
                        st.markdown("#### 💰 Revenue Forecasts")
                        st.dataframe(revenue_forecast, use_container_width=True)
                    
                    upgrades = analyst_data.get('upgrades_downgrades')
                    if upgrades is not None and not upgrades.empty:
                        st.markdown("#### ⬆️⬇️ Upgrades/Downgrades")
                        st.dataframe(upgrades.tail(15), use_container_width=True)
                else:
                    st.warning("⚠️ No analyst data available")

            st.markdown("---")
            st.success(f"✅ Analysis Complete! {len(income.columns) if not income.empty else 0} periods")


if __name__ == "__main__":
    st.set_page_config(page_title="Financial Grapher Pro", page_icon="📈", layout="wide")
    st.markdown("""
        <style>
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)
    render_grapher()
