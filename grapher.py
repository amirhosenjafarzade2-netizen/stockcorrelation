"""
Ultimate Financial Grapher - WORKING VERSION
Combines working data extraction with comprehensive features
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
    """Fetch from yfinance"""
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS - USING THE WORKING VERSION'S APPROACH
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """
    Get row from DataFrame with intelligent fallback to alternative field names.
    This is the WORKING version's implementation.
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    if key in df.index:
        return df.loc[key]
    
    # Comprehensive alternatives mapping - EXACTLY as in working version
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
    st.caption("Professional-grade financial analysis with comprehensive metrics")

    # INPUTS
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    with col2:
        frequency = st.selectbox("Frequency", ["Annual", "Quarterly"], index=0)

    with col3:
        price_years = st.slider("Price History (Years)", 1, 30, 10)

    if not ticker:
        st.info("👆 Enter a ticker")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * price_years)

    if st.button("🚀 Load & Analyze", type="primary", use_container_width=True):
        with st.spinner(f"Fetching comprehensive data for {ticker}..."):
            
            # Fetch data
            st.info("📊 Using Yahoo Finance...")
            income, balance, cashflow, info = fetch_yfinance_fundamentals(ticker, frequency)
            prices = fetch_price_data(ticker, start_date, end_date)

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
                    st.success(f"✅ **Fundamentals**: {len(income.columns)} periods")

            # EXTRACT METRICS - using working version's approach
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
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Price & Revenue", 
                "💰 Profitability", 
                "💵 Cash Flow",
                "📊 Balance Sheet",
                "📐 Ratios & Metrics"
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
                if not revenue.empty and not gross_profit.empty and not operating_income.empty and not net_income.empty:
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
                
                st.markdown("### Absolute Profitability")
                if not operating_income.empty or not net_income.empty:
                    profit_df = pd.DataFrame({"Operating Income": operating_income, "Net Income": net_income})
                    plot_multi(profit_df, "Operating vs Net Income", "Income (USD)", ["#17becf", "#bcbd22"])
                
                st.markdown("### Operating Expenses")
                if not rd_expense.empty or not sga_expense.empty:
                    opex_df = pd.DataFrame({"R&D": rd_expense, "SG&A": sga_expense})
                    plot_multi(opex_df, "R&D and SG&A Expenses", "Expense (USD)", ["#e377c2", "#7f7f7f"])

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
                    eq_df = pd.DataFrame({"Operating Cash Flow": ocf, "Net Income": net_income})
                    plot_multi(eq_df, "Cash Flow vs Earnings", "Amount (USD)", ["#1f77b4", "#ff7f0e"])
                    st.info("💡 High-quality earnings: OCF ≥ Net Income")
                
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
                    bal_df = pd.DataFrame({
                        "Total Assets": total_assets,
                        "Total Liabilities": total_liabilities,
                        "Equity": stockholders_equity if not stockholders_equity.empty else total_assets - total_liabilities
                    })
                    plot_multi(bal_df, "Balance Sheet Components", "Amount (USD)", ["#2ca02c", "#d62728", "#1f77b4"])
                
                st.markdown("### Debt & Cash Position")
                if not total_debt.empty or not cash.empty:
                    debt_df = pd.DataFrame({
                        "Total Debt": total_debt,
                        "Cash": cash,
                        "Net Debt": total_debt - cash
                    })
                    plot_multi(debt_df, "Debt vs Cash", "Amount (USD)", ["#d62728", "#2ca02c", "#ff7f0e"])
                    
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
                    shares_df = pd.DataFrame({"Basic Shares": shares_basic, "Diluted Shares": shares_diluted})
                    plot_multi(shares_df, "Outstanding Shares", "Shares", ["#1f77b4", "#ff7f0e"])
                    
                    if not shares_diluted.empty and len(shares_diluted.dropna()) > 1:
                        share_change = shares_diluted.pct_change() * -100
                        if not share_change.dropna().empty:
                            st.markdown("### Share Count Change")
                            plot_line(share_change.dropna(), "Share Change (+ = Buyback, - = Dilution)", "% Change", "#e377c2")

            # TAB 5: RATIOS
            with tab5:
                st.markdown("### Return on Invested Capital (ROIC)")
                if not net_income.empty and not total_assets.empty:
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
                
                st.markdown("### Return on Equity (ROE)")
                if not net_income.empty and not stockholders_equity.empty:
                    equity_lag = stockholders_equity.shift(1)
                    roe = (net_income / equity_lag) * 100
                    roe = roe.replace([np.inf, -np.inf], np.nan)
                    
                    if not roe.dropna().empty:
                        plot_line(roe.dropna(), "Return on Equity", "ROE (%)", "#17becf")
                        latest = float(roe.dropna().iloc[-1]) if len(roe.dropna()) > 0 else 0
                        st.metric("Latest ROE", f"{latest:.1f}%")
                
                st.markdown("### Debt-to-Equity Ratio")
                if not total_debt.empty and not stockholders_equity.empty:
                    de = total_debt / stockholders_equity
                    de = de.replace([np.inf, -np.inf], np.nan)
                    
                    if not de.dropna().empty:
                        plot_line(de.dropna(), "Debt-to-Equity Ratio", "Ratio", "#d62728")
                        latest = float(de.dropna().iloc[-1]) if len(de.dropna()) > 0 else 0
                        st.metric("Latest D/E", f"{latest:.2f}x")
                        st.info("💡 Benchmark: < 1.0 conservative (varies by industry)")
                
                st.markdown("### Per-Share Metrics")
                if not net_income.empty and not shares_diluted.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        eps = net_income / shares_diluted
                        if not eps.dropna().empty:
                            plot_line(eps.dropna(), "Earnings Per Share (EPS)", "EPS ($)", "#2ca02c", True)
                    with col2:
                        if not ocf.empty:
                            cfps = ocf / shares_diluted
                            if not cfps.dropna().empty:
                                plot_line(cfps.dropna(), "Cash Flow Per Share", "CFPS ($)", "#1f77b4", True)

            # COMPLETION
            st.markdown("---")
            st.success(f"✅ **Analysis Complete!** {len(income.columns) if not income.empty else 0} periods analyzed")
            
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
