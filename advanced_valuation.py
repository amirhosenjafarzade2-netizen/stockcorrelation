# advanced_valuation.py
# Comprehensive stock valuation module with multiple models, Monte Carlo, sensitivity, and S&P 500 undervalued screener
# Data from Yahoo Finance + Finviz fallback

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# ── Constants & Helpers ─────────────────────────────────────────────────────

DEFAULT_MOS = 25.0
TRADING_DAYS_PER_YEAR = 252

# ── Finviz Data Scraper (fallback / enhancement) ────────────────────────────

@st.cache_data(ttl=3600)
def get_finviz_data(ticker: str) -> dict:
    """Scrape key fundamentals from Finviz as fallback or supplement to yfinance."""
    data = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'snapshot-table2'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                for i in range(0, len(cells), 2):
                    if i + 1 < len(cells):
                        key = cells[i].text.strip()
                        value = cells[i+1].text.strip()
                        # Clean and convert numeric values
                        try:
                            clean_val = value.replace('%', '').replace(',', '').replace('$', '')
                            data[key] = float(clean_val) if clean_val.replace('.', '', 1).isdigit() else value
                        except:
                            data[key] = value
        return data
    except Exception as e:
        st.warning(f"Finviz scrape failed for {ticker}: {str(e)}")
        return {}

# ── Data Fetching (Yahoo + Finviz hybrid) ───────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str) -> dict:
    """Fetch stock fundamentals — primarily from yfinance, enhanced/fallback with Finviz."""
    try:
        ticker_clean = ticker.replace('.', '-')
        stock = yf.Ticker(ticker_clean)
        info = stock.info or {}

        # Core data from yfinance
        data = {
            'ticker': ticker_clean,
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 100.0)),
            'current_eps': info.get('trailingEps', 5.0),
            'forward_eps': info.get('forwardEps', 5.5),
            'dividend_per_share': info.get('dividendRate', info.get('trailingAnnualDividendRate', 1.0)),
            'beta': info.get('beta', 1.0),
            'book_value': info.get('bookValue', 20.0),
            'roe': info.get('returnOnEquity', 0.15) * 100 if info.get('returnOnEquity') else 15.0,
            'analyst_growth': info.get('earningsGrowth', 0.10) * 100,
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0,
            'desired_return': 10.0,
            'years_high_growth': 5,
            'core_mos': DEFAULT_MOS,
            'dividend_mos': DEFAULT_MOS,
            'dcf_mos': DEFAULT_MOS,
            'ri_mos': DEFAULT_MOS,
            'fcf': info.get('freeCashflow', 0.0),
            'dividend_growth': 5.0,
            'monte_carlo_runs': 1000,
            'growth_adj': 10.0,
            'wacc_adj': 10.0,
            'market_cap': info.get('marketCap', 0.0),
            'historical_pe': info.get('trailingPE', 15.0),
            'exit_pe': info.get('forwardPE', 15.0) or 15.0,
            # Extra for new models
            'ebitda': info.get('ebitda', 0.0),
            'net_debt': info.get('netDebt', 0.0),
            'nopat': info.get('operatingCashflow', 0.0) * (1 - 0.25),  # rough
            'invested_capital': info.get('totalAssets', 0.0) - info.get('totalCurrentLiabilities', 0.0),
            'shares_outstanding': info.get('sharesOutstanding', 1e9),
            'total_assets': info.get('totalAssets', 0.0),
            'debt': info.get('totalDebt', 0.0),
        }

        # Clamp / sanitize
        data['current_price'] = max(0.01, min(data['current_price'], 100000))
        data['current_eps'] = max(-1000, min(data['current_eps'], 1000))
        data['forward_eps'] = max(0.01, min(data['forward_eps'], 1000))
        data['beta'] = max(0.0, min(data['beta'], 10.0))
        data['roe'] = max(-200, min(data['roe'], 200))
        data['analyst_growth'] = max(0.0, min(data['analyst_growth'], 100.0))

        # Enhance / fallback with Finviz
        finviz = get_finviz_data(ticker_clean)
        if finviz:
            data.update({
                'historical_pe': finviz.get('P/E', data['historical_pe']),
                'current_eps': finviz.get('EPS (ttm)', data['current_eps']),
                'forward_eps': finviz.get('EPS next Y', data['forward_eps']),
                'beta': finviz.get('Beta', data['beta']),
                'roe': finviz.get('ROE', data['roe']),
                'market_cap': finviz.get('Market Cap', data['market_cap']) * 1e6 if isinstance(finviz.get('Market Cap'), (int, float)) else data['market_cap'],
            })

        return data

    except Exception as e:
        st.error(f"Data fetch failed for {ticker}: {str(e)}. Using defaults.")
        return {
            'ticker': ticker,
            'current_price': 100.0,
            'current_eps': 5.0,
            'forward_eps': 5.5,
            'dividend_per_share': 1.0,
            'beta': 1.0,
            'book_value': 20.0,
            'roe': 15.0,
            'analyst_growth': 10.0,
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0,
            'desired_return': 10.0,
            'years_high_growth': 5,
            'core_mos': DEFAULT_MOS,
            'dividend_mos': DEFAULT_MOS,
            'dcf_mos': DEFAULT_MOS,
            'ri_mos': DEFAULT_MOS,
            'fcf': 0.0,
            'dividend_growth': 5.0,
            'historical_pe': 15.0,
            'exit_pe': 15.0,
            'market_cap': 0.0,
        }

# ── Valuation Models ────────────────────────────────────────────────────────

def core_valuation(inputs): ...
# (keep all your original functions: core, lynch, ddm, two_stage_dcf, residual_income, reverse_dcf, graham_intrinsic_value)

# ── Newly Added Valuation Methods ───────────────────────────────────────────

def comparable_company_analysis(inputs):
    """Comparable Companies (P/E multiple of peers)"""
    try:
        stock = yf.Ticker(inputs['ticker'])
        peers = stock.major_holders.get('peers', [])[:5] or []
        if not peers:
            return {'intrinsic_value': inputs['current_eps'] * inputs['historical_pe']}
        avg_pe = 0
        count = 0
        for p in peers:
            try:
                pe = yf.Ticker(p).info.get('trailingPE', 0)
                if pe > 0:
                    avg_pe += pe
                    count += 1
            except:
                continue
        avg_pe = avg_pe / count if count > 0 else inputs['historical_pe']
        return {'intrinsic_value': inputs['current_eps'] * avg_pe}
    except:
        return {'intrinsic_value': inputs['current_eps'] * inputs['historical_pe']}

def precedent_transactions(inputs):
    """Approximated Precedent Transactions (Comps + control premium)"""
    comps_val = comparable_company_analysis(inputs)['intrinsic_value']
    premium = 25.0  # typical M&A control premium
    return {'intrinsic_value': comps_val * (1 + premium / 100)}

def asset_based_valuation(inputs):
    """Net Asset Value / Liquidation Value approximation"""
    book_value = inputs['book_value']
    # Conservative haircut (e.g. 70% of book for liquidation)
    nav = book_value * 0.7
    return {'intrinsic_value': max(nav, 0)}

def sum_of_the_parts(inputs):
    """Simplified SOTP — assumes two business segments"""
    segment1 = two_stage_dcf(inputs)['intrinsic_value'] * 0.6
    segment2 = two_stage_dcf(inputs)['intrinsic_value'] * 0.4
    net_debt = inputs.get('net_debt', 0)
    return {'intrinsic_value': segment1 + segment2 - net_debt}

def eva_based_valuation(inputs):
    """Economic Value Added (perpetual growth approximation)"""
    eva = inputs['nopat'] - (inputs['invested_capital'] * inputs['wacc'] / 100)
    if eva > 0 and inputs['wacc'] > 0:
        mva = eva / (inputs['wacc'] / 100)
        return {'intrinsic_value': inputs['book_value'] + mva}
    return {'intrinsic_value': inputs['book_value']}

def adjusted_present_value(inputs):
    """APV = Unlevered DCF + PV of Tax Shield"""
    unlevered = two_stage_dcf(inputs)['intrinsic_value']
    tax_shield = inputs['tax_rate']/100 * inputs.get('debt', 0) * 0.05  # assume 5% cost of debt
    pv_shield = tax_shield / 0.05 if 0.05 > 0 else 0
    return {'intrinsic_value': unlevered + pv_shield}

def owner_earnings(inputs):
    """Buffett-style Owner Earnings discounted (10y + terminal)"""
    oe = inputs['fcf'] if inputs['fcf'] > 0 else inputs['current_eps'] * inputs['shares_outstanding']
    g = inputs['analyst_growth'] / 100
    r = inputs['desired_return'] / 100
    if r <= g:
        return {'intrinsic_value': 0}
    pv = sum([oe * (1+g)**t / (1+r)**t for t in range(1,11)])
    terminal = (oe * (1+g)**10 * (1 + g) / (r - g)) / (1+r)**10
    return {'intrinsic_value': pv + terminal}

# ── Unified Valuation Dispatcher ────────────────────────────────────────────

def calculate_valuation(inputs):
    model = inputs['model']
    current_price = inputs['current_price']
    mos = inputs.get(f"{model.split()[0].lower()}_mos", DEFAULT_MOS)

    model_map = {
        "Core Valuation (Excel)": core_valuation,
        "Lynch Method": lynch_method,
        "Discounted Cash Flow (DCF)": two_stage_dcf,
        "Dividend Discount Model (DDM)": ddm_valuation,
        "Two-Stage DCF": two_stage_dcf,
        "Residual Income (RI)": residual_income,
        "Reverse DCF": reverse_dcf,
        "Graham Intrinsic Value": graham_intrinsic_value,
        "Comparable Company Analysis": comparable_company_analysis,
        "Precedent Transactions": precedent_transactions,
        "Asset-Based Valuation": asset_based_valuation,
        "Sum-of-the-Parts": sum_of_the_parts,
        "EVA-Based": eva_based_valuation,
        "Adjusted Present Value": adjusted_present_value,
        "Owner Earnings": owner_earnings,
    }

    if model not in model_map:
        return {'intrinsic_value': 0, 'error': 'Unknown model'}

    base_results = model_map[model](inputs)
    intrinsic = base_results.get('intrinsic_value', 0) * (1 - mos / 100)
    undervaluation = ((intrinsic - current_price) / current_price * 100) if current_price > 0 else 0

    results = {
        'intrinsic_value': max(intrinsic, 0),
        'safe_buy_price': intrinsic * (1 - 0.1),  # extra 10% buffer example
        'undervaluation': undervaluation,
        'verdict': "Strong Buy" if undervaluation > 30 else "Buy" if undervaluation > 10 else "Hold" if undervaluation > -10 else "Sell",
        **base_results
    }
    return results

# ── Monte Carlo, Plots, Validation, etc. ────────────────────────────────────
# (Keep your existing monte_carlo, plot_heatmap, plot_monte_carlo, plot_model_comparison, validate_inputs functions here)

# ── S&P 500 List ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)  # 24 hours
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df = pd.read_html(url)[0]
        df = df[['Symbol', 'Security', 'GICS Sector']]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df
    except:
        # Fallback minimal list
        return pd.DataFrame({
            'Symbol': ['AAPL','MSFT','GOOGL','AMZN','META'],
            'Security': ['Apple','Microsoft','Alphabet','Amazon','Meta'],
            'GICS Sector': ['Information Technology']*5
        })

# ── Undervalued Screener ────────────────────────────────────────────────────

def run_undervalued_screener(model, min_undervaluation=10.0, selected_sectors=None):
    sp500 = get_sp500_tickers()
    if selected_sectors:
        sp500 = sp500[sp500['GICS Sector'].isin(selected_sectors)]
    
    results = []
    progress = st.progress(0)
    total = len(sp500)

    for i, row in sp500.iterrows():
        ticker = row['Symbol']
        try:
            inputs = fetch_stock_data(ticker)
            inputs['model'] = model
            if validate_inputs(inputs):
                val = calculate_valuation(inputs)
                underv = val.get('undervaluation', 0)
                if underv >= min_undervaluation:
                    results.append({
                        'Ticker': ticker,
                        'Name': row['Security'],
                        'Sector': row['GICS Sector'],
                        'Price': inputs['current_price'],
                        'Intrinsic': val['intrinsic_value'],
                        'Undervaluation %': underv,
                        'Verdict': val['verdict'],
                        'Market Cap (B)': inputs['market_cap'] / 1e9 if inputs['market_cap'] else None
                    })
        except:
            pass
        progress.progress((i+1)/total)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values('Market Cap (B)', ascending=False, na_position='last')

# ── Main Render Function ────────────────────────────────────────────────────

def render_advanced_valuation():
    st.subheader("Advanced Valuation • Multi-Model + Screener")
    st.caption("Data: Yahoo Finance + Finviz • Not financial advice")

    tab_val, tab_screen = st.tabs(["Single Stock Valuation", "S&P 500 Undervalued Screener"])

    # ── Tab 1: Single Stock Valuation ───────────────────────────────────────
    with tab_val:
        # Sidebar inputs (keep your existing layout, just update model list)
        model = st.sidebar.selectbox("Valuation Model", [
            "Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)",
            "Dividend Discount Model (DDM)", "Two-Stage DCF", "Residual Income (RI)",
            "Reverse DCF", "Graham Intrinsic Value",
            "Comparable Company Analysis", "Precedent Transactions",
            "Asset-Based Valuation", "Sum-of-the-Parts",
            "EVA-Based", "Adjusted Present Value", "Owner Earnings"
        ])

        ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()

        if st.sidebar.button("Fetch & Calculate", type="primary"):
            with st.spinner(f"Analyzing {ticker}..."):
                inputs = fetch_stock_data(ticker)
                inputs['model'] = model
                if validate_inputs(inputs):
                    results = calculate_valuation(inputs)
                    # Display results (metrics, charts, Monte Carlo, etc.)
                    # ... your existing display code here ...
                    st.success("Analysis complete")
                else:
                    st.error("Invalid input parameters — check sidebar values")

    # ── Tab 2: Undervalued Screener ─────────────────────────────────────────
    with tab_screen:
        st.markdown("**Screen S&P 500 for undervalued stocks** using chosen model")

        model_screen = st.selectbox("Screening Model", [
            "Core Valuation (Excel)", "Graham Intrinsic Value", "Lynch Method",
            "Discounted Cash Flow (DCF)", "Owner Earnings"  # faster ones recommended
        ], index=0)

        min_underv = st.number_input("Minimum Undervaluation %", 0.0, 100.0, 15.0, step=5.0)
        sectors = sorted(get_sp500_tickers()['GICS Sector'].unique())
        selected_sectors = st.multiselect("Filter Sectors", sectors, default=sectors)

        if st.button("Run Screener (may take 2–5 min)", type="primary"):
            with st.spinner("Screening S&P 500..."):
                df = run_undervalued_screener(model_screen, min_underv, selected_sectors)
                if df.empty:
                    st.info("No stocks met the criteria.")
                else:
                    st.dataframe(
                        df.style.format({
                            'Price': '${:.2f}',
                            'Intrinsic': '${:.2f}',
                            'Undervaluation %': '{:.1f}%',
                            'Market Cap (B)': '{:.1f}'
                        }).background_gradient(subset=['Undervaluation %'], cmap='Greens'),
                        use_container_width=True
                    )
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, f"undervalued_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# ── Standalone Test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(page_title="Advanced Valuation", layout="wide")
    render_advanced_valuation()
