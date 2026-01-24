# advanced_valuation.py
# Comprehensive stock valuation module with multiple models, sensitivity analysis, and screeners
# Data from Yahoo Finance + Finviz fallback
# Version 2.0 - Added Sector Screener, Removed Monte Carlo

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time

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
                        try:
                            clean_val = value.replace('%', '').replace(',', '').replace('$', '')
                            data[key] = float(clean_val) if clean_val.replace('.', '', 1).lstrip('-').isdigit() else value
                        except:
                            data[key] = value
        return data
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_finviz_sector_tickers(sector: str) -> list:
    """Fetch all tickers from a Finviz sector with caching"""
    sector_map = {
        "Technology": "sec_technology",
        "Healthcare": "sec_healthcare",
        "Financials": "sec_financial",
        "Energy": "sec_energy",
        "Consumer Discretionary": "sec_consumercyclical",
        "Consumer Staples": "sec_consumernoncyclical",
        "Industrials": "sec_industrials",
        "Basic Materials": "sec_basicmaterials",
        "Communication Services": "sec_communicationservices",
        "Utilities": "sec_utilities",
        "Real Estate": "sec_realestate"
    }
    
    if sector not in sector_map:
        return []
    
    tickers = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for page_num in range(1, 51):  # Max 50 pages
        start = (page_num - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            found_tickers = False
            for link in soup.find_all('a', {'class': 'tab-link'}):
                ticker = link.text.strip()
                if ticker and len(ticker) <= 5:
                    tickers.append(ticker)
                    found_tickers = True
            
            if not found_tickers:
                break
            
            time.sleep(0.3)
            
        except Exception:
            break
    
    return list(set(tickers))

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
            'analyst_growth': info.get('earningsGrowth', 0.10) * 100 if info.get('earningsGrowth') else 10.0,
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
            'market_cap': info.get('marketCap', 0.0),
            'historical_pe': info.get('trailingPE', 15.0),
            'exit_pe': info.get('forwardPE', 15.0) or 15.0,
            'ebitda': info.get('ebitda', 0.0),
            'net_debt': info.get('netDebt', 0.0),
            'nopat': info.get('operatingCashflow', 0.0) * (1 - 0.25),
            'invested_capital': info.get('totalAssets', 0.0) - info.get('totalCurrentLiabilities', 0.0),
            'shares_outstanding': info.get('sharesOutstanding', 1e9),
            'total_assets': info.get('totalAssets', 0.0),
            'debt': info.get('totalDebt', 0.0),
            'name': info.get('longName', info.get('shortName', ticker_clean)),
            'sector': info.get('sector', 'N/A'),
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
            data['historical_pe'] = finviz.get('P/E', data['historical_pe'])
            data['current_eps'] = finviz.get('EPS (ttm)', data['current_eps'])
            data['forward_eps'] = finviz.get('EPS next Y', data['forward_eps'])
            data['beta'] = finviz.get('Beta', data['beta'])
            data['roe'] = finviz.get('ROE', data['roe'])
            if isinstance(finviz.get('Market Cap'), (int, float)):
                data['market_cap'] = finviz.get('Market Cap') * 1e6

        return data

    except Exception as e:
        return None

# ── Valuation Models ────────────────────────────────────────────────────────

def core_valuation(inputs):
    """Original Excel-style core valuation"""
    eps = inputs['forward_eps']
    pe = inputs['historical_pe']
    growth = inputs['analyst_growth'] / 100
    years = inputs['years_high_growth']
    desired_return = inputs['desired_return'] / 100
    
    if desired_return <= 0:
        return {'intrinsic_value': 0}
    
    future_eps = eps * ((1 + growth) ** years)
    future_price = future_eps * pe
    intrinsic = future_price / ((1 + desired_return) ** years)
    
    return {'intrinsic_value': intrinsic}

def lynch_method(inputs):
    """Peter Lynch's PEG-based valuation"""
    eps = inputs['current_eps']
    growth = inputs['analyst_growth']
    dividend_yield = (inputs['dividend_per_share'] / inputs['current_price']) * 100 if inputs['current_price'] > 0 else 0
    
    fair_pe = growth + dividend_yield
    intrinsic = eps * fair_pe
    
    return {'intrinsic_value': max(intrinsic, 0)}

def ddm_valuation(inputs):
    """Dividend Discount Model (Gordon Growth)"""
    dividend = inputs['dividend_per_share']
    growth = inputs['dividend_growth'] / 100
    required_return = inputs['desired_return'] / 100
    
    if required_return <= growth or dividend <= 0:
        return {'intrinsic_value': 0}
    
    intrinsic = dividend * (1 + growth) / (required_return - growth)
    return {'intrinsic_value': intrinsic}

def two_stage_dcf(inputs):
    """Two-Stage DCF Model"""
    fcf = inputs['fcf']
    high_growth = inputs['analyst_growth'] / 100
    stable_growth = inputs['stable_growth'] / 100
    wacc = inputs['wacc'] / 100
    years = inputs['years_high_growth']
    shares = inputs['shares_outstanding']
    
    if wacc <= 0 or shares <= 0 or fcf <= 0:
        return {'intrinsic_value': 0}
    
    # High growth phase
    pv_fcf = 0
    for t in range(1, years + 1):
        fcf_t = fcf * ((1 + high_growth) ** t)
        pv_fcf += fcf_t / ((1 + wacc) ** t)
    
    # Terminal value
    terminal_fcf = fcf * ((1 + high_growth) ** years) * (1 + stable_growth)
    if wacc > stable_growth:
        terminal_value = terminal_fcf / (wacc - stable_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** years)
    else:
        pv_terminal = 0
    
    total_value = pv_fcf + pv_terminal
    intrinsic = total_value / shares
    
    return {'intrinsic_value': max(intrinsic, 0)}

def residual_income(inputs):
    """Residual Income Model"""
    book_value = inputs['book_value']
    roe = inputs['roe'] / 100
    cost_of_equity = inputs['desired_return'] / 100
    growth = inputs['analyst_growth'] / 100
    years = 10
    
    if cost_of_equity <= growth:
        return {'intrinsic_value': book_value}
    
    residual = book_value * (roe - cost_of_equity)
    pv_ri = 0
    
    for t in range(1, years + 1):
        ri_t = residual * ((1 + growth) ** t)
        pv_ri += ri_t / ((1 + cost_of_equity) ** t)
    
    intrinsic = book_value + pv_ri
    return {'intrinsic_value': max(intrinsic, 0)}

def reverse_dcf(inputs):
    """Reverse DCF - implied growth rate"""
    current_price = inputs['current_price']
    fcf = inputs['fcf']
    wacc = inputs['wacc'] / 100
    shares = inputs['shares_outstanding']
    stable_growth = inputs['stable_growth'] / 100
    
    if shares <= 0 or fcf <= 0 or current_price <= 0:
        return {'intrinsic_value': current_price, 'implied_growth': 0}
    
    market_value = current_price * shares
    
    # Solve for implied growth (simplified)
    implied_growth = (market_value * wacc / fcf) - 1
    implied_growth = max(0, min(implied_growth, 0.5))
    
    return {
        'intrinsic_value': current_price,
        'implied_growth': implied_growth * 100
    }

def graham_intrinsic_value(inputs):
    """Benjamin Graham's Formula"""
    eps = inputs['current_eps']
    growth = inputs['analyst_growth']
    
    if eps <= 0:
        return {'intrinsic_value': 0}
    
    intrinsic = eps * (8.5 + 2 * growth)
    return {'intrinsic_value': max(intrinsic, 0)}

def comparable_company_analysis(inputs):
    """Comparable Companies (P/E multiple)"""
    eps = inputs['current_eps']
    pe = inputs['historical_pe']
    intrinsic = eps * pe
    return {'intrinsic_value': max(intrinsic, 0)}

def precedent_transactions(inputs):
    """Precedent Transactions (with control premium)"""
    comps_val = comparable_company_analysis(inputs)['intrinsic_value']
    premium = 25.0
    return {'intrinsic_value': comps_val * (1 + premium / 100)}

def asset_based_valuation(inputs):
    """Net Asset Value"""
    book_value = inputs['book_value']
    nav = book_value * 0.7
    return {'intrinsic_value': max(nav, 0)}

def sum_of_the_parts(inputs):
    """Sum-of-the-Parts Valuation"""
    segment1 = two_stage_dcf(inputs)['intrinsic_value'] * 0.6
    segment2 = two_stage_dcf(inputs)['intrinsic_value'] * 0.4
    net_debt = inputs.get('net_debt', 0)
    return {'intrinsic_value': max(segment1 + segment2 - net_debt, 0)}

def eva_based_valuation(inputs):
    """Economic Value Added"""
    eva = inputs['nopat'] - (inputs['invested_capital'] * inputs['wacc'] / 100)
    if eva > 0 and inputs['wacc'] > 0:
        mva = eva / (inputs['wacc'] / 100)
        return {'intrinsic_value': inputs['book_value'] + mva}
    return {'intrinsic_value': inputs['book_value']}

def adjusted_present_value(inputs):
    """Adjusted Present Value"""
    unlevered = two_stage_dcf(inputs)['intrinsic_value']
    tax_shield = inputs['tax_rate']/100 * inputs.get('debt', 0) * 0.05
    pv_shield = tax_shield / 0.05 if 0.05 > 0 else 0
    return {'intrinsic_value': unlevered + pv_shield}

def owner_earnings(inputs):
    """Owner Earnings (Buffett-style)"""
    oe = inputs['fcf'] if inputs['fcf'] > 0 else inputs['current_eps'] * inputs['shares_outstanding']
    g = inputs['analyst_growth'] / 100
    r = inputs['desired_return'] / 100
    
    if r <= g or oe <= 0:
        return {'intrinsic_value': 0}
    
    pv = sum([oe * (1+g)**t / (1+r)**t for t in range(1, 11)])
    terminal = (oe * (1+g)**10 * (1 + g) / (r - g)) / (1+r)**10
    return {'intrinsic_value': pv + terminal}

# ── Unified Valuation Dispatcher ────────────────────────────────────────────

def calculate_valuation(inputs):
    """Calculate intrinsic value using selected model"""
    model = inputs['model']
    current_price = inputs['current_price']
    mos = inputs.get('core_mos', DEFAULT_MOS)

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

    try:
        base_results = model_map[model](inputs)
        intrinsic = base_results.get('intrinsic_value', 0) * (1 - mos / 100)
        undervaluation = ((intrinsic - current_price) / current_price * 100) if current_price > 0 else 0

        results = {
            'intrinsic_value': max(intrinsic, 0),
            'safe_buy_price': intrinsic * 0.9,
            'undervaluation': undervaluation,
            'verdict': "Strong Buy" if undervaluation > 30 else "Buy" if undervaluation > 10 else "Hold" if undervaluation > -10 else "Sell",
            **base_results
        }
        return results
    except Exception as e:
        return {'intrinsic_value': 0, 'error': str(e)}

# ── Validation ──────────────────────────────────────────────────────────────

def validate_inputs(inputs):
    """Validate input parameters"""
    if inputs['current_price'] <= 0:
        return False
    if inputs['current_eps'] == 0 and inputs['fcf'] <= 0:
        return False
    if inputs['desired_return'] <= 0:
        return False
    return True

# ── S&P 500 List ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Fetch S&P 500 ticker list"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df = pd.read_html(url)[0]
        df = df[['Symbol', 'Security', 'GICS Sector']]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df
    except:
        return pd.DataFrame({
            'Symbol': ['AAPL','MSFT','GOOGL','AMZN','META'],
            'Security': ['Apple','Microsoft','Alphabet','Amazon','Meta'],
            'GICS Sector': ['Information Technology']*5
        })

# ── Screener Functions ──────────────────────────────────────────────────────

def run_sp500_screener(model, min_undervaluation=10.0, selected_sectors=None, max_stocks=100):
    """Screen S&P 500 for undervalued stocks"""
    sp500 = get_sp500_tickers()
    if selected_sectors:
        sp500 = sp500[sp500['GICS Sector'].isin(selected_sectors)]
    
    sp500 = sp500.head(max_stocks)
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(sp500)

    for i, row in sp500.iterrows():
        ticker = row['Symbol']
        status.text(f"Analyzing {ticker}... ({i+1}/{total})")
        
        try:
            inputs = fetch_stock_data(ticker)
            if inputs and validate_inputs(inputs):
                inputs['model'] = model
                val = calculate_valuation(inputs)
                underv = val.get('undervaluation', 0)
                
                if underv >= min_undervaluation:
                    results.append({
                        'Ticker': ticker,
                        'Name': inputs['name'],
                        'Sector': inputs['sector'],
                        'Price': inputs['current_price'],
                        'Intrinsic': val['intrinsic_value'],
                        'Undervaluation %': underv,
                        'Verdict': val['verdict'],
                        'Market Cap (B)': inputs['market_cap'] / 1e9 if inputs['market_cap'] else None,
                        'P/E': inputs['historical_pe'],
                        'EPS': inputs['current_eps']
                    })
        except Exception as e:
            pass
        
        progress.progress((i+1)/total)

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values('Undervaluation %', ascending=False)

def run_sector_screener(sector, model, min_undervaluation=10.0, max_stocks=50):
    """Screen a specific sector from Finviz"""
    status = st.empty()
    status.text(f"Fetching {sector} tickers from Finviz...")
    
    tickers = get_finviz_sector_tickers(sector)
    
    if not tickers:
        status.empty()
        return pd.DataFrame()
    
    tickers = tickers[:max_stocks]
    
    results = []
    progress = st.progress(0)
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        status.text(f"Analyzing {ticker}... ({i+1}/{total})")
        
        try:
            inputs = fetch_stock_data(ticker)
            if inputs and validate_inputs(inputs):
                inputs['model'] = model
                val = calculate_valuation(inputs)
                underv = val.get('undervaluation', 0)
                
                if underv >= min_undervaluation:
                    results.append({
                        'Ticker': ticker,
                        'Name': inputs['name'],
                        'Sector': inputs['sector'],
                        'Price': inputs['current_price'],
                        'Intrinsic': val['intrinsic_value'],
                        'Undervaluation %': underv,
                        'Verdict': val['verdict'],
                        'Market Cap (B)': inputs['market_cap'] / 1e9 if inputs['market_cap'] else None,
                        'P/E': inputs['historical_pe'],
                        'EPS': inputs['current_eps']
                    })
        except Exception:
            pass
        
        progress.progress((i+1)/total)
        
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values('Undervaluation %', ascending=False)

# ── Sensitivity Analysis ────────────────────────────────────────────────────

def sensitivity_analysis(inputs, param1='analyst_growth', param2='wacc', steps=10):
    """Run sensitivity analysis on two parameters"""
    base_val = calculate_valuation(inputs)['intrinsic_value']
    
    # Define ranges
    param1_range = np.linspace(
        inputs[param1] * 0.5,
        inputs[param1] * 1.5,
        steps
    )
    param2_range = np.linspace(
        inputs[param2] * 0.5,
        inputs[param2] * 1.5,
        steps
    )
    
    results = np.zeros((steps, steps))
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            temp_inputs = inputs.copy()
            temp_inputs[param1] = p1
            temp_inputs[param2] = p2
            results[i, j] = calculate_valuation(temp_inputs)['intrinsic_value']
    
    return param1_range, param2_range, results

# ── Main Render Function ────────────────────────────────────────────────────

def render_advanced_valuation():
    st.subheader("💹 Advanced Valuation & Screener")
    st.caption("Multi-model intrinsic value calculator with S&P 500 and sector-based undervalued stock screeners")

    tab_val, tab_sp500, tab_sector = st.tabs([
        "📊 Single Stock Valuation",
        "🏛️ S&P 500 Screener",
        "🎯 Sector Screener"
    ])

    # ── Tab 1: Single Stock Valuation ───────────────────────────────────────
    with tab_val:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### ⚙️ Configuration")
            
            ticker = st.text_input("Ticker Symbol", "AAPL").upper().strip()
            
            model = st.selectbox("Valuation Model", [
                "Core Valuation (Excel)",
                "Graham Intrinsic Value",
                "Lynch Method",
                "Two-Stage DCF",
                "Dividend Discount Model (DDM)",
                "Residual Income (RI)",
                "Owner Earnings",
                "Comparable Company Analysis",
                "Precedent Transactions",
                "Asset-Based Valuation",
                "Sum-of-the-Parts",
                "EVA-Based",
                "Adjusted Present Value"
            ])
            
            mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5)
            
            with st.expander("📝 Advanced Parameters"):
                analyst_growth = st.number_input("Growth Rate (%)", 0.0, 100.0, 10.0, 1.0)
                wacc = st.number_input("WACC (%)", 0.0, 30.0, 8.0, 0.5)
                desired_return = st.number_input("Required Return (%)", 0.0, 30.0, 10.0, 0.5)
                years_high_growth = st.slider("High Growth Years", 1, 15, 5)
            
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        with col2:
            if analyze_btn:
                with st.spinner(f"Analyzing {ticker}..."):
                    inputs = fetch_stock_data(ticker)
                    
                    if not inputs:
                        st.error(f"❌ Could not fetch data for {ticker}. Check ticker symbol.")
                    else:
                        # Override with user inputs
                        inputs['model'] = model
                        inputs['core_mos'] = mos
                        inputs['analyst_growth'] = analyst_growth
                        inputs['wacc'] = wacc
                        inputs['desired_return'] = desired_return
                        inputs['years_high_growth'] = years_high_growth
                        
                        if not validate_inputs(inputs):
                            st.error("❌ Invalid input parameters. Check the data.")
                        else:
                            results = calculate_valuation(inputs)
                            
                            # Display results
                            st.markdown(f"### {inputs['name']} ({ticker})")
                            st.caption(f"Sector: {inputs['sector']} • Model: {model}")
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            with col_a:
                                st.metric("Current Price", f"${inputs['current_price']:.2f}")
                            with col_b:
                                st.metric("Intrinsic Value", f"${results['intrinsic_value']:.2f}")
                            with col_c:
                                delta_color = "normal" if results['undervaluation'] > 0 else "inverse"
                                st.metric("Undervaluation", f"{results['undervaluation']:.1f}%", 
                                         delta=results['verdict'])
                            with col_d:
                                st.metric("Safe Buy Price", f"${results['safe_buy_price']:.2f}")
                            
                            # Verdict badge
                            verdict_colors = {
                                "Strong Buy": "🟢",
                                "Buy": "🟡",
                                "Hold": "🟠",
                                "Sell": "🔴"
                            }
                            st.markdown(f"## {verdict_colors.get(results['verdict'], '⚪')} {results['verdict']}")
                            
                            # Key metrics table
                            st.markdown("#### 📋 Key Metrics")
                            metrics_df = pd.DataFrame({
                                'Metric': ['EPS (Current)', 'EPS (Forward)', 'P/E Ratio', 'Beta', 'ROE (%)', 
                                          'Market Cap (B)', 'Book Value', 'Growth Rate (%)'],
                                'Value': [
                                    f"${inputs['current_eps']:.2f}",
                                    f"${inputs['forward_eps']:.2f}",
                                    f"{inputs['historical_pe']:.2f}",
                                    f"{inputs['beta']:.2f}",
                                    f"{inputs['roe']:.1f}%",
                                    f"${inputs['market_cap']/1e9:.2f}B" if inputs['market_cap'] else "N/A",
                                    f"${inputs['book_value']:.2f}",
                                    f"{inputs['analyst_growth']:.1f}%"
                                ]
                            })
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                            
                            # Sensitivity Analysis
                            with st.expander("📊 Sensitivity Analysis", expanded=False):
                                st.markdown("**How intrinsic value changes with different assumptions**")
                                
                                col_s1, col_s2 = st.columns(2)
                                with col_s1:
                                    param1 = st.selectbox("X-Axis Parameter", 
                                                         ['analyst_growth', 'wacc', 'desired_return', 'years_high_growth'],
                                                         index=0)
                                with col_s2:
                                    param2 = st.selectbox("Y-Axis Parameter", 
                                                         ['wacc', 'analyst_growth', 'desired_return', 'years_high_growth'],
                                                         index=0)
                                
                                if st.button("Run Sensitivity Analysis"):
                                    with st.spinner("Calculating..."):
                                        p1_range, p2_range, sens_results = sensitivity_analysis(inputs, param1, param2, steps=10)
                                        
                                        fig = go.Figure(data=go.Heatmap(
                                            z=sens_results,
                                            x=p2_range,
                                            y=p1_range,
                                            colorscale='RdYlGn',
                                            text=np.round(sens_results, 2),
                                            texttemplate='$%{text:.2f}',
                                            textfont={"size": 10},
                                            colorbar=dict(title="Intrinsic Value")
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"Sensitivity: {param1.replace('_', ' ').title()} vs {param2.replace('_', ' ').title()}",
                                            xaxis_title=param2.replace('_', ' ').title(),
                                            yaxis_title=param1.replace('_', ' ').title(),
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Model comparison
                            with st.expander("🔬 Compare All Models", expanded=False):
                                st.markdown("**Intrinsic value across all valuation models**")
                                
                                all_models = [
                                    "Core Valuation (Excel)", "Graham Intrinsic Value", "Lynch Method",
                                    "Two-Stage DCF", "Dividend Discount Model (DDM)", "Residual Income (RI)",
                                    "Owner Earnings", "Comparable Company Analysis"
                                ]
                                
                                comparison = []
                                for m in all_models:
                                    temp_inputs = inputs.copy()
                                    temp_inputs['model'] = m
                                    try:
                                        res = calculate_valuation(temp_inputs)
                                        comparison.append({
                                            'Model': m,
                                            'Intrinsic Value': res['intrinsic_value'],
                                            'Undervaluation %': res['undervaluation'],
                                            'Verdict': res['verdict']
                                        })
                                    except:
                                        pass
                                
                                if comparison:
                                    comp_df = pd.DataFrame(comparison)
                                    
                                    # Bar chart
                                    fig = px.bar(comp_df, x='Model', y='Intrinsic Value', 
                                               title='Intrinsic Value by Model',
                                               color='Undervaluation %',
                                               color_continuous_scale='RdYlGn')
                                    fig.add_hline(y=inputs['current_price'], line_dash="dash", 
                                                line_color="red", annotation_text="Current Price")
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Table
                                    st.dataframe(
                                        comp_df.style.format({
                                            'Intrinsic Value': '${:.2f}',
                                            'Undervaluation %': '{:.1f}%'
                                        }).background_gradient(subset=['Undervaluation %'], cmap='RdYlGn'),
                                        use_container_width=True,
                                        hide_index=True
                                    )

    # ── Tab 2: S&P 500 Screener ─────────────────────────────────────────────
    with tab_sp500:
        st.markdown("### 🏛️ S&P 500 Undervalued Stock Screener")
        st.caption("Screen all S&P 500 stocks for undervalued opportunities using your chosen valuation model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sp500_model = st.selectbox("Screening Model", [
                "Graham Intrinsic Value",
                "Core Valuation (Excel)",
                "Lynch Method",
                "Two-Stage DCF",
                "Owner Earnings"
            ], index=0, key="sp500_model")
        
        with col2:
            sp500_min_underv = st.number_input("Min Undervaluation (%)", 0.0, 100.0, 15.0, 5.0, key="sp500_underv")
        
        with col3:
            sp500_max_stocks = st.slider("Max Stocks to Screen", 10, 500, 100, 10, key="sp500_max")
        
        # Sector filter
        sp500_df = get_sp500_tickers()
        sectors = sorted(sp500_df['GICS Sector'].unique())
        selected_sp500_sectors = st.multiselect("Filter Sectors (optional)", sectors, default=sectors, key="sp500_sectors")
        
        if st.button("🚀 Run S&P 500 Screener", type="primary", key="sp500_run"):
            with st.spinner(f"Screening S&P 500 (max {sp500_max_stocks} stocks)..."):
                results_df = run_sp500_screener(
                    sp500_model,
                    sp500_min_underv,
                    selected_sp500_sectors,
                    sp500_max_stocks
                )
                
                if results_df.empty:
                    st.info(f"No stocks found with undervaluation ≥ {sp500_min_underv}%. Try lowering the threshold.")
                else:
                    st.success(f"✅ Found {len(results_df)} undervalued stocks in S&P 500")
                    
                    # Display results
                    st.dataframe(
                        results_df.style.format({
                            'Price': '${:.2f}',
                            'Intrinsic': '${:.2f}',
                            'Undervaluation %': '{:.1f}%',
                            'Market Cap (B)': '{:.2f}',
                            'P/E': '{:.2f}',
                            'EPS': '${:.2f}'
                        }).background_gradient(subset=['Undervaluation %'], cmap='Greens'),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"sp500_undervalued_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary stats
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Avg Undervaluation", f"{results_df['Undervaluation %'].mean():.1f}%")
                    with col_b:
                        st.metric("Max Undervaluation", f"{results_df['Undervaluation %'].max():.1f}%")
                    with col_c:
                        st.metric("Strong Buys", len(results_df[results_df['Undervaluation %'] > 30]))
                    with col_d:
                        st.metric("Avg Market Cap (B)", f"${results_df['Market Cap (B)'].mean():.1f}B")

    # ── Tab 3: Sector Screener ──────────────────────────────────────────────
    with tab_sector:
        st.markdown("### 🎯 Sector-Based Undervalued Stock Screener")
        st.caption("Screen stocks by sector using Finviz data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sector_choice = st.selectbox("Select Sector", [
                "Technology", "Healthcare", "Financials", "Energy",
                "Consumer Discretionary", "Consumer Staples", "Industrials",
                "Basic Materials", "Communication Services", "Utilities", "Real Estate"
            ], key="sector_choice")
        
        with col2:
            sector_model = st.selectbox("Screening Model", [
                "Graham Intrinsic Value",
                "Core Valuation (Excel)",
                "Lynch Method",
                "Two-Stage DCF",
                "Owner Earnings"
            ], index=0, key="sector_model")
        
        with col3:
            sector_min_underv = st.number_input("Min Undervaluation (%)", 0.0, 100.0, 15.0, 5.0, key="sector_underv")
        
        sector_max_stocks = st.slider("Max Stocks to Screen", 10, 200, 50, 10, key="sector_max")
        
        if st.button("🚀 Run Sector Screener", type="primary", key="sector_run"):
            with st.spinner(f"Screening {sector_choice} sector (max {sector_max_stocks} stocks)..."):
                results_df = run_sector_screener(
                    sector_choice,
                    sector_model,
                    sector_min_underv,
                    sector_max_stocks
                )
                
                if results_df.empty:
                    st.info(f"No stocks found in {sector_choice} with undervaluation ≥ {sector_min_underv}%. Try lowering the threshold or check if Finviz data is available.")
                else:
                    st.success(f"✅ Found {len(results_df)} undervalued stocks in {sector_choice}")
                    
                    # Display results
                    st.dataframe(
                        results_df.style.format({
                            'Price': '${:.2f}',
                            'Intrinsic': '${:.2f}',
                            'Undervaluation %': '{:.1f}%',
                            'Market Cap (B)': '{:.2f}',
                            'P/E': '{:.2f}',
                            'EPS': '${:.2f}'
                        }).background_gradient(subset=['Undervaluation %'], cmap='Greens'),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"{sector_choice.lower().replace(' ', '_')}_undervalued_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary stats
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Avg Undervaluation", f"{results_df['Undervaluation %'].mean():.1f}%")
                    with col_b:
                        st.metric("Max Undervaluation", f"{results_df['Undervaluation %'].max():.1f}%")
                    with col_c:
                        st.metric("Strong Buys", len(results_df[results_df['Undervaluation %'] > 30]))
                    with col_d:
                        avg_mcap = results_df['Market Cap (B)'].mean()
                        st.metric("Avg Market Cap (B)", f"${avg_mcap:.1f}B" if pd.notna(avg_mcap) else "N/A")
                    
                    # Top picks chart
                    if len(results_df) >= 5:
                        st.markdown("#### 📊 Top 10 Most Undervalued")
                        top_10 = results_df.head(10)
                        
                        fig = px.bar(
                            top_10,
                            x='Ticker',
                            y='Undervaluation %',
                            color='Undervaluation %',
                            color_continuous_scale='Greens',
                            title=f"Top Undervalued Stocks in {sector_choice}",
                            hover_data=['Name', 'Price', 'Intrinsic']
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)


# ── Standalone Test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(page_title="Advanced Valuation & Screener", layout="wide")
    render_advanced_valuation()
