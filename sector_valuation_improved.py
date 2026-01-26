# sector_valuation_improved.py - Enhanced Sector-Tuned Valuations + SWOT Module
# Comprehensive sector analysis with DCF, multiples, comparables, and SWOT
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== COMPREHENSIVE SECTOR MULTIPLIERS ====================

SECTOR_MULTIPLIERS = {
    "Technology": {
        "pe_median": 28.0, "pe_range": (18, 45),
        "pb_median": 6.5, "pb_range": (3.0, 12.0),
        "ps_median": 5.5, "ps_range": (2.0, 10.0),
        "ev_ebitda_median": 18.0, "ev_ebitda_range": (12, 28),
        "peg_median": 2.0, "peg_range": (1.0, 3.5),
        "roe_benchmark": 20.0,
        "margin_benchmark": 25.0,
        "growth_rate": 15.0
    },
    "Financial Services": {
        "pe_median": 12.0, "pe_range": (8, 18),
        "pb_median": 1.4, "pb_range": (0.8, 2.5),
        "ps_median": 2.5, "ps_range": (1.0, 4.0),
        "ev_ebitda_median": 10.0, "ev_ebitda_range": (6, 15),
        "peg_median": 1.5, "peg_range": (0.8, 2.5),
        "roe_benchmark": 12.0,
        "margin_benchmark": 30.0,
        "growth_rate": 8.0
    },
    "Healthcare": {
        "pe_median": 22.0, "pe_range": (15, 35),
        "pb_median": 4.0, "pb_range": (2.0, 7.0),
        "ps_median": 3.5, "ps_range": (1.5, 6.0),
        "ev_ebitda_median": 16.0, "ev_ebitda_range": (10, 25),
        "peg_median": 2.2, "peg_range": (1.2, 3.5),
        "roe_benchmark": 15.0,
        "margin_benchmark": 20.0,
        "growth_rate": 10.0
    },
    "Consumer Cyclical": {
        "pe_median": 18.0, "pe_range": (12, 28),
        "pb_median": 3.5, "pb_range": (1.5, 6.0),
        "ps_median": 1.2, "ps_range": (0.5, 2.5),
        "ev_ebitda_median": 12.0, "ev_ebitda_range": (8, 18),
        "peg_median": 1.8, "peg_range": (1.0, 3.0),
        "roe_benchmark": 15.0,
        "margin_benchmark": 8.0,
        "growth_rate": 12.0
    },
    "Consumer Defensive": {
        "pe_median": 20.0, "pe_range": (15, 30),
        "pb_median": 4.5, "pb_range": (2.0, 8.0),
        "ps_median": 1.8, "ps_range": (0.8, 3.5),
        "ev_ebitda_median": 14.0, "ev_ebitda_range": (10, 20),
        "peg_median": 2.5, "peg_range": (1.5, 4.0),
        "roe_benchmark": 18.0,
        "margin_benchmark": 12.0,
        "growth_rate": 6.0
    },
    "Energy": {
        "pe_median": 12.0, "pe_range": (6, 20),
        "pb_median": 1.5, "pb_range": (0.8, 2.5),
        "ps_median": 1.0, "ps_range": (0.4, 2.0),
        "ev_ebitda_median": 8.0, "ev_ebitda_range": (5, 12),
        "peg_median": 1.2, "peg_range": (0.5, 2.0),
        "roe_benchmark": 12.0,
        "margin_benchmark": 10.0,
        "growth_rate": 5.0
    },
    "Industrials": {
        "pe_median": 20.0, "pe_range": (14, 28),
        "pb_median": 3.2, "pb_range": (1.5, 5.5),
        "ps_median": 1.5, "ps_range": (0.8, 2.8),
        "ev_ebitda_median": 13.0, "ev_ebitda_range": (9, 18),
        "peg_median": 2.0, "peg_range": (1.2, 3.0),
        "roe_benchmark": 14.0,
        "margin_benchmark": 10.0,
        "growth_rate": 9.0
    },
    "Basic Materials": {
        "pe_median": 15.0, "pe_range": (8, 22),
        "pb_median": 2.0, "pb_range": (1.0, 3.5),
        "ps_median": 1.2, "ps_range": (0.5, 2.2),
        "ev_ebitda_median": 9.0, "ev_ebitda_range": (6, 14),
        "peg_median": 1.5, "peg_range": (0.8, 2.5),
        "roe_benchmark": 10.0,
        "margin_benchmark": 12.0,
        "growth_rate": 7.0
    },
    "Real Estate": {
        "pe_median": 25.0, "pe_range": (15, 40),
        "pb_median": 1.8, "pb_range": (1.0, 3.0),
        "ps_median": 8.0, "ps_range": (4.0, 15.0),
        "ev_ebitda_median": 16.0, "ev_ebitda_range": (12, 22),
        "peg_median": 3.0, "peg_range": (1.5, 5.0),
        "roe_benchmark": 8.0,
        "margin_benchmark": 40.0,
        "growth_rate": 6.0
    },
    "Utilities": {
        "pe_median": 18.0, "pe_range": (12, 25),
        "pb_median": 1.6, "pb_range": (1.0, 2.5),
        "ps_median": 2.5, "ps_range": (1.5, 4.0),
        "ev_ebitda_median": 11.0, "ev_ebitda_range": (8, 15),
        "peg_median": 3.0, "peg_range": (2.0, 5.0),
        "roe_benchmark": 9.0,
        "margin_benchmark": 15.0,
        "growth_rate": 4.0
    },
    "Communication Services": {
        "pe_median": 20.0, "pe_range": (12, 32),
        "pb_median": 3.0, "pb_range": (1.5, 5.5),
        "ps_median": 3.0, "ps_range": (1.5, 5.5),
        "ev_ebitda_median": 12.0, "ev_ebitda_range": (8, 18),
        "peg_median": 2.0, "peg_range": (1.0, 3.5),
        "roe_benchmark": 12.0,
        "margin_benchmark": 20.0,
        "growth_rate": 10.0
    }
}

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=3600)
def fetch_company_data(ticker: str) -> Dict:
    """Fetch comprehensive company data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financial statements
        try:
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
        except:
            income_stmt = pd.DataFrame()
            balance_sheet = pd.DataFrame()
            cash_flow = pd.DataFrame()
        
        return {
            'info': info,
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'history': stock.history(period='1y')
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_sector_peers(ticker: str, max_peers: int = 5) -> List[str]:
    """Get peer companies in the same sector."""
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', '')
        industry = stock.info.get('industry', '')
        
        # This is a simplified version - in production, you'd use a proper API
        # For now, return some well-known peers by sector
        peer_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Industrials': ['BA', 'CAT', 'GE', 'HON', 'UPS'],
            'Basic Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'T']
        }
        
        peers = peer_mapping.get(sector, [])
        # Remove the input ticker from peers
        peers = [p for p in peers if p != ticker.upper()]
        
        return peers[:max_peers]
    except:
        return []

# ==================== VALUATION METHODS ====================

def calculate_multiples_valuation(data: Dict, sector: str) -> Dict:
    """Calculate fair value using various multiples."""
    info = data['info']
    sector_data = SECTOR_MULTIPLIERS.get(sector, SECTOR_MULTIPLIERS['Technology'])
    
    # Get current metrics
    current_price = info.get('currentPrice', 0)
    book_value = info.get('bookValue', 0)
    eps = info.get('trailingEps', 0)
    revenue_per_share = info.get('revenuePerShare', 0)
    
    # Market cap and enterprise value
    market_cap = info.get('marketCap', 0)
    enterprise_value = info.get('enterpriseValue', market_cap)
    ebitda = info.get('ebitda', 0)
    
    valuations = {}
    
    # P/E based valuation
    if eps and eps > 0:
        pe_fair_value = eps * sector_data['pe_median']
        pe_low = eps * sector_data['pe_range'][0]
        pe_high = eps * sector_data['pe_range'][1]
        valuations['PE'] = {
            'fair_value': pe_fair_value,
            'low': pe_low,
            'high': pe_high,
            'current': current_price
        }
    
    # P/B based valuation
    if book_value and book_value > 0:
        pb_fair_value = book_value * sector_data['pb_median']
        pb_low = book_value * sector_data['pb_range'][0]
        pb_high = book_value * sector_data['pb_range'][1]
        valuations['PB'] = {
            'fair_value': pb_fair_value,
            'low': pb_low,
            'high': pb_high,
            'current': current_price
        }
    
    # P/S based valuation
    if revenue_per_share and revenue_per_share > 0:
        ps_fair_value = revenue_per_share * sector_data['ps_median']
        ps_low = revenue_per_share * sector_data['ps_range'][0]
        ps_high = revenue_per_share * sector_data['ps_range'][1]
        valuations['PS'] = {
            'fair_value': ps_fair_value,
            'low': ps_low,
            'high': ps_high,
            'current': current_price
        }
    
    # EV/EBITDA based valuation
    if ebitda and ebitda > 0 and market_cap > 0:
        shares_outstanding = market_cap / current_price if current_price > 0 else 1
        ev_ebitda_fair_value = (ebitda * sector_data['ev_ebitda_median']) / shares_outstanding
        ev_ebitda_low = (ebitda * sector_data['ev_ebitda_range'][0]) / shares_outstanding
        ev_ebitda_high = (ebitda * sector_data['ev_ebitda_range'][1]) / shares_outstanding
        valuations['EV_EBITDA'] = {
            'fair_value': ev_ebitda_fair_value,
            'low': ev_ebitda_low,
            'high': ev_ebitda_high,
            'current': current_price
        }
    
    # Calculate weighted average (equal weights for simplicity)
    if valuations:
        avg_fair_value = np.mean([v['fair_value'] for v in valuations.values()])
        avg_low = np.mean([v['low'] for v in valuations.values()])
        avg_high = np.mean([v['high'] for v in valuations.values()])
        
        valuations['AVERAGE'] = {
            'fair_value': avg_fair_value,
            'low': avg_low,
            'high': avg_high,
            'current': current_price
        }
    
    return valuations

def calculate_dcf_valuation(data: Dict, sector: str) -> Dict:
    """Calculate intrinsic value using Discounted Cash Flow method."""
    info = data['info']
    sector_data = SECTOR_MULTIPLIERS.get(sector, SECTOR_MULTIPLIERS['Technology'])
    
    # Get free cash flow
    try:
        fcf = data['cash_flow'].loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in data['cash_flow'].index else 0
    except:
        fcf = info.get('freeCashflow', 0)
    
    if not fcf or fcf <= 0:
        return {}
    
    # Parameters
    growth_rate = sector_data['growth_rate'] / 100  # High growth period
    terminal_growth = 0.025  # 2.5% perpetual growth
    discount_rate = 0.10  # 10% WACC (simplified)
    growth_years = 5
    
    # Project cash flows
    projected_fcf = []
    for year in range(1, growth_years + 1):
        projected_fcf.append(fcf * (1 + growth_rate) ** year)
    
    # Calculate present value of projected cash flows
    pv_fcf = sum([cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(projected_fcf)])
    
    # Terminal value
    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** growth_years
    
    # Enterprise value
    enterprise_value = pv_fcf + pv_terminal
    
    # Equity value
    cash = info.get('totalCash', 0)
    debt = info.get('totalDebt', 0)
    equity_value = enterprise_value + cash - debt
    
    # Per share value
    shares_outstanding = info.get('sharesOutstanding', 0)
    if shares_outstanding > 0:
        intrinsic_value = equity_value / shares_outstanding
    else:
        intrinsic_value = 0
    
    return {
        'intrinsic_value': intrinsic_value,
        'enterprise_value': enterprise_value,
        'terminal_value': terminal_value,
        'pv_fcf': pv_fcf,
        'growth_rate': growth_rate * 100,
        'discount_rate': discount_rate * 100,
        'projected_fcf': projected_fcf
    }

def calculate_comparable_companies(ticker: str, peers: List[str]) -> pd.DataFrame:
    """Analyze comparable companies and their valuations."""
    comp_data = []
    
    for peer_ticker in [ticker] + peers:
        try:
            stock = yf.Ticker(peer_ticker)
            info = stock.info
            
            comp_data.append({
                'Ticker': peer_ticker,
                'Company': info.get('shortName', peer_ticker),
                'Market Cap (B)': info.get('marketCap', 0) / 1e9,
                'P/E': info.get('trailingPE', np.nan),
                'P/B': info.get('priceToBook', np.nan),
                'P/S': info.get('priceToSalesTrailing12Months', np.nan),
                'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
                'Profit Margin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
                'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
                'Debt/Equity': info.get('debtToEquity', np.nan) / 100 if info.get('debtToEquity') else np.nan
            })
        except:
            continue
    
    return pd.DataFrame(comp_data)

# ==================== SWOT ANALYSIS ====================

def generate_swot_analysis(data: Dict, sector: str) -> Dict:
    """Generate comprehensive SWOT analysis based on financial metrics."""
    info = data['info']
    sector_data = SECTOR_MULTIPLIERS.get(sector, SECTOR_MULTIPLIERS['Technology'])
    
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []
    
    # Financial metrics
    pe = info.get('trailingPE', 0)
    pb = info.get('priceToBook', 0)
    roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
    profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
    debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
    current_ratio = info.get('currentRatio', 0)
    revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
    earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
    
    # Strengths
    if roe > sector_data['roe_benchmark']:
        strengths.append(f"Strong ROE of {roe:.1f}% (above sector benchmark of {sector_data['roe_benchmark']:.1f}%)")
    
    if profit_margin > sector_data['margin_benchmark']:
        strengths.append(f"High profit margin of {profit_margin:.1f}% (above sector average of {sector_data['margin_benchmark']:.1f}%)")
    
    if debt_to_equity < 0.5:
        strengths.append(f"Low debt-to-equity ratio of {debt_to_equity:.2f} indicates strong financial health")
    
    if current_ratio > 1.5:
        strengths.append(f"Healthy current ratio of {current_ratio:.2f} suggests good liquidity")
    
    if revenue_growth > sector_data['growth_rate']:
        strengths.append(f"Revenue growth of {revenue_growth:.1f}% exceeds sector average of {sector_data['growth_rate']:.1f}%")
    
    # Market position
    market_cap = info.get('marketCap', 0)
    if market_cap > 100e9:
        strengths.append(f"Large market cap of ${market_cap/1e9:.1f}B provides market stability")
    
    # Weaknesses
    if roe < sector_data['roe_benchmark'] * 0.7:
        weaknesses.append(f"ROE of {roe:.1f}% is significantly below sector benchmark")
    
    if profit_margin < sector_data['margin_benchmark'] * 0.7:
        weaknesses.append(f"Profit margin of {profit_margin:.1f}% lags sector peers")
    
    if debt_to_equity > 1.5:
        weaknesses.append(f"High debt-to-equity ratio of {debt_to_equity:.2f} increases financial risk")
    
    if current_ratio < 1.0:
        weaknesses.append(f"Current ratio of {current_ratio:.2f} below 1.0 may indicate liquidity concerns")
    
    if revenue_growth < 0:
        weaknesses.append(f"Negative revenue growth of {revenue_growth:.1f}% signals declining sales")
    
    # Opportunities
    if pe < sector_data['pe_median'] * 0.8 and pe > 0:
        opportunities.append(f"P/E ratio of {pe:.1f} suggests stock may be undervalued (sector median: {sector_data['pe_median']:.1f})")
    
    if pb < sector_data['pb_median'] * 0.8 and pb > 0:
        opportunities.append(f"P/B ratio of {pb:.1f} indicates potential value investment (sector median: {sector_data['pb_median']:.1f})")
    
    if earnings_growth > 15:
        opportunities.append(f"Strong earnings growth of {earnings_growth:.1f}% creates expansion potential")
    
    if revenue_growth > 10:
        opportunities.append("Double-digit revenue growth indicates market share expansion opportunities")
    
    # Threats
    if pe > sector_data['pe_median'] * 1.5 and pe > 0:
        threats.append(f"High P/E ratio of {pe:.1f} suggests overvaluation risk (sector median: {sector_data['pe_median']:.1f})")
    
    if pb > sector_data['pb_median'] * 1.5 and pb > 0:
        threats.append(f"Elevated P/B ratio of {pb:.1f} may indicate limited upside (sector median: {sector_data['pb_median']:.1f})")
    
    if debt_to_equity > 2.0:
        threats.append("Very high leverage creates significant financial risk")
    
    if earnings_growth < -10:
        threats.append(f"Declining earnings growth of {earnings_growth:.1f}% threatens profitability")
    
    # Add general market and sector specific items
    if sector in ['Technology', 'Consumer Cyclical']:
        opportunities.append("Digital transformation trends driving sector growth")
        threats.append("Rapid technological change requires continuous innovation")
    elif sector == 'Energy':
        opportunities.append("Energy transition creating new market segments")
        threats.append("Regulatory pressures and environmental concerns")
    elif sector == 'Healthcare':
        opportunities.append("Aging demographics driving healthcare demand")
        threats.append("Regulatory and reimbursement pressures")
    
    # Ensure we have at least some items in each category
    if not strengths:
        strengths.append("Company maintains operations in stable sector")
    if not weaknesses:
        weaknesses.append("Limited data available for comprehensive analysis")
    if not opportunities:
        opportunities.append("Potential for operational improvements")
    if not threats:
        threats.append("General market volatility and economic uncertainty")
    
    return {
        'Strengths': strengths,
        'Weaknesses': weaknesses,
        'Opportunities': opportunities,
        'Threats': threats
    }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_valuation_chart(valuations: Dict, current_price: float) -> go.Figure:
    """Create valuation comparison chart."""
    methods = []
    fair_values = []
    lows = []
    highs = []
    
    for method, vals in valuations.items():
        if method != 'AVERAGE':  # Plot average separately
            methods.append(method)
            fair_values.append(vals['fair_value'])
            lows.append(vals['low'])
            highs.append(vals['high'])
    
    fig = go.Figure()
    
    # Fair value bars
    fig.add_trace(go.Bar(
        name='Fair Value',
        x=methods,
        y=fair_values,
        marker_color='lightblue',
        text=[f'${v:.2f}' for v in fair_values],
        textposition='auto'
    ))
    
    # Range bars (using error bars)
    errors_y = {
        'type': 'data',
        'symmetric': False,
        'array': [h - fv for h, fv in zip(highs, fair_values)],
        'arrayminus': [fv - l for l, fv in zip(lows, fair_values)]
    }
    
    fig.add_trace(go.Scatter(
        name='Range',
        x=methods,
        y=fair_values,
        error_y=errors_y,
        mode='markers',
        marker=dict(size=0),
        showlegend=True
    ))
    
    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="right"
    )
    
    # Average fair value line
    if 'AVERAGE' in valuations:
        avg_fv = valuations['AVERAGE']['fair_value']
        fig.add_hline(
            y=avg_fv,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Average: ${avg_fv:.2f}",
            annotation_position="left"
        )
    
    fig.update_layout(
        title='Valuation Analysis by Method',
        xaxis_title='Valuation Method',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_peer_comparison_chart(comp_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create peer comparison visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P/E Ratio', 'P/B Ratio', 'ROE (%)', 'Profit Margin (%)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Color coding: highlight the target ticker
    colors = ['lightcoral' if t == ticker else 'lightblue' for t in comp_df['Ticker']]
    
    # P/E
    fig.add_trace(
        go.Bar(x=comp_df['Ticker'], y=comp_df['P/E'], marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # P/B
    fig.add_trace(
        go.Bar(x=comp_df['Ticker'], y=comp_df['P/B'], marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # ROE
    fig.add_trace(
        go.Bar(x=comp_df['Ticker'], y=comp_df['ROE'], marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # Profit Margin
    fig.add_trace(
        go.Bar(x=comp_df['Ticker'], y=comp_df['Profit Margin'], marker_color=colors, showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text='Peer Company Comparison',
        template='plotly_white',
        height=700
    )
    
    return fig

# ==================== MAIN MODULE ====================

def sector_valuation_module(analysis_context: Optional[Dict] = None):
    """
    Enhanced sector-tuned valuation and SWOT analysis module.
    
    Args:
        analysis_context: Optional context from main app
    """
    st.title("🎯 Sector-Tuned Valuation & SWOT Analysis")
    st.markdown("""
    Comprehensive valuation analysis using sector-specific multiples, DCF modeling,
    comparable company analysis, and automated SWOT generation.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Analysis Configuration")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        sector = st.selectbox(
            "Sector",
            options=list(SECTOR_MULTIPLIERS.keys()),
            help="Select the company's sector for appropriate benchmarks"
        )
        
        analysis_type = st.multiselect(
            "Analysis Type",
            ["Multiples Valuation", "DCF Analysis", "Peer Comparison", "SWOT Analysis"],
            default=["Multiples Valuation", "SWOT Analysis"]
        )
    
    if st.button("🔍 Run Analysis", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            # Fetch company data
            data = fetch_company_data(ticker)
            
            if not data or not data['info']:
                st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                return
            
            info = data['info']
            
            # Display company header
            st.header(f"📊 {info.get('longName', ticker)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = info.get('currentPrice', 0)
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
            
            with col2:
                market_cap = info.get('marketCap', 0) / 1e9
                st.metric("Market Cap", f"${market_cap:.2f}B")
            
            with col3:
                pe = info.get('trailingPE', 0)
                st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
            
            with col4:
                actual_sector = info.get('sector', sector)
                st.metric("Sector", actual_sector)
            
            # Update sector if fetched from API
            if actual_sector and actual_sector in SECTOR_MULTIPLIERS:
                sector = actual_sector
            
            st.divider()
            
            # ==================== MULTIPLES VALUATION ====================
            if "Multiples Valuation" in analysis_type:
                st.header("💰 Multiples-Based Valuation")
                
                valuations = calculate_multiples_valuation(data, sector)
                
                if valuations:
                    # Summary metrics
                    if 'AVERAGE' in valuations:
                        avg_fv = valuations['AVERAGE']['fair_value']
                        upside = ((avg_fv - current_price) / current_price * 100)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Average Fair Value", f"${avg_fv:.2f}")
                        with col2:
                            st.metric("Potential Upside/Downside", f"{upside:+.1f}%",
                                    delta_color="normal")
                        with col3:
                            if upside > 20:
                                recommendation = "🟢 UNDERVALUED"
                                color = "green"
                            elif upside < -20:
                                recommendation = "🔴 OVERVALUED"
                                color = "red"
                            else:
                                recommendation = "🟡 FAIRLY VALUED"
                                color = "orange"
                            st.markdown(f"**Recommendation:** :{color}[{recommendation}]")
                    
                    # Detailed valuation table
                    st.subheader("📊 Valuation by Method")
                    
                    val_df = pd.DataFrame({
                        'Method': [k for k in valuations.keys() if k != 'AVERAGE'],
                        'Fair Value': [v['fair_value'] for k, v in valuations.items() if k != 'AVERAGE'],
                        'Low Estimate': [v['low'] for k, v in valuations.items() if k != 'AVERAGE'],
                        'High Estimate': [v['high'] for k, v in valuations.items() if k != 'AVERAGE'],
                        'Current Price': [v['current'] for k, v in valuations.items() if k != 'AVERAGE'],
                        'Upside (%)': [((v['fair_value'] - v['current']) / v['current'] * 100) 
                                      for k, v in valuations.items() if k != 'AVERAGE']
                    })
                    
                    st.dataframe(
                        val_df.style.format({
                            'Fair Value': '${:.2f}',
                            'Low Estimate': '${:.2f}',
                            'High Estimate': '${:.2f}',
                            'Current Price': '${:.2f}',
                            'Upside (%)': '{:+.1f}'
                        }).background_gradient(subset=['Upside (%)'], cmap='RdYlGn'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualization
                    fig_val = create_valuation_chart(valuations, current_price)
                    st.plotly_chart(fig_val, use_container_width=True)
                    
                    # Sector benchmarks
                    st.subheader("📈 Sector Benchmarks")
                    
                    sector_data = SECTOR_MULTIPLIERS[sector]
                    
                    benchmark_df = pd.DataFrame({
                        'Metric': ['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA', 'ROE', 'Profit Margin'],
                        'Current': [
                            info.get('trailingPE', 0),
                            info.get('priceToBook', 0),
                            info.get('priceToSalesTrailing12Months', 0),
                            info.get('enterpriseToEbitda', 0),
                            info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                            info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
                        ],
                        'Sector Median': [
                            sector_data['pe_median'],
                            sector_data['pb_median'],
                            sector_data['ps_median'],
                            sector_data['ev_ebitda_median'],
                            sector_data['roe_benchmark'],
                            sector_data['margin_benchmark']
                        ],
                        'vs Sector': [
                            'Above' if info.get('trailingPE', 0) > sector_data['pe_median'] else 'Below',
                            'Above' if info.get('priceToBook', 0) > sector_data['pb_median'] else 'Below',
                            'Above' if info.get('priceToSalesTrailing12Months', 0) > sector_data['ps_median'] else 'Below',
                            'Above' if info.get('enterpriseToEbitda', 0) > sector_data['ev_ebitda_median'] else 'Below',
                            'Above' if (info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0) > sector_data['roe_benchmark'] else 'Below',
                            'Above' if (info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0) > sector_data['margin_benchmark'] else 'Below'
                        ]
                    })
                    
                    st.dataframe(
                        benchmark_df.style.format({
                            'Current': '{:.2f}',
                            'Sector Median': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("⚠️ Insufficient data for multiples valuation")
            
            # ==================== DCF ANALYSIS ====================
            if "DCF Analysis" in analysis_type:
                st.header("📉 Discounted Cash Flow (DCF) Analysis")
                
                dcf_result = calculate_dcf_valuation(data, sector)
                
                if dcf_result:
                    intrinsic_value = dcf_result['intrinsic_value']
                    upside_dcf = ((intrinsic_value - current_price) / current_price * 100)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Intrinsic Value (DCF)", f"${intrinsic_value:.2f}")
                    with col2:
                        st.metric("Upside/Downside", f"{upside_dcf:+.1f}%",
                                delta_color="normal")
                    with col3:
                        st.metric("Enterprise Value", f"${dcf_result['enterprise_value']/1e9:.2f}B")
                    
                    # DCF assumptions
                    st.subheader("🔧 DCF Assumptions")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Growth Rate", f"{dcf_result['growth_rate']:.1f}%")
                    with col2:
                        st.metric("Discount Rate (WACC)", f"{dcf_result['discount_rate']:.1f}%")
                    with col3:
                        st.metric("Terminal Growth", "2.5%")
                    
                    # Projected cash flows
                    st.subheader("📊 Projected Free Cash Flows")
                    
                    fcf_df = pd.DataFrame({
                        'Year': range(1, len(dcf_result['projected_fcf']) + 1),
                        'Projected FCF': dcf_result['projected_fcf']
                    })
                    
                    fig_fcf = px.bar(
                        fcf_df,
                        x='Year',
                        y='Projected FCF',
                        title='5-Year Free Cash Flow Projection',
                        labels={'Projected FCF': 'Free Cash Flow ($)'}
                    )
                    fig_fcf.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig_fcf, use_container_width=True)
                    
                    # Sensitivity analysis
                    st.subheader("🎯 Sensitivity Analysis")
                    
                    st.write("**Impact of changing assumptions on intrinsic value:**")
                    
                    # Growth rate sensitivity
                    growth_rates = np.linspace(0.05, 0.20, 7)
                    intrinsic_values = []
                    
                    for gr in growth_rates:
                        # Recalculate with different growth rate
                        # Simplified - just scale proportionally
                        scaled_value = intrinsic_value * (1 + (gr - dcf_result['growth_rate']/100) * 2)
                        intrinsic_values.append(scaled_value)
                    
                    sens_df = pd.DataFrame({
                        'Growth Rate (%)': growth_rates * 100,
                        'Intrinsic Value ($)': intrinsic_values,
                        'Upside (%)': [(iv - current_price) / current_price * 100 for iv in intrinsic_values]
                    })
                    
                    st.dataframe(
                        sens_df.style.format({
                            'Growth Rate (%)': '{:.1f}',
                            'Intrinsic Value ($)': '${:.2f}',
                            'Upside (%)': '{:+.1f}'
                        }).background_gradient(subset=['Upside (%)'], cmap='RdYlGn'),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("⚠️ Insufficient cash flow data for DCF analysis")
            
            # ==================== PEER COMPARISON ====================
            if "Peer Comparison" in analysis_type:
                st.header("⚖️ Comparable Company Analysis")
                
                with st.spinner("Fetching peer company data..."):
                    peers = get_sector_peers(ticker, 5)
                    
                    if peers:
                        st.info(f"**Peer Companies:** {', '.join(peers)}")
                        
                        comp_df = calculate_comparable_companies(ticker, peers)
                        
                        if not comp_df.empty:
                            # Display comparison table
                            st.subheader("📊 Peer Comparison Table")
                            
                            st.dataframe(
                                comp_df.style.format({
                                    'Market Cap (B)': '${:.2f}',
                                    'P/E': '{:.2f}',
                                    'P/B': '{:.2f}',
                                    'P/S': '{:.2f}',
                                    'EV/EBITDA': '{:.2f}',
                                    'Profit Margin': '{:.1f}',
                                    'ROE': '{:.1f}',
                                    'Debt/Equity': '{:.2f}'
                                }).background_gradient(subset=['P/E', 'ROE'], cmap='RdYlGn_r'),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Visualization
                            fig_peer = create_peer_comparison_chart(comp_df, ticker)
                            st.plotly_chart(fig_peer, use_container_width=True)
                            
                            # Relative valuation
                            st.subheader("📈 Relative Valuation")
                            
                            # Calculate percentile rankings
                            ticker_row = comp_df[comp_df['Ticker'] == ticker].iloc[0]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pe_rank = (comp_df['P/E'] < ticker_row['P/E']).sum() / len(comp_df) * 100
                                st.metric("P/E Percentile", f"{pe_rank:.0f}%",
                                        help="Higher percentile = more expensive vs peers")
                            
                            with col2:
                                roe_rank = (comp_df['ROE'] < ticker_row['ROE']).sum() / len(comp_df) * 100
                                st.metric("ROE Percentile", f"{roe_rank:.0f}%",
                                        help="Higher percentile = better performance")
                            
                            with col3:
                                margin_rank = (comp_df['Profit Margin'] < ticker_row['Profit Margin']).sum() / len(comp_df) * 100
                                st.metric("Margin Percentile", f"{margin_rank:.0f}%",
                                        help="Higher percentile = more profitable")
                        else:
                            st.warning("⚠️ Could not fetch peer company data")
                    else:
                        st.warning("⚠️ No peer companies identified for this sector")
            
            # ==================== SWOT ANALYSIS ====================
            if "SWOT Analysis" in analysis_type:
                st.header("🎯 SWOT Analysis")
                
                swot = generate_swot_analysis(data, sector)
                
                # Display SWOT in quadrants
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💪 Strengths")
                    st.success("**Internal Positive Factors**")
                    for strength in swot['Strengths']:
                        st.write(f"✓ {strength}")
                    
                    st.divider()
                    
                    st.subheader("🎯 Opportunities")
                    st.info("**External Positive Factors**")
                    for opportunity in swot['Opportunities']:
                        st.write(f"✓ {opportunity}")
                
                with col2:
                    st.subheader("⚠️ Weaknesses")
                    st.warning("**Internal Negative Factors**")
                    for weakness in swot['Weaknesses']:
                        st.write(f"✗ {weakness}")
                    
                    st.divider()
                    
                    st.subheader("⚡ Threats")
                    st.error("**External Negative Factors**")
                    for threat in swot['Threats']:
                        st.write(f"✗ {threat}")
                
                # SWOT summary
                st.subheader("📋 SWOT Summary")
                
                swot_score = (len(swot['Strengths']) + len(swot['Opportunities']) - 
                             len(swot['Weaknesses']) - len(swot['Threats']))
                
                if swot_score > 2:
                    st.success(f"**Overall Assessment: POSITIVE** (Score: +{swot_score})")
                    st.write("The company demonstrates more positive than negative factors, suggesting favorable fundamentals.")
                elif swot_score < -2:
                    st.error(f"**Overall Assessment: NEGATIVE** (Score: {swot_score})")
                    st.write("Significant challenges and risks identified that may impact performance.")
                else:
                    st.info(f"**Overall Assessment: NEUTRAL** (Score: {swot_score:+d})")
                    st.write("Balanced profile with both positive and negative factors to consider.")
    
    # Educational section
    with st.expander("ℹ️ Understanding Valuation Methods"):
        st.markdown("""
        ### Valuation Methods Explained
        
        **1. P/E Ratio (Price-to-Earnings)**
        - Compares stock price to earnings per share
        - Lower P/E may indicate undervaluation
        - Sector-specific ranges are important
        
        **2. P/B Ratio (Price-to-Book)**
        - Compares market value to book value
        - Useful for asset-heavy companies
        - Below 1.0 may indicate undervaluation
        
        **3. P/S Ratio (Price-to-Sales)**
        - Compares market cap to revenue
        - Useful for companies with no profits yet
        - Lower ratios generally better
        
        **4. EV/EBITDA**
        - Enterprise Value to Earnings Before Interest, Taxes, Depreciation & Amortization
        - Accounts for debt and cash
        - Good for cross-company comparisons
        
        **5. DCF (Discounted Cash Flow)**
        - Estimates intrinsic value based on future cash flows
        - Most theoretically sound but assumption-dependent
        - Sensitive to growth and discount rates
        
        ### SWOT Analysis
        - **Strengths & Weaknesses**: Internal factors the company controls
        - **Opportunities & Threats**: External market factors
        - Use alongside valuation for complete picture
        """)
    
    # Footer
    st.divider()
    st.caption("🎯 Sector Valuation Module | Comprehensive valuation and strategic analysis")

# Standalone execution for testing
if __name__ == "__main__":
    sector_valuation_module(None)
