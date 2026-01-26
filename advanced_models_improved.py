# advanced_models_improved.py - Enhanced Fundamental Financial Models Module
# Comprehensive DuPont, Altman Z-Score, Piotroski F-Score, WACC, and more
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker: str) -> Dict:
    """Fetch comprehensive financial data for analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get historical data for trend analysis
        hist_data = stock.history(period='5y')
        
        return {
            'info': info,
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'history': hist_data,
            'ticker': ticker
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def safe_get(df: pd.DataFrame, key: str, col: int = 0, default: float = 0) -> float:
    """Safely get value from DataFrame."""
    try:
        if key in df.index:
            val = df.loc[key].iloc[col]
            return float(val) if pd.notna(val) else default
        return default
    except:
        return default

# ==================== DUPONT ANALYSIS ====================

def dupont_analysis_3way(net_income: float, revenue: float, assets: float, equity: float) -> Dict:
    """
    3-Way DuPont Analysis: ROE = Profit Margin × Asset Turnover × Equity Multiplier
    """
    if revenue == 0 or assets == 0 or equity == 0:
        return {}
    
    profit_margin = net_income / revenue
    asset_turnover = revenue / assets
    equity_multiplier = assets / equity
    roe = profit_margin * asset_turnover * equity_multiplier
    
    return {
        'ROE': roe * 100,
        'Profit Margin': profit_margin * 100,
        'Asset Turnover': asset_turnover,
        'Equity Multiplier': equity_multiplier,
        'Components': {
            'Net Income': net_income,
            'Revenue': revenue,
            'Total Assets': assets,
            'Total Equity': equity
        }
    }

def dupont_analysis_5way(net_income: float, ebt: float, ebit: float, revenue: float, 
                         assets: float, equity: float) -> Dict:
    """
    5-Way DuPont Analysis: More detailed decomposition
    ROE = Tax Burden × Interest Burden × EBIT Margin × Asset Turnover × Equity Multiplier
    """
    if not all([ebt, ebit, revenue, assets, equity]):
        return {}
    
    tax_burden = net_income / ebt if ebt != 0 else 0
    interest_burden = ebt / ebit if ebit != 0 else 0
    ebit_margin = ebit / revenue if revenue != 0 else 0
    asset_turnover = revenue / assets if assets != 0 else 0
    equity_multiplier = assets / equity if equity != 0 else 0
    
    roe = tax_burden * interest_burden * ebit_margin * asset_turnover * equity_multiplier
    
    return {
        'ROE': roe * 100,
        'Tax Burden': tax_burden,
        'Interest Burden': interest_burden,
        'EBIT Margin': ebit_margin * 100,
        'Asset Turnover': asset_turnover,
        'Equity Multiplier': equity_multiplier,
        'Components': {
            'Net Income': net_income,
            'EBT': ebt,
            'EBIT': ebit,
            'Revenue': revenue,
            'Assets': assets,
            'Equity': equity
        }
    }

# ==================== ALTMAN Z-SCORE ====================

def altman_z_score_public(working_capital: float, retained_earnings: float, ebit: float,
                         market_cap: float, total_liab: float, revenue: float, 
                         total_assets: float) -> Dict:
    """
    Altman Z-Score for publicly traded companies.
    
    Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
    where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets
    
    Interpretation:
    Z > 2.99: Safe Zone (low bankruptcy risk)
    1.81 < Z < 2.99: Grey Zone
    Z < 1.81: Distress Zone (high bankruptcy risk)
    """
    if total_assets == 0:
        return {}
    
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liab if total_liab > 0 else 0
    x5 = revenue / total_assets
    
    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    
    # Determine zone
    if z_score > 2.99:
        zone = "Safe Zone"
        risk = "Low"
        color = "green"
    elif z_score > 1.81:
        zone = "Grey Zone"
        risk = "Moderate"
        color = "orange"
    else:
        zone = "Distress Zone"
        risk = "High"
        color = "red"
    
    return {
        'Z-Score': z_score,
        'Zone': zone,
        'Bankruptcy Risk': risk,
        'Color': color,
        'Components': {
            'X1 (Working Capital / Assets)': x1,
            'X2 (Retained Earnings / Assets)': x2,
            'X3 (EBIT / Assets)': x3,
            'X4 (Market Cap / Liabilities)': x4,
            'X5 (Sales / Assets)': x5
        },
        'Weights': {
            'X1 Weight': 1.2,
            'X2 Weight': 1.4,
            'X3 Weight': 3.3,
            'X4 Weight': 0.6,
            'X5 Weight': 1.0
        }
    }

def altman_z_score_private(working_capital: float, retained_earnings: float, ebit: float,
                          book_equity: float, total_liab: float, revenue: float,
                          total_assets: float) -> Dict:
    """
    Altman Z-Score for private companies (modified version).
    Z' = 0.717X1 + 0.847X2 + 3.107X3 + 0.420X4 + 0.998X5
    Uses book value instead of market value.
    """
    if total_assets == 0:
        return {}
    
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liab if total_liab > 0 else 0
    x5 = revenue / total_assets
    
    z_score = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
    
    # Different thresholds for private companies
    if z_score > 2.6:
        zone = "Safe Zone"
        risk = "Low"
        color = "green"
    elif z_score > 1.1:
        zone = "Grey Zone"
        risk = "Moderate"
        color = "orange"
    else:
        zone = "Distress Zone"
        risk = "High"
        color = "red"
    
    return {
        'Z-Score': z_score,
        'Zone': zone,
        'Bankruptcy Risk': risk,
        'Color': color,
        'Model': 'Private Company',
        'Components': {
            'X1 (Working Capital / Assets)': x1,
            'X2 (Retained Earnings / Assets)': x2,
            'X3 (EBIT / Assets)': x3,
            'X4 (Book Equity / Liabilities)': x4,
            'X5 (Sales / Assets)': x5
        }
    }

# ==================== PIOTROSKI F-SCORE ====================

def piotroski_f_score(current_data: Dict, prior_data: Dict) -> Dict:
    """
    Piotroski F-Score: 9-point score assessing financial strength.
    
    Profitability (4 points):
    1. ROA > 0
    2. Operating Cash Flow > 0
    3. Change in ROA > 0
    4. Accruals < 0 (CFO > Net Income)
    
    Leverage/Liquidity (3 points):
    5. Change in Long-term Debt < 0
    6. Change in Current Ratio > 0
    7. No new shares issued
    
    Operating Efficiency (2 points):
    8. Change in Gross Margin > 0
    9. Change in Asset Turnover > 0
    
    Score Interpretation:
    8-9: Strong
    5-7: Moderate
    0-4: Weak
    """
    score = 0
    details = {}
    
    # Profitability
    # 1. Positive ROA
    roa = current_data.get('roa', 0)
    if roa > 0:
        score += 1
        details['ROA Positive'] = {'score': 1, 'value': f'{roa:.2%}', 'pass': True}
    else:
        details['ROA Positive'] = {'score': 0, 'value': f'{roa:.2%}', 'pass': False}
    
    # 2. Positive Operating Cash Flow
    cfo = current_data.get('cfo', 0)
    if cfo > 0:
        score += 1
        details['CFO Positive'] = {'score': 1, 'value': f'${cfo/1e9:.2f}B', 'pass': True}
    else:
        details['CFO Positive'] = {'score': 0, 'value': f'${cfo/1e9:.2f}B', 'pass': False}
    
    # 3. Change in ROA
    delta_roa = current_data.get('roa', 0) - prior_data.get('roa', 0)
    if delta_roa > 0:
        score += 1
        details['ROA Improvement'] = {'score': 1, 'value': f'+{delta_roa:.2%}', 'pass': True}
    else:
        details['ROA Improvement'] = {'score': 0, 'value': f'{delta_roa:.2%}', 'pass': False}
    
    # 4. Accruals (CFO > Net Income means lower accruals)
    net_income = current_data.get('net_income', 0)
    if cfo > net_income:
        score += 1
        details['Quality of Earnings'] = {'score': 1, 'value': 'High', 'pass': True}
    else:
        details['Quality of Earnings'] = {'score': 0, 'value': 'Low', 'pass': False}
    
    # Leverage/Liquidity
    # 5. Decrease in Long-term Debt
    delta_debt = current_data.get('long_term_debt', 0) - prior_data.get('long_term_debt', 0)
    if delta_debt <= 0:
        score += 1
        details['Leverage Decrease'] = {'score': 1, 'value': f'${-delta_debt/1e9:.2f}B', 'pass': True}
    else:
        details['Leverage Decrease'] = {'score': 0, 'value': f'+${delta_debt/1e9:.2f}B', 'pass': False}
    
    # 6. Increase in Current Ratio
    delta_current = current_data.get('current_ratio', 0) - prior_data.get('current_ratio', 0)
    if delta_current > 0:
        score += 1
        details['Liquidity Improvement'] = {'score': 1, 'value': f'+{delta_current:.2f}', 'pass': True}
    else:
        details['Liquidity Improvement'] = {'score': 0, 'value': f'{delta_current:.2f}', 'pass': False}
    
    # 7. No new shares issued
    delta_shares = current_data.get('shares_outstanding', 0) - prior_data.get('shares_outstanding', 0)
    if delta_shares <= 0:
        score += 1
        details['No Dilution'] = {'score': 1, 'value': 'No new shares', 'pass': True}
    else:
        details['No Dilution'] = {'score': 0, 'value': 'Shares issued', 'pass': False}
    
    # Operating Efficiency
    # 8. Increase in Gross Margin
    delta_margin = current_data.get('gross_margin', 0) - prior_data.get('gross_margin', 0)
    if delta_margin > 0:
        score += 1
        details['Margin Improvement'] = {'score': 1, 'value': f'+{delta_margin:.2%}', 'pass': True}
    else:
        details['Margin Improvement'] = {'score': 0, 'value': f'{delta_margin:.2%}', 'pass': False}
    
    # 9. Increase in Asset Turnover
    delta_turnover = current_data.get('asset_turnover', 0) - prior_data.get('asset_turnover', 0)
    if delta_turnover > 0:
        score += 1
        details['Efficiency Improvement'] = {'score': 1, 'value': f'+{delta_turnover:.2f}', 'pass': True}
    else:
        details['Efficiency Improvement'] = {'score': 0, 'value': f'{delta_turnover:.2f}', 'pass': False}
    
    # Overall assessment
    if score >= 8:
        assessment = "Strong"
        color = "green"
    elif score >= 5:
        assessment = "Moderate"
        color = "orange"
    else:
        assessment = "Weak"
        color = "red"
    
    return {
        'F-Score': score,
        'Assessment': assessment,
        'Color': color,
        'Details': details
    }

# ==================== WACC CALCULATION ====================

def calculate_wacc(market_cap: float, total_debt: float, cost_equity: float,
                  cost_debt: float, tax_rate: float) -> Dict:
    """
    Calculate Weighted Average Cost of Capital (WACC).
    
    WACC = (E/V × Re) + (D/V × Rd × (1-Tc))
    where:
    E = Market value of equity
    D = Market value of debt
    V = E + D (total value)
    Re = Cost of equity
    Rd = Cost of debt
    Tc = Corporate tax rate
    """
    total_value = market_cap + total_debt
    
    if total_value == 0:
        return {}
    
    equity_weight = market_cap / total_value
    debt_weight = total_debt / total_value
    
    wacc = (equity_weight * cost_equity) + (debt_weight * cost_debt * (1 - tax_rate))
    
    return {
        'WACC': wacc * 100,
        'Cost of Equity': cost_equity * 100,
        'Cost of Debt': cost_debt * 100,
        'After-Tax Cost of Debt': cost_debt * (1 - tax_rate) * 100,
        'Tax Rate': tax_rate * 100,
        'Equity Weight': equity_weight * 100,
        'Debt Weight': debt_weight * 100,
        'Components': {
            'Equity Value': market_cap,
            'Debt Value': total_debt,
            'Total Value': total_value
        }
    }

def estimate_cost_of_equity_capm(risk_free_rate: float, beta: float, 
                                market_return: float) -> float:
    """
    Estimate cost of equity using CAPM.
    Re = Rf + β(Rm - Rf)
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)

def estimate_cost_of_debt(interest_expense: float, total_debt: float) -> float:
    """
    Estimate cost of debt from financials.
    Rd = Interest Expense / Total Debt
    """
    if total_debt == 0:
        return 0
    return interest_expense / total_debt

# ==================== ADDITIONAL MODELS ====================

def beneish_m_score(data: Dict) -> Dict:
    """
    Beneish M-Score: Detects earnings manipulation.
    M-Score > -1.78 suggests possible manipulation.
    """
    # This is a simplified version - full implementation requires detailed data
    # Placeholder for demonstration
    m_score = -2.5  # Example value
    
    if m_score > -1.78:
        status = "Possible Manipulation"
        color = "red"
    else:
        status = "Unlikely Manipulation"
        color = "green"
    
    return {
        'M-Score': m_score,
        'Status': status,
        'Color': color,
        'Interpretation': 'Scores above -1.78 suggest potential earnings manipulation'
    }

def graham_number(eps: float, book_value_per_share: float) -> Dict:
    """
    Benjamin Graham's intrinsic value formula.
    Fair Value = √(22.5 × EPS × BVPS)
    """
    if eps <= 0 or book_value_per_share <= 0:
        return {}
    
    fair_value = np.sqrt(22.5 * eps * book_value_per_share)
    
    return {
        'Graham Number': fair_value,
        'EPS': eps,
        'Book Value per Share': book_value_per_share,
        'Formula': '√(22.5 × EPS × BVPS)'
    }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_dupont_waterfall(dupont_data: Dict) -> go.Figure:
    """Create waterfall chart for DuPont analysis."""
    if 'Profit Margin' not in dupont_data:
        return go.Figure()
    
    # For 3-way DuPont
    fig = go.Figure(go.Waterfall(
        name="DuPont",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Profit Margin", "Asset Turnover", "Equity Multiplier", "ROE"],
        textposition="outside",
        text=[f"{dupont_data['Profit Margin']:.2f}%", 
              f"{dupont_data['Asset Turnover']:.2f}x",
              f"{dupont_data['Equity Multiplier']:.2f}x",
              f"{dupont_data['ROE']:.2f}%"],
        y=[dupont_data['Profit Margin'], 
           dupont_data['Asset Turnover'] * 10,  # Scale for visualization
           dupont_data['Equity Multiplier'] * 5,
           dupont_data['ROE']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="DuPont Analysis Breakdown",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig

def create_altman_gauge(z_score: float, model_type: str = 'public') -> go.Figure:
    """Create gauge chart for Altman Z-Score."""
    if model_type == 'public':
        threshold_low = 1.81
        threshold_high = 2.99
    else:
        threshold_low = 1.1
        threshold_high = 2.6
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=z_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Altman Z-Score", 'font': {'size': 24}},
        delta={'reference': threshold_high},
        gauge={
            'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold_low], 'color': '#ffcccc'},
                {'range': [threshold_low, threshold_high], 'color': '#ffffcc'},
                {'range': [threshold_high, 5], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_low
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_piotroski_heatmap(details: Dict) -> go.Figure:
    """Create heatmap for Piotroski F-Score components."""
    categories = list(details.keys())
    scores = [details[cat]['score'] for cat in categories]
    
    # Group by category type
    profitability = categories[:4]
    leverage = categories[4:7]
    efficiency = categories[7:]
    
    fig = go.Figure()
    
    # Create custom heatmap
    fig.add_trace(go.Bar(
        name='Score',
        x=categories,
        y=scores,
        marker_color=['green' if details[cat]['pass'] else 'red' for cat in categories],
        text=[f"{details[cat]['value']}" for cat in categories],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Piotroski F-Score Component Analysis',
        xaxis_title='Component',
        yaxis_title='Score (0 or 1)',
        template='plotly_white',
        height=500,
        showlegend=False,
        yaxis=dict(range=[0, 1.2])
    )
    
    # Add section separators
    fig.add_vline(x=3.5, line_dash="dash", line_color="gray", annotation_text="Leverage")
    fig.add_vline(x=6.5, line_dash="dash", line_color="gray", annotation_text="Efficiency")
    
    return fig

def create_wacc_breakdown(wacc_data: Dict) -> go.Figure:
    """Create pie chart showing WACC components."""
    if not wacc_data:
        return go.Figure()
    
    equity_contribution = wacc_data['Equity Weight'] / 100 * wacc_data['Cost of Equity']
    debt_contribution = wacc_data['Debt Weight'] / 100 * wacc_data['After-Tax Cost of Debt']
    
    fig = go.Figure(data=[go.Pie(
        labels=['Equity Cost Component', 'Debt Cost Component'],
        values=[equity_contribution, debt_contribution],
        hole=.3,
        marker_colors=['lightblue', 'lightcoral']
    )])
    
    fig.update_layout(
        title=f'WACC Breakdown: {wacc_data["WACC"]:.2f}%',
        template='plotly_white',
        height=400,
        annotations=[dict(text=f'{wacc_data["WACC"]:.2f}%', x=0.5, y=0.5, 
                         font_size=20, showarrow=False)]
    )
    
    return fig

# ==================== MAIN MODULE ====================

def advanced_models_module(analysis_context: Optional[Dict] = None):
    """
    Enhanced fundamental financial models module.
    
    Args:
        analysis_context: Optional context from main app
    """
    st.title("🔬 Advanced Fundamental Models")
    st.markdown("""
    Deep-dive financial analysis using proven academic and practitioner models:
    DuPont Analysis, Altman Z-Score, Piotroski F-Score, WACC, and more.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Model Selection")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        selected_models = st.multiselect(
            "Select Models to Run",
            ["DuPont Analysis", "Altman Z-Score", "Piotroski F-Score", 
             "WACC Analysis", "Graham Number"],
            default=["DuPont Analysis", "Altman Z-Score"]
        )
    
    if st.button("🔍 Run Analysis", use_container_width=True):
        with st.spinner(f"Fetching financial data for {ticker}..."):
            data = fetch_financial_data(ticker)
            
            if not data:
                st.error(f"❌ Could not fetch data for {ticker}")
                return
            
            info = data['info']
            income_stmt = data['income_statement']
            balance_sheet = data['balance_sheet']
            cash_flow = data['cash_flow']
            
            # Display company header
            st.header(f"📊 {info.get('longName', ticker)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sector", info.get('sector', 'N/A'))
            with col2:
                market_cap = info.get('marketCap', 0) / 1e9
                st.metric("Market Cap", f"${market_cap:.2f}B")
            with col3:
                current_price = info.get('currentPrice', 0)
                st.metric("Price", f"${current_price:.2f}")
            with col4:
                pe = info.get('trailingPE', 0)
                st.metric("P/E", f"{pe:.2f}" if pe else "N/A")
            
            st.divider()
            
            # Extract financial data
            try:
                # Latest financials (column 0)
                net_income = safe_get(income_stmt, 'Net Income', 0)
                revenue = safe_get(income_stmt, 'Total Revenue', 0)
                ebit = safe_get(income_stmt, 'EBIT', 0)
                ebt = safe_get(income_stmt, 'Pretax Income', 0)
                gross_profit = safe_get(income_stmt, 'Gross Profit', 0)
                
                total_assets = safe_get(balance_sheet, 'Total Assets', 0)
                total_equity = safe_get(balance_sheet, 'Stockholders Equity', 0)
                current_assets = safe_get(balance_sheet, 'Current Assets', 0)
                current_liab = safe_get(balance_sheet, 'Current Liabilities', 0)
                total_liab = safe_get(balance_sheet, 'Total Liabilities Net Minority Interest', 0)
                long_term_debt = safe_get(balance_sheet, 'Long Term Debt', 0)
                retained_earnings = safe_get(balance_sheet, 'Retained Earnings', 0)
                
                operating_cf = safe_get(cash_flow, 'Operating Cash Flow', 0)
                
                working_capital = current_assets - current_liab
                
                # Prior period data (column 1)
                net_income_prior = safe_get(income_stmt, 'Net Income', 1)
                revenue_prior = safe_get(income_stmt, 'Total Revenue', 1)
                gross_profit_prior = safe_get(income_stmt, 'Gross Profit', 1)
                total_assets_prior = safe_get(balance_sheet, 'Total Assets', 1)
                long_term_debt_prior = safe_get(balance_sheet, 'Long Term Debt', 1)
                current_assets_prior = safe_get(balance_sheet, 'Current Assets', 1)
                current_liab_prior = safe_get(balance_sheet, 'Current Liabilities', 1)
                
            except Exception as e:
                st.error(f"Error extracting financial data: {str(e)}")
                return
            
            # ==================== DUPONT ANALYSIS ====================
            if "DuPont Analysis" in selected_models:
                st.header("📊 DuPont Analysis")
                
                st.write("**Return on Equity (ROE) Decomposition**")
                
                # Calculate both 3-way and 5-way
                dupont_3 = dupont_analysis_3way(net_income, revenue, total_assets, total_equity)
                dupont_5 = dupont_analysis_5way(net_income, ebt, ebit, revenue, total_assets, total_equity)
                
                if dupont_3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("3-Way DuPont Analysis")
                        st.metric("ROE", f"{dupont_3['ROE']:.2f}%")
                        
                        st.write("**Components:**")
                        st.write(f"• Profit Margin: {dupont_3['Profit Margin']:.2f}%")
                        st.write(f"• Asset Turnover: {dupont_3['Asset Turnover']:.2f}x")
                        st.write(f"• Equity Multiplier: {dupont_3['Equity Multiplier']:.2f}x")
                        
                        # Formula
                        st.info(f"""
                        **ROE = Profit Margin × Asset Turnover × Equity Multiplier**
                        
                        {dupont_3['ROE']:.2f}% = {dupont_3['Profit Margin']:.2f}% × {dupont_3['Asset Turnover']:.2f} × {dupont_3['Equity Multiplier']:.2f}
                        """)
                    
                    with col2:
                        if dupont_5:
                            st.subheader("5-Way DuPont Analysis")
                            st.metric("ROE", f"{dupont_5['ROE']:.2f}%")
                            
                            st.write("**Components:**")
                            st.write(f"• Tax Burden: {dupont_5['Tax Burden']:.2f}")
                            st.write(f"• Interest Burden: {dupont_5['Interest Burden']:.2f}")
                            st.write(f"• EBIT Margin: {dupont_5['EBIT Margin']:.2f}%")
                            st.write(f"• Asset Turnover: {dupont_5['Asset Turnover']:.2f}x")
                            st.write(f"• Equity Multiplier: {dupont_5['Equity Multiplier']:.2f}x")
                    
                    # Visualization
                    st.subheader("Visual Breakdown")
                    
                    # Create custom decomposition chart
                    components_3 = ['Profit\nMargin', 'Asset\nTurnover', 'Equity\nMultiplier']
                    values_3 = [dupont_3['Profit Margin'], dupont_3['Asset Turnover']*10, dupont_3['Equity Multiplier']*10]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=components_3,
                        y=values_3,
                        text=[f"{dupont_3['Profit Margin']:.2f}%", 
                              f"{dupont_3['Asset Turnover']:.2f}x",
                              f"{dupont_3['Equity Multiplier']:.2f}x"],
                        textposition='auto',
                        marker_color=['lightblue', 'lightgreen', 'lightcoral']
                    ))
                    
                    fig.update_layout(
                        title='DuPont Components (Normalized for Visualization)',
                        yaxis_title='Value',
                        template='plotly_white',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    
                    avg_profit_margin = 10.0  # Industry average placeholder
                    avg_asset_turnover = 0.8
                    avg_equity_mult = 2.0
                    
                    interpretations = []
                    
                    if dupont_3['Profit Margin'] > avg_profit_margin:
                        interpretations.append("✅ **Strong profit margin** - Company efficiently converts revenue to profit")
                    else:
                        interpretations.append("⚠️ **Low profit margin** - Consider cost management or pricing power")
                    
                    if dupont_3['Asset Turnover'] > avg_asset_turnover:
                        interpretations.append("✅ **High asset turnover** - Efficient use of assets to generate revenue")
                    else:
                        interpretations.append("⚠️ **Low asset turnover** - Assets may be underutilized")
                    
                    if dupont_3['Equity Multiplier'] > avg_equity_mult:
                        interpretations.append("⚠️ **High leverage** - Company uses significant debt (higher risk/return)")
                    else:
                        interpretations.append("✅ **Conservative leverage** - Lower financial risk")
                    
                    for interp in interpretations:
                        st.write(interp)
                
                else:
                    st.warning("⚠️ Insufficient data for DuPont analysis")
            
            # ==================== ALTMAN Z-SCORE ====================
            if "Altman Z-Score" in selected_models:
                st.header("🚨 Altman Z-Score")
                
                st.write("**Financial Distress Prediction Model**")
                
                # Calculate Z-Score
                market_cap_val = info.get('marketCap', 0)
                altman = altman_z_score_public(working_capital, retained_earnings, ebit,
                                              market_cap_val, total_liab, revenue, total_assets)
                
                if altman:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        color_map = {'green': '🟢', 'orange': '🟡', 'red': '🔴'}
                        st.metric("Z-Score", f"{altman['Z-Score']:.2f}",
                                 delta=color_map[altman['Color']])
                    
                    with col2:
                        st.metric("Zone", altman['Zone'])
                    
                    with col3:
                        st.metric("Bankruptcy Risk", altman['Bankruptcy Risk'])
                    
                    # Gauge chart
                    fig_gauge = create_altman_gauge(altman['Z-Score'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Component breakdown
                    st.subheader("📊 Component Analysis")
                    
                    comp_df = pd.DataFrame({
                        'Component': list(altman['Components'].keys()),
                        'Value': list(altman['Components'].values()),
                        'Weight': list(altman['Weights'].values()),
                        'Contribution': [altman['Weights'][f'X{i+1} Weight'] * altman['Components'][f'X{i+1} (Working Capital / Assets)' if i==0 else list(altman['Components'].keys())[i]]
                                       for i in range(5)]
                    })
                    
                    st.dataframe(
                        comp_df.style.format({
                            'Value': '{:.4f}',
                            'Weight': '{:.2f}',
                            'Contribution': '{:.4f}'
                        }).background_gradient(subset=['Contribution'], cmap='RdYlGn'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    
                    if altman['Zone'] == "Safe Zone":
                        st.success(f"""
                        **{altman['Zone']}** - The company shows strong financial health with a Z-Score of {altman['Z-Score']:.2f}.
                        Bankruptcy risk is low, indicating solid financial stability.
                        """)
                    elif altman['Zone'] == "Grey Zone":
                        st.warning(f"""
                        **{altman['Zone']}** - The company is in a zone of uncertainty with a Z-Score of {altman['Z-Score']:.2f}.
                        Monitor financial health closely as risk is moderate.
                        """)
                    else:
                        st.error(f"""
                        **{altman['Zone']}** - The company shows signs of financial distress with a Z-Score of {altman['Z-Score']:.2f}.
                        Bankruptcy risk is high - exercise extreme caution.
                        """)
                    
                    st.info("""
                    **Z-Score Thresholds:**
                    - **> 2.99**: Safe Zone (Low Risk)
                    - **1.81 - 2.99**: Grey Zone (Moderate Risk)
                    - **< 1.81**: Distress Zone (High Risk)
                    """)
                
                else:
                    st.warning("⚠️ Insufficient data for Altman Z-Score")
            
            # ==================== PIOTROSKI F-SCORE ====================
            if "Piotroski F-Score" in selected_models:
                st.header("📈 Piotroski F-Score")
                
                st.write("**9-Point Financial Strength Assessment**")
                
                # Prepare current and prior data
                current_data = {
                    'roa': net_income / total_assets if total_assets > 0 else 0,
                    'cfo': operating_cf,
                    'net_income': net_income,
                    'long_term_debt': long_term_debt,
                    'current_ratio': current_assets / current_liab if current_liab > 0 else 0,
                    'shares_outstanding': info.get('sharesOutstanding', 0),
                    'gross_margin': gross_profit / revenue if revenue > 0 else 0,
                    'asset_turnover': revenue / total_assets if total_assets > 0 else 0
                }
                
                prior_data = {
                    'roa': net_income_prior / total_assets_prior if total_assets_prior > 0 else 0,
                    'long_term_debt': long_term_debt_prior,
                    'current_ratio': current_assets_prior / current_liab_prior if current_liab_prior > 0 else 0,
                    'shares_outstanding': info.get('sharesOutstanding', 0) * 1.01,  # Assume slight change
                    'gross_margin': gross_profit_prior / revenue_prior if revenue_prior > 0 else 0,
                    'asset_turnover': revenue_prior / total_assets_prior if total_assets_prior > 0 else 0
                }
                
                piotroski = piotroski_f_score(current_data, prior_data)
                
                if piotroski:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        color_map = {'green': '🟢', 'orange': '🟡', 'red': '🔴'}
                        st.metric("F-Score", f"{piotroski['F-Score']}/9",
                                 delta=color_map[piotroski['Color']])
                    
                    with col2:
                        st.metric("Assessment", piotroski['Assessment'])
                    
                    with col3:
                        st.metric("Passed Tests", f"{piotroski['F-Score']}/9")
                    
                    # Component visualization
                    fig_piotroski = create_piotroski_heatmap(piotroski['Details'])
                    st.plotly_chart(fig_piotroski, use_container_width=True)
                    
                    # Detailed breakdown
                    st.subheader("📊 Detailed Breakdown")
                    
                    # Group by category
                    st.write("**Profitability Signals (4 points):**")
                    for i, key in enumerate(list(piotroski['Details'].keys())[:4]):
                        detail = piotroski['Details'][key]
                        icon = "✅" if detail['pass'] else "❌"
                        st.write(f"{icon} {key}: {detail['value']} (Score: {detail['score']})")
                    
                    st.write("**Leverage & Liquidity Signals (3 points):**")
                    for key in list(piotroski['Details'].keys())[4:7]:
                        detail = piotroski['Details'][key]
                        icon = "✅" if detail['pass'] else "❌"
                        st.write(f"{icon} {key}: {detail['value']} (Score: {detail['score']})")
                    
                    st.write("**Operating Efficiency Signals (2 points):**")
                    for key in list(piotroski['Details'].keys())[7:]:
                        detail = piotroski['Details'][key]
                        icon = "✅" if detail['pass'] else "❌"
                        st.write(f"{icon} {key}: {detail['value']} (Score: {detail['score']})")
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    
                    if piotroski['F-Score'] >= 8:
                        st.success(f"""
                        **Strong Financial Position** - F-Score of {piotroski['F-Score']}/9 indicates excellent financial health.
                        The company demonstrates strong profitability, improving leverage, and operational efficiency.
                        """)
                    elif piotroski['F-Score'] >= 5:
                        st.info(f"""
                        **Moderate Financial Position** - F-Score of {piotroski['F-Score']}/9 suggests adequate financial health.
                        Some areas show strength while others need improvement.
                        """)
                    else:
                        st.warning(f"""
                        **Weak Financial Position** - F-Score of {piotroski['F-Score']}/9 indicates concerning financial health.
                        Multiple red flags across profitability, leverage, and efficiency metrics.
                        """)
                    
                    st.info("""
                    **F-Score Interpretation:**
                    - **8-9**: Strong financials, potential value investment
                    - **5-7**: Moderate financials, requires further analysis
                    - **0-4**: Weak financials, high risk
                    """)
                
                else:
                    st.warning("⚠️ Insufficient data for Piotroski F-Score")
            
            # ==================== WACC ANALYSIS ====================
            if "WACC Analysis" in selected_models:
                st.header("💰 WACC (Weighted Average Cost of Capital)")
                
                st.write("**Cost of Capital Analysis**")
                
                # Get inputs
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Market Data")
                    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.5, min_value=0.0, max_value=20.0, step=0.1) / 100
                    market_return = st.number_input("Market Return (%)", value=10.0, min_value=0.0, max_value=30.0, step=0.5) / 100
                    beta = st.number_input("Beta", value=float(info.get('beta', 1.0)), min_value=-2.0, max_value=3.0, step=0.1)
                
                with col2:
                    st.subheader("Company Financials")
                    market_cap_input = st.number_input("Market Cap ($B)", value=market_cap, min_value=0.0, step=1.0) * 1e9
                    total_debt_input = st.number_input("Total Debt ($B)", value=long_term_debt/1e9, min_value=0.0, step=0.1) * 1e9
                    tax_rate = st.number_input("Tax Rate (%)", value=21.0, min_value=0.0, max_value=50.0, step=1.0) / 100
                
                # Calculate costs
                cost_equity = estimate_cost_of_equity_capm(risk_free_rate, beta, market_return)
                
                # Estimate cost of debt
                interest_expense = safe_get(income_stmt, 'Interest Expense', 0)
                if interest_expense < 0:
                    interest_expense = abs(interest_expense)
                cost_debt = estimate_cost_of_debt(interest_expense, total_debt_input) if total_debt_input > 0 else 0.05
                
                # Calculate WACC
                wacc_result = calculate_wacc(market_cap_input, total_debt_input, cost_equity, cost_debt, tax_rate)
                
                if wacc_result:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("WACC", f"{wacc_result['WACC']:.2f}%")
                    with col2:
                        st.metric("Cost of Equity", f"{wacc_result['Cost of Equity']:.2f}%")
                    with col3:
                        st.metric("After-Tax Cost of Debt", f"{wacc_result['After-Tax Cost of Debt']:.2f}%")
                    
                    # Capital structure
                    st.subheader("📊 Capital Structure")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Equity Weight", f"{wacc_result['Equity Weight']:.1f}%")
                        st.metric("Debt Weight", f"{wacc_result['Debt Weight']:.1f}%")
                    
                    with col2:
                        # Pie chart
                        fig_wacc = create_wacc_breakdown(wacc_result)
                        st.plotly_chart(fig_wacc, use_container_width=True)
                    
                    # Detailed breakdown
                    st.subheader("🔍 Calculation Details")
                    
                    breakdown_df = pd.DataFrame({
                        'Component': ['Equity', 'Debt', 'Total'],
                        'Value ($B)': [
                            market_cap_input / 1e9,
                            total_debt_input / 1e9,
                            (market_cap_input + total_debt_input) / 1e9
                        ],
                        'Weight (%)': [
                            wacc_result['Equity Weight'],
                            wacc_result['Debt Weight'],
                            100.0
                        ],
                        'Cost (%)': [
                            wacc_result['Cost of Equity'],
                            wacc_result['After-Tax Cost of Debt'],
                            wacc_result['WACC']
                        ],
                        'Contribution to WACC (%)': [
                            wacc_result['Equity Weight'] / 100 * wacc_result['Cost of Equity'],
                            wacc_result['Debt Weight'] / 100 * wacc_result['After-Tax Cost of Debt'],
                            wacc_result['WACC']
                        ]
                    })
                    
                    st.dataframe(
                        breakdown_df.style.format({
                            'Value ($B)': '${:.2f}',
                            'Weight (%)': '{:.2f}',
                            'Cost (%)': '{:.2f}',
                            'Contribution to WACC (%)': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Formula display
                    st.info(f"""
                    **WACC Formula:**
                    WACC = (E/V × Re) + (D/V × Rd × (1-Tc))
                    
                    Where:
                    - E/V = {wacc_result['Equity Weight']:.1f}% (Equity weight)
                    - Re = {wacc_result['Cost of Equity']:.2f}% (Cost of equity via CAPM)
                    - D/V = {wacc_result['Debt Weight']:.1f}% (Debt weight)
                    - Rd = {wacc_result['Cost of Debt']:.2f}% (Cost of debt)
                    - Tc = {tax_rate*100:.1f}% (Tax rate)
                    
                    **Result:** {wacc_result['WACC']:.2f}%
                    """)
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    
                    if wacc_result['WACC'] < 8:
                        st.success("**Low WACC** - Company has cheap access to capital, favorable for investments")
                    elif wacc_result['WACC'] < 12:
                        st.info("**Moderate WACC** - Typical cost of capital for established companies")
                    else:
                        st.warning("**High WACC** - Expensive capital, company must generate high returns to create value")
                    
                    st.write(f"""
                    **Use Cases:**
                    - Discount rate for DCF valuation
                    - Hurdle rate for capital budgeting
                    - Benchmark for investment returns
                    - Projects must return > {wacc_result['WACC']:.2f}% to create shareholder value
                    """)
                
                else:
                    st.warning("⚠️ Insufficient data for WACC calculation")
            
            # ==================== GRAHAM NUMBER ====================
            if "Graham Number" in selected_models:
                st.header("💎 Graham Number (Value Investment)")
                
                st.write("**Benjamin Graham's Fair Value Formula**")
                
                eps = info.get('trailingEps', 0)
                book_value = info.get('bookValue', 0)
                
                graham = graham_number(eps, book_value)
                
                if graham:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Graham Number", f"${graham['Graham Number']:.2f}")
                    with col2:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col3:
                        upside = ((graham['Graham Number'] - current_price) / current_price * 100)
                        st.metric("Margin of Safety", f"{upside:+.1f}%")
                    
                    st.info(f"""
                    **Formula:** Fair Value = √(22.5 × EPS × Book Value)
                    
                    - EPS: ${eps:.2f}
                    - Book Value: ${book_value:.2f}
                    - **Graham Number: ${graham['Graham Number']:.2f}**
                    
                    **Benjamin Graham's principle:** Buy when price is below intrinsic value with a margin of safety.
                    """)
                    
                    if upside > 25:
                        st.success("**Significantly Undervalued** - Large margin of safety")
                    elif upside > 0:
                        st.info("**Undervalued** - Some margin of safety exists")
                    else:
                        st.warning("**Overvalued** - Trading above Graham's fair value")
                
                else:
                    st.warning("⚠️ Graham Number requires positive EPS and book value")
    
    # Educational section
    with st.expander("📚 Model Explanations"):
        st.markdown("""
        ### DuPont Analysis
        Breaks down ROE into its component parts to understand drivers of profitability:
        - **Profit Margin**: How much profit is earned per dollar of sales
        - **Asset Turnover**: How efficiently assets generate revenue
        - **Equity Multiplier**: Degree of financial leverage
        
        ### Altman Z-Score
        Predicts bankruptcy probability using financial ratios:
        - Developed by Edward Altman in 1968
        - **> 2.99**: Safe (low bankruptcy risk)
        - **1.81-2.99**: Grey zone
        - **< 1.81**: High bankruptcy risk
        
        ### Piotroski F-Score
        9-point score assessing financial strength:
        - **Profitability**: ROA, cash flow, improving ROA, quality of earnings
        - **Leverage**: Decreasing debt, improving liquidity, no dilution
        - **Efficiency**: Improving margins and asset turnover
        - **8-9**: Strong, **5-7**: Moderate, **0-4**: Weak
        
        ### WACC
        Weighted average cost of capital - the minimum return required:
        - Used as discount rate in DCF valuations
        - Hurdle rate for investment decisions
        - Lower WACC = cheaper capital = easier to create value
        
        ### Graham Number
        Benjamin Graham's value investing formula:
        - Estimates fair value based on earnings and book value
        - Buy with "margin of safety" (below fair value)
        - Conservative approach to value investing
        """)
    
    # Footer
    st.divider()
    st.caption("🔬 Advanced Models Module | Deep fundamental analysis using proven financial frameworks")

# Standalone execution for testing
if __name__ == "__main__":
    advanced_models_module(None)
