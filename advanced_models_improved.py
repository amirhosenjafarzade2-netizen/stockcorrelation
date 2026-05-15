# advanced_models_improved.py - Enhanced Fundamental Financial Models Module
# Comprehensive DuPont, Altman Z-Score, Piotroski F-Score, WACC, Graham Number,
# Beneish M-Score, EV/EBITDA, Greenblatt Magic Formula, and more.

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

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Advanced Financial Models",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 4px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }
    .badge-green { background: #d4edda; color: #155724; }
    .badge-orange { background: #fff3cd; color: #856404; }
    .badge-red { background: #f8d7da; color: #721c24; }
    .section-header {
        font-size: 1.4em;
        font-weight: 700;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stMetric { border-radius: 8px; }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker: str) -> Optional[Dict]:
    """Fetch comprehensive financial data for analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
            return None
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
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
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def safe_get(df: pd.DataFrame, key: str, col: int = 0, default: float = 0.0) -> float:
    """Safely extract a float from a DataFrame by row name and column index."""
    try:
        if df is None or df.empty:
            return default
        if key in df.index:
            val = df.loc[key].iloc[col]
            return float(val) if pd.notna(val) else default
        return default
    except Exception:
        return default


def fmt_billions(val: float) -> str:
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"

# ==================== DUPONT ANALYSIS ====================

def dupont_analysis_3way(net_income, revenue, assets, equity) -> Dict:
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
    }


def dupont_analysis_5way(net_income, ebt, ebit, revenue, assets, equity) -> Dict:
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
    }

# ==================== ALTMAN Z-SCORE ====================

def altman_z_score_public(working_capital, retained_earnings, ebit,
                          market_cap, total_liab, revenue, total_assets) -> Dict:
    if total_assets == 0:
        return {}
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liab if total_liab > 0 else 0
    x5 = revenue / total_assets
    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    zone, risk, color = ("Safe Zone", "Low", "green") if z > 2.99 else \
                        ("Grey Zone", "Moderate", "orange") if z > 1.81 else \
                        ("Distress Zone", "High", "red")
    return {
        'Z-Score': z, 'Zone': zone, 'Bankruptcy Risk': risk, 'Color': color,
        'Components': {'X1 Working Capital/Assets': x1, 'X2 Retained Earnings/Assets': x2,
                       'X3 EBIT/Assets': x3, 'X4 MktCap/Liabilities': x4, 'X5 Sales/Assets': x5},
        'Weights': {'X1': 1.2, 'X2': 1.4, 'X3': 3.3, 'X4': 0.6, 'X5': 1.0}
    }

# ==================== PIOTROSKI F-SCORE ====================

def piotroski_f_score(curr: Dict, prev: Dict) -> Dict:
    score = 0
    details = {}

    def check(label: str, passed: bool, value: str, category: str):
        nonlocal score
        if passed:
            score += 1
        details[label] = {'score': int(passed), 'value': value, 'pass': passed, 'category': category}

    # Profitability
    roa = curr.get('roa', 0)
    check('ROA Positive', roa > 0, f'{roa:.2%}', 'Profitability')
    cfo = curr.get('cfo', 0)
    check('CFO Positive', cfo > 0, fmt_billions(cfo), 'Profitability')
    delta_roa = curr.get('roa', 0) - prev.get('roa', 0)
    check('ROA Improving', delta_roa > 0, f'{delta_roa:+.2%}', 'Profitability')
    ni = curr.get('net_income', 0)
    check('Quality of Earnings (CFO>NI)', cfo > ni, 'High' if cfo > ni else 'Low', 'Profitability')

    # Leverage / Liquidity
    delta_debt = curr.get('long_term_debt', 0) - prev.get('long_term_debt', 0)
    check('Leverage Decreasing', delta_debt <= 0, fmt_billions(-delta_debt), 'Leverage')
    delta_cr = curr.get('current_ratio', 0) - prev.get('current_ratio', 0)
    check('Current Ratio Improving', delta_cr > 0, f'{delta_cr:+.2f}', 'Leverage')
    delta_shares = curr.get('shares_outstanding', 0) - prev.get('shares_outstanding', 0)
    check('No Share Dilution', delta_shares <= 0, 'No new shares' if delta_shares <= 0 else f'+{delta_shares/1e6:.1f}M', 'Leverage')

    # Efficiency
    delta_gm = curr.get('gross_margin', 0) - prev.get('gross_margin', 0)
    check('Gross Margin Improving', delta_gm > 0, f'{delta_gm:+.2%}', 'Efficiency')
    delta_at = curr.get('asset_turnover', 0) - prev.get('asset_turnover', 0)
    check('Asset Turnover Improving', delta_at > 0, f'{delta_at:+.2f}', 'Efficiency')

    assessment, color = ('Strong', 'green') if score >= 8 else ('Moderate', 'orange') if score >= 5 else ('Weak', 'red')
    return {'F-Score': score, 'Assessment': assessment, 'Color': color, 'Details': details}

# ==================== WACC ====================

def calculate_wacc(market_cap, total_debt, cost_equity, cost_debt, tax_rate) -> Dict:
    total_value = market_cap + total_debt
    if total_value == 0:
        return {}
    ew = market_cap / total_value
    dw = total_debt / total_value
    wacc = ew * cost_equity + dw * cost_debt * (1 - tax_rate)
    return {
        'WACC': wacc * 100,
        'Cost of Equity': cost_equity * 100,
        'Cost of Debt': cost_debt * 100,
        'After-Tax Cost of Debt': cost_debt * (1 - tax_rate) * 100,
        'Tax Rate': tax_rate * 100,
        'Equity Weight': ew * 100,
        'Debt Weight': dw * 100,
        'Equity Value': market_cap,
        'Debt Value': total_debt,
        'Total Value': total_value,
    }


def capm(rf: float, beta: float, rm: float) -> float:
    return rf + beta * (rm - rf)

# ==================== GRAHAM NUMBER ====================

def graham_number(eps: float, bvps: float) -> Dict:
    if eps <= 0 or bvps <= 0:
        return {}
    gn = np.sqrt(22.5 * eps * bvps)
    return {'Graham Number': gn, 'EPS': eps, 'Book Value per Share': bvps}

# ==================== BENEISH M-SCORE ====================

def beneish_m_score(curr: Dict, prev: Dict) -> Dict:
    """
    Beneish M-Score (8-variable model).
    M > -1.78 → possible earnings manipulation.
    
    Variables:
    DSRI  = Days Sales in Receivables Index
    GMI   = Gross Margin Index
    AQI   = Asset Quality Index
    SGI   = Sales Growth Index
    DEPI  = Depreciation Index
    SGAI  = SG&A Index
    LVGI  = Leverage Index
    TATA  = Total Accruals to Total Assets
    """
    def safe_div(a, b, default=1.0):
        return a / b if b and b != 0 else default

    # DSRI
    rec_curr = curr.get('receivables', 0)
    rev_curr = curr.get('revenue', 1)
    rec_prev = prev.get('receivables', 0)
    rev_prev = prev.get('revenue', 1)
    days_rec_curr = safe_div(rec_curr, rev_curr) * 365
    days_rec_prev = safe_div(rec_prev, rev_prev) * 365
    dsri = safe_div(days_rec_curr, days_rec_prev)

    # GMI
    gm_curr = safe_div(curr.get('gross_profit', 0), rev_curr)
    gm_prev = safe_div(prev.get('gross_profit', 0), rev_prev)
    gmi = safe_div(gm_prev, gm_curr)

    # AQI
    ta_curr = curr.get('total_assets', 1)
    ca_curr = curr.get('current_assets', 0)
    ppe_curr = curr.get('ppe', 0)
    ta_prev = prev.get('total_assets', 1)
    ca_prev = prev.get('current_assets', 0)
    ppe_prev = prev.get('ppe', 0)
    aqi = safe_div(1 - safe_div(ca_curr + ppe_curr, ta_curr),
                   1 - safe_div(ca_prev + ppe_prev, ta_prev))

    # SGI
    sgi = safe_div(rev_curr, rev_prev)

    # DEPI
    dep_curr = curr.get('depreciation', 0)
    dep_prev = prev.get('depreciation', 0)
    depi = safe_div(
        safe_div(dep_prev, dep_prev + ppe_prev) if (dep_prev + ppe_prev) else 0,
        safe_div(dep_curr, dep_curr + ppe_curr) if (dep_curr + ppe_curr) else 1
    )

    # SGAI
    sga_curr = curr.get('sga', 0)
    sga_prev = prev.get('sga', 0)
    sgai = safe_div(safe_div(sga_curr, rev_curr), safe_div(sga_prev, rev_prev))

    # LVGI
    ltd_curr = curr.get('long_term_debt', 0)
    ltd_prev = prev.get('long_term_debt', 0)
    cl_curr = curr.get('current_liabilities', 0)
    cl_prev = prev.get('current_liabilities', 0)
    lvgi = safe_div(
        safe_div(ltd_curr + cl_curr, ta_curr),
        safe_div(ltd_prev + cl_prev, ta_prev)
    )

    # TATA
    cfo = curr.get('cfo', 0)
    ni = curr.get('net_income', 0)
    tata = safe_div(ni - cfo, ta_curr)

    m = (-4.840
         + 0.920 * dsri
         + 0.528 * gmi
         + 0.404 * aqi
         + 0.892 * sgi
         + 0.115 * depi
         - 0.172 * sgai
         + 4.679 * tata
         - 0.327 * lvgi)

    if m > -1.78:
        status, color = "Possible Manipulation", "red"
    else:
        status, color = "Unlikely Manipulation", "green"

    return {
        'M-Score': m, 'Status': status, 'Color': color,
        'Components': {
            'DSRI (Receivables)': dsri, 'GMI (Gross Margin)': gmi,
            'AQI (Asset Quality)': aqi, 'SGI (Sales Growth)': sgi,
            'DEPI (Depreciation)': depi, 'SGAI (SG&A)': sgai,
            'LVGI (Leverage)': lvgi, 'TATA (Accruals)': tata,
        }
    }

# ==================== EV/EBITDA ====================

def ev_ebitda_analysis(market_cap, total_debt, cash, ebitda, revenue,
                       net_income, total_assets, sector='N/A') -> Dict:
    ev = market_cap + total_debt - cash
    ev_ebitda = ev / ebitda if ebitda > 0 else None
    ev_revenue = ev / revenue if revenue > 0 else None
    ev_assets = ev / total_assets if total_assets > 0 else None
    return {
        'Enterprise Value': ev,
        'EV/EBITDA': ev_ebitda,
        'EV/Revenue': ev_revenue,
        'EV/Assets': ev_assets,
        'EBITDA': ebitda,
        'EV': ev,
    }

# ==================== MAGIC FORMULA (GREENBLATT) ====================

def magic_formula(ebit, enterprise_value, ebit_net_working_capital, tangible_assets) -> Dict:
    """
    Greenblatt Magic Formula:
    Earnings Yield = EBIT / Enterprise Value
    Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
    """
    earnings_yield = ebit / enterprise_value if enterprise_value > 0 else None
    roc = ebit / tangible_assets if tangible_assets > 0 else None
    return {
        'Earnings Yield': earnings_yield * 100 if earnings_yield else None,
        'Return on Capital': roc * 100 if roc else None,
    }

# ==================== VISUALIZATIONS ====================

def gauge_chart(value: float, title: str, min_val: float, max_val: float,
                thresholds: List[Tuple], unit: str = '') -> go.Figure:
    """Generic gauge chart."""
    steps = []
    prev = min_val
    for thresh, color in thresholds:
        steps.append({'range': [prev, thresh], 'color': color})
        prev = thresh
    steps.append({'range': [prev, max_val], 'color': thresholds[-1][1]})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': unit, 'font': {'size': 28}},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': '#2c3e50', 'thickness': 0.25},
            'steps': steps,
            'borderwidth': 2,
            'bordercolor': '#ccc',
        }
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def piotroski_bar(details: Dict) -> go.Figure:
    cats = list(details.keys())
    colors = ['#27ae60' if details[c]['pass'] else '#e74c3c' for c in cats]
    fig = go.Figure(go.Bar(
        x=cats, y=[details[c]['score'] for c in cats],
        marker_color=colors,
        text=[details[c]['value'] for c in cats],
        textposition='outside',
        width=0.6,
    ))
    fig.update_layout(
        title='Piotroski F-Score — Component Pass/Fail',
        yaxis=dict(range=[0, 1.5], title='Pass (1) / Fail (0)'),
        xaxis=dict(tickangle=-30),
        template='plotly_white', height=420,
        showlegend=False,
    )
    fig.add_vline(x=3.5, line_dash='dash', line_color='#aaa', annotation_text='Leverage →', annotation_position='top right')
    fig.add_vline(x=6.5, line_dash='dash', line_color='#aaa', annotation_text='Efficiency →', annotation_position='top right')
    return fig


def wacc_waterfall(wacc_data: Dict) -> go.Figure:
    ew = wacc_data['Equity Weight'] / 100
    dw = wacc_data['Debt Weight'] / 100
    re = wacc_data['Cost of Equity'] / 100
    rd_at = wacc_data['After-Tax Cost of Debt'] / 100
    equity_contrib = ew * re * 100
    debt_contrib = dw * rd_at * 100

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=['relative', 'relative', 'total'],
        x=['Equity Component\n(w×Re)', 'Debt Component\n(w×Rd×(1-T))', 'WACC'],
        y=[equity_contrib, debt_contrib, 0],
        text=[f'{equity_contrib:.2f}%', f'{debt_contrib:.2f}%', f'{wacc_data["WACC"]:.2f}%'],
        textposition='outside',
        connector={'line': {'color': '#636363'}},
        increasing={'marker': {'color': '#3498db'}},
        totals={'marker': {'color': '#2c3e50'}},
    ))
    fig.update_layout(title='WACC Build-Up', template='plotly_white', height=380,
                      yaxis_title='Rate (%)')
    return fig


def dupont_tree(d3: Dict, d5: Optional[Dict] = None) -> go.Figure:
    """Sankey-style DuPont breakdown."""
    labels = ['ROE', 'Profit Margin', 'Asset Turnover', 'Equity Multiplier']
    values = [abs(d3['ROE']), abs(d3['Profit Margin']), abs(d3['Asset Turnover'])*10, abs(d3['Equity Multiplier'])*10]
    colors = ['#2c3e50', '#3498db', '#27ae60', '#e67e22']

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{d3['ROE']:.2f}%", f"{d3['Profit Margin']:.2f}%",
              f"{d3['Asset Turnover']:.2f}x", f"{d3['Equity Multiplier']:.2f}x"],
        textposition='outside',
    ))
    fig.update_layout(title='3-Way DuPont Components (normalised for comparison)',
                      template='plotly_white', height=380, showlegend=False,
                      yaxis_title='Normalised Value')
    return fig


def beneish_radar(components: Dict) -> go.Figure:
    labels = list(components.keys())
    vals = [abs(v) for v in components.values()]
    # close the polygon
    labels_closed = labels + [labels[0]]
    vals_closed = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=labels_closed, fill='toself',
        fillcolor='rgba(231,76,60,0.25)', line_color='#e74c3c',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title='Beneish M-Score Variable Profile',
        template='plotly_white', height=420,
    )
    return fig

# ==================== SCORE DISPLAY HELPERS ====================

def score_badge(label: str, value: str, color: str):
    color_map = {'green': 'badge-green', 'orange': 'badge-orange', 'red': 'badge-red'}
    css = color_map.get(color, 'badge-green')
    st.markdown(f'<span class="score-badge {css}">{label}: {value}</span>', unsafe_allow_html=True)


def render_score_summary(title: str, score_str: str, assessment: str, color: str, explanation: str):
    color_hex = {'green': '#27ae60', 'orange': '#f39c12', 'red': '#e74c3c'}.get(color, '#2c3e50')
    bg = {'green': '#d4edda', 'orange': '#fff3cd', 'red': '#f8d7da'}.get(color, '#f8f9fa')
    st.markdown(f"""
    <div style="background:{bg}; border-left:6px solid {color_hex};
                padding:16px 20px; border-radius:8px; margin:10px 0;">
        <div style="font-size:1.5em; font-weight:800; color:{color_hex};">{title}: {score_str}</div>
        <div style="font-size:1.1em; font-weight:600; color:#333; margin-top:4px;">{assessment}</div>
        <div style="color:#555; margin-top:6px;">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN MODULE ====================

def advanced_models_module(analysis_context: Optional[Dict] = None):
    st.title("🔬 Advanced Fundamental Models")
    st.markdown("Deep-dive financial analysis: DuPont · Altman Z · Piotroski F · WACC · Graham · Beneish M · EV/EBITDA · Magic Formula")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.text_input("Ticker Symbol", value="AAPL",
                               help="NYSE / NASDAQ ticker, e.g. AAPL, MSFT, TSLA").upper().strip()

        all_models = [
            "DuPont Analysis",
            "Altman Z-Score",
            "Piotroski F-Score",
            "WACC Analysis",
            "Graham Number",
            "Beneish M-Score",
            "EV/EBITDA Multiples",
            "Magic Formula (Greenblatt)",
        ]
        selected_models = st.multiselect("Models to run", all_models, default=all_models[:5])

        st.divider()
        st.subheader("WACC Inputs")
        rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.5, 0.1) / 100
        mkt_return = st.number_input("Market Return (%)", 0.0, 30.0, 10.0, 0.5) / 100
        custom_tax = st.number_input("Tax Rate (%)", 0.0, 50.0, 21.0, 1.0) / 100

        run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    if not run_btn:
        st.info("👈 Configure settings in the sidebar and press **Run Analysis**.")
        return

    # ---- Fetch data ----
    with st.spinner(f"Fetching data for **{ticker}** …"):
        data = fetch_financial_data(ticker)

    if not data:
        st.error(f"❌ Could not fetch data for **{ticker}**. Check the ticker and try again.")
        return

    info = data['info']
    inc = data['income_statement']
    bs = data['balance_sheet']
    cf = data['cash_flow']

    # ---- Company header ----
    name = info.get('longName', ticker)
    sector = info.get('sector', 'N/A')
    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
    market_cap = info.get('marketCap', 0)
    beta_val = info.get('beta', 1.0) or 1.0

    st.header(f"📊 {name}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sector", sector)
    c2.metric("Market Cap", fmt_billions(market_cap))
    c3.metric("Price", f"${current_price:.2f}")
    c4.metric("P/E (TTM)", f"{info.get('trailingPE', 0):.1f}" if info.get('trailingPE') else "N/A")
    c5.metric("Beta", f"{beta_val:.2f}")
    st.divider()

    # ---- Extract financials ----
    try:
        # Current period (col 0)
        net_income   = safe_get(inc, 'Net Income', 0)
        revenue      = safe_get(inc, 'Total Revenue', 0)
        ebit         = safe_get(inc, 'EBIT', 0)
        ebt          = safe_get(inc, 'Pretax Income', 0)
        gross_profit = safe_get(inc, 'Gross Profit', 0)
        int_expense  = abs(safe_get(inc, 'Interest Expense', 0))
        sga          = abs(safe_get(inc, 'Selling General Administrative', 0))

        total_assets   = safe_get(bs, 'Total Assets', 0)
        total_equity   = safe_get(bs, 'Stockholders Equity', 0)
        current_assets = safe_get(bs, 'Current Assets', 0)
        current_liab   = safe_get(bs, 'Current Liabilities', 0)
        total_liab     = safe_get(bs, 'Total Liabilities Net Minority Interest', 0)
        long_term_debt = safe_get(bs, 'Long Term Debt', 0)
        retained_earn  = safe_get(bs, 'Retained Earnings', 0)
        cash           = safe_get(bs, 'Cash And Cash Equivalents', 0)
        receivables    = safe_get(bs, 'Receivables', 0)
        ppe            = safe_get(bs, 'Net PPE', 0)
        depreciation   = abs(safe_get(cf, 'Depreciation And Amortization', 0))

        operating_cf   = safe_get(cf, 'Operating Cash Flow', 0)
        working_capital = current_assets - current_liab

        # Prior period (col 1)
        ni_prev       = safe_get(inc, 'Net Income', 1)
        rev_prev      = safe_get(inc, 'Total Revenue', 1)
        gp_prev       = safe_get(inc, 'Gross Profit', 1)
        sga_prev      = abs(safe_get(inc, 'Selling General Administrative', 1))
        ta_prev       = safe_get(bs, 'Total Assets', 1)
        ltd_prev      = safe_get(bs, 'Long Term Debt', 1)
        ca_prev       = safe_get(bs, 'Current Assets', 1)
        cl_prev       = safe_get(bs, 'Current Liabilities', 1)
        rec_prev      = safe_get(bs, 'Receivables', 1)
        ppe_prev      = safe_get(bs, 'Net PPE', 1)
        dep_prev      = abs(safe_get(cf, 'Depreciation And Amortization', 1))
        cfo_prev      = safe_get(cf, 'Operating Cash Flow', 1)

        shares_out    = info.get('sharesOutstanding', 0)

        # Derived
        ebitda = ebit + depreciation

    except Exception as e:
        st.error(f"Error extracting financials: {e}")
        return

    # ===========================================================
    # DUPONT
    # ===========================================================
    if "DuPont Analysis" in selected_models:
        st.subheader("📊 DuPont Analysis")
        d3 = dupont_analysis_3way(net_income, revenue, total_assets, total_equity)
        d5 = dupont_analysis_5way(net_income, ebt, ebit, revenue, total_assets, total_equity)

        if d3:
            render_score_summary(
                "ROE (3-Way)", f"{d3['ROE']:.2f}%",
                f"Profit Margin {d3['Profit Margin']:.2f}% × Asset Turnover {d3['Asset Turnover']:.2f}x × Equity Multiplier {d3['Equity Multiplier']:.2f}x",
                'green' if d3['ROE'] > 15 else 'orange' if d3['ROE'] > 5 else 'red',
                "ROE above 15% is generally considered strong for most sectors."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**3-Way Components**")
                st.metric("Profit Margin", f"{d3['Profit Margin']:.2f}%")
                st.metric("Asset Turnover", f"{d3['Asset Turnover']:.2f}x")
                st.metric("Equity Multiplier", f"{d3['Equity Multiplier']:.2f}x")
            with col2:
                if d5:
                    st.markdown("**5-Way Components**")
                    st.metric("Tax Burden", f"{d5['Tax Burden']:.3f}")
                    st.metric("Interest Burden", f"{d5['Interest Burden']:.3f}")
                    st.metric("EBIT Margin", f"{d5['EBIT Margin']:.2f}%")
                    st.metric("Asset Turnover", f"{d5['Asset Turnover']:.2f}x")
                    st.metric("Equity Multiplier", f"{d5['Equity Multiplier']:.2f}x")

            st.plotly_chart(dupont_tree(d3, d5), use_container_width=True)
        else:
            st.warning("Insufficient data for DuPont analysis.")
        st.divider()

    # ===========================================================
    # ALTMAN Z-SCORE
    # ===========================================================
    if "Altman Z-Score" in selected_models:
        st.subheader("🚨 Altman Z-Score")
        altman = altman_z_score_public(working_capital, retained_earn, ebit,
                                       market_cap, total_liab, revenue, total_assets)
        if altman:
            render_score_summary(
                "Altman Z-Score", f"{altman['Z-Score']:.2f}",
                f"{altman['Zone']}  ·  Bankruptcy Risk: {altman['Bankruptcy Risk']}",
                altman['Color'],
                "Z > 2.99 = Safe · 1.81–2.99 = Grey Zone · < 1.81 = Distress"
            )
            fig_gauge = gauge_chart(
                altman['Z-Score'], 'Altman Z-Score', 0, 5,
                [(1.81, '#f8d7da'), (2.99, '#fff3cd')], ''
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                comp = altman['Components']
                wts = altman['Weights']
                keys = list(comp.keys())
                wt_vals = list(wts.values())
                comp_vals = list(comp.values())
                contrib = [w * v for w, v in zip(wt_vals, comp_vals)]
                df_comp = pd.DataFrame({
                    'Variable': keys,
                    'Raw Value': comp_vals,
                    'Weight': wt_vals,
                    'Contribution': contrib
                })
                st.dataframe(
                    df_comp.style.format({'Raw Value': '{:.4f}', 'Weight': '{:.2f}', 'Contribution': '{:.4f}'})
                                 .background_gradient(subset=['Contribution'], cmap='RdYlGn'),
                    use_container_width=True, hide_index=True
                )
        else:
            st.warning("Insufficient data for Altman Z-Score.")
        st.divider()

    # ===========================================================
    # PIOTROSKI F-SCORE
    # ===========================================================
    if "Piotroski F-Score" in selected_models:
        st.subheader("📈 Piotroski F-Score")

        curr_data = {
            'roa': net_income / total_assets if total_assets else 0,
            'cfo': operating_cf, 'net_income': net_income,
            'long_term_debt': long_term_debt,
            'current_ratio': current_assets / current_liab if current_liab else 0,
            'shares_outstanding': shares_out,
            'gross_margin': gross_profit / revenue if revenue else 0,
            'asset_turnover': revenue / total_assets if total_assets else 0,
        }
        prev_data = {
            'roa': ni_prev / ta_prev if ta_prev else 0,
            'long_term_debt': ltd_prev,
            'current_ratio': ca_prev / cl_prev if cl_prev else 0,
            'shares_outstanding': shares_out * 1.01,
            'gross_margin': gp_prev / rev_prev if rev_prev else 0,
            'asset_turnover': rev_prev / ta_prev if ta_prev else 0,
        }

        pio = piotroski_f_score(curr_data, prev_data)

        render_score_summary(
            "Piotroski F-Score", f"{pio['F-Score']} / 9",
            pio['Assessment'],
            pio['Color'],
            "8–9 = Strong (potential value play) · 5–7 = Moderate · 0–4 = Weak / high risk"
        )

        # Detailed component table
        details = pio['Details']
        rows = []
        for name_k, d in details.items():
            rows.append({
                'Category': d['category'],
                'Signal': name_k,
                'Value': d['value'],
                'Pass': '✅' if d['pass'] else '❌',
                'Score': d['score'],
            })
        df_pio = pd.DataFrame(rows)

        col1, col2 = st.columns([2, 3])
        with col1:
            # Score breakdown by category
            by_cat = df_pio.groupby('Category')['Score'].agg(['sum', 'count']).reset_index()
            by_cat.columns = ['Category', 'Scored', 'Max']
            for _, row in by_cat.iterrows():
                pct = row['Scored'] / row['Max']
                color_cat = 'green' if pct >= 0.67 else 'orange' if pct >= 0.34 else 'red'
                score_badge(row['Category'], f"{int(row['Scored'])}/{int(row['Max'])}", color_cat)
                st.markdown("")
        with col2:
            st.dataframe(df_pio[['Category', 'Signal', 'Value', 'Pass', 'Score']],
                         use_container_width=True, hide_index=True)

        st.plotly_chart(piotroski_bar(details), use_container_width=True)
        st.divider()

    # ===========================================================
    # WACC
    # ===========================================================
    if "WACC Analysis" in selected_models:
        st.subheader("💰 WACC — Weighted Average Cost of Capital")

        cost_eq = capm(rf_rate, beta_val, mkt_return)
        cost_dt = int_expense / long_term_debt if long_term_debt > 0 else 0.05

        wacc_res = calculate_wacc(market_cap, long_term_debt, cost_eq, cost_dt, custom_tax)

        if wacc_res:
            render_score_summary(
                "WACC", f"{wacc_res['WACC']:.2f}%",
                f"Cost of Equity {wacc_res['Cost of Equity']:.2f}%  ·  After-Tax Cost of Debt {wacc_res['After-Tax Cost of Debt']:.2f}%",
                'green' if wacc_res['WACC'] < 8 else 'orange' if wacc_res['WACC'] < 12 else 'red',
                "Projects must earn above WACC to create shareholder value. Lower WACC = cheaper capital."
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("WACC", f"{wacc_res['WACC']:.2f}%")
            c2.metric("Cost of Equity (CAPM)", f"{wacc_res['Cost of Equity']:.2f}%")
            c3.metric("After-Tax Cost of Debt", f"{wacc_res['After-Tax Cost of Debt']:.2f}%")
            c4.metric("Equity / Debt Split", f"{wacc_res['Equity Weight']:.0f}% / {wacc_res['Debt Weight']:.0f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(wacc_waterfall(wacc_res), use_container_width=True)
            with col2:
                fig_pie = go.Figure(go.Pie(
                    labels=['Equity', 'Debt'],
                    values=[wacc_res['Equity Weight'], wacc_res['Debt Weight']],
                    hole=0.45,
                    marker_colors=['#3498db', '#e74c3c'],
                ))
                fig_pie.update_layout(title='Capital Structure', height=380,
                                      template='plotly_white',
                                      annotations=[dict(text=f"{wacc_res['WACC']:.2f}%\nWACC",
                                                        x=0.5, y=0.5, font_size=14, showarrow=False)])
                st.plotly_chart(fig_pie, use_container_width=True)

            with st.expander("🔍 Detailed WACC Calculation"):
                st.markdown(f"""
| | Equity | Debt | **Total** |
|---|---|---|---|
| **Value** | {fmt_billions(wacc_res['Equity Value'])} | {fmt_billions(wacc_res['Debt Value'])} | {fmt_billions(wacc_res['Total Value'])} |
| **Weight** | {wacc_res['Equity Weight']:.2f}% | {wacc_res['Debt Weight']:.2f}% | 100% |
| **Pre-tax Cost** | {wacc_res['Cost of Equity']:.2f}% | {wacc_res['Cost of Debt']:.2f}% | – |
| **After-tax Cost** | {wacc_res['Cost of Equity']:.2f}% | {wacc_res['After-Tax Cost of Debt']:.2f}% | – |
| **WACC Contribution** | {wacc_res['Equity Weight']/100*wacc_res['Cost of Equity']:.2f}% | {wacc_res['Debt Weight']/100*wacc_res['After-Tax Cost of Debt']:.2f}% | **{wacc_res['WACC']:.2f}%** |

CAPM: Re = {rf_rate*100:.2f}% + {beta_val:.2f} × ({mkt_return*100:.2f}% − {rf_rate*100:.2f}%) = **{cost_eq*100:.2f}%**
                """)
        else:
            st.warning("Insufficient data for WACC calculation.")
        st.divider()

    # ===========================================================
    # GRAHAM NUMBER
    # ===========================================================
    if "Graham Number" in selected_models:
        st.subheader("💎 Graham Number")
        eps = info.get('trailingEps', 0) or 0
        bvps = info.get('bookValue', 0) or 0
        graham = graham_number(eps, bvps)

        if graham:
            gn = graham['Graham Number']
            upside = (gn - current_price) / current_price * 100 if current_price else 0
            color = 'green' if upside > 25 else 'orange' if upside > 0 else 'red'
            render_score_summary(
                "Graham Number", f"${gn:.2f}",
                f"Current Price ${current_price:.2f}  ·  Margin of Safety {upside:+.1f}%",
                color,
                "Buy when price < Graham Number with ≥25% margin of safety (Benjamin Graham)."
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Graham Fair Value", f"${gn:.2f}")
            c2.metric("Current Price", f"${current_price:.2f}")
            c3.metric("Margin of Safety", f"{upside:+.1f}%",
                      delta_color="normal" if upside > 0 else "inverse")

            # Visual price comparison
            fig_gn = go.Figure()
            fig_gn.add_trace(go.Bar(x=['Graham Number', 'Current Price'],
                                    y=[gn, current_price],
                                    marker_color=['#27ae60', '#3498db'],
                                    text=[f'${gn:.2f}', f'${current_price:.2f}'],
                                    textposition='outside'))
            fig_gn.update_layout(title='Graham Number vs Current Price',
                                  template='plotly_white', height=340,
                                  yaxis_title='Price ($)')
            st.plotly_chart(fig_gn, use_container_width=True)
        else:
            st.warning("⚠️ Graham Number requires positive EPS and book value per share.")
        st.divider()

    # ===========================================================
    # BENEISH M-SCORE
    # ===========================================================
    if "Beneish M-Score" in selected_models:
        st.subheader("🕵️ Beneish M-Score — Earnings Manipulation Detection")

        curr_b = {
            'receivables': receivables, 'revenue': revenue, 'gross_profit': gross_profit,
            'current_assets': current_assets, 'ppe': ppe, 'total_assets': total_assets,
            'depreciation': depreciation, 'sga': sga,
            'long_term_debt': long_term_debt, 'current_liabilities': current_liab,
            'cfo': operating_cf, 'net_income': net_income,
        }
        prev_b = {
            'receivables': rec_prev, 'revenue': rev_prev, 'gross_profit': gp_prev,
            'current_assets': ca_prev, 'ppe': ppe_prev, 'total_assets': ta_prev,
            'depreciation': dep_prev, 'sga': sga_prev,
            'long_term_debt': ltd_prev, 'current_liabilities': cl_prev,
            'cfo': cfo_prev, 'net_income': ni_prev,
        }

        m_res = beneish_m_score(curr_b, prev_b)

        render_score_summary(
            "M-Score", f"{m_res['M-Score']:.3f}",
            m_res['Status'],
            m_res['Color'],
            "M > −1.78 signals possible earnings manipulation (Beneish 1999). Use alongside other checks."
        )

        c1, c2 = st.columns(2)
        with c1:
            comp = m_res['Components']
            df_m = pd.DataFrame({'Variable': list(comp.keys()), 'Value': list(comp.values())})
            st.dataframe(df_m.style.format({'Value': '{:.4f}'}), use_container_width=True, hide_index=True)
        with c2:
            st.plotly_chart(beneish_radar(comp), use_container_width=True)
        st.divider()

    # ===========================================================
    # EV/EBITDA
    # ===========================================================
    if "EV/EBITDA Multiples" in selected_models:
        st.subheader("📐 EV / EBITDA & Enterprise Multiples")

        ev_res = ev_ebitda_analysis(market_cap, long_term_debt, cash, ebitda,
                                    revenue, net_income, total_assets, sector)

        if ebitda > 0:
            ev_mult = ev_res['EV/EBITDA']
            color_ev = 'green' if ev_mult and ev_mult < 12 else 'orange' if ev_mult and ev_mult < 20 else 'red'
            render_score_summary(
                "EV/EBITDA", f"{ev_mult:.1f}x" if ev_mult else "N/A",
                f"Enterprise Value {fmt_billions(ev_res['Enterprise Value'])}  ·  EBITDA {fmt_billions(ebitda)}",
                color_ev,
                "EV/EBITDA < 10x often considered cheap; > 20x expensive (varies by sector & growth)."
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Enterprise Value", fmt_billions(ev_res['Enterprise Value']))
        c2.metric("EBITDA", fmt_billions(ev_res['EBITDA']))
        c3.metric("EV/EBITDA", f"{ev_res['EV/EBITDA']:.1f}x" if ev_res['EV/EBITDA'] else "N/A")
        c4.metric("EV/Revenue", f"{ev_res['EV/Revenue']:.2f}x" if ev_res['EV/Revenue'] else "N/A")

        # Waterfall: EV bridge
        fig_ev = go.Figure(go.Waterfall(
            orientation='v',
            measure=['relative', 'relative', 'relative', 'total'],
            x=['Market Cap', '+ Total Debt', '− Cash & Equiv', 'Enterprise Value'],
            y=[market_cap, long_term_debt, -cash, 0],
            text=[fmt_billions(market_cap), fmt_billions(long_term_debt),
                  f'−{fmt_billions(cash)}', fmt_billions(ev_res['Enterprise Value'])],
            textposition='outside',
            connector={'line': {'color': '#636363'}},
            increasing={'marker': {'color': '#3498db'}},
            decreasing={'marker': {'color': '#e74c3c'}},
            totals={'marker': {'color': '#2c3e50'}},
        ))
        fig_ev.update_layout(title='Enterprise Value Bridge', template='plotly_white', height=380)
        st.plotly_chart(fig_ev, use_container_width=True)
        st.divider()

    # ===========================================================
    # MAGIC FORMULA
    # ===========================================================
    if "Magic Formula (Greenblatt)" in selected_models:
        st.subheader("🧙 Magic Formula — Greenblatt")

        tangible = (working_capital if working_capital > 0 else 0) + ppe
        ev_for_magic = market_cap + long_term_debt - cash
        mf = magic_formula(ebit, ev_for_magic, working_capital, tangible)

        if mf['Earnings Yield'] is not None and mf['Return on Capital'] is not None:
            ey = mf['Earnings Yield']
            roc = mf['Return on Capital']
            color_mf = 'green' if ey > 8 and roc > 15 else 'orange' if ey > 4 else 'red'
            render_score_summary(
                "Magic Formula",
                f"EY {ey:.1f}%  ·  ROC {roc:.1f}%",
                "High Earnings Yield + High Return on Capital = Greenblatt's ideal",
                color_mf,
                "Earnings Yield > 8% and ROC > 15% are typical thresholds for inclusion."
            )
            c1, c2 = st.columns(2)
            c1.metric("Earnings Yield (EBIT/EV)", f"{ey:.2f}%")
            c2.metric("Return on Capital (EBIT/Tangible)", f"{roc:.2f}%")

            fig_mf = go.Figure()
            fig_mf.add_trace(go.Bar(x=['Earnings Yield', 'Return on Capital'],
                                    y=[ey, roc],
                                    marker_color=['#3498db', '#27ae60'],
                                    text=[f'{ey:.2f}%', f'{roc:.2f}%'],
                                    textposition='outside'))
            fig_mf.add_hline(y=8, line_dash='dash', line_color='gray',
                             annotation_text='EY threshold (8%)')
            fig_mf.add_hline(y=15, line_dash='dot', line_color='gray',
                              annotation_text='ROC threshold (15%)')
            fig_mf.update_layout(title='Magic Formula Metrics', template='plotly_white',
                                  height=360, yaxis_title='%')
            st.plotly_chart(fig_mf, use_container_width=True)
        else:
            st.warning("Insufficient data for Magic Formula calculation.")
        st.divider()

    # ===========================================================
    # EDUCATION EXPANDER
    # ===========================================================
    with st.expander("📚 Model Reference Guide"):
        st.markdown("""
| Model | Purpose | Key Threshold |
|---|---|---|
| **DuPont (3-Way)** | Decompose ROE into margin × turnover × leverage | ROE > 15% = strong |
| **DuPont (5-Way)** | Add tax & interest burden dimensions | Identify drag on ROE |
| **Altman Z-Score** | Predict bankruptcy probability | > 2.99 safe, < 1.81 distress |
| **Piotroski F-Score** | 9-point quality screen | 8–9 strong, 0–4 weak |
| **WACC** | Cost of capital / DCF discount rate | Lower is better |
| **Graham Number** | Value investing fair price | Buy below with ≥25% margin |
| **Beneish M-Score** | Earnings manipulation detection | M > −1.78 = possible fraud |
| **EV/EBITDA** | Capital-structure-neutral valuation | < 10x cheap, > 20x pricey |
| **Magic Formula** | Combined quality + value screen | EY > 8%, ROC > 15% |
        """)

    st.caption("🔬 Advanced Models Module — all figures sourced from Yahoo Finance via yfinance. Not financial advice.")


if __name__ == "__main__":
    advanced_models_module(None)
