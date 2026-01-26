# fixed_income_improved.py - Enhanced Fixed Income / Bond Calculations Module
# Comprehensive bond pricing, analytics, yield curves, and portfolio management
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import fsolve, minimize_scalar
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ==================== CORE BOND CALCULATIONS ====================

def bond_price(par: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """
    Calculate bond price using present value of cash flows.
    
    Args:
        par: Par/face value of bond
        coupon_rate: Annual coupon rate (as decimal)
        years: Years to maturity
        ytm: Yield to maturity (as decimal)
        freq: Payment frequency per year (2=semi-annual, 4=quarterly, 12=monthly)
    
    Returns:
        Bond price
    """
    if years <= 0:
        return par
    
    periods = int(years * freq)
    coupon = par * coupon_rate / freq
    ytm_period = ytm / freq
    
    if ytm_period == 0:
        return par + coupon * periods
    
    # Present value of coupons
    pv_coupons = coupon * (1 - (1 + ytm_period)**(-periods)) / ytm_period
    
    # Present value of par
    pv_par = par / (1 + ytm_period)**periods
    
    return pv_coupons + pv_par

def bond_ytm(par: float, price: float, coupon_rate: float, years: float, freq: int = 2, guess: float = 0.05) -> float:
    """
    Calculate yield to maturity using numerical optimization.
    
    Args:
        par: Par value
        price: Current market price
        coupon_rate: Annual coupon rate
        years: Years to maturity
        freq: Payment frequency
        guess: Initial guess for YTM
    
    Returns:
        Yield to maturity
    """
    def objective(y):
        return bond_price(par, coupon_rate, years, y, freq) - price
    
    try:
        result = fsolve(objective, guess, full_output=True)
        if result[2] == 1:  # Solution found
            return float(result[0][0])
        else:
            # Fallback to bounded optimization
            result = minimize_scalar(lambda y: abs(objective(y)), bounds=(0.0001, 1.0), method='bounded')
            return result.x
    except:
        return np.nan

def bond_duration(par: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """
    Calculate Macaulay duration.
    
    Returns:
        Duration in years
    """
    if years <= 0:
        return 0.0
    
    periods = int(years * freq)
    coupon = par * coupon_rate / freq
    ytm_period = ytm / freq
    price = bond_price(par, coupon_rate, years, ytm, freq)
    
    if price == 0:
        return 0.0
    
    # Weighted present value of cash flows
    weighted_sum = 0
    for t in range(1, periods + 1):
        pv = coupon / (1 + ytm_period)**t
        weighted_sum += (t / freq) * pv
    
    # Add principal payment
    pv_par = par / (1 + ytm_period)**periods
    weighted_sum += years * pv_par
    
    return weighted_sum / price

def modified_duration(par: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """Calculate modified duration (price sensitivity to yield changes)."""
    mac_dur = bond_duration(par, coupon_rate, years, ytm, freq)
    return mac_dur / (1 + ytm / freq)

def bond_convexity(par: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """
    Calculate bond convexity (second-order price sensitivity).
    
    Returns:
        Convexity
    """
    if years <= 0:
        return 0.0
    
    periods = int(years * freq)
    coupon = par * coupon_rate / freq
    ytm_period = ytm / freq
    price = bond_price(par, coupon_rate, years, ytm, freq)
    
    if price == 0:
        return 0.0
    
    weighted_sum = 0
    for t in range(1, periods + 1):
        pv = coupon / (1 + ytm_period)**t
        weighted_sum += (t / freq) * (t / freq + 1 / freq) * pv
    
    # Add principal payment
    pv_par = par / (1 + ytm_period)**periods
    weighted_sum += years * (years + 1 / freq) * pv_par
    
    return weighted_sum / (price * (1 + ytm_period)**2)

def dv01(par: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """
    Calculate DV01 (dollar value of 1 basis point).
    
    Returns:
        Dollar change for 1bp yield change
    """
    price_base = bond_price(par, coupon_rate, years, ytm, freq)
    price_up = bond_price(par, coupon_rate, years, ytm + 0.0001, freq)
    return abs(price_base - price_up)

def current_yield(par: float, coupon_rate: float, price: float) -> float:
    """Calculate current yield (annual coupon / current price)."""
    if price == 0:
        return 0.0
    return (par * coupon_rate) / price

def accrued_interest(par: float, coupon_rate: float, freq: int, days_since_last: int) -> float:
    """
    Calculate accrued interest since last coupon payment.
    
    Args:
        days_since_last: Days since last coupon payment
    """
    days_in_period = 365 / freq
    coupon = par * coupon_rate / freq
    return coupon * (days_since_last / days_in_period)

# ==================== ADVANCED ANALYTICS ====================

def price_yield_sensitivity(par: float, coupon_rate: float, years: float, base_ytm: float, 
                           freq: int = 2, yield_range: float = 0.03) -> pd.DataFrame:
    """
    Generate price-yield sensitivity table.
    
    Args:
        yield_range: Range of yield changes (e.g., 0.03 = +/- 3%)
    
    Returns:
        DataFrame with yield changes and corresponding prices
    """
    yields = np.linspace(base_ytm - yield_range, base_ytm + yield_range, 21)
    prices = [bond_price(par, coupon_rate, years, y, freq) for y in yields]
    
    base_price = bond_price(par, coupon_rate, years, base_ytm, freq)
    price_changes = [(p - base_price) / base_price * 100 for p in prices]
    yield_changes = [(y - base_ytm) * 100 for y in yields]
    
    return pd.DataFrame({
        'Yield (%)': [y * 100 for y in yields],
        'Yield Change (bp)': [int(yc * 100) for yc in yield_changes],
        'Price': prices,
        'Price Change (%)': price_changes
    })

def immunization_analysis(target_duration: float, available_bonds: List[Dict]) -> Dict:
    """
    Find bond portfolio that matches target duration (immunization strategy).
    
    Args:
        target_duration: Desired portfolio duration
        available_bonds: List of bond dictionaries with keys: par, coupon_rate, years, ytm, freq
    
    Returns:
        Dictionary with recommended weights
    """
    durations = []
    for bond in available_bonds:
        dur = bond_duration(bond['par'], bond['coupon_rate'], bond['years'], 
                           bond['ytm'], bond.get('freq', 2))
        durations.append(dur)
    
    # Simple two-bond immunization (can be extended to N bonds)
    if len(durations) >= 2:
        # Find weights that match target duration
        # w1 * D1 + w2 * D2 = Dtarget, where w1 + w2 = 1
        w1 = (target_duration - durations[1]) / (durations[0] - durations[1])
        w2 = 1 - w1
        
        return {
            'weights': [w1, w2] + [0] * (len(durations) - 2),
            'portfolio_duration': target_duration,
            'individual_durations': durations
        }
    
    return {}

def yield_curve_interpolation(maturities: List[float], yields: List[float], 
                              target_maturities: np.ndarray) -> np.ndarray:
    """
    Interpolate yield curve using cubic spline.
    
    Args:
        maturities: Known maturities
        yields: Known yields
        target_maturities: Maturities to interpolate
    
    Returns:
        Interpolated yields
    """
    from scipy.interpolate import CubicSpline
    
    cs = CubicSpline(maturities, yields)
    return cs(target_maturities)

# ==================== VISUALIZATION FUNCTIONS ====================

def create_price_yield_chart(par: float, coupon_rate: float, years: float, 
                            current_ytm: float, freq: int = 2) -> go.Figure:
    """Create interactive price-yield relationship chart."""
    ytms = np.linspace(0.001, 0.15, 100)
    prices = [bond_price(par, coupon_rate, years, y, freq) for y in ytms]
    current_price = bond_price(par, coupon_rate, years, current_ytm, freq)
    
    fig = go.Figure()
    
    # Price-yield curve
    fig.add_trace(go.Scatter(
        x=ytms * 100,
        y=prices,
        mode='lines',
        name='Price-Yield Curve',
        line=dict(color='blue', width=3),
        hovertemplate='YTM: %{x:.2f}%<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Current position
    fig.add_trace(go.Scatter(
        x=[current_ytm * 100],
        y=[current_price],
        mode='markers',
        name='Current Position',
        marker=dict(size=12, color='red', symbol='star'),
        hovertemplate='Current YTM: %{x:.2f}%<br>Current Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Par value reference line
    fig.add_hline(y=par, line_dash="dash", line_color="green", 
                  annotation_text="Par Value", annotation_position="right")
    
    fig.update_layout(
        title='Bond Price vs Yield to Maturity',
        xaxis_title='Yield to Maturity (%)',
        yaxis_title='Bond Price ($)',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        height=500
    )
    
    return fig

def create_duration_convexity_chart(par: float, coupon_rate: float, years: float, 
                                   ytm: float, freq: int = 2) -> go.Figure:
    """Create chart showing duration and convexity approximations."""
    base_price = bond_price(par, coupon_rate, years, ytm, freq)
    mod_dur = modified_duration(par, coupon_rate, years, ytm, freq)
    conv = bond_convexity(par, coupon_rate, years, ytm, freq)
    
    yield_changes = np.linspace(-0.03, 0.03, 100)
    
    # Actual prices
    actual_prices = [bond_price(par, coupon_rate, years, ytm + dy, freq) for dy in yield_changes]
    
    # Duration approximation
    duration_prices = [base_price * (1 - mod_dur * dy) for dy in yield_changes]
    
    # Duration + Convexity approximation
    dur_conv_prices = [base_price * (1 - mod_dur * dy + 0.5 * conv * dy**2) for dy in yield_changes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yield_changes * 100,
        y=actual_prices,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=yield_changes * 100,
        y=duration_prices,
        mode='lines',
        name='Duration Approximation',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=yield_changes * 100,
        y=dur_conv_prices,
        mode='lines',
        name='Duration + Convexity',
        line=dict(color='green', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title='Price Change Approximations: Duration vs Convexity',
        xaxis_title='Yield Change (bp)',
        yaxis_title='Bond Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_cash_flow_diagram(par: float, coupon_rate: float, years: float, freq: int = 2) -> go.Figure:
    """Create cash flow timeline diagram."""
    periods = int(years * freq)
    times = np.arange(1, periods + 1) / freq
    coupon = par * coupon_rate / freq
    
    cash_flows = [coupon] * periods
    cash_flows[-1] += par  # Add principal to last payment
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=times,
        y=cash_flows,
        name='Cash Flows',
        marker_color=['lightblue' if i < periods - 1 else 'darkblue' for i in range(periods)],
        hovertemplate='Time: %{x:.2f} years<br>Cash Flow: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Bond Cash Flow Timeline ({freq}x per year)',
        xaxis_title='Years',
        yaxis_title='Cash Flow ($)',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return fig

def create_yield_curve_chart(maturities: List[float], yields: List[float]) -> go.Figure:
    """Create yield curve visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=[y * 100 for y in yields],
        mode='lines+markers',
        name='Yield Curve',
        line=dict(color='darkblue', width=3),
        marker=dict(size=8),
        hovertemplate='Maturity: %{x:.1f} years<br>Yield: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Treasury Yield Curve',
        xaxis_title='Maturity (Years)',
        yaxis_title='Yield (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

# ==================== MAIN MODULE ====================

def fixed_income_module(analysis_context: Optional[Dict] = None):
    """
    Enhanced fixed income analysis module.
    
    Args:
        analysis_context: Optional context from main app
    """
    st.title("💰 Fixed Income / Bond Analytics")
    st.markdown("""
    Comprehensive bond analysis including pricing, yield calculations, duration, convexity,
    and portfolio immunization strategies.
    """)
    
    # Sidebar for mode selection
    with st.sidebar:
        st.header("Analysis Mode")
        mode = st.selectbox(
            "Select Analysis Type",
            ["Single Bond Analysis", "Bond Comparison", "Portfolio Analysis", "Yield Curve"]
        )
    
    # ==================== SINGLE BOND ANALYSIS ====================
    if mode == "Single Bond Analysis":
        st.header("📊 Single Bond Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bond Specifications")
            par = st.number_input("Par Value ($)", value=1000.0, min_value=1.0, step=100.0)
            coupon_rate = st.number_input("Annual Coupon Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.25) / 100
            years = st.number_input("Years to Maturity", value=10.0, min_value=0.1, max_value=30.0, step=0.5)
            freq = st.selectbox("Payment Frequency", [1, 2, 4, 12], index=1, 
                               format_func=lambda x: {1: "Annual", 2: "Semi-Annual", 4: "Quarterly", 12: "Monthly"}[x])
        
        with col2:
            st.subheader("Market Data")
            calculation_type = st.radio("Calculate:", ["Price from YTM", "YTM from Price"])
            
            if calculation_type == "Price from YTM":
                ytm = st.number_input("Yield to Maturity (%)", value=5.0, min_value=0.1, max_value=20.0, step=0.25) / 100
                price = bond_price(par, coupon_rate, years, ytm, freq)
                st.metric("Calculated Price", f"${price:.2f}")
            else:
                price = st.number_input("Current Market Price ($)", value=1000.0, min_value=1.0, step=10.0)
                ytm = bond_ytm(par, price, coupon_rate, years, freq)
                st.metric("Calculated YTM", f"{ytm*100:.4f}%")
            
            days_since = st.number_input("Days Since Last Coupon", value=0, min_value=0, max_value=365)
        
        if st.button("🔍 Analyze Bond", use_container_width=True):
            # Calculate all metrics
            dur = bond_duration(par, coupon_rate, years, ytm, freq)
            mod_dur = modified_duration(par, coupon_rate, years, ytm, freq)
            conv = bond_convexity(par, coupon_rate, years, ytm, freq)
            dv = dv01(par, coupon_rate, years, ytm, freq)
            cy = current_yield(par, coupon_rate, price)
            accrued = accrued_interest(par, coupon_rate, freq, days_since)
            
            # Display metrics in organized layout
            st.subheader("📈 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bond Price", f"${price:.2f}")
                st.metric("Yield to Maturity", f"{ytm*100:.4f}%")
            with col2:
                st.metric("Current Yield", f"{cy*100:.4f}%")
                st.metric("Accrued Interest", f"${accrued:.2f}")
            with col3:
                st.metric("Macaulay Duration", f"{dur:.4f} years")
                st.metric("Modified Duration", f"{mod_dur:.4f}")
            with col4:
                st.metric("Convexity", f"{conv:.4f}")
                st.metric("DV01", f"${dv:.4f}")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📉 Price-Yield", "⚡ Sensitivity", "💵 Cash Flows", "📊 Data"])
            
            with tab1:
                st.plotly_chart(create_price_yield_chart(par, coupon_rate, years, ytm, freq), 
                              use_container_width=True)
                
                st.plotly_chart(create_duration_convexity_chart(par, coupon_rate, years, ytm, freq),
                              use_container_width=True)
            
            with tab2:
                st.subheader("Price Sensitivity to Yield Changes")
                sensitivity_df = price_yield_sensitivity(par, coupon_rate, years, ytm, freq)
                
                # Highlight current yield
                def highlight_current(row):
                    if abs(row['Yield Change (bp)']) == 0:
                        return ['background-color: #ffeb3b'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    sensitivity_df.style.apply(highlight_current, axis=1).format({
                        'Yield (%)': '{:.4f}',
                        'Price': '${:.2f}',
                        'Price Change (%)': '{:.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Risk interpretation
                st.info(f"""
                **Risk Interpretation:**
                - A 100bp (1%) increase in yield would change price by approximately **{-mod_dur*100:.2f}%**
                - The DV01 of **${dv:.4f}** means each 1bp move costs **${dv:.2f}** per $1M face value
                - Convexity of **{conv:.2f}** provides cushioning against large yield changes
                """)
            
            with tab3:
                st.plotly_chart(create_cash_flow_diagram(par, coupon_rate, years, freq),
                              use_container_width=True)
                
                # Cash flow table
                periods = int(years * freq)
                coupon = par * coupon_rate / freq
                times = np.arange(1, periods + 1) / freq
                cash_flows = [coupon] * periods
                cash_flows[-1] += par
                
                pv_factors = [(1 + ytm/freq)**(-t) for t in range(1, periods + 1)]
                pv_cash_flows = [cf * pv for cf, pv in zip(cash_flows, pv_factors)]
                
                cf_df = pd.DataFrame({
                    'Period': range(1, periods + 1),
                    'Time (Years)': times,
                    'Cash Flow': cash_flows,
                    'PV Factor': pv_factors,
                    'Present Value': pv_cash_flows
                })
                
                st.dataframe(
                    cf_df.style.format({
                        'Time (Years)': '{:.2f}',
                        'Cash Flow': '${:.2f}',
                        'PV Factor': '{:.6f}',
                        'Present Value': '${:.2f}'
                    }),
                    use_container_width=True
                )
                
                st.metric("Total Present Value", f"${sum(pv_cash_flows):.2f}")
            
            with tab4:
                st.subheader("Complete Bond Data")
                
                summary_data = {
                    'Specification': [
                        'Par Value', 'Coupon Rate', 'Years to Maturity', 
                        'Payment Frequency', 'Market Price', 'Yield to Maturity'
                    ],
                    'Value': [
                        f'${par:.2f}', f'{coupon_rate*100:.2f}%', f'{years:.2f}',
                        {1: 'Annual', 2: 'Semi-Annual', 4: 'Quarterly', 12: 'Monthly'}[freq],
                        f'${price:.2f}', f'{ytm*100:.4f}%'
                    ]
                }
                
                metrics_data = {
                    'Metric': [
                        'Current Yield', 'Macaulay Duration', 'Modified Duration',
                        'Convexity', 'DV01', 'Accrued Interest'
                    ],
                    'Value': [
                        f'{cy*100:.4f}%', f'{dur:.4f} years', f'{mod_dur:.4f}',
                        f'{conv:.4f}', f'${dv:.4f}', f'${accrued:.2f}'
                    ]
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                with col2:
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    # ==================== BOND COMPARISON ====================
    elif mode == "Bond Comparison":
        st.header("⚖️ Bond Comparison")
        st.write("Compare multiple bonds side-by-side")
        
        num_bonds = st.slider("Number of Bonds to Compare", 2, 5, 3)
        
        bonds = []
        cols = st.columns(num_bonds)
        
        for i, col in enumerate(cols):
            with col:
                st.subheader(f"Bond {i+1}")
                par = st.number_input(f"Par Value {i+1}", value=1000.0, key=f"par_{i}")
                coupon = st.number_input(f"Coupon Rate (%) {i+1}", value=5.0 + i, key=f"coup_{i}") / 100
                years = st.number_input(f"Years {i+1}", value=10.0 - i, key=f"years_{i}")
                ytm = st.number_input(f"YTM (%) {i+1}", value=5.5 + i*0.5, key=f"ytm_{i}") / 100
                
                bonds.append({
                    'name': f'Bond {i+1}',
                    'par': par,
                    'coupon_rate': coupon,
                    'years': years,
                    'ytm': ytm,
                    'freq': 2
                })
        
        if st.button("🔍 Compare Bonds"):
            # Calculate metrics for all bonds
            comparison_data = []
            
            for bond in bonds:
                price = bond_price(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                dur = bond_duration(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                mod_dur = modified_duration(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                conv = bond_convexity(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                cy = current_yield(bond['par'], bond['coupon_rate'], price)
                
                comparison_data.append({
                    'Bond': bond['name'],
                    'Price': price,
                    'Coupon (%)': bond['coupon_rate'] * 100,
                    'YTM (%)': bond['ytm'] * 100,
                    'Current Yield (%)': cy * 100,
                    'Duration': dur,
                    'Modified Duration': mod_dur,
                    'Convexity': conv
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            st.subheader("Comparison Table")
            st.dataframe(
                comp_df.style.format({
                    'Price': '${:.2f}',
                    'Coupon (%)': '{:.2f}',
                    'YTM (%)': '{:.4f}',
                    'Current Yield (%)': '{:.4f}',
                    'Duration': '{:.4f}',
                    'Modified Duration': '{:.4f}',
                    'Convexity': '{:.4f}'
                }).background_gradient(subset=['Duration', 'YTM (%)'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            # Visual comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Comparison', 'Duration Comparison', 
                              'YTM Comparison', 'Convexity Comparison')
            )
            
            fig.add_trace(go.Bar(x=comp_df['Bond'], y=comp_df['Price'], name='Price'), row=1, col=1)
            fig.add_trace(go.Bar(x=comp_df['Bond'], y=comp_df['Duration'], name='Duration'), row=1, col=2)
            fig.add_trace(go.Bar(x=comp_df['Bond'], y=comp_df['YTM (%)'], name='YTM'), row=2, col=1)
            fig.add_trace(go.Bar(x=comp_df['Bond'], y=comp_df['Convexity'], name='Convexity'), row=2, col=2)
            
            fig.update_layout(height=700, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.subheader("💡 Recommendations")
            best_yield = comp_df.loc[comp_df['YTM (%)'].idxmax(), 'Bond']
            lowest_risk = comp_df.loc[comp_df['Duration'].idxmin(), 'Bond']
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Highest Yield:** {best_yield}")
            with col2:
                st.info(f"**Lowest Interest Rate Risk:** {lowest_risk}")
    
    # ==================== PORTFOLIO ANALYSIS ====================
    elif mode == "Portfolio Analysis":
        st.header("📊 Bond Portfolio Analysis")
        st.write("Analyze portfolio metrics and immunization strategies")
        
        st.subheader("Portfolio Bonds")
        num_bonds = st.number_input("Number of Bonds in Portfolio", 2, 10, 3)
        
        portfolio_bonds = []
        weights = []
        
        for i in range(num_bonds):
            with st.expander(f"Bond {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    par = st.number_input(f"Par Value", value=1000.0, key=f"p_par_{i}")
                    coupon = st.number_input(f"Coupon (%)", value=5.0, key=f"p_coup_{i}") / 100
                    years = st.number_input(f"Years", value=10.0, key=f"p_years_{i}")
                with col2:
                    ytm = st.number_input(f"YTM (%)", value=5.0, key=f"p_ytm_{i}") / 100
                    weight = st.number_input(f"Weight (%)", value=100/num_bonds, key=f"p_weight_{i}")
                
                portfolio_bonds.append({
                    'par': par, 'coupon_rate': coupon, 'years': years, 
                    'ytm': ytm, 'freq': 2
                })
                weights.append(weight / 100)
        
        if st.button("📊 Analyze Portfolio"):
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Calculate individual and portfolio metrics
            portfolio_value = 0
            portfolio_duration = 0
            portfolio_convexity = 0
            
            metrics = []
            
            for bond, weight in zip(portfolio_bonds, weights):
                price = bond_price(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                dur = bond_duration(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                conv = bond_convexity(bond['par'], bond['coupon_rate'], bond['years'], bond['ytm'], bond['freq'])
                
                bond_value = price * weight
                portfolio_value += bond_value
                portfolio_duration += dur * weight
                portfolio_convexity += conv * weight
                
                metrics.append({
                    'Bond': f"Bond {len(metrics)+1}",
                    'Weight (%)': weight * 100,
                    'Price': price,
                    'Duration': dur,
                    'Convexity': conv,
                    'Value': bond_value
                })
            
            metrics_df = pd.DataFrame(metrics)
            
            st.subheader("Portfolio Composition")
            st.dataframe(
                metrics_df.style.format({
                    'Weight (%)': '{:.2f}',
                    'Price': '${:.2f}',
                    'Duration': '{:.4f}',
                    'Convexity': '{:.4f}',
                    'Value': '${:.2f}'
                }),
                use_container_width=True
            )
            
            # Portfolio summary
            st.subheader("Portfolio Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Duration", f"{portfolio_duration:.4f} years")
            with col2:
                st.metric("Portfolio Convexity", f"{portfolio_convexity:.4f}")
            with col3:
                st.metric("Total Value", f"${portfolio_value:.2f}")
            
            # Pie chart
            fig = px.pie(metrics_df, values='Value', names='Bond', 
                        title='Portfolio Allocation')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== YIELD CURVE ====================
    elif mode == "Yield Curve":
        st.header("📈 Treasury Yield Curve")
        st.write("Analyze and visualize the yield curve")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Yield Data")
            
            # Default US Treasury yields (example)
            default_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
            default_yields = [4.5, 4.6, 4.7, 4.5, 4.4, 4.3, 4.35, 4.4, 4.6, 4.65]
            
            st.write("Enter yields for different maturities:")
            
            maturities = []
            yields_list = []
            
            for i, (mat, yld) in enumerate(zip(default_maturities, default_yields)):
                col_a, col_b = st.columns(2)
                with col_a:
                    maturity = st.number_input(f"Maturity {i+1} (years)", value=mat, key=f"mat_{i}")
                with col_b:
                    yld_val = st.number_input(f"Yield {i+1} (%)", value=yld, key=f"yld_{i}")
                
                maturities.append(maturity)
                yields_list.append(yld_val / 100)
        
        with col2:
            st.subheader("Curve Analysis")
            
            if len(maturities) >= 2:
                # Calculate spreads
                spread_2_10 = (yields_list[default_maturities.index(10)] - 
                              yields_list[default_maturities.index(2)]) * 100
                spread_3m_10 = (yields_list[default_maturities.index(10)] - 
                               yields_list[default_maturities.index(0.25)]) * 100
                
                st.metric("2-10 Spread", f"{spread_2_10:.2f} bp")
                st.metric("3M-10Y Spread", f"{spread_3m_10:.2f} bp")
                
                if spread_2_10 < 0:
                    st.warning("⚠️ Inverted Yield Curve")
                elif spread_2_10 < 50:
                    st.info("📊 Flat Yield Curve")
                else:
                    st.success("📈 Normal Yield Curve")
        
        # Plot yield curve
        fig = create_yield_curve_chart(maturities, yields_list)
        st.plotly_chart(fig, use_container_width=True)
        
        # Yield curve table
        curve_df = pd.DataFrame({
            'Maturity (Years)': maturities,
            'Yield (%)': [y * 100 for y in yields_list]
        })
        
        st.dataframe(
            curve_df.style.format({'Yield (%)': '{:.4f}'}).background_gradient(
                subset=['Yield (%)'], cmap='RdYlGn_r'
            ),
            use_container_width=True
        )
    
    # Footer
    st.divider()
    st.caption("💡 Fixed Income Analytics Module | Bond calculations and portfolio analysis")

# Standalone execution for testing
if __name__ == "__main__":
    fixed_income_module(None)
