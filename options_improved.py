# options_improved.py - Enhanced Options Pricing & Greeks Module
# Comprehensive options analytics with Black-Scholes, Greeks, strategies, and IV analysis
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# ==================== BLACK-SCHOLES CORE FUNCTIONS ====================

def get_d1(S: float, K: float, r: float, sigma: float, T: float, q: float = 0) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def get_d2(d1: float, sigma: float, T: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    if T <= 0:
        return d1
    return d1 - sigma * np.sqrt(T)

def black_scholes(S: float, K: float, r: float, sigma: float, T: float, 
                 q: float = 0, option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        T: Time to expiration (years)
        q: Dividend yield (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Option price
    """
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = get_d1(S, K, r, sigma, T, q)
    d2 = get_d2(d1, sigma, T)
    
    if option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:  # call
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ==================== GREEKS ====================

def delta(S: float, K: float, r: float, sigma: float, T: float, 
         q: float = 0, option_type: str = 'call') -> float:
    """
    Calculate option delta (sensitivity to stock price).
    Range: Call [0, 1], Put [-1, 0]
    """
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = get_d1(S, K, r, sigma, T, q)
    
    if option_type == 'put':
        return -np.exp(-q * T) * norm.cdf(-d1)
    else:
        return np.exp(-q * T) * norm.cdf(d1)

def gamma(S: float, K: float, r: float, sigma: float, T: float, q: float = 0) -> float:
    """
    Calculate gamma (rate of change of delta).
    Same for calls and puts.
    """
    if T <= 0 or S <= 0:
        return 0.0
    
    d1 = get_d1(S, K, r, sigma, T, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S: float, K: float, r: float, sigma: float, T: float, q: float = 0) -> float:
    """
    Calculate vega (sensitivity to volatility).
    Returns change per 1% change in volatility.
    Same for calls and puts.
    """
    if T <= 0:
        return 0.0
    
    d1 = get_d1(S, K, r, sigma, T, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change

def theta(S: float, K: float, r: float, sigma: float, T: float, 
         q: float = 0, option_type: str = 'call') -> float:
    """
    Calculate theta (time decay).
    Returns change per day (divided by 365).
    """
    if T <= 0:
        return 0.0
    
    d1 = get_d1(S, K, r, sigma, T, q)
    d2 = get_d2(d1, sigma, T)
    
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    
    if option_type == 'put':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
        return (term1 + term2 + term3) / 365
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(d1)
        return (term1 + term2 + term3) / 365

def rho(S: float, K: float, r: float, sigma: float, T: float, 
       q: float = 0, option_type: str = 'call') -> float:
    """
    Calculate rho (sensitivity to interest rate).
    Returns change per 1% change in interest rate.
    """
    if T <= 0:
        return 0.0
    
    d1 = get_d1(S, K, r, sigma, T, q)
    d2 = get_d2(d1, sigma, T)
    
    if option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100

# ==================== ADVANCED ANALYTICS ====================

def implied_volatility(price: float, S: float, K: float, r: float, T: float, 
                      q: float = 0, option_type: str = 'call') -> float:
    """
    Calculate implied volatility using Brent's method.
    
    Args:
        price: Observed option price
        Other parameters same as black_scholes
    
    Returns:
        Implied volatility (annual)
    """
    if T <= 0:
        return np.nan
    
    def objective(sigma):
        return black_scholes(S, K, r, sigma, T, q, option_type) - price
    
    try:
        # Try Brent's method first (more robust)
        iv = brentq(objective, 0.001, 5.0, maxiter=100)
        return iv
    except:
        try:
            # Fallback to bounded optimization
            result = minimize_scalar(lambda sig: abs(objective(sig)), 
                                    bounds=(0.001, 5.0), method='bounded')
            return result.x if result.success else np.nan
        except:
            return np.nan

def binomial_tree_american(S: float, K: float, r: float, sigma: float, T: float, 
                          N: int = 100, q: float = 0, option_type: str = 'call') -> float:
    """
    Price American options using binomial tree.
    
    Args:
        N: Number of time steps
    
    Returns:
        American option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    ST = np.array([S * (u ** (N - i)) * (d ** i) for i in range(N + 1)])
    
    # Initialize option values at maturity
    if option_type == 'call':
        V = np.maximum(ST - K, 0)
    else:
        V = np.maximum(K - ST, 0)
    
    # Backward induction
    for j in range(N - 1, -1, -1):
        ST = np.array([S * (u ** (j - i)) * (d ** i) for i in range(j + 1)])
        
        # Option value from continuation
        V = np.exp(-r * dt) * (p * V[:-1] + (1 - p) * V[1:])
        
        # Early exercise value
        if option_type == 'call':
            exercise_value = np.maximum(ST - K, 0)
        else:
            exercise_value = np.maximum(K - ST, 0)
        
        # American option: take max of continuation and exercise
        V = np.maximum(V, exercise_value)
    
    return V[0]

def monte_carlo_pricing(S: float, K: float, r: float, sigma: float, T: float,
                       n_simulations: int = 10000, q: float = 0, 
                       option_type: str = 'call') -> Tuple[float, float]:
    """
    Price European options using Monte Carlo simulation.
    
    Returns:
        Tuple of (price, standard_error)
    """
    np.random.seed(42)
    
    # Simulate stock price paths
    Z = np.random.standard_normal(n_simulations)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discount to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    
    return price, std_error

def option_pnl(S_initial: float, S_final: float, K: float, r: float, 
              sigma: float, T_initial: float, T_final: float, 
              premium: float, q: float = 0, option_type: str = 'call',
              position: str = 'long') -> Dict:
    """
    Calculate P&L and Greeks changes for an option position.
    
    Args:
        S_initial, S_final: Initial and final stock prices
        T_initial, T_final: Initial and final time to expiration
        premium: Initial option premium paid/received
        position: 'long' or 'short'
    
    Returns:
        Dictionary with P&L breakdown
    """
    # Initial option value
    initial_value = black_scholes(S_initial, K, r, sigma, T_initial, q, option_type)
    
    # Final option value
    final_value = black_scholes(S_final, K, r, sigma, T_final, q, option_type) if T_final > 0 else (
        max(0, S_final - K) if option_type == 'call' else max(0, K - S_final)
    )
    
    # Calculate P&L components
    if position == 'long':
        total_pnl = final_value - premium
        intrinsic_change = final_value - initial_value
    else:  # short
        total_pnl = premium - final_value
        intrinsic_change = initial_value - final_value
    
    # Greeks at initial time
    initial_delta = delta(S_initial, K, r, sigma, T_initial, q, option_type)
    initial_gamma = gamma(S_initial, K, r, sigma, T_initial, q)
    initial_theta = theta(S_initial, K, r, sigma, T_initial, q, option_type)
    initial_vega = vega(S_initial, K, r, sigma, T_initial, q)
    
    return {
        'total_pnl': total_pnl,
        'initial_value': initial_value,
        'final_value': final_value,
        'intrinsic_change': intrinsic_change,
        'initial_delta': initial_delta,
        'initial_gamma': initial_gamma,
        'initial_theta': initial_theta,
        'initial_vega': initial_vega,
        'premium_paid': premium if position == 'long' else -premium
    }

# ==================== OPTION STRATEGIES ====================

def strategy_payoff(S_range: np.ndarray, legs: List[Dict]) -> np.ndarray:
    """
    Calculate payoff for multi-leg option strategy.
    
    Args:
        S_range: Array of stock prices
        legs: List of dictionaries with keys: type, K, position, quantity, premium
    
    Returns:
        Array of payoffs at each stock price
    """
    total_payoff = np.zeros_like(S_range)
    
    for leg in legs:
        K = leg['K']
        premium = leg.get('premium', 0)
        quantity = leg.get('quantity', 1)
        
        if leg['type'] == 'call':
            intrinsic = np.maximum(S_range - K, 0)
        else:  # put
            intrinsic = np.maximum(K - S_range, 0)
        
        if leg['position'] == 'long':
            leg_payoff = (intrinsic - premium) * quantity
        else:  # short
            leg_payoff = (premium - intrinsic) * quantity
        
        total_payoff += leg_payoff
    
    return total_payoff

STRATEGY_TEMPLATES = {
    'Covered Call': [
        {'type': 'stock', 'position': 'long', 'quantity': 100},
        {'type': 'call', 'position': 'short', 'quantity': 1}
    ],
    'Protective Put': [
        {'type': 'stock', 'position': 'long', 'quantity': 100},
        {'type': 'put', 'position': 'long', 'quantity': 1}
    ],
    'Bull Call Spread': [
        {'type': 'call', 'position': 'long', 'quantity': 1},
        {'type': 'call', 'position': 'short', 'quantity': 1}
    ],
    'Bear Put Spread': [
        {'type': 'put', 'position': 'long', 'quantity': 1},
        {'type': 'put', 'position': 'short', 'quantity': 1}
    ],
    'Long Straddle': [
        {'type': 'call', 'position': 'long', 'quantity': 1},
        {'type': 'put', 'position': 'long', 'quantity': 1}
    ],
    'Long Strangle': [
        {'type': 'call', 'position': 'long', 'quantity': 1},
        {'type': 'put', 'position': 'long', 'quantity': 1}
    ],
    'Iron Condor': [
        {'type': 'put', 'position': 'long', 'quantity': 1},
        {'type': 'put', 'position': 'short', 'quantity': 1},
        {'type': 'call', 'position': 'short', 'quantity': 1},
        {'type': 'call', 'position': 'long', 'quantity': 1}
    ],
    'Butterfly Spread': [
        {'type': 'call', 'position': 'long', 'quantity': 1},
        {'type': 'call', 'position': 'short', 'quantity': 2},
        {'type': 'call', 'position': 'long', 'quantity': 1}
    ]
}

# ==================== VISUALIZATION FUNCTIONS ====================

def create_greeks_surface(S_range: np.ndarray, T_range: np.ndarray, K: float, 
                         r: float, sigma: float, q: float, greek: str, 
                         option_type: str) -> go.Figure:
    """Create 3D surface plot for a Greek."""
    greek_funcs = {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }
    
    Z = np.zeros((len(T_range), len(S_range)))
    
    for i, T in enumerate(T_range):
        for j, S in enumerate(S_range):
            if greek in ['delta', 'theta', 'rho']:
                Z[i, j] = greek_funcs[greek](S, K, r, sigma, T, q, option_type)
            else:  # gamma, vega (same for calls and puts)
                Z[i, j] = greek_funcs[greek](S, K, r, sigma, T, q)
    
    fig = go.Figure(data=[go.Surface(x=S_range, y=T_range, z=Z, colorscale='Viridis')])
    
    fig.update_layout(
        title=f'{greek.capitalize()} Surface ({option_type.capitalize()} Option)',
        scene=dict(
            xaxis_title='Stock Price',
            yaxis_title='Time to Expiration (Years)',
            zaxis_title=greek.capitalize()
        ),
        template='plotly_white',
        height=600
    )
    
    return fig

def create_payoff_diagram(S_range: np.ndarray, legs: List[Dict], current_S: float) -> go.Figure:
    """Create strategy payoff diagram."""
    payoffs = strategy_payoff(S_range, legs)
    
    fig = go.Figure()
    
    # Payoff line
    fig.add_trace(go.Scatter(
        x=S_range,
        y=payoffs,
        mode='lines',
        name='Strategy Payoff',
        line=dict(color='blue', width=3),
        hovertemplate='Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    
    # Current stock price
    current_payoff = strategy_payoff(np.array([current_S]), legs)[0]
    fig.add_trace(go.Scatter(
        x=[current_S],
        y=[current_payoff],
        mode='markers',
        name='Current Position',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    # Profit/Loss regions
    fig.add_hrect(y0=0, y1=max(payoffs), fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=min(payoffs), y1=0, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    fig.update_layout(
        title='Strategy Payoff Diagram at Expiration',
        xaxis_title='Stock Price at Expiration ($)',
        yaxis_title='Profit/Loss ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_volatility_smile(strikes: List[float], ivs: List[float], current_K: float) -> go.Figure:
    """Create implied volatility smile chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strikes,
        y=[iv * 100 for iv in ivs],
        mode='lines+markers',
        name='Implied Volatility',
        line=dict(color='purple', width=3),
        marker=dict(size=8),
        hovertemplate='Strike: $%{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
    ))
    
    # Highlight ATM strike
    if current_K in strikes:
        idx = strikes.index(current_K)
        fig.add_trace(go.Scatter(
            x=[current_K],
            y=[ivs[idx] * 100],
            mode='markers',
            name='ATM Strike',
            marker=dict(size=15, color='red', symbol='star')
        ))
    
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price ($)',
        yaxis_title='Implied Volatility (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

# ==================== MAIN MODULE ====================

def options_module(analysis_context: Optional[Dict] = None):
    """
    Enhanced options pricing and analytics module.
    
    Args:
        analysis_context: Optional context from main app
    """
    st.title("📊 Options Pricing & Analytics")
    st.markdown("""
    Comprehensive options analysis using Black-Scholes model, Greeks calculation,
    strategy analysis, and implied volatility computation.
    """)
    
    # Sidebar for mode selection
    with st.sidebar:
        st.header("Analysis Mode")
        mode = st.selectbox(
            "Select Analysis Type",
            ["Single Option Pricing", "Greeks Analysis", "Strategy Builder", 
             "Implied Volatility", "Comparison Tools"]
        )
    
    # ==================== SINGLE OPTION PRICING ====================
    if mode == "Single Option Pricing":
        st.header("💵 Option Pricing Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Parameters")
            S = st.number_input("Stock Price ($)", value=100.0, min_value=0.01, step=1.0)
            K = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0)
            r = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.25) / 100
            sigma = st.number_input("Volatility (% annual)", value=20.0, min_value=0.1, max_value=200.0, step=1.0) / 100
            
        with col2:
            st.subheader("Option Specifications")
            T = st.number_input("Time to Expiration (years)", value=1.0, min_value=0.001, max_value=10.0, step=0.083)
            T_days = st.number_input("or Days to Expiration", value=int(T*365), min_value=1, step=1)
            T = T_days / 365  # Use days input
            q = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=10.0, step=0.25) / 100
            option_type = st.radio("Option Type", ["call", "put"])
        
        # Pricing method selection
        pricing_method = st.selectbox(
            "Pricing Method",
            ["Black-Scholes (European)", "Binomial Tree (American)", "Monte Carlo Simulation"]
        )
        
        if st.button("🔍 Calculate Option Value", use_container_width=True):
            # Calculate option price based on method
            if pricing_method == "Black-Scholes (European)":
                price = black_scholes(S, K, r, sigma, T, q, option_type)
                method_note = "European option (exercise only at expiration)"
            elif pricing_method == "Binomial Tree (American)":
                n_steps = st.slider("Number of Steps", 50, 500, 100)
                price = binomial_tree_american(S, K, r, sigma, T, n_steps, q, option_type)
                method_note = "American option (can exercise early)"
            else:  # Monte Carlo
                n_sims = st.slider("Number of Simulations", 1000, 50000, 10000)
                price, std_err = monte_carlo_pricing(S, K, r, sigma, T, n_sims, q, option_type)
                method_note = f"European option (MC std error: ${std_err:.4f})"
            
            # Calculate all Greeks
            del_val = delta(S, K, r, sigma, T, q, option_type)
            gam_val = gamma(S, K, r, sigma, T, q)
            veg_val = vega(S, K, r, sigma, T, q)
            theta_val = theta(S, K, r, sigma, T, q, option_type)
            rho_val = rho(S, K, r, sigma, T, q, option_type)
            
            # Intrinsic and time value
            if option_type == 'call':
                intrinsic = max(0, S - K)
            else:
                intrinsic = max(0, K - S)
            time_value = price - intrinsic
            
            # Display results
            st.success(method_note)
            
            st.subheader("💰 Valuation")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Option Price", f"${price:.4f}")
            with col2:
                st.metric("Intrinsic Value", f"${intrinsic:.4f}")
            with col3:
                st.metric("Time Value", f"${time_value:.4f}")
            with col4:
                moneyness = "ATM" if abs(S - K) < 0.01 * S else ("ITM" if intrinsic > 0 else "OTM")
                st.metric("Moneyness", moneyness)
            
            st.subheader("📊 The Greeks")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Delta (Δ)",
                    f"{del_val:.4f}",
                    help="Change in option price for $1 change in stock price"
                )
            with col2:
                st.metric(
                    "Gamma (Γ)",
                    f"{gam_val:.4f}",
                    help="Change in delta for $1 change in stock price"
                )
            with col3:
                st.metric(
                    "Vega (ν)",
                    f"${veg_val:.4f}",
                    help="Change in option price for 1% change in volatility"
                )
            with col4:
                st.metric(
                    "Theta (Θ)",
                    f"${theta_val:.4f}",
                    help="Change in option price per day (time decay)"
                )
            with col5:
                st.metric(
                    "Rho (ρ)",
                    f"${rho_val:.4f}",
                    help="Change in option price for 1% change in interest rate"
                )
            
            # Visualizations
            tab1, tab2, tab3 = st.tabs(["📈 Price Sensitivity", "⏰ Time Decay", "📊 Volatility"])
            
            with tab1:
                # Price vs Stock Price
                S_range = np.linspace(S * 0.5, S * 1.5, 100)
                prices = [black_scholes(s, K, r, sigma, T, q, option_type) for s in S_range]
                intrinsics = [max(0, s - K) if option_type == 'call' else max(0, K - s) for s in S_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name='Option Price',
                                        line=dict(color='blue', width=3)))
                fig.add_trace(go.Scatter(x=S_range, y=intrinsics, mode='lines', name='Intrinsic Value',
                                        line=dict(color='green', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=[S], y=[price], mode='markers', name='Current',
                                        marker=dict(size=12, color='red', symbol='star')))
                fig.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")
                
                fig.update_layout(
                    title=f'{option_type.capitalize()} Option Price vs Stock Price',
                    xaxis_title='Stock Price ($)',
                    yaxis_title='Option Value ($)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Time Decay
                T_range = np.linspace(T, 0.001, 100)
                prices_time = [black_scholes(S, K, r, sigma, t, q, option_type) for t in T_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=T_range * 365, y=prices_time, mode='lines',
                                        line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=[T * 365], y=[price], mode='markers',
                                        marker=dict(size=12, color='blue', symbol='star')))
                
                fig.update_layout(
                    title='Option Price Time Decay',
                    xaxis_title='Days to Expiration',
                    yaxis_title='Option Price ($)',
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Time Decay**: This option loses approximately **${abs(theta_val):.4f}** per day due to theta decay.")
            
            with tab3:
                # Volatility Sensitivity
                vol_range = np.linspace(max(0.05, sigma * 0.5), min(2.0, sigma * 2), 100)
                prices_vol = [black_scholes(S, K, r, v, T, q, option_type) for v in vol_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vol_range * 100, y=prices_vol, mode='lines',
                                        line=dict(color='purple', width=3)))
                fig.add_trace(go.Scatter(x=[sigma * 100], y=[price], mode='markers',
                                        marker=dict(size=12, color='red', symbol='star')))
                
                fig.update_layout(
                    title='Option Price vs Volatility',
                    xaxis_title='Volatility (%)',
                    yaxis_title='Option Price ($)',
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Vega**: A 1% increase in volatility increases option value by **${veg_val:.4f}**.")
    
    # ==================== GREEKS ANALYSIS ====================
    elif mode == "Greeks Analysis":
        st.header("📐 Greeks Analysis & Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01, key="greeks_S")
            K = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, key="greeks_K")
            r = st.number_input("Risk-Free Rate (%)", value=5.0, key="greeks_r") / 100
            
        with col2:
            sigma = st.number_input("Volatility (%)", value=20.0, key="greeks_sigma") / 100
            T = st.number_input("Time to Expiration (years)", value=1.0, min_value=0.001, key="greeks_T")
            q = st.number_input("Dividend Yield (%)", value=0.0, key="greeks_q") / 100
        
        option_type = st.radio("Option Type", ["call", "put"], key="greeks_type")
        
        if st.button("📊 Analyze Greeks"):
            # Calculate Greeks
            del_val = delta(S, K, r, sigma, T, q, option_type)
            gam_val = gamma(S, K, r, sigma, T, q)
            veg_val = vega(S, K, r, sigma, T, q)
            theta_val = theta(S, K, r, sigma, T, q, option_type)
            rho_val = rho(S, K, r, sigma, T, q, option_type)
            
            # Display current Greeks
            st.subheader("Current Greeks Values")
            
            greeks_df = pd.DataFrame({
                'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                'Value': [del_val, gam_val, veg_val, theta_val, rho_val],
                'Interpretation': [
                    f"${del_val:.4f} change per $1 stock move",
                    f"{gam_val:.4f} change in delta per $1 stock move",
                    f"${veg_val:.4f} change per 1% volatility increase",
                    f"${theta_val:.4f} decay per day",
                    f"${rho_val:.4f} change per 1% rate increase"
                ]
            })
            
            st.dataframe(
                greeks_df.style.format({'Value': '{:.6f}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Greeks vs Stock Price
            st.subheader("Greeks vs Stock Price")
            
            S_range = np.linspace(S * 0.7, S * 1.3, 100)
            
            deltas = [delta(s, K, r, sigma, T, q, option_type) for s in S_range]
            gammas = [gamma(s, K, r, sigma, T, q) for s in S_range]
            vegas = [vega(s, K, r, sigma, T, q) for s in S_range]
            thetas = [theta(s, K, r, sigma, T, q, option_type) for s in S_range]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Delta vs Stock Price', 'Gamma vs Stock Price',
                              'Vega vs Stock Price', 'Theta vs Stock Price')
            )
            
            fig.add_trace(go.Scatter(x=S_range, y=deltas, mode='lines', name='Delta',
                                    line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=S_range, y=gammas, mode='lines', name='Gamma',
                                    line=dict(color='green')), row=1, col=2)
            fig.add_trace(go.Scatter(x=S_range, y=vegas, mode='lines', name='Vega',
                                    line=dict(color='purple')), row=2, col=1)
            fig.add_trace(go.Scatter(x=S_range, y=thetas, mode='lines', name='Theta',
                                    line=dict(color='red')), row=2, col=2)
            
            # Add current position markers
            for row, col, greek_val, greek_name in [(1, 1, del_val, 'Delta'), (1, 2, gam_val, 'Gamma'),
                                                      (2, 1, veg_val, 'Vega'), (2, 2, theta_val, 'Theta')]:
                fig.add_trace(go.Scatter(x=[S], y=[greek_val], mode='markers',
                                        marker=dict(size=10, color='red', symbol='star'),
                                        showlegend=False), row=row, col=col)
            
            fig.update_xaxes(title_text="Stock Price ($)")
            fig.update_layout(height=800, template='plotly_white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 3D Surface for selected Greek
            st.subheader("3D Greeks Surface")
            selected_greek = st.selectbox("Select Greek for 3D Visualization",
                                         ['delta', 'gamma', 'vega', 'theta', 'rho'])
            
            with st.spinner("Generating 3D surface..."):
                S_range_3d = np.linspace(S * 0.7, S * 1.3, 30)
                T_range_3d = np.linspace(0.01, min(2, T * 2), 30)
                
                fig_3d = create_greeks_surface(S_range_3d, T_range_3d, K, r, sigma, q,
                                              selected_greek, option_type)
                st.plotly_chart(fig_3d, use_container_width=True)
    
    # ==================== STRATEGY BUILDER ====================
    elif mode == "Strategy Builder":
        st.header("🎯 Options Strategy Builder")
        
        st.write("Build and analyze multi-leg option strategies")
        
        # Select strategy template or custom
        strategy_choice = st.selectbox(
            "Select Strategy",
            ["Custom"] + list(STRATEGY_TEMPLATES.keys())
        )
        
        # Market parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            current_S = st.number_input("Current Stock Price ($)", value=100.0, key="strat_S")
        with col2:
            r_strat = st.number_input("Risk-Free Rate (%)", value=5.0, key="strat_r") / 100
        with col3:
            sigma_strat = st.number_input("Volatility (%)", value=20.0, key="strat_sigma") / 100
        
        T_strat = st.slider("Time to Expiration (days)", 1, 365, 30) / 365
        
        # Build strategy legs
        st.subheader("Strategy Legs")
        
        if strategy_choice != "Custom":
            st.info(f"Using template: **{strategy_choice}**. Modify strikes and premiums below.")
        
        legs = []
        num_legs = st.number_input("Number of Legs", 1, 6, 2 if strategy_choice == "Custom" else len(STRATEGY_TEMPLATES.get(strategy_choice, [])))
        
        for i in range(int(num_legs)):
            with st.expander(f"Leg {i+1}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    leg_type = st.selectbox(f"Type {i+1}", ["call", "put"], key=f"leg_type_{i}")
                with col2:
                    position = st.selectbox(f"Position {i+1}", ["long", "short"], key=f"leg_pos_{i}")
                with col3:
                    strike = st.number_input(f"Strike {i+1} ($)", value=100.0 + i*5, key=f"leg_K_{i}")
                with col4:
                    quantity = st.number_input(f"Quantity {i+1}", value=1, min_value=1, key=f"leg_qty_{i}")
                
                # Calculate premium using Black-Scholes
                premium = black_scholes(current_S, strike, r_strat, sigma_strat, T_strat, 0, leg_type)
                premium_input = st.number_input(f"Premium {i+1} ($)", value=float(premium), key=f"leg_prem_{i}")
                
                legs.append({
                    'type': leg_type,
                    'position': position,
                    'K': strike,
                    'quantity': quantity,
                    'premium': premium_input
                })
        
        if st.button("📊 Analyze Strategy"):
            # Calculate payoff diagram
            S_range = np.linspace(current_S * 0.7, current_S * 1.3, 200)
            
            fig_payoff = create_payoff_diagram(S_range, legs, current_S)
            st.plotly_chart(fig_payoff, use_container_width=True)
            
            # Calculate key metrics
            payoffs = strategy_payoff(S_range, legs)
            max_profit = np.max(payoffs)
            max_loss = np.min(payoffs)
            
            # Find break-even points
            breakevens = []
            for i in range(len(payoffs) - 1):
                if payoffs[i] * payoffs[i+1] < 0:  # Sign change
                    breakevens.append(S_range[i])
            
            # Initial cost
            initial_cost = sum([
                leg['premium'] * leg['quantity'] * (1 if leg['position'] == 'long' else -1)
                for leg in legs
            ])
            
            # Display metrics
            st.subheader("Strategy Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Cost", f"${initial_cost:.2f}")
            with col2:
                st.metric("Max Profit", f"${max_profit:.2f}" if max_profit < 1e6 else "Unlimited")
            with col3:
                st.metric("Max Loss", f"${abs(max_loss):.2f}" if max_loss > -1e6 else "Unlimited")
            with col4:
                profit_prob = np.sum(payoffs > 0) / len(payoffs) * 100
                st.metric("Profit Probability", f"{profit_prob:.1f}%")
            
            if breakevens:
                st.write("**Break-even Points:**", ", ".join([f"${be:.2f}" for be in breakevens]))
            
            # Strategy table
            st.subheader("Position Details")
            
            legs_df = pd.DataFrame([{
                'Leg': i+1,
                'Type': leg['type'].upper(),
                'Position': leg['position'].upper(),
                'Strike': f"${leg['K']:.2f}",
                'Quantity': leg['quantity'],
                'Premium': f"${leg['premium']:.2f}",
                'Cost': f"${leg['premium'] * leg['quantity'] * (1 if leg['position'] == 'long' else -1):.2f}"
            } for i, leg in enumerate(legs)])
            
            st.dataframe(legs_df, use_container_width=True, hide_index=True)
    
    # ==================== IMPLIED VOLATILITY ====================
    elif mode == "Implied Volatility":
        st.header("📉 Implied Volatility Calculator")
        
        st.write("Calculate implied volatility from market prices and visualize volatility smile/skew")
        
        tab1, tab2 = st.tabs(["Single Option IV", "Volatility Smile"])
        
        with tab1:
            st.subheader("Calculate IV from Market Price")
            
            col1, col2 = st.columns(2)
            
            with col1:
                S_iv = st.number_input("Stock Price ($)", value=100.0, key="iv_S")
                K_iv = st.number_input("Strike Price ($)", value=105.0, key="iv_K")
                market_price = st.number_input("Market Option Price ($)", value=7.50, key="iv_price")
            
            with col2:
                r_iv = st.number_input("Risk-Free Rate (%)", value=5.0, key="iv_r") / 100
                T_iv = st.number_input("Time to Expiration (years)", value=0.5, key="iv_T")
                q_iv = st.number_input("Dividend Yield (%)", value=0.0, key="iv_q") / 100
            
            option_type_iv = st.radio("Option Type", ["call", "put"], key="iv_type")
            
            if st.button("Calculate Implied Volatility"):
                with st.spinner("Calculating IV..."):
                    iv = implied_volatility(market_price, S_iv, K_iv, r_iv, T_iv, q_iv, option_type_iv)
                    
                    if np.isnan(iv):
                        st.error("❌ Could not calculate implied volatility. Check if option price is valid.")
                    else:
                        st.success(f"**Implied Volatility: {iv * 100:.2f}%**")
                        
                        # Compare with theoretical price at different volatilities
                        vol_range = np.linspace(0.05, 1.0, 100)
                        theoretical_prices = [black_scholes(S_iv, K_iv, r_iv, v, T_iv, q_iv, option_type_iv) 
                                            for v in vol_range]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=vol_range * 100, y=theoretical_prices, 
                                                mode='lines', name='Theoretical Price',
                                                line=dict(color='blue', width=3)))
                        fig.add_trace(go.Scatter(x=[iv * 100], y=[market_price],
                                                mode='markers', name='Market Price',
                                                marker=dict(size=15, color='red', symbol='star')))
                        fig.add_hline(y=market_price, line_dash="dash", line_color="red")
                        
                        fig.update_layout(
                            title='Option Price vs Volatility',
                            xaxis_title='Volatility (%)',
                            yaxis_title='Option Price ($)',
                            template='plotly_white',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Greeks at implied volatility
                        st.subheader("Greeks at Implied Volatility")
                        
                        del_iv = delta(S_iv, K_iv, r_iv, iv, T_iv, q_iv, option_type_iv)
                        gam_iv = gamma(S_iv, K_iv, r_iv, iv, T_iv, q_iv)
                        veg_iv = vega(S_iv, K_iv, r_iv, iv, T_iv, q_iv)
                        theta_iv = theta(S_iv, K_iv, r_iv, iv, T_iv, q_iv, option_type_iv)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Delta", f"{del_iv:.4f}")
                        col2.metric("Gamma", f"{gam_iv:.6f}")
                        col3.metric("Vega", f"${veg_iv:.4f}")
                        col4.metric("Theta", f"${theta_iv:.4f}")
        
        with tab2:
            st.subheader("Volatility Smile/Skew")
            
            st.write("Enter market prices for different strikes to visualize the volatility smile")
            
            S_smile = st.number_input("Current Stock Price ($)", value=100.0, key="smile_S")
            r_smile = st.number_input("Risk-Free Rate (%)", value=5.0, key="smile_r") / 100
            T_smile = st.number_input("Time to Expiration (years)", value=0.25, key="smile_T")
            option_type_smile = st.radio("Option Type", ["call", "put"], key="smile_type")
            
            num_strikes = st.slider("Number of Strikes", 3, 10, 5)
            
            strikes = []
            prices = []
            
            st.write("**Enter Strike Prices and Market Prices:**")
            cols = st.columns(num_strikes)
            
            for i, col in enumerate(cols):
                with col:
                    strike = st.number_input(f"K{i+1}", value=float(S_smile - 10 + i * 5), key=f"smile_K_{i}")
                    price = st.number_input(f"Price{i+1}", value=10.0 - i, key=f"smile_P_{i}")
                    strikes.append(strike)
                    prices.append(price)
            
            if st.button("Calculate Volatility Smile"):
                with st.spinner("Calculating implied volatilities..."):
                    ivs = []
                    for K, price in zip(strikes, prices):
                        iv = implied_volatility(price, S_smile, K, r_smile, T_smile, 0, option_type_smile)
                        ivs.append(iv if not np.isnan(iv) else None)
                    
                    # Filter out failed calculations
                    valid_data = [(k, iv) for k, iv in zip(strikes, ivs) if iv is not None]
                    
                    if not valid_data:
                        st.error("Could not calculate IV for any strikes. Check input prices.")
                    else:
                        valid_strikes, valid_ivs = zip(*valid_data)
                        
                        fig_smile = create_volatility_smile(list(valid_strikes), list(valid_ivs), S_smile)
                        st.plotly_chart(fig_smile, use_container_width=True)
                        
                        # Display IV table
                        smile_df = pd.DataFrame({
                            'Strike ($)': valid_strikes,
                            'Market Price ($)': [p for k, p in zip(strikes, prices) if k in valid_strikes],
                            'Implied Vol (%)': [iv * 100 for iv in valid_ivs],
                            'Moneyness': ['ATM' if abs(k - S_smile) < 0.01 * S_smile 
                                        else ('ITM' if (k < S_smile and option_type_smile == 'call') or 
                                                      (k > S_smile and option_type_smile == 'put')
                                             else 'OTM') 
                                        for k in valid_strikes]
                        })
                        
                        st.dataframe(
                            smile_df.style.format({
                                'Strike ($)': '{:.2f}',
                                'Market Price ($)': '{:.2f}',
                                'Implied Vol (%)': '{:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Analyze skew
                        min_iv = min(valid_ivs)
                        max_iv = max(valid_ivs)
                        iv_spread = (max_iv - min_iv) * 100
                        
                        st.info(f"""
                        **Volatility Smile Analysis:**
                        - IV Range: {min_iv*100:.2f}% - {max_iv*100:.2f}%
                        - IV Spread: {iv_spread:.2f}%
                        - ATM IV: {valid_ivs[len(valid_ivs)//2]*100:.2f}% (approx)
                        """)
    
    # ==================== COMPARISON TOOLS ====================
    elif mode == "Comparison Tools":
        st.header("⚖️ Options Comparison")
        
        st.write("Compare multiple options side-by-side")
        
        # Common parameters
        st.subheader("Market Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            S_comp = st.number_input("Stock Price ($)", value=100.0, key="comp_S")
        with col2:
            r_comp = st.number_input("Risk-Free Rate (%)", value=5.0, key="comp_r") / 100
        with col3:
            sigma_comp = st.number_input("Volatility (%)", value=20.0, key="comp_sigma") / 100
        with col4:
            T_comp = st.number_input("Time (years)", value=1.0, key="comp_T")
        
        # Options to compare
        st.subheader("Options to Compare")
        num_options = st.slider("Number of Options", 2, 5, 3)
        
        options = []
        cols = st.columns(num_options)
        
        for i, col in enumerate(cols):
            with col:
                st.write(f"**Option {i+1}**")
                opt_type = st.selectbox(f"Type", ["call", "put"], key=f"comp_type_{i}")
                strike = st.number_input(f"Strike ($)", value=95.0 + i * 5, key=f"comp_K_{i}")
                
                options.append({
                    'name': f'Option {i+1}',
                    'type': opt_type,
                    'strike': strike
                })
        
        if st.button("Compare Options"):
            # Calculate metrics for all options
            comparison_data = []
            
            for opt in options:
                price = black_scholes(S_comp, opt['strike'], r_comp, sigma_comp, T_comp, 0, opt['type'])
                del_val = delta(S_comp, opt['strike'], r_comp, sigma_comp, T_comp, 0, opt['type'])
                gam_val = gamma(S_comp, opt['strike'], r_comp, sigma_comp, T_comp, 0)
                veg_val = vega(S_comp, opt['strike'], r_comp, sigma_comp, T_comp, 0)
                theta_val = theta(S_comp, opt['strike'], r_comp, sigma_comp, T_comp, 0, opt['type'])
                
                if opt['type'] == 'call':
                    intrinsic = max(0, S_comp - opt['strike'])
                else:
                    intrinsic = max(0, opt['strike'] - S_comp)
                
                comparison_data.append({
                    'Option': opt['name'],
                    'Type': opt['type'].upper(),
                    'Strike': opt['strike'],
                    'Price': price,
                    'Intrinsic': intrinsic,
                    'Time Value': price - intrinsic,
                    'Delta': del_val,
                    'Gamma': gam_val,
                    'Vega': veg_val,
                    'Theta': theta_val
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            st.subheader("Comparison Table")
            st.dataframe(
                comp_df.style.format({
                    'Strike': '${:.2f}',
                    'Price': '${:.4f}',
                    'Intrinsic': '${:.4f}',
                    'Time Value': '${:.4f}',
                    'Delta': '{:.4f}',
                    'Gamma': '{:.6f}',
                    'Vega': '${:.4f}',
                    'Theta': '${:.4f}'
                }).background_gradient(subset=['Price', 'Delta'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
            
            # Visual comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Comparison', 'Delta Comparison',
                              'Time Value Comparison', 'Theta Comparison')
            )
            
            fig.add_trace(go.Bar(x=comp_df['Option'], y=comp_df['Price'], name='Price'), row=1, col=1)
            fig.add_trace(go.Bar(x=comp_df['Option'], y=comp_df['Delta'], name='Delta'), row=1, col=2)
            fig.add_trace(go.Bar(x=comp_df['Option'], y=comp_df['Time Value'], name='Time Value'), row=2, col=1)
            fig.add_trace(go.Bar(x=comp_df['Option'], y=comp_df['Theta'], name='Theta'), row=2, col=2)
            
            fig.update_layout(height=700, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Analysis")
            
            highest_delta_idx = comp_df['Delta'].abs().idxmax()
            lowest_theta_idx = comp_df['Theta'].abs().idxmin()
            best_value_idx = (comp_df['Price'] / comp_df['Delta'].abs()).idxmin()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Most Sensitive to Price:**  \n{comp_df.loc[highest_delta_idx, 'Option']} (Delta: {comp_df.loc[highest_delta_idx, 'Delta']:.4f})")
            with col2:
                st.success(f"**Lowest Time Decay:**  \n{comp_df.loc[lowest_theta_idx, 'Option']} (Theta: {comp_df.loc[lowest_theta_idx, 'Theta']:.4f})")
            with col3:
                st.warning(f"**Best Delta per Dollar:**  \n{comp_df.loc[best_value_idx, 'Option']}")
    
    # Footer
    st.divider()
    st.caption("💡 Options Analytics Module | Black-Scholes pricing and Greeks analysis")

# Standalone execution for testing
if __name__ == "__main__":
    options_module(None)
