# advanced_risk_improved.py - Enhanced Advanced Risk Analytics Module
# Comprehensive risk measurement: VaR, CVaR, GARCH, stress testing, risk decomposition
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== VALUE AT RISK (VaR) ====================

def var_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Historical VaR using empirical distribution.
    
    Args:
        returns: Array of returns
        alpha: Confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
        VaR value (negative for losses)
    """
    if len(returns) == 0:
        return np.nan
    return np.percentile(returns, alpha * 100)

def var_parametric(returns: np.ndarray, alpha: float = 0.05, 
                   distribution: str = 'normal') -> float:
    """
    Calculate Parametric VaR assuming normal or t-distribution.
    
    Args:
        distribution: 'normal' or 't' (Student's t)
    
    Returns:
        VaR value
    """
    if len(returns) == 0:
        return np.nan
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    if distribution == 'normal':
        z = norm.ppf(alpha)
        var = mu + z * sigma
    else:  # Student's t
        df = len(returns) - 1
        t_value = student_t.ppf(alpha, df)
        var = mu + t_value * sigma
    
    return var

def var_cornish_fisher(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Modified VaR using Cornish-Fisher expansion.
    Accounts for skewness and kurtosis.
    """
    if len(returns) < 4:
        return var_parametric(returns, alpha)
    
    from scipy.stats import skew, kurtosis
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    s = skew(returns)
    k = kurtosis(returns)
    
    z = norm.ppf(alpha)
    
    # Cornish-Fisher expansion
    z_cf = (z + 
            (z**2 - 1) * s / 6 + 
            (z**3 - 3*z) * k / 24 - 
            (2*z**3 - 5*z) * s**2 / 36)
    
    return mu + z_cf * sigma

def var_monte_carlo(returns: np.ndarray, alpha: float = 0.05, 
                   n_simulations: int = 10000) -> Tuple[float, np.ndarray]:
    """
    Calculate VaR using Monte Carlo simulation.
    
    Returns:
        Tuple of (VaR value, simulated returns)
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    np.random.seed(42)
    simulated_returns = np.random.normal(mu, sigma, n_simulations)
    
    var = np.percentile(simulated_returns, alpha * 100)
    
    return var, simulated_returns

# ==================== CONDITIONAL VALUE AT RISK (CVaR/ES) ====================

def cvar_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall) using historical method.
    CVaR is the expected loss given that loss exceeds VaR.
    """
    if len(returns) == 0:
        return np.nan
    
    var = var_historical(returns, alpha)
    # Expected value of returns below VaR
    tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        return var
    
    return np.mean(tail_returns)

def cvar_parametric(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate parametric CVaR assuming normal distribution.
    """
    if len(returns) == 0:
        return np.nan
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    z = norm.ppf(alpha)
    # For normal distribution: CVaR = μ - σ * φ(z)/α
    # where φ is the standard normal PDF
    cvar = mu - sigma * norm.pdf(z) / alpha
    
    return cvar

# ==================== GARCH MODELING ====================

def garch_11_estimate(returns: np.ndarray, max_iter: int = 1000) -> Tuple[float, float, float]:
    """
    Estimate GARCH(1,1) parameters: omega, alpha, beta
    Using maximum likelihood estimation.
    
    Returns:
        Tuple of (omega, alpha, beta)
    """
    if len(returns) < 10:
        # Return default parameters if insufficient data
        return 0.0001, 0.05, 0.90
    
    returns = returns - np.mean(returns)  # Demean
    
    def garch_loglikelihood(params):
        omega, alpha, beta = params
        
        # Constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
            # Prevent numerical issues
            if sigma2[t] <= 0:
                sigma2[t] = 1e-6
        
        # Log-likelihood
        loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        
        return -loglik  # Minimize negative log-likelihood
    
    # Initial guess
    initial_params = [0.0001, 0.05, 0.90]
    
    # Bounds
    bounds = [(1e-6, 1.0), (0, 1.0), (0, 1.0)]
    
    # Optimization
    try:
        result = minimize(garch_loglikelihood, initial_params, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': max_iter})
        
        if result.success:
            omega, alpha, beta = result.x
            
            # Verify stationarity
            if alpha + beta < 1:
                return omega, alpha, beta
    except:
        pass
    
    # Return defaults if optimization fails
    return 0.0001, 0.05, 0.90

def garch_11_forecast(returns: np.ndarray, steps: int = 10, 
                     params: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    Forecast volatility using GARCH(1,1) model.
    
    Args:
        returns: Historical returns
        steps: Number of periods to forecast
        params: Optional pre-estimated parameters (omega, alpha, beta)
    
    Returns:
        Array of forecasted volatilities (standard deviations)
    """
    if len(returns) < 10:
        # Use simple volatility if insufficient data
        return np.array([np.std(returns)] * steps)
    
    # Estimate parameters if not provided
    if params is None:
        omega, alpha, beta = garch_11_estimate(returns)
    else:
        omega, alpha, beta = params
    
    returns = returns - np.mean(returns)
    
    # Calculate current conditional variance
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Forecast
    forecast = np.zeros(steps)
    current_sigma2 = sigma2[-1]
    
    # Long-run variance
    long_run_var = omega / (1 - alpha - beta)
    
    for i in range(steps):
        if i == 0:
            forecast[i] = np.sqrt(omega + alpha * returns[-1]**2 + beta * current_sigma2)
        else:
            # Mean reversion to long-run variance
            weight = (alpha + beta) ** i
            forecast[i] = np.sqrt(weight * forecast[0]**2 + (1 - weight) * long_run_var)
    
    return forecast

def garch_conditional_variance(returns: np.ndarray, 
                               params: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    Calculate conditional variance series using GARCH(1,1).
    """
    if params is None:
        omega, alpha, beta = garch_11_estimate(returns)
    else:
        omega, alpha, beta = params
    
    returns = returns - np.mean(returns)
    
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    return np.sqrt(sigma2)

# ==================== STRESS TESTING ====================

def historical_stress_test(returns: pd.DataFrame, 
                          scenarios: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """
    Perform historical stress tests based on specific time periods.
    
    Args:
        returns: DataFrame of returns with date index
        scenarios: Dict mapping scenario names to (start_date, end_date) tuples
    
    Returns:
        DataFrame with scenario impacts
    """
    results = []
    
    for scenario_name, (start, end) in scenarios.items():
        try:
            scenario_returns = returns.loc[start:end]
            
            if len(scenario_returns) > 0:
                total_return = (1 + scenario_returns).prod() - 1
                max_drawdown = (scenario_returns.cumsum().cummax() - scenario_returns.cumsum()).max()
                volatility = scenario_returns.std() * np.sqrt(252)
                
                results.append({
                    'Scenario': scenario_name,
                    'Period': f"{start} to {end}",
                    'Total Return': total_return,
                    'Max Drawdown': max_drawdown,
                    'Volatility (Annual)': volatility,
                    'Days': len(scenario_returns)
                })
        except:
            continue
    
    return pd.DataFrame(results)

def parametric_stress_test(returns: np.ndarray, shock_sizes: List[float]) -> pd.DataFrame:
    """
    Perform parametric stress tests with various shock sizes.
    
    Args:
        returns: Array of returns
        shock_sizes: List of shock magnitudes (e.g., [-0.10, -0.15, -0.20])
    
    Returns:
        DataFrame with stress test results
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    current_value = 100  # Normalized to 100
    
    results = []
    
    for shock in shock_sizes:
        shocked_return = shock
        new_value = current_value * (1 + shocked_return)
        loss = current_value - new_value
        
        # Calculate probability (assuming normal distribution)
        z_score = (shock - mu) / sigma
        probability = norm.cdf(z_score) * 100
        
        results.append({
            'Shock Size': f"{shock*100:.1f}%",
            'Portfolio Value': new_value,
            'Loss': loss,
            'Loss %': shock * 100,
            'Probability': probability
        })
    
    return pd.DataFrame(results)

# ==================== RISK DECOMPOSITION ====================

def risk_contribution(returns: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """
    Calculate risk contribution of each asset to portfolio risk.
    
    Args:
        returns: DataFrame with returns for each asset
        weights: Array of portfolio weights
    
    Returns:
        DataFrame with risk contributions
    """
    # Covariance matrix
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Portfolio volatility
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Marginal contribution to risk
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    
    # Risk contribution
    risk_contrib = weights * marginal_contrib
    
    # Percentage contribution
    pct_contrib = (risk_contrib / portfolio_vol) * 100
    
    results = pd.DataFrame({
        'Asset': returns.columns,
        'Weight (%)': weights * 100,
        'Marginal Risk': marginal_contrib,
        'Risk Contribution': risk_contrib,
        'Risk Contribution (%)': pct_contrib
    })
    
    return results

def marginal_var(returns: pd.DataFrame, weights: np.ndarray, 
                alpha: float = 0.05) -> np.ndarray:
    """
    Calculate marginal VaR for each position.
    
    Returns:
        Array of marginal VaR values
    """
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_var = var_historical(portfolio_returns.values, alpha)
    
    # Calculate marginal VaR by finite differences
    epsilon = 0.01
    marginal_vars = np.zeros(len(weights))
    
    for i in range(len(weights)):
        weights_up = weights.copy()
        weights_up[i] += epsilon
        weights_up = weights_up / weights_up.sum()  # Renormalize
        
        portfolio_returns_up = (returns * weights_up).sum(axis=1)
        var_up = var_historical(portfolio_returns_up.values, alpha)
        
        marginal_vars[i] = (var_up - portfolio_var) / epsilon
    
    return marginal_vars

# ==================== BACKTESTING ====================

def var_backtest(returns: np.ndarray, var_estimates: np.ndarray, 
                alpha: float = 0.05) -> Dict:
    """
    Backtest VaR estimates using actual returns.
    
    Args:
        returns: Actual returns
        var_estimates: VaR estimates for each period
        alpha: Confidence level
    
    Returns:
        Dictionary with backtest results
    """
    # Violations (when loss exceeds VaR)
    violations = returns < var_estimates
    n_violations = np.sum(violations)
    expected_violations = int(len(returns) * alpha)
    
    violation_rate = n_violations / len(returns)
    
    # Kupiec's POF test (Proportion of Failures)
    from scipy.stats import chi2
    
    if n_violations > 0:
        lr = -2 * (
            np.log((1 - alpha)**(len(returns) - n_violations) * alpha**n_violations) -
            np.log((1 - violation_rate)**(len(returns) - n_violations) * violation_rate**n_violations)
        )
        p_value = 1 - chi2.cdf(lr, df=1)
    else:
        lr = np.nan
        p_value = np.nan
    
    # Christoffersen test for independence
    # (simplified version)
    
    results = {
        'Number of Violations': n_violations,
        'Expected Violations': expected_violations,
        'Violation Rate': violation_rate * 100,
        'Expected Rate': alpha * 100,
        'LR Statistic': lr,
        'P-Value': p_value,
        'Test Result': 'Pass' if p_value > 0.05 else 'Fail' if not np.isnan(p_value) else 'N/A'
    }
    
    return results

# ==================== VISUALIZATION FUNCTIONS ====================

def create_var_comparison_chart(returns: np.ndarray, alpha: float = 0.05) -> go.Figure:
    """Create chart comparing different VaR methods."""
    var_hist = var_historical(returns, alpha)
    var_param = var_parametric(returns, alpha, 'normal')
    var_cf = var_cornish_fisher(returns, alpha)
    var_mc, sim_returns = var_monte_carlo(returns, alpha)
    
    fig = go.Figure()
    
    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=returns,
        name='Returns Distribution',
        nbinsx=50,
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # VaR lines
    methods = [
        ('Historical', var_hist, 'red'),
        ('Parametric', var_param, 'blue'),
        ('Cornish-Fisher', var_cf, 'green'),
        ('Monte Carlo', var_mc, 'purple')
    ]
    
    for name, var_val, color in methods:
        fig.add_vline(
            x=var_val,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{name}: {var_val*100:.2f}%",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f'VaR Comparison (α={alpha*100:.0f}%)',
        xaxis_title='Returns',
        yaxis_title='Density',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_garch_forecast_chart(returns: np.ndarray, forecast: np.ndarray, 
                               historical_vol: np.ndarray) -> go.Figure:
    """Create GARCH volatility forecast visualization."""
    fig = go.Figure()
    
    # Historical volatility
    dates_hist = pd.date_range(end=datetime.now(), periods=len(historical_vol), freq='D')
    fig.add_trace(go.Scatter(
        x=dates_hist,
        y=historical_vol * 100,
        mode='lines',
        name='Historical Volatility',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    dates_forecast = pd.date_range(start=datetime.now() + timedelta(days=1), 
                                   periods=len(forecast), freq='D')
    fig.add_trace(go.Scatter(
        x=dates_forecast,
        y=forecast * 100,
        mode='lines',
        name='GARCH Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Long-run average
    long_run = np.mean(historical_vol[-60:]) * 100  # Last 60 days average
    fig.add_hline(
        y=long_run,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Long-run avg: {long_run:.2f}%"
    )
    
    fig.update_layout(
        title='GARCH Volatility Forecast',
        xaxis_title='Date',
        yaxis_title='Volatility (% annual)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_var_backtest_chart(returns: np.ndarray, var_estimates: np.ndarray) -> go.Figure:
    """Create VaR backtest visualization."""
    dates = pd.date_range(end=datetime.now(), periods=len(returns), freq='D')
    
    fig = go.Figure()
    
    # Returns
    fig.add_trace(go.Scatter(
        x=dates,
        y=returns * 100,
        mode='lines',
        name='Actual Returns',
        line=dict(color='blue', width=1)
    ))
    
    # VaR estimates
    fig.add_trace(go.Scatter(
        x=dates,
        y=var_estimates * 100,
        mode='lines',
        name='VaR Estimate',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Highlight violations
    violations = returns < var_estimates
    violation_dates = dates[violations]
    violation_returns = returns[violations] * 100
    
    fig.add_trace(go.Scatter(
        x=violation_dates,
        y=violation_returns,
        mode='markers',
        name='VaR Violations',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title='VaR Backtest',
        xaxis_title='Date',
        yaxis_title='Returns (%)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ==================== MAIN MODULE ====================

def advanced_risk_module(analysis_context: Optional[Dict] = None):
    """
    Enhanced advanced risk analytics module.
    
    Args:
        analysis_context: Optional context from main app with historical data
    """
    st.title("⚠️ Advanced Risk Analytics")
    st.markdown("""
    Comprehensive risk measurement including VaR, CVaR, GARCH volatility modeling,
    stress testing, and portfolio risk decomposition.
    """)
    
    # Sidebar for mode selection
    with st.sidebar:
        st.header("Analysis Mode")
        mode = st.selectbox(
            "Select Analysis Type",
            ["VaR & CVaR Analysis", "GARCH Modeling", "Stress Testing", 
             "Risk Decomposition", "VaR Backtesting"]
        )
    
    # Data input section
    st.header("📊 Data Input")
    
    data_source = st.radio("Data Source", ["Use Sample Data", "Upload CSV", "Use Context Data"])
    
    returns_df = None
    returns = None
    
    if data_source == "Use Sample Data":
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        returns = np.random.normal(0.0005, 0.02, 500)
        returns_df = pd.DataFrame({'Returns': returns}, index=dates)
        st.success("✅ Using sample return data (500 days)")
        
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload returns data (CSV with date index)", type=['csv'])
        if uploaded_file:
            returns_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            returns = returns_df.iloc[:, 0].values
            st.success(f"✅ Loaded {len(returns)} data points")
        else:
            st.warning("Please upload a CSV file")
            return
            
    else:  # Use context data
        if analysis_context and 'historical_data' in analysis_context:
            returns_df = analysis_context['historical_data']
            if 'Returns' in returns_df.columns:
                returns = returns_df['Returns'].values
            elif 'Return' in returns_df.columns:
                returns = returns_df['Return'].values
            else:
                # Calculate returns from price data
                if 'Close' in returns_df.columns:
                    returns = returns_df['Close'].pct_change().dropna().values
                    returns_df = pd.DataFrame({'Returns': returns}, 
                                             index=returns_df.index[1:])
                else:
                    st.error("No suitable return data found in context")
                    return
            st.success(f"✅ Using context data ({len(returns)} returns)")
        else:
            st.warning("No context data available. Please select another data source.")
            return
    
    # ==================== VAR & CVAR ANALYSIS ====================
    if mode == "VaR & CVaR Analysis":
        st.header("📉 Value at Risk (VaR) & Conditional VaR Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.selectbox(
                "Confidence Level",
                [90, 95, 99],
                index=1
            )
            alpha = (100 - confidence_level) / 100
        
        with col2:
            holding_period = st.number_input(
                "Holding Period (days)",
                min_value=1,
                max_value=252,
                value=1
            )
        
        with col3:
            portfolio_value = st.number_input(
                "Portfolio Value ($)",
                min_value=1000.0,
                value=1000000.0,
                step=10000.0,
                format="%.0f"
            )
        
        if st.button("🔍 Calculate Risk Metrics"):
            # Adjust for holding period
            returns_scaled = returns * np.sqrt(holding_period)
            
            # Calculate VaR using different methods
            var_hist = var_historical(returns_scaled, alpha)
            var_param = var_parametric(returns_scaled, alpha, 'normal')
            var_t = var_parametric(returns_scaled, alpha, 't')
            var_cf = var_cornish_fisher(returns_scaled, alpha)
            var_mc, sim_returns = var_monte_carlo(returns_scaled, alpha, 10000)
            
            # Calculate CVaR
            cvar_hist = cvar_historical(returns_scaled, alpha)
            cvar_param = cvar_parametric(returns_scaled, alpha)
            
            # Display results
            st.subheader(f"📊 Risk Metrics ({confidence_level}% Confidence)")
            
            # VaR comparison
            var_results = pd.DataFrame({
                'Method': ['Historical', 'Parametric (Normal)', 'Parametric (t-dist)', 
                          'Cornish-Fisher', 'Monte Carlo'],
                'VaR (%)': [var_hist * 100, var_param * 100, var_t * 100, 
                           var_cf * 100, var_mc * 100],
                'VaR ($)': [var_hist * portfolio_value, var_param * portfolio_value,
                           var_t * portfolio_value, var_cf * portfolio_value,
                           var_mc * portfolio_value]
            })
            
            st.dataframe(
                var_results.style.format({
                    'VaR (%)': '{:.4f}',
                    'VaR ($)': '${:,.2f}'
                }).background_gradient(subset=['VaR (%)'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
            
            # CVaR comparison
            st.subheader("📊 Conditional VaR (Expected Shortfall)")
            
            cvar_results = pd.DataFrame({
                'Method': ['Historical', 'Parametric'],
                'CVaR (%)': [cvar_hist * 100, cvar_param * 100],
                'CVaR ($)': [cvar_hist * portfolio_value, cvar_param * portfolio_value]
            })
            
            st.dataframe(
                cvar_results.style.format({
                    'CVaR (%)': '{:.4f}',
                    'CVaR ($)': '${:,.2f}'
                }).background_gradient(subset=['CVaR (%)'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Historical VaR",
                    f"${abs(var_hist * portfolio_value):,.0f}",
                    f"{var_hist * 100:.2f}%",
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Historical CVaR",
                    f"${abs(cvar_hist * portfolio_value):,.0f}",
                    f"{cvar_hist * 100:.2f}%",
                    delta_color="inverse"
                )
            
            with col3:
                ratio = abs(cvar_hist / var_hist) if var_hist != 0 else 1
                st.metric(
                    "CVaR/VaR Ratio",
                    f"{ratio:.2f}",
                    help="Higher ratio indicates fatter tails"
                )
            
            # Visualization
            st.subheader("📈 VaR Visualization")
            
            tab1, tab2 = st.tabs(["VaR Comparison", "Distribution Analysis"])
            
            with tab1:
                fig_var = create_var_comparison_chart(returns_scaled, alpha)
                st.plotly_chart(fig_var, use_container_width=True)
                
                st.info(f"""
                **Interpretation:**
                - With {confidence_level}% confidence, the maximum expected loss over {holding_period} day(s) is approximately **${abs(var_hist * portfolio_value):,.0f}**
                - If losses exceed VaR, the expected loss (CVaR) is **${abs(cvar_hist * portfolio_value):,.0f}**
                - This represents a **{abs(var_hist) * 100:.2f}%** decline in portfolio value
                """)
            
            with tab2:
                from scipy.stats import skew, kurtosis
                
                # Distribution statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Return", f"{np.mean(returns_scaled) * 100:.4f}%")
                with col2:
                    st.metric("Volatility", f"{np.std(returns_scaled) * 100:.2f}%")
                with col3:
                    st.metric("Skewness", f"{skew(returns_scaled):.2f}")
                with col4:
                    st.metric("Kurtosis", f"{kurtosis(returns_scaled):.2f}")
                
                # Q-Q plot
                fig_qq = go.Figure()
                
                sorted_returns = np.sort(returns_scaled)
                theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(returns_scaled)))
                
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Actual',
                    marker=dict(size=4)
                ))
                
                # 45-degree line
                fig_qq.add_trace(go.Scatter(
                    x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                    y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_qq.update_layout(
                    title='Q-Q Plot (Normal Distribution)',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_qq, use_container_width=True)
    
    # ==================== GARCH MODELING ====================
    elif mode == "GARCH Modeling":
        st.header("📊 GARCH Volatility Modeling")
        
        st.write("Estimate and forecast volatility using GARCH(1,1) model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider(
                "Forecast Horizon (days)",
                min_value=1,
                max_value=252,
                value=30
            )
        
        with col2:
            estimation_window = st.slider(
                "Estimation Window (days)",
                min_value=100,
                max_value=len(returns),
                value=min(252, len(returns))
            )
        
        if st.button("📈 Estimate GARCH Model"):
            with st.spinner("Estimating GARCH parameters..."):
                # Use recent data for estimation
                returns_estimation = returns[-estimation_window:]
                
                # Estimate parameters
                omega, alpha, beta = garch_11_estimate(returns_estimation)
                
                # Display parameters
                st.subheader("🔢 GARCH(1,1) Parameters")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Omega (ω)", f"{omega:.6f}")
                with col2:
                    st.metric("Alpha (α)", f"{alpha:.4f}")
                with col3:
                    st.metric("Beta (β)", f"{beta:.4f}")
                with col4:
                    persistence = alpha + beta
                    st.metric("Persistence (α+β)", f"{persistence:.4f}")
                
                # Check stationarity
                if persistence < 1:
                    st.success(f"✅ Model is stationary (α+β = {persistence:.4f} < 1)")
                else:
                    st.warning(f"⚠️ Model may not be stationary (α+β = {persistence:.4f} ≥ 1)")
                
                # Calculate long-run variance
                if persistence < 1:
                    long_run_vol = np.sqrt(omega / (1 - persistence)) * np.sqrt(252) * 100
                    st.info(f"**Long-run volatility:** {long_run_vol:.2f}% (annualized)")
                
                # Calculate conditional volatility
                conditional_vol = garch_conditional_variance(returns_estimation, (omega, alpha, beta))
                
                # Forecast
                forecast = garch_11_forecast(returns_estimation, forecast_days, (omega, alpha, beta))
                
                # Visualization
                st.subheader("📈 Volatility Analysis")
                
                fig_garch = create_garch_forecast_chart(returns_estimation, forecast, conditional_vol)
                st.plotly_chart(fig_garch, use_container_width=True)
                
                # Forecast table
                st.subheader("📊 Volatility Forecast")
                
                forecast_df = pd.DataFrame({
                    'Day': range(1, forecast_days + 1),
                    'Volatility (daily %)': forecast * 100,
                    'Volatility (annual %)': forecast * np.sqrt(252) * 100
                })
                
                st.dataframe(
                    forecast_df.style.format({
                        'Volatility (daily %)': '{:.4f}',
                        'Volatility (annual %)': '{:.2f}'
                    }).background_gradient(subset=['Volatility (annual %)'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    height=400
                )
                
                # Download forecast
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast (CSV)",
                    data=csv,
                    file_name="garch_forecast.csv",
                    mime="text/csv"
                )
                
                # Model interpretation
                st.subheader("💡 Model Interpretation")
                
                st.markdown(f"""
                **GARCH(1,1) Model:** σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
                
                - **Omega (ω = {omega:.6f})**: Base level of volatility
                - **Alpha (α = {alpha:.4f})**: Reaction to market shocks (news impact)
                - **Beta (β = {beta:.4f})**: Volatility persistence (memory effect)
                
                **Key Insights:**
                - {'High' if beta > 0.85 else 'Moderate' if beta > 0.70 else 'Low'} persistence: Volatility shocks {'persist for a long time' if beta > 0.85 else 'have moderate duration' if beta > 0.70 else 'dissipate quickly'}
                - {'High' if alpha > 0.10 else 'Moderate' if alpha > 0.05 else 'Low'} sensitivity to news
                - Half-life of shocks: ~{int(np.log(0.5) / np.log(alpha + beta))} days
                """)
    
    # ==================== STRESS TESTING ====================
    elif mode == "Stress Testing":
        st.header("🔥 Stress Testing & Scenario Analysis")
        
        st.write("Analyze portfolio performance under extreme market conditions")
        
        tab1, tab2 = st.tabs(["Historical Scenarios", "Parametric Shocks"])
        
        with tab1:
            st.subheader("📅 Historical Stress Scenarios")
            
            # Define historical scenarios
            default_scenarios = {
                '2008 Financial Crisis': ('2008-09-01', '2009-03-31'),
                'COVID-19 Crash': ('2020-02-01', '2020-04-30'),
                'Dot-com Bubble': ('2000-03-01', '2002-10-31'),
                'Black Monday 1987': ('1987-10-01', '1987-11-30'),
                'Flash Crash 2010': ('2010-05-01', '2010-06-30')
            }
            
            # Allow custom scenarios
            use_custom = st.checkbox("Add custom scenario")
            
            scenarios = default_scenarios.copy()
            
            if use_custom:
                col1, col2, col3 = st.columns(3)
                with col1:
                    custom_name = st.text_input("Scenario Name", "Custom Crisis")
                with col2:
                    custom_start = st.date_input("Start Date", datetime(2020, 1, 1))
                with col3:
                    custom_end = st.date_input("End Date", datetime(2020, 12, 31))
                
                scenarios[custom_name] = (custom_start.strftime('%Y-%m-%d'), 
                                         custom_end.strftime('%Y-%m-%d'))
            
            if st.button("🔍 Run Historical Stress Tests"):
                if returns_df is not None and len(returns_df) > 0:
                    stress_results = historical_stress_test(returns_df, scenarios)
                    
                    if not stress_results.empty:
                        st.subheader("📊 Stress Test Results")
                        
                        st.dataframe(
                            stress_results.style.format({
                                'Total Return': '{:.2%}',
                                'Max Drawdown': '{:.2%}',
                                'Volatility (Annual)': '{:.2%}'
                            }).background_gradient(subset=['Total Return'], cmap='RdYlGn'),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Visualization
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=stress_results['Scenario'],
                            y=stress_results['Total Return'] * 100,
                            marker_color=['red' if x < 0 else 'green' 
                                         for x in stress_results['Total Return']],
                            text=[f"{x:.1f}%" for x in stress_results['Total Return'] * 100],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title='Historical Scenario Returns',
                            xaxis_title='Scenario',
                            yaxis_title='Total Return (%)',
                            template='plotly_white',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No historical data available for selected scenarios")
                else:
                    st.error("No return data available")
        
        with tab2:
            st.subheader("⚡ Parametric Stress Tests")
            
            st.write("Apply hypothetical shocks to the portfolio")
            
            # Shock configuration
            col1, col2 = st.columns(2)
            
            with col1:
                shock_type = st.selectbox(
                    "Shock Type",
                    ["Market Crash", "Custom Shocks"]
                )
            
            with col2:
                portfolio_value_stress = st.number_input(
                    "Portfolio Value ($)",
                    min_value=1000.0,
                    value=1000000.0,
                    step=10000.0,
                    format="%.0f",
                    key="stress_pv"
                )
            
            if shock_type == "Market Crash":
                shock_sizes = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.40]
                st.info("Simulating market crashes of various magnitudes")
            else:
                num_shocks = st.slider("Number of Shock Scenarios", 3, 10, 5)
                shock_sizes = []
                
                cols = st.columns(num_shocks)
                for i, col in enumerate(cols):
                    with col:
                        shock = st.number_input(
                            f"Shock {i+1} (%)",
                            value=-10.0 - i*5,
                            step=1.0,
                            key=f"shock_{i}"
                        ) / 100
                        shock_sizes.append(shock)
            
            if st.button("🔍 Run Parametric Stress Tests"):
                stress_results = parametric_stress_test(returns, shock_sizes)
                
                # Scale by portfolio value
                stress_results['Portfolio Value'] *= (portfolio_value_stress / 100)
                stress_results['Loss'] *= (portfolio_value_stress / 100)
                
                st.subheader("📊 Stress Test Results")
                
                st.dataframe(
                    stress_results.style.format({
                        'Portfolio Value': '${:,.2f}',
                        'Loss': '${:,.2f}',
                        'Loss %': '{:.2f}',
                        'Probability': '{:.4f}'
                    }).background_gradient(subset=['Loss %'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Portfolio Value Under Stress', 'Loss Distribution')
                )
                
                fig.add_trace(
                    go.Bar(
                        x=stress_results['Shock Size'],
                        y=stress_results['Portfolio Value'],
                        name='Portfolio Value',
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=stress_results['Shock Size'],
                        y=stress_results['Loss'],
                        name='Loss Amount',
                        marker_color='red'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Shock Size", row=1, col=1)
                fig.update_xaxes(title_text="Shock Size", row=1, col=2)
                fig.update_yaxes(title_text="Value ($)", row=1, col=1)
                fig.update_yaxes(title_text="Loss ($)", row=1, col=2)
                
                fig.update_layout(
                    template='plotly_white',
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Worst case analysis
                worst_case = stress_results.loc[stress_results['Loss'].idxmax()]
                
                st.error(f"""
                **Worst Case Scenario:**
                - Shock Size: {worst_case['Shock Size']}
                - Portfolio Value: ${worst_case['Portfolio Value']:,.2f}
                - Total Loss: ${worst_case['Loss']:,.2f}
                - Probability: {worst_case['Probability']:.4f}%
                """)
    
    # ==================== RISK DECOMPOSITION ====================
    elif mode == "Risk Decomposition":
        st.header("🔍 Portfolio Risk Decomposition")
        
        st.write("Analyze the contribution of each asset to total portfolio risk")
        
        # For this mode, we need multi-asset data
        st.subheader("📊 Portfolio Configuration")
        
        # Check if we have multi-asset data
        if returns_df is not None and len(returns_df.columns) > 1:
            st.success(f"✅ Multi-asset portfolio detected ({len(returns_df.columns)} assets)")
            
            # Weight configuration
            st.write("**Asset Weights:**")
            
            weights = []
            cols = st.columns(min(5, len(returns_df.columns)))
            
            for i, asset in enumerate(returns_df.columns):
                with cols[i % len(cols)]:
                    weight = st.number_input(
                        f"{asset} (%)",
                        value=100.0 / len(returns_df.columns),
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        key=f"weight_{i}"
                    ) / 100
                    weights.append(weight)
            
            weights = np.array(weights)
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                st.info(f"Weights normalized to sum to 100%")
            
            if st.button("🔍 Analyze Risk Decomposition"):
                # Calculate risk contributions
                risk_contrib_df = risk_contribution(returns_df, weights)
                
                st.subheader("📊 Risk Contribution Analysis")
                
                st.dataframe(
                    risk_contrib_df.style.format({
                        'Weight (%)': '{:.2f}',
                        'Marginal Risk': '{:.6f}',
                        'Risk Contribution': '{:.6f}',
                        'Risk Contribution (%)': '{:.2f}'
                    }).background_gradient(subset=['Risk Contribution (%)'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Weight vs Risk Contribution
                    fig_pie1 = go.Figure(data=[go.Pie(
                        labels=risk_contrib_df['Asset'],
                        values=risk_contrib_df['Weight (%)'],
                        title='Portfolio Weights'
                    )])
                    fig_pie1.update_layout(height=400)
                    st.plotly_chart(fig_pie1, use_container_width=True)
                
                with col2:
                    fig_pie2 = go.Figure(data=[go.Pie(
                        labels=risk_contrib_df['Asset'],
                        values=risk_contrib_df['Risk Contribution (%)'],
                        title='Risk Contributions'
                    )])
                    fig_pie2.update_layout(height=400)
                    st.plotly_chart(fig_pie2, use_container_width=True)
                
                # Comparison chart
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    name='Weight',
                    x=risk_contrib_df['Asset'],
                    y=risk_contrib_df['Weight (%)']
                ))
                
                fig_comparison.add_trace(go.Bar(
                    name='Risk Contribution',
                    x=risk_contrib_df['Asset'],
                    y=risk_contrib_df['Risk Contribution (%)']
                ))
                
                fig_comparison.update_layout(
                    title='Weight vs Risk Contribution',
                    xaxis_title='Asset',
                    yaxis_title='Percentage (%)',
                    barmode='group',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Calculate marginal VaR
                st.subheader("📉 Marginal VaR Analysis")
                
                alpha_mvar = st.selectbox("Confidence Level for Marginal VaR", [90, 95, 99], index=1)
                mvar = marginal_var(returns_df, weights, (100 - alpha_mvar) / 100)
                
                mvar_df = pd.DataFrame({
                    'Asset': returns_df.columns,
                    'Marginal VaR': mvar,
                    'Interpretation': ['Increase' if mv > 0 else 'Decrease' for mv in mvar]
                })
                
                st.dataframe(
                    mvar_df.style.format({
                        'Marginal VaR': '{:.6f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.info("""
                **Marginal VaR** shows how much the portfolio VaR would change if you 
                increased the allocation to each asset by 1%. Positive values indicate 
                that increasing the position would increase portfolio risk.
                """)
                
        else:
            st.warning("⚠️ Risk decomposition requires multi-asset portfolio data")
            st.info("Please upload a CSV file with multiple asset returns or use appropriate context data")
    
    # ==================== VAR BACKTESTING ====================
    elif mode == "VaR Backtesting":
        st.header("✅ VaR Model Backtesting")
        
        st.write("Validate VaR model accuracy using historical performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha_backtest = st.selectbox("Confidence Level", [90, 95, 99], index=1, key="backtest_alpha")
            alpha = (100 - alpha_backtest) / 100
        
        with col2:
            var_method = st.selectbox(
                "VaR Method",
                ["Historical", "Parametric", "Cornish-Fisher", "GARCH"]
            )
        
        with col3:
            window_size = st.slider(
                "Rolling Window (days)",
                min_value=50,
                max_value=500,
                value=252
            )
        
        if st.button("📊 Run Backtest"):
            with st.spinner("Running backtest..."):
                # Calculate rolling VaR estimates
                n = len(returns)
                var_estimates = np.zeros(n)
                
                for i in range(window_size, n):
                    window_returns = returns[i-window_size:i]
                    
                    if var_method == "Historical":
                        var_estimates[i] = var_historical(window_returns, alpha)
                    elif var_method == "Parametric":
                        var_estimates[i] = var_parametric(window_returns, alpha, 'normal')
                    elif var_method == "Cornish-Fisher":
                        var_estimates[i] = var_cornish_fisher(window_returns, alpha)
                    else:  # GARCH
                        # Use GARCH for 1-day ahead forecast
                        omega, alpha_g, beta_g = garch_11_estimate(window_returns)
                        forecast = garch_11_forecast(window_returns, 1, (omega, alpha_g, beta_g))
                        # Approximate VaR using forecasted volatility
                        z = norm.ppf(alpha)
                        var_estimates[i] = z * forecast[0]
                
                # Backtest
                backtest_returns = returns[window_size:]
                backtest_var = var_estimates[window_size:]
                
                backtest_results = var_backtest(backtest_returns, backtest_var, alpha)
                
                # Display results
                st.subheader("📊 Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Violations",
                        backtest_results['Number of Violations'],
                        f"Expected: {backtest_results['Expected Violations']}"
                    )
                
                with col2:
                    st.metric(
                        "Violation Rate",
                        f"{backtest_results['Violation Rate']:.2f}%",
                        f"Expected: {backtest_results['Expected Rate']:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "Test Statistic",
                        f"{backtest_results['LR Statistic']:.2f}" if not np.isnan(backtest_results['LR Statistic']) else "N/A"
                    )
                
                with col4:
                    result_color = "🟢" if backtest_results['Test Result'] == 'Pass' else "🔴"
                    st.metric(
                        "Test Result",
                        f"{result_color} {backtest_results['Test Result']}"
                    )
                
                # Interpretation
                if backtest_results['Test Result'] == 'Pass':
                    st.success(f"""
                    ✅ **Model Performance: ACCEPTABLE**
                    
                    The VaR model accurately predicted risk at the {alpha_backtest}% confidence level.
                    The number of violations is consistent with the expected rate.
                    """)
                elif backtest_results['Test Result'] == 'Fail':
                    st.error(f"""
                    ❌ **Model Performance: POOR**
                    
                    The VaR model {'underestimated' if backtest_results['Violation Rate'] > backtest_results['Expected Rate'] else 'overestimated'} 
                    risk. Consider recalibrating the model or using a different methodology.
                    """)
                
                # Visualization
                fig_backtest = create_var_backtest_chart(backtest_returns, backtest_var)
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Detailed analysis
                st.subheader("📈 Detailed Analysis")
                
                # Violation clustering
                violations = backtest_returns < backtest_var
                violation_indices = np.where(violations)[0]
                
                if len(violation_indices) > 1:
                    # Calculate gaps between violations
                    gaps = np.diff(violation_indices)
                    avg_gap = np.mean(gaps)
                    
                    st.write(f"**Average gap between violations:** {avg_gap:.1f} days")
                    
                    # Clustering test (simple version)
                    if avg_gap < len(backtest_returns) / backtest_results['Number of Violations'] * 0.5:
                        st.warning("⚠️ Violations appear to be clustered, suggesting model may not capture volatility dynamics well")
                    else:
                        st.success("✅ Violations appear to be well-distributed over time")
                
                # Summary statistics
                summary_df = pd.DataFrame({
                    'Metric': ['Total Observations', 'Number of Violations', 'Expected Violations',
                              'Violation Rate (%)', 'Expected Rate (%)', 'Excess Violations'],
                    'Value': [
                        len(backtest_returns),
                        backtest_results['Number of Violations'],
                        backtest_results['Expected Violations'],
                        backtest_results['Violation Rate'],
                        backtest_results['Expected Rate'],
                        backtest_results['Number of Violations'] - backtest_results['Expected Violations']
                    ]
                })
                
                st.dataframe(
                    summary_df.style.format({'Value': '{:.2f}'}),
                    use_container_width=True,
                    hide_index=True
                )
    
    # Footer
    st.divider()
    st.caption("⚠️ Advanced Risk Analytics Module | VaR, CVaR, GARCH, and stress testing")

# Standalone execution for testing
if __name__ == "__main__":
    advanced_risk_module(None)
