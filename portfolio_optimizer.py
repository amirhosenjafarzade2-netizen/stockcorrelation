# portfolio_optimizer.py
# Portfolio Optimizer Module using Genetic Algorithm and Monte Carlo
# Integrates with app.py - uses yfinance for data, scipy for optimization

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import List, Dict, Tuple

# Assume get_finviz_tickers is available from screener.py (import if needed)
# from screener import get_finviz_tickers  # Uncomment if separate file

# Dummy for testing - replace with actual import
def get_finviz_tickers(sector: str) -> List[str]:
    # Placeholder: In real app, use the function from screener.py
    if sector == "Technology":
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    # Add more sectors...
    return []

def fetch_stock_data(tickers: List[str], years_back: int) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Fetch historical prices and fundamentals."""
    start_date = (date.today() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    prices = prices.dropna(how='all')
    
    fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                'earningsGrowth': info.get('earningsGrowth', 0.0),
                'revenueGrowth': info.get('revenueGrowth', 0.0),
                'dividendYield': info.get('dividendYield', 0.0),
                'trailingPE': info.get('trailingPE', np.nan),
                'returnOnEquity': info.get('returnOnEquity', 0.0),
                'beta': info.get('beta', 1.0),
                'targetMeanPrice': info.get('targetMeanPrice', np.nan)
            }
        except:
            fundamentals[ticker] = {}
    
    return prices, fundamentals

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()

def adjust_expected_returns(historical_returns: np.ndarray, fundamentals: Dict[str, Dict]) -> np.ndarray:
    """Forward-adjust historical returns using fundamentals."""
    adjusted = historical_returns.copy()
    for i, ticker in enumerate(fundamentals.keys()):
        growth = (fundamentals[ticker].get('earningsGrowth', 0) + fundamentals[ticker].get('revenueGrowth', 0)) / 2
        roe_adjust = fundamentals[ticker].get('returnOnEquity', 0) * 0.1  # Small boost for high ROE
        adjusted[i] += growth + roe_adjust
    return adjusted

def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> Tuple[float, float]:
    """Calculate portfolio return and volatility."""
    port_return = np.sum(weights * mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe

def objective_function(weights: np.ndarray, args: Tuple) -> float:
    """Negative Sharpe for minimization (GA maximizes by negating)."""
    mean_returns, cov_matrix, risk_free_rate, max_stocks = args
    # Constrain to max_stocks: set weights to 0 for excess (but GA bounds handle this)
    active = np.sum(weights > 0.01)  # Count stocks with meaningful weight
    if active > max_stocks:
        return np.inf  # Penalty
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe  # Maximize Sharpe

def optimize_portfolio(returns_df: pd.DataFrame, max_stocks: int, use_fundamentals: bool, fundamentals: Dict) -> Dict:
    """Use genetic algorithm (differential_evolution) to optimize."""
    mean_returns = returns_df.mean().values * 252  # Annualize
    cov_matrix = returns_df.cov().values * 252
    
    if use_fundamentals:
        mean_returns = adjust_expected_returns(mean_returns, fundamentals)
    
    num_assets = len(returns_df.columns)
    bounds = [(0.0, 1.0)] * num_assets  # No short-selling
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
    
    args = (mean_returns, cov_matrix, 0.02, max_stocks)  # Risk-free rate 2%
    
    result = differential_evolution(
        objective_function,
        bounds,
        args=(args,),
        constraints=[constraints],
        strategy='best1bin',
        popsize=15,
        maxiter=100,
        tol=0.01,
        workers=1  # Single-thread for Streamlit
    )
    
    if not result.success:
        raise ValueError("Optimization failed.")
    
    weights = result.x
    port_return, port_vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    
    # Select best stocks (non-zero weights)
    best_indices = np.argsort(weights)[-max_stocks:]
    best_tickers = returns_df.columns[best_indices]
    best_weights = weights[best_indices]
    best_weights /= best_weights.sum()  # Renormalize
    
    return {
        'tickers': best_tickers.tolist(),
        'weights': best_weights.tolist(),
        'expected_return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe
    }

def monte_carlo_simulation(portfolio: Dict, prices: pd.DataFrame, iterations: int, timespan_days: int) -> np.ndarray:
    """Monte Carlo for future portfolio value distribution."""
    tickers = portfolio['tickers']
    weights = np.array(portfolio['weights'])
    
    current_prices = prices[tickers].iloc[-1].values
    current_value = np.sum(current_prices * weights * 100)  # Assume $100 initial
    
    returns_df = calculate_returns(prices[tickers])
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    # Simulate paths (geometric Brownian motion)
    sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (iterations, timespan_days))
    sim_prices = current_prices * np.exp(np.cumsum(sim_returns, axis=1))
    
    # Portfolio values
    port_values = np.dot(sim_prices, weights * 100)  # Scale to $100 initial
    
    return port_values[:, -1]  # Final values

def render_portfolio_optimizer() -> None:
    st.subheader("Portfolio Optimizer • Genetic Algorithm + Monte Carlo")

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        max_stocks = st.slider("Max Stocks in Portfolio", 3, 20, 5)
    with col2:
        years_back = st.slider("Historical Years", 1, 10, 5)
    with col3:
        use_fundamentals = st.checkbox("Use Fundamentals (Forward-Adjust)", value=True)

    ticker_mode = st.radio("Ticker Selection", ["Specific Tickers", "Auto from Finviz Sector"])
    
    if ticker_mode == "Specific Tickers":
        tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,META")
        candidate_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        sector = st.selectbox("Sector", ["Technology", "Healthcare", "Financials", "Energy"])  # Add more
        candidate_tickers = get_finviz_tickers(sector)[:50]  # Limit to 50 for perf
    
    if len(candidate_tickers) < max_stocks:
        st.error(f"Not enough candidates ({len(candidate_tickers)}). Need at least {max_stocks}.")
        return
    
    run_opt = st.button("Optimize Portfolio", type="primary")
    
    if run_opt:
        with st.spinner("Fetching data..."):
            prices, fundamentals = fetch_stock_data(candidate_tickers, years_back)
        
        if prices.empty:
            st.error("No data fetched.")
            return
        
        returns_df = calculate_returns(prices)
        
        with st.spinner("Running Genetic Algorithm..."):
            try:
                portfolio = optimize_portfolio(returns_df, max_stocks, use_fundamentals, fundamentals)
            except ValueError as e:
                st.error(str(e))
                return
        
        st.success("Optimization Complete!")
        
        # Display Results
        results_df = pd.DataFrame({
            "Stock": portfolio['tickers'],
            "Weight (%)": [w * 100 for w in portfolio['weights']]
        })
        st.dataframe(results_df.style.format({"Weight (%)": "{:.2f}"}), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Expected Annual Return", f"{portfolio['expected_return']:.2%}")
        with col2: st.metric("Annual Volatility", f"{portfolio['volatility']:.2%}")
        with col3: st.metric("Sharpe Ratio", f"{portfolio['sharpe']:.2f}")
        
        # Pie Chart
        fig_pie = px.pie(results_df, values="Weight (%)", names="Stock", title="Portfolio Allocation")
        st.plotly_chart(fig_pie)
        
        # Monte Carlo Option
        run_mc = st.checkbox("Run Monte Carlo Simulation")
        if run_mc:
            iterations = st.number_input("Iterations", 1000, 100000, 10000)
            timespan_years = st.number_input("Future Timespan (Years)", 1, 10, 1)
            timespan_days = timespan_years * 252
            
            with st.spinner("Running Monte Carlo..."):
                final_values = monte_carlo_simulation(portfolio, prices, iterations, timespan_days)
            
            fig_hist = px.histogram(final_values, nbins=50, title=f"Portfolio Value Distribution After {timespan_years} Years ($100 Initial)")
            fig_hist.update_layout(xaxis_title="Final Value ($)", yaxis_title="Frequency")
            st.plotly_chart(fig_hist)
            
            p5, p50, p95 = np.percentile(final_values, [5, 50, 95])
            st.metric("Median Outcome", f"${p50:.2f}")
            st.metric("5th Percentile (Worst Case)", f"${p5:.2f}")
            st.metric("95th Percentile (Best Case)", f"${p95:.2f}")

if __name__ == "__main__":
    st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
    render_portfolio_optimizer()
