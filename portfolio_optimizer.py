# portfolio_optimizer.py
# Portfolio Optimizer Module using Genetic Algorithm and Monte Carlo
# FIXED VERSION - Constraint handling, UI improvements, better optimization

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional
import time

# Import sector function from screener (with fallback)
try:
    from screener import get_finviz_tickers
    HAS_SCREENER = True
except ImportError:
    HAS_SCREENER = False
    st.warning("⚠️ Screener module not available. Finviz sector selection disabled.")


def get_all_sectors_tickers(max_per_sector: int = 20) -> List[str]:
    """Fetch tickers from all sectors"""
    if not HAS_SCREENER:
        return []
    
    all_sectors = [
        "Technology", "Healthcare", "Financials", "Energy",
        "Consumer Discretionary", "Consumer Staples", "Industrials",
        "Basic Materials", "Communication Services", "Utilities", "Real Estate"
    ]
    
    all_tickers = []
    for sector in all_sectors:
        tickers = get_finviz_tickers(sector)
        all_tickers.extend(tickers[:max_per_sector])
    
    return list(set(all_tickers))  # Remove duplicates


def fetch_stock_data(tickers: List[str], years_back: int) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Fetch historical prices and fundamentals."""
    start_date = (date.today() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    if isinstance(prices, pd.DataFrame):
        if 'Close' in prices.columns:
            prices = prices['Close']
        elif len(prices.columns.levels) > 1 if hasattr(prices.columns, 'levels') else False:
            prices = prices['Close'] if 'Close' in prices.columns.get_level_values(0) else prices
    
    prices = prices.dropna(how='all')
    
    fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                'earningsGrowth': info.get('earningsGrowth', 0.0) or 0.0,
                'revenueGrowth': info.get('revenueGrowth', 0.0) or 0.0,
                'dividendYield': info.get('dividendYield', 0.0) or 0.0,
                'trailingPE': info.get('trailingPE', np.nan),
                'returnOnEquity': info.get('returnOnEquity', 0.0) or 0.0,
                'beta': info.get('beta', 1.0) or 1.0,
                'targetMeanPrice': info.get('targetMeanPrice', np.nan)
            }
        except:
            fundamentals[ticker] = {
                'earningsGrowth': 0.0,
                'revenueGrowth': 0.0,
                'dividendYield': 0.0,
                'trailingPE': np.nan,
                'returnOnEquity': 0.0,
                'beta': 1.0,
                'targetMeanPrice': np.nan
            }
    
    return prices, fundamentals


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def adjust_expected_returns(historical_returns: np.ndarray, fundamentals: Dict[str, Dict], tickers: List[str]) -> np.ndarray:
    """Forward-adjust historical returns using fundamentals."""
    adjusted = historical_returns.copy()
    for i, ticker in enumerate(tickers):
        if ticker in fundamentals:
            growth = (fundamentals[ticker].get('earningsGrowth', 0) + 
                     fundamentals[ticker].get('revenueGrowth', 0)) / 2
            roe_adjust = fundamentals[ticker].get('returnOnEquity', 0) * 0.1
            adjusted[i] += growth + roe_adjust
    return adjusted


def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, 
                         risk_free_rate: float = 0.02) -> Tuple[float, float, float]:
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    port_return = np.sum(weights * mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def negative_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, 
                   risk_free_rate: float) -> float:
    """Negative Sharpe for minimization."""
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe


def optimize_portfolio(returns_df: pd.DataFrame, max_stocks: int, use_fundamentals: bool, 
                      fundamentals: Dict, risk_free_rate: float = 0.02,
                      min_weight: float = 0.01) -> Dict:
    """Optimize portfolio using scipy minimize with SLSQP."""
    mean_returns = returns_df.mean().values * 252  # Annualize
    cov_matrix = returns_df.cov().values * 252
    
    if use_fundamentals:
        mean_returns = adjust_expected_returns(mean_returns, fundamentals, returns_df.columns.tolist())
    
    num_assets = len(returns_df.columns)
    
    # Initial guess: equal weights
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Bounds: 0 to 1 for each weight
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda w: max_stocks - np.sum(w > min_weight)}  # Max stocks constraint
    ]
    
    # Optimize
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        # Fallback: try without max_stocks constraint
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        result = minimize(
            negative_sharpe,
            initial_weights,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    weights = result.x
    
    # Filter to top stocks
    top_indices = np.argsort(weights)[-max_stocks:]
    top_indices = top_indices[weights[top_indices] > min_weight]
    
    if len(top_indices) == 0:
        top_indices = np.argsort(weights)[-max_stocks:]
    
    best_tickers = returns_df.columns[top_indices].tolist()
    best_weights = weights[top_indices]
    best_weights /= best_weights.sum()  # Renormalize
    
    port_return, port_vol, sharpe = portfolio_performance(best_weights, mean_returns[top_indices], 
                                                           cov_matrix[np.ix_(top_indices, top_indices)],
                                                           risk_free_rate)
    
    return {
        'tickers': best_tickers,
        'weights': best_weights.tolist(),
        'expected_return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe
    }


def monte_carlo_simulation(portfolio: Dict, prices: pd.DataFrame, iterations: int, 
                          timespan_days: int, initial_investment: float = 10000) -> np.ndarray:
    """Monte Carlo for future portfolio value distribution."""
    tickers = portfolio['tickers']
    weights = np.array(portfolio['weights'])
    
    # Ensure we have the right tickers in prices
    available_tickers = [t for t in tickers if t in prices.columns]
    if len(available_tickers) != len(tickers):
        missing = set(tickers) - set(available_tickers)
        raise ValueError(f"Missing price data for: {missing}")
    
    # Get current prices
    current_prices = prices[available_tickers].iloc[-1]
    if current_prices.isna().any():
        raise ValueError("Some current prices are NaN")
    
    current_prices = current_prices.values
    
    # Calculate returns
    returns_df = calculate_returns(prices[available_tickers])
    
    # Check for sufficient data
    if len(returns_df) < 20:
        raise ValueError(f"Insufficient data for simulation: only {len(returns_df)} days available")
    
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    # Validate covariance matrix
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        raise ValueError("Invalid covariance matrix (contains NaN or Inf)")
    
    # Make sure cov_matrix is positive semi-definite
    try:
        np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Add small diagonal to make it positive definite
        cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-6
    
    # Simulate paths (geometric Brownian motion)
    try:
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (iterations, timespan_days))
    except Exception as e:
        raise ValueError(f"Failed to generate random returns: {str(e)}")
    
    # Calculate cumulative returns and prices
    cumulative_returns = np.cumsum(sim_returns, axis=1)
    sim_prices = current_prices[np.newaxis, np.newaxis, :] * np.exp(cumulative_returns)
    
    # Calculate shares owned
    shares = (weights * initial_investment) / current_prices
    
    # Portfolio values
    port_values = np.dot(sim_prices[:, :, :], shares)
    
    return port_values[:, -1]  # Final values


def render_portfolio_optimizer() -> None:
    st.subheader("📊 Portfolio Optimizer • Genetic Algorithm + Monte Carlo")
    st.caption("Build optimal portfolios maximizing risk-adjusted returns")

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Optimization Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_stocks = st.slider("Max Stocks in Portfolio", 3, 20, 8,
                                  help="Maximum number of stocks to include")
            years_back = st.slider("Historical Years", 1, 10, 3,
                                  help="Years of price history for optimization")
        
        with col2:
            risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5,
                                            help="For Sharpe ratio calculation") / 100
            min_weight = st.number_input("Min Weight per Stock (%)", 0.1, 10.0, 1.0, 0.5,
                                        help="Minimum allocation per stock") / 100
        
        with col3:
            use_fundamentals = st.checkbox("Use Fundamentals", value=True,
                                          help="Adjust expected returns using fundamentals")
            rebalancing_freq = st.selectbox("Rebalancing Frequency", 
                                           ["Monthly", "Quarterly", "Annually"],
                                           help="Assumed portfolio rebalancing frequency")

    # ═══════════════════════════════════════════════════════════════════════
    # TICKER SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🎯 Stock Universe Selection")
    
    ticker_mode = st.radio("Selection Method", 
                          ["Manual Entry", "Finviz Sectors (Auto)"],
                          horizontal=True)
    
    candidate_tickers = []
    
    if ticker_mode == "Manual Entry":
        tickers_input = st.text_area(
            "Enter Tickers (one per line or comma-separated)",
            "AAPL\nMSFT\nGOOGL\nAMZN\nMETA\nTSLA\nNVDA\nJPM",
            height=150,
            help="Enter stock tickers you want to consider"
        )
        candidate_tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
        
    else:
        if not HAS_SCREENER:
            st.error("❌ Screener module not available. Please use Manual Entry mode.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sector_options = [
                "All Sectors",
                "Technology",
                "Healthcare",
                "Financials",
                "Energy",
                "Consumer Discretionary",
                "Consumer Staples",
                "Industrials",
                "Basic Materials",
                "Communication Services",
                "Utilities",
                "Real Estate"
            ]
            
            sector = st.selectbox("Select Sector", sector_options, index=0)
        
        with col2:
            max_candidates = st.number_input("Max Candidates", 10, 200, 50, 10,
                                            help="Maximum stocks to fetch per sector")
        
        fetch_clicked = st.button("🔍 Fetch Sector Tickers", type="primary", use_container_width=True)
        
        # Auto-fetch on first load if not already fetched
        if fetch_clicked or ('finviz_fetched' not in st.session_state):
            with st.spinner(f"Fetching tickers from {sector}..."):
                try:
                    if sector == "All Sectors":
                        candidate_tickers = get_all_sectors_tickers(max_per_sector=10)[:max_candidates]
                    else:
                        candidate_tickers = get_finviz_tickers(sector)[:max_candidates]
                    
                    if candidate_tickers:
                        st.success(f"✅ Loaded {len(candidate_tickers)} tickers")
                        st.session_state['candidate_tickers'] = candidate_tickers
                        st.session_state['finviz_fetched'] = True
                    else:
                        st.error("❌ No tickers found. Try a different sector or use Manual Entry.")
                        st.session_state['candidate_tickers'] = []
                except Exception as e:
                    st.error(f"❌ Error fetching tickers: {str(e)}")
                    st.session_state['candidate_tickers'] = []
        
        # Use session state if available
        if 'candidate_tickers' in st.session_state:
            candidate_tickers = st.session_state['candidate_tickers']
    
    # Show ticker preview
    if candidate_tickers:
        with st.expander(f"📋 Candidate Tickers ({len(candidate_tickers)})", expanded=False):
            st.write(", ".join(candidate_tickers[:50]))
            if len(candidate_tickers) > 50:
                st.caption(f"... and {len(candidate_tickers) - 50} more")
    
    # Validation
    if len(candidate_tickers) < max_stocks:
        st.error(f"❌ Not enough candidates ({len(candidate_tickers)}). Need at least {max_stocks}.")
        st.stop()
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUN OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    
    run_opt = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)
    
    if run_opt:
        with st.spinner("Fetching price data..."):
            prices, fundamentals = fetch_stock_data(candidate_tickers, years_back)
        
        if prices.empty:
            st.error("❌ No price data available. Check tickers or try different time period.")
            st.stop()
        
        # Filter to stocks with sufficient data
        valid_tickers = [col for col in prices.columns if prices[col].notna().sum() > 252]
        
        if len(valid_tickers) < max_stocks:
            st.warning(f"⚠️ Only {len(valid_tickers)} stocks have sufficient data. Adjusting max_stocks.")
            max_stocks = len(valid_tickers)
        
        prices = prices[valid_tickers]
        returns_df = calculate_returns(prices)
        
        st.info(f"📊 Analyzing {len(valid_tickers)} stocks with {len(prices)} trading days")
        
        with st.spinner("Running optimization algorithm..."):
            try:
                portfolio = optimize_portfolio(
                    returns_df, 
                    max_stocks, 
                    use_fundamentals, 
                    fundamentals,
                    risk_free_rate,
                    min_weight
                )
            except ValueError as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.info("💡 Try: reducing max stocks, increasing history, or using different tickers")
                st.stop()
        
        st.success("✅ Optimization Complete!")
        
        # ═══════════════════════════════════════════════════════════════
        # RESULTS DISPLAY
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📈 Optimized Portfolio")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Annual Return", f"{portfolio['expected_return']:.2%}")
        with col2:
            st.metric("Annual Volatility", f"{portfolio['volatility']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio['sharpe']:.2f}")
        with col4:
            st.metric("Number of Stocks", len(portfolio['tickers']))
        
        # Allocation Table
        st.markdown("#### 📊 Portfolio Allocation")
        
        results_df = pd.DataFrame({
            "Stock": portfolio['tickers'],
            "Weight (%)": [w * 100 for w in portfolio['weights']],
            "Expected Return (%)": [returns_df[t].mean() * 252 * 100 for t in portfolio['tickers']],
            "Volatility (%)": [returns_df[t].std() * np.sqrt(252) * 100 for t in portfolio['tickers']]
        })
        
        # Add fundamental data if available
        if use_fundamentals:
            results_df["P/E Ratio"] = [fundamentals.get(t, {}).get('trailingPE', np.nan) 
                                       for t in portfolio['tickers']]
            results_df["ROE (%)"] = [fundamentals.get(t, {}).get('returnOnEquity', 0) * 100 
                                     for t in portfolio['tickers']]
        
        st.dataframe(
            results_df.style.format({
                "Weight (%)": "{:.2f}",
                "Expected Return (%)": "{:.2f}",
                "Volatility (%)": "{:.2f}",
                "P/E Ratio": "{:.1f}",
                "ROE (%)": "{:.1f}"
            }).background_gradient(subset=["Weight (%)"], cmap="Greens"),
            use_container_width=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie Chart
            st.markdown("#### 🥧 Allocation Breakdown")
            fig_pie = px.pie(
                results_df, 
                values="Weight (%)", 
                names="Stock", 
                title="Portfolio Weights",
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Risk-Return Scatter
            st.markdown("#### 📍 Risk-Return Profile")
            fig_scatter = px.scatter(
                results_df,
                x="Volatility (%)",
                y="Expected Return (%)",
                size="Weight (%)",
                text="Stock",
                title="Individual Stock Positions",
                color="Weight (%)",
                color_continuous_scale="Viridis"
            )
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Historical Performance
        st.markdown("#### 📈 Historical Portfolio Performance")
        
        portfolio_hist = (returns_df[portfolio['tickers']] * portfolio['weights']).sum(axis=1)
        cumulative_returns = (1 + portfolio_hist).cumprod() * 100
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark if SPY is available
        if 'SPY' in prices.columns:
            spy_returns = calculate_returns(prices[['SPY']])
            spy_cumulative = (1 + spy_returns['SPY']).cumprod() * 100
            fig_perf.add_trace(go.Scatter(
                x=spy_cumulative.index,
                y=spy_cumulative.values,
                mode='lines',
                name='SPY (Benchmark)',
                line=dict(color='gray', width=1, dash='dash')
            ))
        
        fig_perf.update_layout(
            title="Growth of $100",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════
        # MONTE CARLO SIMULATION
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🎲 Monte Carlo Simulation")
        
        with st.expander("Run Monte Carlo Future Value Projection", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mc_iterations = st.number_input("Iterations", 1000, 100000, 10000, 1000,
                                               help="More iterations = more accurate")
            with col2:
                mc_years = st.number_input("Years to Project", 1, 30, 5,
                                          help="Future timespan")
            with col3:
                initial_investment = st.number_input("Initial Investment ($)", 1000, 1000000, 10000, 1000)
            
            if st.button("🎯 Run Simulation"):
                with st.spinner(f"Running {mc_iterations:,} Monte Carlo simulations..."):
                    timespan_days = mc_years * 252
                    final_values = monte_carlo_simulation(
                        portfolio, prices, mc_iterations, timespan_days, initial_investment
                    )
                
                # Results
                p5, p25, p50, p75, p95 = np.percentile(final_values, [5, 25, 50, 75, 95])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Median Outcome", f"${p50:,.0f}", 
                             delta=f"{((p50/initial_investment - 1) * 100):.1f}%")
                with col2:
                    st.metric("5th Percentile", f"${p5:,.0f}",
                             help="Worst case (95% confidence)")
                with col3:
                    st.metric("95th Percentile", f"${p95:,.0f}",
                             help="Best case (95% confidence)")
                with col4:
                    prob_loss = (final_values < initial_investment).sum() / len(final_values) * 100
                    st.metric("Probability of Loss", f"{prob_loss:.1f}%")
                
                # Distribution plot
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    name='Distribution',
                    marker_color='lightblue'
                ))
                
                # Add percentile lines
                for pct, val, name, color in [
                    (5, p5, '5th', 'red'),
                    (50, p50, 'Median', 'green'),
                    (95, p95, '95th', 'blue')
                ]:
                    fig_hist.add_vline(x=val, line_dash="dash", line_color=color,
                                      annotation_text=f"{name}: ${val:,.0f}")
                
                fig_hist.update_layout(
                    title=f"Portfolio Value Distribution After {mc_years} Years (${initial_investment:,} initial)",
                    xaxis_title="Final Value ($)",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════
        # DOWNLOAD & EXPORT
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Allocation (CSV)",
                data=csv_data,
                file_name=f"optimized_portfolio_{date.today()}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create summary report
            report = f"""
PORTFOLIO OPTIMIZATION REPORT
Generated: {date.today()}

PORTFOLIO METRICS:
- Expected Annual Return: {portfolio['expected_return']:.2%}
- Annual Volatility: {portfolio['volatility']:.2%}
- Sharpe Ratio: {portfolio['sharpe']:.2f}
- Number of Stocks: {len(portfolio['tickers'])}

HOLDINGS:
"""
            for ticker, weight in zip(portfolio['tickers'], portfolio['weights']):
                report += f"\n{ticker}: {weight*100:.2f}%"
            
            report += f"\n\nOPTIMIZATION PARAMETERS:\n"
            report += f"- Historical Period: {years_back} years\n"
            report += f"- Risk-Free Rate: {risk_free_rate*100:.1f}%\n"
            report += f"- Fundamentals Adjustment: {use_fundamentals}\n"
            report += f"- Candidate Universe: {len(candidate_tickers)} stocks\n"
            
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report,
                file_name=f"portfolio_report_{date.today()}.txt",
                mime="text/plain"
            )


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
    render_portfolio_optimizer()
