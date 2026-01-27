import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Enhanced performance and risk analytics engine."""
    
    def __init__(self, df: pd.DataFrame, tickers: List[str], trading_days: int = 252):
        self.df = df
        self.tickers = tickers
        self.trading_days = trading_days
        self.returns = np.log(df / df.shift(1)).dropna()
        self.simple_returns = df.pct_change().dropna()
        
    def calculate_beta_alpha(self, ticker: str, benchmark_returns: pd.Series) -> Dict:
        """Calculate beta, alpha, and R² for a ticker vs benchmark."""
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                benchmark_returns, self.returns[ticker]
            )
            return {
                "Beta": slope,
                "Alpha (ann.)": intercept * self.trading_days,
                "R²": r_value ** 2,
                "P-value": p_value,
                "Std Error": std_err
            }
        except Exception as e:
            return {
                "Beta": np.nan,
                "Alpha (ann.)": np.nan,
                "R²": np.nan,
                "P-value": np.nan,
                "Std Error": np.nan
            }
    
    def calculate_drawdown_metrics(self, ticker: str) -> Dict:
        """Comprehensive drawdown analysis."""
        prices = self.df[ticker]
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        
        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1]
        
        # Recovery analysis
        is_recovering = current_dd == 0
        days_underwater = 0
        if not is_recovering:
            # Count days since last peak
            last_peak_idx = cummax[cummax == prices].index[-1] if len(cummax[cummax == prices]) > 0 else prices.index[0]
            days_underwater = len(prices.loc[last_peak_idx:]) - 1
        
        # Find longest drawdown period
        is_dd = drawdown < 0
        dd_periods = is_dd.astype(int).groupby((~is_dd).cumsum()).sum()
        max_dd_duration = dd_periods.max() if len(dd_periods) > 0 else 0
        
        # Calmar and MAR ratios
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = len(prices) / self.trading_days
        cagr = (1 + total_return) ** (1/years) - 1
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.inf
        
        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        return {
            "Max Drawdown": max_dd,
            "Current Drawdown": current_dd,
            "Avg Drawdown": avg_dd,
            "Max DD Duration (days)": max_dd_duration,
            "Days Underwater": days_underwater,
            "Calmar Ratio": calmar,
            "Recovery Status": "Recovered" if is_recovering else "In Drawdown"
        }
    
    def calculate_risk_metrics(self, ticker: str) -> Dict:
        """Advanced risk and distribution metrics."""
        ret = self.returns[ticker]
        simple_ret = self.simple_returns[ticker]
        
        # Volatility measures
        ann_vol = ret.std() * np.sqrt(self.trading_days)
        downside = ret[ret < 0]
        downside_std = downside.std() * np.sqrt(self.trading_days) if len(downside) > 0 else 0
        upside = ret[ret > 0]
        upside_std = upside.std() * np.sqrt(self.trading_days) if len(upside) > 0 else 0
        
        # Return metrics
        mean_ret = ret.mean() * self.trading_days
        
        # Sharpe, Sortino, and Omega
        sharpe = mean_ret / ann_vol if ann_vol > 0 else 0
        sortino = mean_ret / downside_std if downside_std > 0 else 0
        
        # Information ratio (assuming zero benchmark for simplicity here)
        tracking_error = ann_vol
        info_ratio = mean_ret / tracking_error if tracking_error > 0 else 0
        
        # VaR and CVaR at multiple confidence levels
        var_95 = np.percentile(simple_ret, 5)
        var_99 = np.percentile(simple_ret, 1)
        cvar_95 = simple_ret[simple_ret <= var_95].mean() if len(simple_ret[simple_ret <= var_95]) > 0 else var_95
        cvar_99 = simple_ret[simple_ret <= var_99].mean() if len(simple_ret[simple_ret <= var_99]) > 0 else var_99
        
        # Distribution moments
        skew = stats.skew(ret.dropna())
        kurt = stats.kurtosis(ret.dropna())
        
        # Win/Loss metrics
        win_rate = (simple_ret > 0).sum() / len(simple_ret)
        avg_gain = simple_ret[simple_ret > 0].mean() if len(simple_ret[simple_ret > 0]) > 0 else 0
        avg_loss = simple_ret[simple_ret < 0].mean() if len(simple_ret[simple_ret < 0]) > 0 else 0
        profit_factor = abs(avg_gain / avg_loss) if avg_loss != 0 else np.inf
        
        # Gain-to-Pain ratio
        total_gain = simple_ret[simple_ret > 0].sum()
        total_pain = abs(simple_ret[simple_ret < 0].sum())
        gain_to_pain = total_gain / total_pain if total_pain > 0 else np.inf
        
        return {
            "Ann. Volatility": ann_vol,
            "Downside Vol": downside_std,
            "Upside Vol": upside_std,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Information Ratio": info_ratio,
            "VaR (95%)": var_95,
            "CVaR (95%)": cvar_95,
            "VaR (99%)": var_99,
            "CVaR (99%)": cvar_99,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Win Rate": win_rate,
            "Avg Gain": avg_gain,
            "Avg Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Gain-to-Pain": gain_to_pain
        }
    
    def calculate_performance_metrics(self, ticker: str) -> Dict:
        """Comprehensive performance metrics."""
        prices = self.df[ticker]
        ret = self.simple_returns[ticker]
        
        # Return calculations
        total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = len(prices) / self.trading_days
        cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        # Best/Worst periods
        best_day = ret.max()
        worst_day = ret.min()
        best_month = ret.rolling(21).sum().max()
        worst_month = ret.rolling(21).sum().min()
        
        # Streaks
        gains = (ret > 0).astype(int)
        gain_streaks = gains.groupby((gains != gains.shift()).cumsum()).sum()
        max_gain_streak = gain_streaks.max() if len(gain_streaks) > 0 else 0
        
        losses = (ret < 0).astype(int)
        loss_streaks = losses.groupby((losses != losses.shift()).cumsum()).sum()
        max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        # Consistency metrics
        positive_months = (ret.rolling(21).sum() > 0).sum()
        total_months = len(ret.rolling(21).sum().dropna())
        consistency = positive_months / total_months if total_months > 0 else 0
        
        # Ulcer Index (pain metric)
        drawdown = (prices - prices.cummax()) / prices.cummax()
        ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
        
        return {
            "Total Return": total_ret,
            "CAGR": cagr,
            "Best Day": best_day,
            "Worst Day": worst_day,
            "Best Month": best_month,
            "Worst Month": worst_month,
            "Max Gain Streak": max_gain_streak,
            "Max Loss Streak": max_loss_streak,
            "Monthly Consistency": consistency,
            "Ulcer Index": ulcer_index
        }


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Return Correlation Matrix",
        height=400,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    return fig


def create_scatter_matrix(returns: pd.DataFrame, tickers: List[str]) -> go.Figure:
    """Create scatter matrix for returns."""
    from plotly.figure_factory import create_scatterplotmatrix
    
    # Limit to first 5 tickers for readability
    display_tickers = tickers[:5] if len(tickers) > 5 else tickers
    returns_subset = returns[display_tickers]
    
    fig = create_scatterplotmatrix(
        returns_subset,
        diag='histogram',
        height=600,
        width=800,
        title="Return Distribution Scatter Matrix"
    )
    
    return fig


def create_efficient_frontier_plot(returns: pd.DataFrame, tickers: List[str]) -> go.Figure:
    """Create an efficient frontier visualization."""
    # Calculate portfolio metrics
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Individual assets
    volatilities = np.sqrt(np.diag(cov_matrix))
    
    fig = go.Figure()
    
    # Plot individual assets
    fig.add_trace(go.Scatter(
        x=volatilities * 100,
        y=mean_returns * 100,
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=tickers,
        textposition="top center",
        name='Individual Assets',
        hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>'
    ))
    
    # Generate random portfolios for frontier
    n_portfolios = 5000
    results = np.zeros((3, n_portfolios))
    
    for i in range(n_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_std
        
        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = sharpe
    
    # Plot efficient frontier
    fig.add_trace(go.Scatter(
        x=results[0] * 100,
        y=results[1] * 100,
        mode='markers',
        marker=dict(
            size=3,
            color=results[2],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Portfolio Combinations',
        hovertemplate='Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Efficient Frontier & Asset Positioning",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        height=500,
        showlegend=True
    )
    
    return fig


def render_additional_metrics(
    df: pd.DataFrame,
    tickers: List[str],
    benchmark_index: Optional[int] = 0
) -> None:
    """
    Render comprehensive performance and risk metrics with enhanced visualizations.
    
    Args:
        df: DataFrame with dates as index and tickers as columns (adjusted Close prices)
        tickers: List of ticker symbols (must match df columns)
        benchmark_index: Which ticker to use as benchmark for beta (default: first one)
    """
    # ══════════════════════════════════════════════════════════════════════════
    # VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    if df.empty:
        st.warning("⚠️ No price data available.")
        return

    if len(tickers) < 2:
        st.info("ℹ️ This analysis requires at least two tickers.")
        return

    # Validate tickers
    available = [t for t in tickers if t in df.columns]
    if len(available) < 2:
        st.error(f"❌ Only found {len(available)} valid ticker(s) in data: {available}")
        return

    df = df[available].copy()
    tickers = available

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("# 📊 Advanced Portfolio Analytics")
    st.markdown(f"**Analysis Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} "
                f"({len(df)} trading days)")

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZE ANALYZER
    # ══════════════════════════════════════════════════════════════════════════
    with st.spinner("🔄 Calculating comprehensive metrics..."):
        analyzer = PerformanceAnalyzer(df, tickers)
        
        if len(analyzer.returns) < 20:
            st.warning("⚠️ Too few overlapping trading days for reliable statistics.")
            return

        benchmark_ticker = tickers[benchmark_index]
        bench_returns = analyzer.returns[benchmark_ticker]

        # ══════════════════════════════════════════════════════════════════════
        # TAB LAYOUT
        # ══════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Correlation Analysis", 
            "📉 Drawdown & Recovery", 
            "🎯 Risk Metrics",
            "📈 Performance",
            "📉 Rolling Analysis",
            "🎨 Portfolio Theory"
        ])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: CORRELATION ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        with tab1:
            st.markdown("### Correlation & Regression Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Correlation matrix
                corr_matrix = analyzer.returns.corr()
                fig_corr = create_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Average correlation
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                st.metric("Average Pairwise Correlation", f"{avg_corr:.3f}")

            with col2:
                st.markdown(f"**Beta & Alpha vs {benchmark_ticker}**")
                
                beta_alpha_data = []
                for ticker in tickers:
                    if ticker == benchmark_ticker:
                        beta_alpha_data.append({
                            "Ticker": ticker,
                            "Beta": 1.000,
                            "Alpha (ann.)": 0.0,
                            "R²": 1.000,
                            "P-value": 0.0
                        })
                    else:
                        metrics = analyzer.calculate_beta_alpha(ticker, bench_returns)
                        beta_alpha_data.append({
                            "Ticker": ticker,
                            **metrics
                        })
                
                beta_df = pd.DataFrame(beta_alpha_data).set_index("Ticker")
                st.dataframe(
                    beta_df.style.format({
                        "Beta": "{:.3f}",
                        "Alpha (ann.)": "{:.2%}",
                        "R²": "{:.3f}",
                        "P-value": "{:.4f}"
                    }).background_gradient(cmap="RdYlGn", subset=["Alpha (ann.)"]),
                    use_container_width=True
                )
                
                st.caption("""
                **Beta**: Sensitivity to benchmark | **Alpha**: Excess return over benchmark | 
                **R²**: Variance explained | **P-value**: Statistical significance
                """)
            
            # Scatter matrix (if not too many tickers)
            if len(tickers) <= 5:
                st.markdown("---")
                st.markdown("### Return Distribution Matrix")
                try:
                    fig_scatter = create_scatter_matrix(analyzer.simple_returns, tickers)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                except:
                    st.info("Scatter matrix unavailable for current selection.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: DRAWDOWN ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        with tab2:
            st.markdown("### Maximum Drawdown & Recovery Analysis")
            
            # Metrics table
            drawdown_data = []
            for ticker in tickers:
                metrics = analyzer.calculate_drawdown_metrics(ticker)
                drawdown_data.append({"Ticker": ticker, **metrics})
            
            dd_df = pd.DataFrame(drawdown_data).set_index("Ticker")
            
            # Reorder columns for better display
            display_cols = ["Max Drawdown", "Current Drawdown", "Avg Drawdown", 
                          "Max DD Duration (days)", "Days Underwater", "Calmar Ratio", "Recovery Status"]
            dd_df = dd_df[display_cols]
            
            st.dataframe(
                dd_df.style.format({
                    "Max Drawdown": "{:.2%}",
                    "Current Drawdown": "{:.2%}",
                    "Avg Drawdown": "{:.2%}",
                    "Max DD Duration (days)": "{:.0f}",
                    "Days Underwater": "{:.0f}",
                    "Calmar Ratio": "{:.2f}"
                }).background_gradient(cmap="RdYlGn_r", subset=["Max Drawdown", "Current Drawdown"])
                  .applymap(lambda x: 'background-color: lightgreen' if x == "Recovered" else '', 
                           subset=["Recovery Status"]),
                use_container_width=True
            )
            
            st.caption("**Calmar Ratio**: CAGR / |Max Drawdown| (higher is better)")
            
            # Drawdown visualization
            st.markdown("---")
            st.markdown("### Drawdown Evolution")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Drawdown Over Time (%)", "Underwater Period"),
                row_heights=[0.6, 0.4],
                vertical_spacing=0.12
            )
            
            for ticker in tickers:
                prices = df[ticker]
                cummax = prices.cummax()
                drawdown = (prices - cummax) / cummax * 100
                
                # Drawdown line
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        name=ticker,
                        mode='lines',
                        fill='tozeroy',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Underwater indicator
                underwater = (drawdown < -0.1).astype(int)
                fig.add_trace(
                    go.Scatter(
                        x=underwater.index,
                        y=underwater,
                        name=f"{ticker} Underwater",
                        mode='lines',
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            fig.update_yaxes(title_text="Drawdown %", row=1, col=1)
            fig.update_yaxes(title_text="Underwater", row=2, col=1)
            fig.update_layout(height=600, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: RISK METRICS
        # ══════════════════════════════════════════════════════════════════════
        with tab3:
            st.markdown("### Comprehensive Risk Analysis")
            
            risk_data = []
            for ticker in tickers:
                metrics = analyzer.calculate_risk_metrics(ticker)
                risk_data.append({"Ticker": ticker, **metrics})
            
            risk_df = pd.DataFrame(risk_data).set_index("Ticker")
            
            # Display in multiple sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Volatility Metrics**")
                vol_cols = ["Ann. Volatility", "Downside Vol", "Upside Vol"]
                st.dataframe(
                    risk_df[vol_cols].style.format("{:.2%}")
                    .background_gradient(cmap="Reds", subset=vol_cols),
                    use_container_width=True
                )
                
                st.markdown("**Risk-Adjusted Returns**")
                ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Information Ratio"]
                st.dataframe(
                    risk_df[ratio_cols].style.format("{:.2f}")
                    .background_gradient(cmap="RdYlGn", subset=ratio_cols),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Value at Risk (VaR)**")
                var_cols = ["VaR (95%)", "CVaR (95%)", "VaR (99%)", "CVaR (99%)"]
                st.dataframe(
                    risk_df[var_cols].style.format("{:.2%}")
                    .background_gradient(cmap="Reds_r", subset=var_cols),
                    use_container_width=True
                )
                
                st.markdown("**Distribution Metrics**")
                dist_cols = ["Skewness", "Kurtosis"]
                st.dataframe(
                    risk_df[dist_cols].style.format("{:.2f}"),
                    use_container_width=True
                )
            
            st.markdown("---")
            st.markdown("**Win/Loss Analysis**")
            win_cols = ["Win Rate", "Avg Gain", "Avg Loss", "Profit Factor", "Gain-to-Pain"]
            st.dataframe(
                risk_df[win_cols].style.format({
                    "Win Rate": "{:.1%}",
                    "Avg Gain": "{:.2%}",
                    "Avg Loss": "{:.2%}",
                    "Profit Factor": "{:.2f}",
                    "Gain-to-Pain": "{:.2f}"
                }).background_gradient(cmap="RdYlGn", subset=["Win Rate", "Profit Factor", "Gain-to-Pain"]),
                use_container_width=True
            )
            
            st.caption("""
            **Sharpe**: Risk-adjusted return | **Sortino**: Penalizes downside only | 
            **VaR**: Worst loss at confidence level | **CVaR**: Average loss beyond VaR |
            **Skewness**: <0 indicates left tail | **Kurtosis**: >0 indicates fat tails |
            **Profit Factor**: Avg Gain / |Avg Loss| | **Gain-to-Pain**: Total gains / Total losses
            """)
            
            # Return distribution histograms
            st.markdown("---")
            st.markdown("### Return Distribution Comparison")
            
            fig = go.Figure()
            for ticker in tickers:
                fig.add_trace(go.Histogram(
                    x=analyzer.simple_returns[ticker] * 100,
                    name=ticker,
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig.update_layout(
                barmode='overlay',
                title="Daily Return Distribution (%)",
                xaxis_title="Daily Return %",
                yaxis_title="Frequency",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4: PERFORMANCE
        # ══════════════════════════════════════════════════════════════════════
        with tab4:
            st.markdown("### Performance Summary")
            
            perf_data = []
            for ticker in tickers:
                metrics = analyzer.calculate_performance_metrics(ticker)
                perf_data.append({"Ticker": ticker, **metrics})
            
            perf_df = pd.DataFrame(perf_data).set_index("Ticker")
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Return Metrics**")
                return_cols = ["Total Return", "CAGR"]
                st.dataframe(
                    perf_df[return_cols].style.format("{:.2%}")
                    .background_gradient(cmap="RdYlGn", subset=return_cols),
                    use_container_width=True
                )
                
                st.markdown("**Extremes**")
                extreme_cols = ["Best Day", "Worst Day", "Best Month", "Worst Month"]
                st.dataframe(
                    perf_df[extreme_cols].style.format("{:.2%}"),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Consistency Metrics**")
                cons_cols = ["Max Gain Streak", "Max Loss Streak", "Monthly Consistency", "Ulcer Index"]
                st.dataframe(
                    perf_df[cons_cols].style.format({
                        "Max Gain Streak": "{:.0f}",
                        "Max Loss Streak": "{:.0f}",
                        "Monthly Consistency": "{:.1%}",
                        "Ulcer Index": "{:.2f}"
                    }),
                    use_container_width=True
                )
                
                st.caption("**Ulcer Index**: Measure of downside risk/pain (lower is better)")
            
            # Cumulative returns
            st.markdown("---")
            st.markdown("### Cumulative Returns (Indexed to 100)")
            
            cum_returns = (1 + analyzer.simple_returns).cumprod() * 100
            
            fig = go.Figure()
            for ticker in tickers:
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns[ticker],
                    name=ticker,
                    mode='lines',
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Growth of $100 Investment",
                yaxis_title="Value ($)",
                yaxis_type="log",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly returns heatmap
            st.markdown("---")
            st.markdown("### Monthly Returns Heatmap")
            
            ticker_select = st.selectbox("Select ticker for monthly heatmap:", tickers, key="monthly_heatmap")
            
            monthly_rets = analyzer.simple_returns[ticker_select].resample('M').sum()
            monthly_pivot = pd.DataFrame({
                'Year': monthly_rets.index.year,
                'Month': monthly_rets.index.month,
                'Return': monthly_rets.values
            }).pivot(index='Month', columns='Year', values='Return')
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=monthly_pivot.values * 100,
                x=monthly_pivot.columns,
                y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                colorscale='RdYlGn',
                zmid=0,
                text=monthly_pivot.values * 100,
                texttemplate='%{text:.1f}%',
                textfont={"size": 9},
                colorbar=dict(title="Return %")
            ))
            
            fig_heatmap.update_layout(
                title=f"{ticker_select} Monthly Returns (%)",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5: ROLLING ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        with tab5:
            st.markdown("### Rolling Window Analysis")
            
            # Window selector
            col1, col2 = st.columns([1, 3])
            with col1:
                window = st.selectbox(
                    "Rolling Window (days):",
                    [30, 60, 90, 120, 252],
                    index=1
                )
            
            # Rolling volatility
            st.markdown(f"**Rolling {window}-Day Annualized Volatility**")
            rolling_vol = analyzer.returns.rolling(window).std() * np.sqrt(252) * 100
            
            fig_vol = go.Figure()
            for ticker in tickers:
                fig_vol.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol[ticker],
                    name=ticker,
                    mode='lines'
                ))
            
            fig_vol.update_layout(
                yaxis_title="Volatility %",
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Rolling Sharpe
            st.markdown(f"**Rolling {window}-Day Sharpe Ratio**")
            rolling_mean = analyzer.returns.rolling(window).mean() * 252
            rolling_std = analyzer.returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_mean / rolling_std
            
            fig_sharpe = go.Figure()
            for ticker in tickers:
                fig_sharpe.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe[ticker],
                    name=ticker,
                    mode='lines'
                ))
            
            fig_sharpe.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_sharpe.update_layout(
                yaxis_title="Sharpe Ratio",
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
            
            # Rolling correlation
            if len(tickers) > 1:
                st.markdown(f"**Rolling {window}-Day Correlation to {benchmark_ticker}**")
                rolling_corr = pd.DataFrame()
                
                for ticker in tickers:
                    if ticker != benchmark_ticker:
                        rolling_corr[ticker] = analyzer.returns[ticker].rolling(window).corr(bench_returns)
                
                if not rolling_corr.empty:
                    fig_corr = go.Figure()
                    for ticker in rolling_corr.columns:
                        fig_corr.add_trace(go.Scatter(
                            x=rolling_corr.index,
                            y=rolling_corr[ticker],
                            name=ticker,
                            mode='lines'
                        ))
                    
                    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_corr.update_layout(
                        yaxis_title="Correlation",
                        hovermode='x unified',
                        height=350
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Rolling Beta
            st.markdown(f"**Rolling {window}-Day Beta vs {benchmark_ticker}**")
            rolling_beta = pd.DataFrame()
            
            for ticker in tickers:
                if ticker != benchmark_ticker:
                    rolling_cov = analyzer.returns[ticker].rolling(window).cov(bench_returns)
                    rolling_var = bench_returns.rolling(window).var()
                    rolling_beta[ticker] = rolling_cov / rolling_var
            
            if not rolling_beta.empty:
                fig_beta = go.Figure()
                for ticker in rolling_beta.columns:
                    fig_beta.add_trace(go.Scatter(
                        x=rolling_beta.index,
                        y=rolling_beta[ticker],
                        name=ticker,
                        mode='lines'
                    ))
                
                fig_beta.add_hline(y=1, line_dash="dash", line_color="red", opacity=0.5, 
                                  annotation_text="Market Beta")
                fig_beta.update_layout(
                    yaxis_title="Beta",
                    hovermode='x unified',
                    height=350
                )
                st.plotly_chart(fig_beta, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6: PORTFOLIO THEORY
        # ══════════════════════════════════════════════════════════════════════
        with tab6:
            st.markdown("### Modern Portfolio Theory Analysis")
            
            if len(tickers) >= 2:
                try:
                    # Efficient frontier
                    st.markdown("**Efficient Frontier**")
                    fig_ef = create_efficient_frontier_plot(analyzer.returns, tickers)
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                    st.caption("""
                    The efficient frontier shows optimal portfolios that maximize return for each level of risk.
                    Points closer to the upper-left are more efficient (higher return per unit of risk).
                    """)
                    
                except Exception as e:
                    st.error(f"Could not generate efficient frontier: {str(e)}")
                
                # Covariance matrix
                st.markdown("---")
                st.markdown("**Annualized Covariance Matrix**")
                cov_matrix = analyzer.returns.cov() * 252
                
                fig_cov = go.Figure(data=go.Heatmap(
                    z=cov_matrix.values,
                    x=cov_matrix.columns,
                    y=cov_matrix.index,
                    colorscale='Blues',
                    text=cov_matrix.values,
                    texttemplate='%{text:.4f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Covariance")
                ))
                
                fig_cov.update_layout(
                    title="Return Covariance Matrix (Annualized)",
                    height=400,
                    yaxis={'autorange': 'reversed'}
                )
                
                st.plotly_chart(fig_cov, use_container_width=True)
                
                # Diversification metrics
                st.markdown("---")
                st.markdown("**Diversification Metrics**")
                
                # Equal-weight portfolio metrics
                n_assets = len(tickers)
                equal_weights = np.ones(n_assets) / n_assets
                
                port_return = np.dot(equal_weights, analyzer.returns.mean() * 252)
                port_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
                
                # Diversification ratio
                weighted_vol = np.dot(equal_weights, np.sqrt(np.diag(cov_matrix)))
                div_ratio = weighted_vol / port_vol
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Equal-Weight Return", f"{port_return*100:.2f}%")
                col2.metric("Equal-Weight Volatility", f"{port_vol*100:.2f}%")
                col3.metric("Equal-Weight Sharpe", f"{port_return/port_vol:.2f}")
                col4.metric("Diversification Ratio", f"{div_ratio:.2f}")
                
                st.caption("**Diversification Ratio**: >1 indicates diversification benefit")
                
            else:
                st.info("Portfolio analysis requires at least 2 tickers.")

        # ══════════════════════════════════════════════════════════════════════
        # DOWNLOAD SECTION
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📥 Export Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv_returns = analyzer.returns.to_csv()
            st.download_button(
                label="📊 Log Returns",
                data=csv_returns,
                file_name=f"log_returns_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_simple = analyzer.simple_returns.to_csv()
            st.download_button(
                label="📈 Simple Returns",
                data=csv_simple,
                file_name=f"simple_returns_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Combine all metrics
            all_metrics = pd.concat([
                dd_df.add_prefix("DD_"),
                risk_df.add_prefix("Risk_"),
                perf_df.add_prefix("Perf_")
            ], axis=1)
            
            csv_metrics = all_metrics.to_csv()
            st.download_button(
                label="🎯 All Metrics",
                data=csv_metrics,
                file_name=f"comprehensive_metrics_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )
        
        with col4:
            csv_prices = df.to_csv()
            st.download_button(
                label="💰 Price Data",
                data=csv_prices,
                file_name=f"prices_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION (for testing)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Portfolio Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Test with sample data
    import yfinance as yf
    from datetime import date, timedelta
    
    st.sidebar.header("Configuration")
    
    # Ticker selection
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
    ticker_input = st.sidebar.text_input(
        "Enter tickers (comma-separated):",
        value=", ".join(default_tickers)
    )
    tickers_test = [t.strip().upper() for t in ticker_input.split(",")]
    
    # Date range
    end_date = date.today()
    start_date = end_date - timedelta(days=3*365)
    
    start_input = st.sidebar.date_input("Start Date", value=start_date)
    end_input = st.sidebar.date_input("End Date", value=end_date)
    
    # Benchmark selection
    benchmark_idx = st.sidebar.selectbox(
        "Benchmark (for Beta/Alpha):",
        range(len(tickers_test)),
        format_func=lambda x: tickers_test[x]
    )
    
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Downloading data..."):
            try:
                df_test = yf.download(
                    tickers_test,
                    start=start_input,
                    end=end_input,
                    auto_adjust=True,
                    progress=False
                )["Close"]
                
                if isinstance(df_test, pd.Series):
                    df_test = df_test.to_frame(name=tickers_test[0])
                
                render_additional_metrics(df_test, tickers_test, benchmark_index=benchmark_idx)
                
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
    else:
        st.info("👈 Configure settings and click 'Run Analysis' to start")
