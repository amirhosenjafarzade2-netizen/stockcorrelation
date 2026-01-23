import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional
import plotly.graph_objects as go
import plotly.express as px


def render_additional_metrics(
    df: pd.DataFrame,
    tickers: List[str],
    benchmark_index: Optional[int] = 0
) -> None:
    """
    Render comprehensive performance and risk metrics.
    
    Args:
        df: DataFrame with dates as index and tickers as columns (adjusted Close prices)
        tickers: List of ticker symbols (must match df columns)
        benchmark_index: Which ticker to use as benchmark for beta (default: first one)
    """
    if df.empty:
        st.warning("No price data available.")
        return

    if len(tickers) < 2:
        st.info("This analysis requires at least two tickers.")
        return

    # Validate tickers
    available = [t for t in tickers if t in df.columns]
    if len(available) < 2:
        st.error(f"Only found {len(available)} valid tickers in data: {available}")
        return

    df = df[available].copy()
    tickers = available

    st.subheader("Additional Metrics • Comprehensive Performance & Risk Analysis")

    with st.spinner("Calculating metrics..."):
        # ── Calculate returns ─────────────────────────────────────────────────
        returns = np.log(df / df.shift(1)).dropna()
        simple_returns = df.pct_change().dropna()

        if len(returns) < 20:
            st.warning("Too few overlapping trading days for reliable statistics.")
            return

        trading_days = 252
        benchmark_ticker = tickers[benchmark_index]
        bench_returns = returns[benchmark_ticker]

        # ── TAB LAYOUT ────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Correlation & Beta", 
            "📉 Drawdown Analysis", 
            "🎯 Risk Metrics",
            "📈 Performance",
            "📉 Rolling Analysis"
        ])

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 1: CORRELATION & BETA
        # ═══════════════════════════════════════════════════════════════════════
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Correlation of Daily Returns**")
                corr_matrix = returns.corr()
                st.dataframe(
                    corr_matrix.style
                        .format("{:.3f}")
                        .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
                        .set_properties(**{'text-align': 'center'}),
                    use_container_width=True
                )

            with col2:
                st.markdown(f"**Beta & Alpha vs {benchmark_ticker}**")
                beta_alpha_data = []
                
                for ticker in tickers:
                    if ticker == benchmark_ticker:
                        beta_alpha_data.append({
                            "Ticker": ticker,
                            "Beta": 1.000,
                            "Alpha (ann.)": 0.0,
                            "R²": 1.000
                        })
                    else:
                        try:
                            slope, intercept, r_value, _, _ = stats.linregress(
                                bench_returns, returns[ticker]
                            )
                            alpha_annual = intercept * trading_days
                            beta_alpha_data.append({
                                "Ticker": ticker,
                                "Beta": slope,
                                "Alpha (ann.)": alpha_annual,
                                "R²": r_value ** 2
                            })
                        except:
                            beta_alpha_data.append({
                                "Ticker": ticker,
                                "Beta": np.nan,
                                "Alpha (ann.)": np.nan,
                                "R²": np.nan
                            })
                
                beta_df = pd.DataFrame(beta_alpha_data).set_index("Ticker")
                st.dataframe(
                    beta_df.style.format({
                        "Beta": "{:.3f}",
                        "Alpha (ann.)": "{:.2%}",
                        "R²": "{:.3f}"
                    }),
                    use_container_width=True
                )
                
                st.caption("Beta: sensitivity to benchmark | Alpha: excess return | R²: variance explained")

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 2: DRAWDOWN ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        with tab2:
            st.markdown("**Maximum Drawdown Analysis**")
            
            drawdown_metrics = []
            
            for ticker in tickers:
                prices = df[ticker]
                cummax = prices.cummax()
                drawdown = (prices - cummax) / cummax
                
                max_dd = drawdown.min()
                current_dd = drawdown.iloc[-1]
                
                # Find drawdown duration
                is_dd = drawdown < 0
                dd_periods = is_dd.astype(int).groupby((~is_dd).cumsum()).sum()
                max_dd_duration = dd_periods.max() if len(dd_periods) > 0 else 0
                
                # Calmar Ratio (annual return / abs(max drawdown))
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                years = len(prices) / trading_days
                cagr = (1 + total_return) ** (1/years) - 1
                calmar = cagr / abs(max_dd) if max_dd < 0 else np.inf
                
                drawdown_metrics.append({
                    "Ticker": ticker,
                    "Max Drawdown": max_dd,
                    "Current Drawdown": current_dd,
                    "Max DD Duration (days)": max_dd_duration,
                    "Calmar Ratio": calmar
                })
            
            dd_df = pd.DataFrame(drawdown_metrics).set_index("Ticker")
            st.dataframe(
                dd_df.style.format({
                    "Max Drawdown": "{:.2%}",
                    "Current Drawdown": "{:.2%}",
                    "Max DD Duration (days)": "{:.0f}",
                    "Calmar Ratio": "{:.2f}"
                }).background_gradient(cmap="RdYlGn_r", subset=["Max Drawdown"]),
                use_container_width=True
            )
            
            # Plot drawdowns
            st.markdown("**Drawdown Over Time**")
            fig = go.Figure()
            
            for ticker in tickers:
                prices = df[ticker]
                cummax = prices.cummax()
                drawdown = (prices - cummax) / cummax * 100
                
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name=ticker,
                    mode='lines',
                    fill='tozeroy'
                ))
            
            fig.update_layout(
                title="Drawdown (%)",
                yaxis_title="Drawdown %",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 3: RISK METRICS
        # ═══════════════════════════════════════════════════════════════════════
        with tab3:
            st.markdown("**Risk & Distribution Metrics**")
            
            risk_metrics = []
            
            for ticker in tickers:
                ret = returns[ticker]
                simple_ret = simple_returns[ticker]
                
                # Annualized volatility
                ann_vol = ret.std() * np.sqrt(trading_days)
                
                # Downside deviation (Sortino)
                downside = ret[ret < 0]
                downside_std = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else 0
                
                # Sharpe & Sortino
                mean_ret = ret.mean() * trading_days
                sharpe = mean_ret / ann_vol if ann_vol > 0 else 0
                sortino = mean_ret / downside_std if downside_std > 0 else 0
                
                # VaR & CVaR (95%)
                var_95 = np.percentile(simple_ret, 5)
                cvar_95 = simple_ret[simple_ret <= var_95].mean()
                
                # Skewness & Kurtosis
                skew = stats.skew(ret.dropna())
                kurt = stats.kurtosis(ret.dropna())
                
                # Win rate
                win_rate = (simple_ret > 0).sum() / len(simple_ret)
                
                risk_metrics.append({
                    "Ticker": ticker,
                    "Ann. Volatility": ann_vol,
                    "Sharpe Ratio": sharpe,
                    "Sortino Ratio": sortino,
                    "VaR (95%)": var_95,
                    "CVaR (95%)": cvar_95,
                    "Skewness": skew,
                    "Kurtosis": kurt,
                    "Win Rate": win_rate
                })
            
            risk_df = pd.DataFrame(risk_metrics).set_index("Ticker")
            st.dataframe(
                risk_df.style.format({
                    "Ann. Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Sortino Ratio": "{:.2f}",
                    "VaR (95%)": "{:.2%}",
                    "CVaR (95%)": "{:.2%}",
                    "Skewness": "{:.2f}",
                    "Kurtosis": "{:.2f}",
                    "Win Rate": "{:.1%}"
                }),
                use_container_width=True
            )
            
            st.caption("""
            **Sharpe**: risk-adjusted return | **Sortino**: penalizes downside only | 
            **VaR**: worst 5% loss threshold | **CVaR**: avg loss beyond VaR |
            **Skewness**: <0 = more left tail | **Kurtosis**: >0 = fat tails
            """)

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 4: PERFORMANCE
        # ═══════════════════════════════════════════════════════════════════════
        with tab4:
            st.markdown("**Performance Summary**")
            
            perf_metrics = []
            
            for ticker in tickers:
                prices = df[ticker]
                ret = simple_returns[ticker]
                
                # Total return
                total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                
                # CAGR
                years = len(prices) / trading_days
                cagr = (1 + total_ret) ** (1/years) - 1
                
                # Best/Worst day
                best_day = ret.max()
                worst_day = ret.min()
                
                # Streaks
                gains = (ret > 0).astype(int)
                gain_streaks = gains.groupby((gains != gains.shift()).cumsum()).sum()
                max_gain_streak = gain_streaks.max()
                
                losses = (ret < 0).astype(int)
                loss_streaks = losses.groupby((losses != losses.shift()).cumsum()).sum()
                max_loss_streak = loss_streaks.max()
                
                perf_metrics.append({
                    "Ticker": ticker,
                    "Total Return": total_ret,
                    "CAGR": cagr,
                    "Best Day": best_day,
                    "Worst Day": worst_day,
                    "Max Gain Streak": max_gain_streak,
                    "Max Loss Streak": max_loss_streak
                })
            
            perf_df = pd.DataFrame(perf_metrics).set_index("Ticker")
            st.dataframe(
                perf_df.style.format({
                    "Total Return": "{:.2%}",
                    "CAGR": "{:.2%}",
                    "Best Day": "{:.2%}",
                    "Worst Day": "{:.2%}",
                    "Max Gain Streak": "{:.0f}",
                    "Max Loss Streak": "{:.0f}"
                }).background_gradient(cmap="RdYlGn", subset=["Total Return", "CAGR"]),
                use_container_width=True
            )
            
            # Cumulative returns chart
            st.markdown("**Cumulative Returns (Indexed to 100)**")
            cum_returns = (1 + simple_returns).cumprod() * 100
            
            fig = px.line(cum_returns, title="Growth of $100")
            fig.update_layout(yaxis_title="Value ($)", hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 5: ROLLING ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        with tab5:
            st.markdown("**Rolling Metrics (60-day window)**")
            
            window = 60
            
            # Rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(trading_days) * 100
            
            fig_vol = px.line(rolling_vol, title=f"Rolling {window}-Day Annualized Volatility (%)")
            fig_vol.update_layout(yaxis_title="Volatility %", hovermode='x unified', height=350)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Rolling Sharpe
            rolling_mean = returns.rolling(window).mean() * trading_days
            rolling_std = returns.rolling(window).std() * np.sqrt(trading_days)
            rolling_sharpe = rolling_mean / rolling_std
            
            fig_sharpe = px.line(rolling_sharpe, title=f"Rolling {window}-Day Sharpe Ratio")
            fig_sharpe.update_layout(yaxis_title="Sharpe Ratio", hovermode='x unified', height=350)
            st.plotly_chart(fig_sharpe, use_container_width=True)
            
            # Rolling correlation to benchmark
            if len(tickers) > 1:
                st.markdown(f"**Rolling Correlation to {benchmark_ticker}**")
                rolling_corr = pd.DataFrame()
                
                for ticker in tickers:
                    if ticker != benchmark_ticker:
                        rolling_corr[ticker] = returns[ticker].rolling(window).corr(bench_returns)
                
                if not rolling_corr.empty:
                    fig_corr = px.line(rolling_corr, title=f"Rolling {window}-Day Correlation")
                    fig_corr.update_layout(yaxis_title="Correlation", hovermode='x unified', height=350)
                    st.plotly_chart(fig_corr, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════════
        # DOWNLOAD OPTIONS
        # ═══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_returns = returns.to_csv()
            st.download_button(
                label="📊 Daily Returns CSV",
                data=csv_returns,
                file_name=f"returns_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Combine all metrics into one CSV
            all_metrics = pd.concat([
                dd_df.add_prefix("DD_"),
                risk_df.add_prefix("Risk_"),
                perf_df.add_prefix("Perf_")
            ], axis=1)
            
            csv_metrics = all_metrics.to_csv()
            st.download_button(
                label="📈 All Metrics CSV",
                data=csv_metrics,
                file_name=f"metrics_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_prices = df.to_csv()
            st.download_button(
                label="💰 Price Data CSV",
                data=csv_prices,
                file_name=f"prices_{'-'.join(tickers[:3])}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Metrics", layout="wide")
    
    import yfinance as yf
    from datetime import date
    
    tickers_test = ["AAPL", "MSFT", "SPY"]
    df_test = yf.download(tickers_test, start="2020-01-01", end=date.today(), auto_adjust=True)["Close"]
    
    render_additional_metrics(df_test, tickers_test, benchmark_index=2)
