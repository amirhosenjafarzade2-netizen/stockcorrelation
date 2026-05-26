import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AnalyticsConfig:
    trading_days: int = 252
    risk_free_rate: float = 0.04
    rolling_window: int = 60
    frontier_points: int = 50
    random_seed: int = 42


# =============================================================================
# ENUMS
# =============================================================================

class RecoveryStatus(str, Enum):
    RECOVERED = "Recovered"
    IN_DRAWDOWN = "In Drawdown"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class RiskMetrics:
    annual_volatility: float
    downside_volatility: float
    upside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    skewness: float
    kurtosis: float
    win_rate: float
    avg_gain: float
    avg_loss: float
    profit_factor: float
    gain_to_pain: float


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float
    max_gain_streak: int
    max_loss_streak: int
    monthly_consistency: float
    ulcer_index: float


@dataclass
class DrawdownMetrics:
    max_drawdown: float
    current_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    days_underwater: int
    calmar_ratio: float
    recovery_status: RecoveryStatus


@dataclass
class BetaAlphaMetrics:
    beta: float
    alpha_annualized: float
    r_squared: float
    p_value: float
    std_error: float


# =============================================================================
# VALIDATION
# =============================================================================


def validate_prices(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending")

    if df.empty:
        raise ValueError("Price DataFrame is empty")

    if (df <= 0).any().any():
        raise ValueError("Prices must be positive")

    if df.isna().all().any():
        raise ValueError("One or more columns contain only NaNs")


# =============================================================================
# HELPERS
# =============================================================================


def compute_drawdown(prices: pd.Series) -> pd.Series:
    running_max = prices.cummax()
    return prices / running_max - 1


# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class PerformanceAnalyzer:

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        config: AnalyticsConfig = AnalyticsConfig(),
    ):
        validate_prices(df)

        self.df = df.copy()
        self.tickers = tickers
        self.config = config

    # -------------------------------------------------------------------------
    # CACHED PROPERTIES
    # -------------------------------------------------------------------------

    @cached_property
    def returns(self) -> pd.DataFrame:
        return np.log(self.df / self.df.shift(1)).dropna()

    @cached_property
    def simple_returns(self) -> pd.DataFrame:
        return self.df.pct_change().dropna()

    @cached_property
    def covariance_matrix(self) -> pd.DataFrame:
        return self.returns.cov() * self.config.trading_days

    @cached_property
    def annualized_returns(self) -> pd.Series:
        return self.returns.mean() * self.config.trading_days

    # -------------------------------------------------------------------------
    # BETA / ALPHA
    # -------------------------------------------------------------------------

    def calculate_beta_alpha(
        self,
        ticker: str,
        benchmark_returns: pd.Series,
    ) -> BetaAlphaMetrics:

        try:
            aligned = pd.concat(
                [benchmark_returns, self.returns[ticker]],
                axis=1,
            ).dropna()

            x = aligned.iloc[:, 0]
            y = aligned.iloc[:, 1]

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            return BetaAlphaMetrics(
                beta=slope,
                alpha_annualized=intercept * self.config.trading_days,
                r_squared=r_value ** 2,
                p_value=p_value,
                std_error=std_err,
            )

        except Exception:
            logger.exception("Beta calculation failed for %s", ticker)

            return BetaAlphaMetrics(
                beta=np.nan,
                alpha_annualized=np.nan,
                r_squared=np.nan,
                p_value=np.nan,
                std_error=np.nan,
            )

    # -------------------------------------------------------------------------
    # DRAWDOWN METRICS
    # -------------------------------------------------------------------------

    def calculate_drawdown_metrics(self, ticker: str) -> DrawdownMetrics:

        prices = self.df[ticker]
        drawdown = compute_drawdown(prices)

        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1]

        is_recovered = current_dd == 0

        if is_recovered:
            days_underwater = 0
        else:
            peak_idx = prices.cummax().idxmax()
            days_underwater = len(prices.loc[peak_idx:]) - 1

        dd_periods = (
            (drawdown < 0)
            .astype(int)
            .groupby((drawdown >= 0).cumsum())
            .sum()
        )

        max_duration = int(dd_periods.max()) if len(dd_periods) else 0

        total_return = prices.iloc[-1] / prices.iloc[0]
        years = len(prices) / self.config.trading_days

        cagr = np.exp(np.log(total_return) / years) - 1

        calmar = cagr / abs(max_dd) if max_dd < 0 else np.inf

        avg_dd = drawdown[drawdown < 0].mean()

        return DrawdownMetrics(
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_duration,
            days_underwater=days_underwater,
            calmar_ratio=calmar,
            recovery_status=(
                RecoveryStatus.RECOVERED
                if is_recovered
                else RecoveryStatus.IN_DRAWDOWN
            ),
        )

    # -------------------------------------------------------------------------
    # RISK METRICS
    # -------------------------------------------------------------------------

    def calculate_risk_metrics(self, ticker: str) -> RiskMetrics:

        ret = self.returns[ticker]
        simple_ret = self.simple_returns[ticker]

        ann_vol = ret.std() * np.sqrt(self.config.trading_days)

        downside = ret[ret < 0]
        upside = ret[ret > 0]

        downside_vol = (
            downside.std() * np.sqrt(self.config.trading_days)
            if len(downside)
            else 0
        )

        upside_vol = (
            upside.std() * np.sqrt(self.config.trading_days)
            if len(upside)
            else 0
        )

        mean_return = ret.mean() * self.config.trading_days

        excess_return = mean_return - self.config.risk_free_rate

        sharpe = excess_return / ann_vol if ann_vol else 0

        sortino = (
            excess_return / downside_vol
            if downside_vol
            else 0
        )

        info_ratio = excess_return / ann_vol if ann_vol else 0

        var_95 = np.percentile(simple_ret, 5)
        var_99 = np.percentile(simple_ret, 1)

        cvar_95 = simple_ret[simple_ret <= var_95].mean()
        cvar_99 = simple_ret[simple_ret <= var_99].mean()

        avg_gain = simple_ret[simple_ret > 0].mean()
        avg_loss = simple_ret[simple_ret < 0].mean()

        total_gain = simple_ret[simple_ret > 0].sum()
        total_pain = abs(simple_ret[simple_ret < 0].sum())

        return RiskMetrics(
            annual_volatility=ann_vol,
            downside_volatility=downside_vol,
            upside_volatility=upside_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            information_ratio=info_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            skewness=stats.skew(ret.dropna()),
            kurtosis=stats.kurtosis(ret.dropna()),
            win_rate=(simple_ret > 0).mean(),
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            profit_factor=(
                abs(avg_gain / avg_loss)
                if avg_loss != 0
                else np.inf
            ),
            gain_to_pain=(
                total_gain / total_pain
                if total_pain > 0
                else np.inf
            ),
        )

    # -------------------------------------------------------------------------
    # PERFORMANCE METRICS
    # -------------------------------------------------------------------------

    def calculate_performance_metrics(
        self,
        ticker: str,
    ) -> PerformanceMetrics:

        prices = self.df[ticker]
        returns = self.simple_returns[ticker]

        total_return = prices.iloc[-1] / prices.iloc[0]
        years = len(prices) / self.config.trading_days

        cagr = np.exp(np.log(total_return) / years) - 1

        monthly_returns = (
            prices.resample("ME")
            .last()
            .pct_change()
            .dropna()
        )

        gains = (returns > 0).astype(int)
        losses = (returns < 0).astype(int)

        gain_streaks = gains.groupby(
            (gains != gains.shift()).cumsum()
        ).sum()

        loss_streaks = losses.groupby(
            (losses != losses.shift()).cumsum()
        ).sum()

        drawdown = compute_drawdown(prices)

        ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100

        return PerformanceMetrics(
            total_return=total_return - 1,
            cagr=cagr,
            best_day=returns.max(),
            worst_day=returns.min(),
            best_month=monthly_returns.max(),
            worst_month=monthly_returns.min(),
            max_gain_streak=int(gain_streaks.max()),
            max_loss_streak=int(loss_streaks.max()),
            monthly_consistency=(monthly_returns > 0).mean(),
            ulcer_index=ulcer_index,
        )

    # -------------------------------------------------------------------------
    # EFFICIENT FRONTIER
    # -------------------------------------------------------------------------

    def efficient_frontier(self):

        mean_returns = self.annualized_returns
        cov_matrix = self.covariance_matrix

        n_assets = len(self.tickers)

        def portfolio_performance(weights):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return ret, vol

        def minimize_volatility(weights):
            return portfolio_performance(weights)[1]

        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)

        bounds = tuple((0, 1) for _ in range(n_assets))

        target_returns = np.linspace(
            mean_returns.min(),
            mean_returns.max(),
            self.config.frontier_points,
        )

        frontier = []

        for target in target_returns:
            cons = constraints + (
                {
                    "type": "eq",
                    "fun": lambda x, target=target: (
                        np.dot(x, mean_returns) - target
                    ),
                },
            )

            result = minimize(
                minimize_volatility,
                x0=np.ones(n_assets) / n_assets,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )

            if result.success:
                vol = portfolio_performance(result.x)[1]
                frontier.append((vol, target))

        return frontier


# =============================================================================
# VISUALIZATION
# =============================================================================


def create_line_chart(
    df: pd.DataFrame,
    title: str,
    yaxis_title: str,
) -> go.Figure:

    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=400,
    )

    return fig


# =============================================================================
# MAIN UI
# =============================================================================


def render_dashboard(
    analyzer: PerformanceAnalyzer,
    benchmark_ticker: str,
):

    st.title("Advanced Portfolio Analytics")

    tab1, tab2, tab3 = st.tabs(
        [
            "Performance",
            "Risk",
            "Portfolio Theory",
        ]
    )

    # -------------------------------------------------------------------------
    # PERFORMANCE TAB
    # -------------------------------------------------------------------------

    with tab1:

        rows = []

        for ticker in analyzer.tickers:
            metrics = analyzer.calculate_performance_metrics(ticker)

            rows.append(
                {
                    "Ticker": ticker,
                    "CAGR": metrics.cagr,
                    "Total Return": metrics.total_return,
                    "Best Day": metrics.best_day,
                    "Worst Day": metrics.worst_day,
                    "Ulcer Index": metrics.ulcer_index,
                }
            )

        perf_df = pd.DataFrame(rows).set_index("Ticker")

        st.dataframe(
            perf_df.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Total Return": "{:.2%}",
                    "Best Day": "{:.2%}",
                    "Worst Day": "{:.2%}",
                    "Ulcer Index": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        cumulative = (1 + analyzer.simple_returns).cumprod() * 100

        fig = create_line_chart(
            cumulative,
            title="Growth of $100",
            yaxis_title="Portfolio Value",
        )

        fig.update_layout(yaxis_type="log")

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # RISK TAB
    # -------------------------------------------------------------------------

    with tab2:

        rows = []

        for ticker in analyzer.tickers:
            metrics = analyzer.calculate_risk_metrics(ticker)

            rows.append(
                {
                    "Ticker": ticker,
                    "Sharpe": metrics.sharpe_ratio,
                    "Sortino": metrics.sortino_ratio,
                    "Volatility": metrics.annual_volatility,
                    "VaR 95": metrics.var_95,
                    "CVaR 95": metrics.cvar_95,
                }
            )

        risk_df = pd.DataFrame(rows).set_index("Ticker")

        st.dataframe(
            risk_df.style.format(
                {
                    "Sharpe": "{:.2f}",
                    "Sortino": "{:.2f}",
                    "Volatility": "{:.2%}",
                    "VaR 95": "{:.2%}",
                    "CVaR 95": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        rolling_vol = (
            analyzer.returns
            .rolling(analyzer.config.rolling_window)
            .std()
            * np.sqrt(analyzer.config.trading_days)
            * 100
        )

        fig = create_line_chart(
            rolling_vol,
            title="Rolling Volatility",
            yaxis_title="Volatility %",
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # PORTFOLIO THEORY TAB
    # -------------------------------------------------------------------------

    with tab3:

        frontier = analyzer.efficient_frontier()

        fig = go.Figure()

        frontier_x = [x[0] * 100 for x in frontier]
        frontier_y = [x[1] * 100 for x in frontier]

        fig.add_trace(
            go.Scatter(
                x=frontier_x,
                y=frontier_y,
                mode="lines",
                name="Efficient Frontier",
            )
        )

        asset_vols = np.sqrt(np.diag(analyzer.covariance_matrix)) * 100
        asset_returns = analyzer.annualized_returns * 100

        fig.add_trace(
            go.Scatter(
                x=asset_vols,
                y=asset_returns,
                mode="markers+text",
                text=analyzer.tickers,
                textposition="top center",
                name="Assets",
            )
        )

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility %",
            yaxis_title="Return %",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

if __name__ == "__main__":

    import yfinance as yf
    from datetime import date, timedelta

    st.set_page_config(
        page_title="Portfolio Analytics",
        page_icon="📊",
        layout="wide",
    )

    st.sidebar.header("Configuration")

    default_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "SPY",
    ]

    ticker_input = st.sidebar.text_input(
        "Tickers",
        value=", ".join(default_tickers),
    )

    tickers = [
        t.strip().upper()
        for t in ticker_input.split(",")
    ]

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 3)

    start_input = st.sidebar.date_input(
        "Start Date",
        value=start_date,
    )

    end_input = st.sidebar.date_input(
        "End Date",
        value=end_date,
    )

    benchmark = st.sidebar.selectbox(
        "Benchmark",
        tickers,
    )

    if st.sidebar.button("Run Analysis"):

        with st.spinner("Downloading data..."):

            try:

                df = yf.download(
                    tickers,
                    start=start_input,
                    end=end_input,
                    auto_adjust=True,
                    progress=False,
                )["Close"]

                if isinstance(df, pd.Series):
                    df = df.to_frame(name=tickers[0])

                config = AnalyticsConfig(
                    trading_days=252,
                    risk_free_rate=0.04,
                    rolling_window=60,
                    frontier_points=50,
                    random_seed=42,
                )

                analyzer = PerformanceAnalyzer(
                    df=df,
                    tickers=tickers,
                    config=config,
                )

                render_dashboard(
                    analyzer,
                    benchmark_ticker=benchmark,
                )

            except Exception as e:
                logger.exception("Application failed")
                st.error(str(e))
