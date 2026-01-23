# additional_metrics.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional


def render_additional_metrics(
    df: pd.DataFrame,
    tickers: List[str],
    benchmark_index: Optional[int] = 0
) -> None:
    """
    Render returns correlation, annualized volatility, and beta metrics.
    
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

    # Make sure all requested tickers exist in the DataFrame
    available = [t for t in tickers if t in df.columns]
    if len(available) < 2:
        st.error(f"Only found {len(available)} valid tickers in data: {available}")
        return

    # Use only available columns
    df = df[available].copy()
    tickers = available  # update list to match reality

    st.subheader("Additional Metrics • Returns • Volatility • Beta")

    with st.spinner("Calculating metrics..."):
        # ── Daily log returns (more stable for stats) ─────────────────────────────
        returns = np.log(df / df.shift(1)).dropna()

        if len(returns) < 20:
            st.warning("Too few overlapping trading days for reliable statistics.")
            return

        # ── Correlation of daily returns ──────────────────────────────────────────
        corr_matrix = returns.corr()

        st.markdown("**Correlation of Daily Returns**")
        st.dataframe(
            corr_matrix.style
                .format("{:.3f}")
                .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
                .set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )

        # ── Annualized Volatility ─────────────────────────────────────────────────
        trading_days_per_year = 252
        annualized_vol = returns.std() * np.sqrt(trading_days_per_year)

        vol_df = pd.DataFrame({
            "Ticker": annualized_vol.index,
            "Annualized Volatility": annualized_vol.values
        }).set_index("Ticker")

        st.markdown("**Annualized Volatility** (252 trading days)")
        st.dataframe(
            vol_df.style.format("{:.2%}").set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )

        # ── Beta (each ticker vs chosen benchmark) ────────────────────────────────
        benchmark_ticker = tickers[benchmark_index]
        bench_returns = returns[benchmark_ticker]

        betas = {}
        for col in tickers:
            if col == benchmark_ticker:
                betas[col] = 1.00
            else:
                try:
                    slope, _, _, _, _ = stats.linregress(bench_returns, returns[col])
                    betas[col] = round(slope, 3)
                except:
                    betas[col] = np.nan

        beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=["Beta"])
        beta_df.index.name = "Ticker"

        st.markdown(f"**Beta** — relative to benchmark **{benchmark_ticker}**")
        st.dataframe(
            beta_df.style.format("{:.3f}").set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )

        # ── Quick summary stats table ─────────────────────────────────────────────
        with st.expander("Summary Table", expanded=False):
            summary = pd.DataFrame({
                "Mean Daily Return": returns.mean(),
                "Annualized Return": returns.mean() * trading_days_per_year,
                "Annualized Vol": annualized_vol,
                "Sharpe (risk-free=0)": (returns.mean() * trading_days_per_year) / annualized_vol
            }).T

            st.dataframe(
                summary.style.format({
                    "Mean Daily Return": "{:.4f}",
                    "Annualized Return": "{:.2%}",
                    "Annualized Vol": "{:.2%}",
                    "Sharpe (risk-free=0)": "{:.2f}"
                }),
                use_container_width=True
            )

        # ── Download ──────────────────────────────────────────────────────────────
        csv_data = returns.to_csv()
        st.download_button(
            label="Download daily returns CSV",
            data=csv_data,
            file_name=f"daily_returns_{'_'.join(tickers[:3])}.csv",
            mime="text/csv",
            use_container_width=False
        )


# For quick local testing / development (optional)
if __name__ == "__main__":
    st.set_page_config(page_title="Metrics Test", layout="wide")
    
    import yfinance as yf
    from datetime import date
    
    tickers_test = ["AAPL", "MSFT", "GC=F"]
    df_test = yf.download(tickers_test, start="2022-01-01", end=date.today(), auto_adjust=True)["Close"]
    
    render_additional_metrics(df_test, tickers_test)
