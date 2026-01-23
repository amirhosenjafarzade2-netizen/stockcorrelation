# correlation_finder.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def render_correlation_finder(
    df: pd.DataFrame,
    tickers: List[str],
    method: str = "pearson",
    min_periods: int = 20
) -> None:
    """
    Display correlation matrix + heatmap for 2+ assets.
    
    Args:
        df: DataFrame with dates as index and tickers as columns (adjusted Close prices)
        tickers: List of ticker symbols (must match df columns)
        method: Correlation method ('pearson', 'kendall', 'spearman')
        min_periods: Minimum number of observations required per pair
    """
    if df.empty:
        st.warning("No price data available.")
        return

    if len(tickers) < 2:
        st.info("Correlation analysis requires at least two tickers.")
        return

    # Filter to available columns only
    available = [t for t in tickers if t in df.columns]
    if len(available) < 2:
        st.error(f"Only {len(available)} valid tickers found in data.")
        return

    df = df[available].copy()
    tickers = available

    st.subheader("Correlation Finder • Multi-Asset")

    with st.spinner("Computing correlations..."):
        # ── Daily returns (log returns are more appropriate for correlation) ─────
        returns = np.log(df / df.shift(1)).dropna(how="all")

        if len(returns) < min_periods:
            st.warning(f"Too few overlapping days ({len(returns)}) — correlations may be unreliable.")
            return

        if len(returns) < 10:
            st.error("Not enough data points to compute meaningful correlations.")
            return

        # ── Correlation matrix ───────────────────────────────────────────────────
        corr_matrix = returns.corr(method=method, min_periods=min_periods)

        # ── Styled table view ────────────────────────────────────────────────────
        st.markdown(f"**Pairwise Correlation Matrix** ({method.title()})")

        styled = corr_matrix.style\
            .format("{:.3f}")\
            .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)\
            .set_properties(**{
                'text-align': 'center',
                'font-size': '14px',
                'border': '1px solid #ddd'
            })\
            .set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '16px')]}
            ])

        st.dataframe(styled, use_container_width=True)

        # ── Heatmap ──────────────────────────────────────────────────────────────
        st.markdown("**Correlation Heatmap**")

        fig, ax = plt.subplots(figsize=(max(6, len(tickers)*0.7), max(5, len(tickers)*0.7)))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(f"Correlation Heatmap ({method.title()}) – {len(returns)} overlapping days")
        plt.tight_layout()

        st.pyplot(fig)

        # Optional: larger / interactive version with altair or plotly could be added later

        # ── Quick insights ───────────────────────────────────────────────────────
        with st.expander("Quick Insights", expanded=False):
            # Find strongest positive & negative pairs (excluding self)
            off_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
            max_pair = off_diag.max().max()
            min_pair = off_diag.min().min()

            max_loc = off_diag.stack().idxmax()
            min_loc = off_diag.stack().idxmin()

            st.markdown(f"**Strongest positive correlation**: {max_pair:.3f} between **{max_loc[0]}** and **{max_loc[1]}**")
            st.markdown(f"**Strongest negative correlation**: {min_pair:.3f} between **{min_loc[0]}** and **{min_loc[1]}**")

            avg_corr = off_diag.mean().mean()
            st.markdown(f"**Average pairwise correlation**: {avg_corr:.3f}")

        # ── Download ─────────────────────────────────────────────────────────────
        csv_data = corr_matrix.to_csv()
        st.download_button(
            label="Download Correlation Matrix (CSV)",
            data=csv_data,
            file_name=f"correlation_matrix_{'_'.join(tickers[:4])}.csv",
            mime="text/csv"
        )


# ── Quick standalone test (run this file directly for development) ─────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Correlation Test", layout="wide")

    import yfinance as yf
    from datetime import date

    test_tickers = ["AAPL", "MSFT", "GOOGL", "GC=F", "BTC-USD"]
    df_test = yf.download(
        test_tickers,
        start="2022-01-01",
        end=date.today(),
        auto_adjust=True
    )["Close"]

    render_correlation_finder(df_test, test_tickers, method="pearson")
