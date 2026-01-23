import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from typing import List, Optional


def render_correlation_finder(
    df: pd.DataFrame,
    tickers: List[str],
    method: str = "pearson",
    min_periods: int = 20
) -> None:
    """
    Advanced correlation analysis with rolling correlations, clustering, and network analysis.
    
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

    # Validate tickers
    available = [t for t in tickers if t in df.columns]
    if len(available) < 2:
        st.error(f"Only {len(available)} valid tickers found in data.")
        return

    df = df[available].copy()
    tickers = available

    st.subheader("Correlation Finder • Advanced Multi-Asset Analysis")

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Analysis Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            method = st.selectbox(
                "Correlation Method",
                ["pearson", "spearman", "kendall"],
                help="Pearson: linear | Spearman: monotonic | Kendall: rank-based"
            )
        
        with col2:
            window_size = st.slider(
                "Rolling Window (days)",
                min_value=20,
                max_value=252,
                value=60,
                step=10
            )
        
        with col3:
            cluster_method = st.selectbox(
                "Clustering Method",
                ["average", "complete", "single", "ward"],
                help="Hierarchical clustering linkage method"
            )

    with st.spinner("Computing correlations..."):
        # ── Calculate returns ─────────────────────────────────────────────────
        returns = np.log(df / df.shift(1)).dropna(how="all")

        if len(returns) < min_periods:
            st.warning(f"Too few overlapping days ({len(returns)}) — correlations may be unreliable.")
            return

        if len(returns) < 10:
            st.error("Not enough data points to compute meaningful correlations.")
            return

        # ── Correlation matrix ────────────────────────────────────────────────
        corr_matrix = returns.corr(method=method, min_periods=min_periods)

        # ══════════════════════════════════════════════════════════════════════
        # TAB LAYOUT
        # ══════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Correlation Matrix",
            "📈 Rolling Correlations", 
            "🌳 Hierarchical Clustering",
            "🔗 Network Graph",
            "📉 Correlation Breakdown"
        ])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: STATIC CORRELATION MATRIX
        # ══════════════════════════════════════════════════════════════════════
        with tab1:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"**Correlation Matrix** ({method.title()} • {len(returns)} days)")
                
                styled = corr_matrix.style\
                    .format("{:.3f}")\
                    .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)\
                    .set_properties(**{'text-align': 'center'})\
                    .set_table_styles([
                        {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]}
                    ])
                
                st.dataframe(styled, use_container_width=True)
            
            with col2:
                st.markdown("**Distribution of Correlations**")
                
                # Get off-diagonal values
                mask = ~np.eye(len(corr_matrix), dtype=bool)
                off_diag_vals = corr_matrix.where(mask).stack().values
                
                fig_hist = px.histogram(
                    off_diag_vals,
                    nbins=30,
                    title="Correlation Distribution",
                    labels={"value": "Correlation", "count": "Frequency"}
                )
                fig_hist.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Stats
                st.metric("Mean Correlation", f"{off_diag_vals.mean():.3f}")
                st.metric("Median Correlation", f"{np.median(off_diag_vals):.3f}")
                st.metric("Std Dev", f"{off_diag_vals.std():.3f}")

            # Interactive heatmap with Plotly
            st.markdown("**Interactive Heatmap**")
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig_heatmap.update_layout(
                title=f"Correlation Heatmap ({method.title()})",
                height=max(400, len(tickers) * 40),
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Insights
            with st.expander("📊 Key Insights", expanded=True):
                off_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
                
                max_pair = off_diag.max().max()
                min_pair = off_diag.min().min()
                max_loc = off_diag.stack().idxmax()
                min_loc = off_diag.stack().idxmin()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Strongest Positive",
                        f"{max_pair:.3f}",
                        delta=f"{max_loc[0]} ↔ {max_loc[1]}"
                    )
                
                with col2:
                    st.metric(
                        "Strongest Negative",
                        f"{min_pair:.3f}",
                        delta=f"{min_loc[0]} ↔ {min_loc[1]}"
                    )
                
                with col3:
                    avg_corr = off_diag.mean().mean()
                    st.metric("Average Correlation", f"{avg_corr:.3f}")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: ROLLING CORRELATIONS
        # ══════════════════════════════════════════════════════════════════════
        with tab2:
            st.markdown(f"**Rolling Correlation Analysis** ({window_size}-day window)")
            
            # Let user select pairs to plot
            if len(tickers) > 2:
                st.info("Select specific pairs to visualize rolling correlations")
                
                ticker_pairs = []
                for i in range(len(tickers)):
                    for j in range(i+1, len(tickers)):
                        ticker_pairs.append(f"{tickers[i]} ↔ {tickers[j]}")
                
                # Limit to top 5 pairs by default
                default_pairs = ticker_pairs[:min(5, len(ticker_pairs))]
                selected_pairs = st.multiselect(
                    "Select pairs to plot",
                    ticker_pairs,
                    default=default_pairs
                )
            else:
                selected_pairs = [f"{tickers[0]} ↔ {tickers[1]}"]
            
            if selected_pairs:
                fig_rolling = go.Figure()
                
                for pair_str in selected_pairs:
                    t1, t2 = pair_str.split(" ↔ ")
                    
                    if t1 in returns.columns and t2 in returns.columns:
                        rolling_corr = returns[t1].rolling(window_size).corr(returns[t2])
                        
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_corr.index,
                            y=rolling_corr,
                            name=pair_str,
                            mode='lines'
                        ))
                
                fig_rolling.update_layout(
                    title=f"Rolling {window_size}-Day Correlations",
                    yaxis_title="Correlation",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Correlation stability analysis
                st.markdown("**Correlation Stability Metrics**")
                
                stability_data = []
                for pair_str in selected_pairs:
                    t1, t2 = pair_str.split(" ↔ ")
                    
                    if t1 in returns.columns and t2 in returns.columns:
                        rolling_corr = returns[t1].rolling(window_size).corr(returns[t2]).dropna()
                        
                        stability_data.append({
                            "Pair": pair_str,
                            "Mean": rolling_corr.mean(),
                            "Std Dev": rolling_corr.std(),
                            "Min": rolling_corr.min(),
                            "Max": rolling_corr.max(),
                            "Range": rolling_corr.max() - rolling_corr.min()
                        })
                
                if stability_data:
                    stability_df = pd.DataFrame(stability_data).set_index("Pair")
                    st.dataframe(
                        stability_df.style.format({
                            "Mean": "{:.3f}",
                            "Std Dev": "{:.3f}",
                            "Min": "{:.3f}",
                            "Max": "{:.3f}",
                            "Range": "{:.3f}"
                        }).background_gradient(cmap="YlOrRd", subset=["Std Dev", "Range"]),
                        use_container_width=True
                    )
                    
                    st.caption("High Std Dev or Range indicates unstable correlation over time")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: HIERARCHICAL CLUSTERING
        # ══════════════════════════════════════════════════════════════════════
        with tab3:
            st.markdown("**Hierarchical Clustering of Assets**")
            st.caption("Groups assets by similarity in returns behavior")
            
            # Convert correlation to distance
            distance_matrix = 1 - corr_matrix.abs()
            
            # Perform hierarchical clustering
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = hierarchy.linkage(condensed_dist, method=cluster_method)
            
            # Dendrogram
            fig_dend, ax = plt.subplots(figsize=(12, 6))
            
            dendrogram = hierarchy.dendrogram(
                linkage_matrix,
                labels=tickers,
                ax=ax,
                color_threshold=0.7 * max(linkage_matrix[:, 2])
            )
            
            ax.set_title(f"Hierarchical Clustering Dendrogram ({cluster_method} linkage)")
            ax.set_xlabel("Ticker")
            ax.set_ylabel("Distance (1 - |correlation|)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig_dend)
            
            # Reorder correlation matrix by clustering
            st.markdown("**Clustered Correlation Matrix**")
            
            cluster_order = dendrogram['leaves']
            reordered_tickers = [tickers[i] for i in cluster_order]
            clustered_corr = corr_matrix.loc[reordered_tickers, reordered_tickers]
            
            fig_clustered = go.Figure(data=go.Heatmap(
                z=clustered_corr.values,
                x=clustered_corr.columns,
                y=clustered_corr.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=clustered_corr.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 9},
                colorbar=dict(title="Correlation")
            ))
            
            fig_clustered.update_layout(
                title="Clustered Heatmap (assets grouped by similarity)",
                height=max(400, len(tickers) * 40),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig_clustered, use_container_width=True)
            
            st.info("💡 Assets close together in the dendrogram have similar price movements")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4: NETWORK GRAPH
        # ══════════════════════════════════════════════════════════════════════
        with tab4:
            st.markdown("**Correlation Network Graph**")
            
            # Threshold for drawing edges
            threshold = st.slider(
                "Correlation threshold (only show edges above this)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            # Build network
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for ticker in tickers:
                G.add_node(ticker)
            
            # Add edges for strong correlations
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val >= threshold:
                        G.add_edge(
                            tickers[i],
                            tickers[j],
                            weight=corr_val,
                            color='green' if corr_matrix.iloc[i, j] > 0 else 'red'
                        )
            
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=edge[2]['weight'] * 3,
                        color=edge[2]['color']
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=30,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                hoverinfo='text',
                hovertext=[f"{node}<br>Connections: {G.degree(node)}" for node in G.nodes()]
            )
            
            # Create figure
            fig_network = go.Figure(data=edge_traces + [node_trace])
            
            fig_network.update_layout(
                title=f"Correlation Network (threshold: {threshold:.2f})",
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            st.plotly_chart(fig_network, use_container_width=True)
            
            st.caption("🟢 Green edges: positive correlation | 🔴 Red edges: negative correlation | Thickness: correlation strength")
            
            # Network metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nodes (Assets)", len(G.nodes()))
            with col2:
                st.metric("Edges (Connections)", len(G.edges()))
            with col3:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.metric("Avg Connections", f"{avg_degree:.1f}")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5: CORRELATION BREAKDOWN
        # ══════════════════════════════════════════════════════════════════════
        with tab5:
            st.markdown("**Detailed Pairwise Analysis**")
            
            # Create detailed table
            pairs_data = []
            
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    corr_val = corr_matrix.loc[t1, t2]
                    
                    # Calculate rolling correlation stats
                    rolling_corr = returns[t1].rolling(window_size).corr(returns[t2]).dropna()
                    
                    pairs_data.append({
                        "Pair": f"{t1} ↔ {t2}",
                        "Correlation": corr_val,
                        "Abs Correlation": abs(corr_val),
                        "Rolling Mean": rolling_corr.mean(),
                        "Rolling Std": rolling_corr.std(),
                        "Type": "Positive" if corr_val > 0 else "Negative"
                    })
            
            pairs_df = pd.DataFrame(pairs_data).sort_values("Abs Correlation", ascending=False)
            
            st.dataframe(
                pairs_df.style.format({
                    "Correlation": "{:.3f}",
                    "Abs Correlation": "{:.3f}",
                    "Rolling Mean": "{:.3f}",
                    "Rolling Std": "{:.3f}"
                }).background_gradient(cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1)
                .background_gradient(cmap="YlOrRd", subset=["Rolling Std"]),
                use_container_width=True
            )
            
            # Top correlations chart
            st.markdown("**Top 10 Strongest Correlations (by absolute value)**")
            
            top_10 = pairs_df.nlargest(min(10, len(pairs_df)), "Abs Correlation")
            
            fig_bar = px.bar(
                top_10,
                x="Pair",
                y="Correlation",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
                title="Strongest Correlations"
            )
            
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # DOWNLOAD OPTIONS
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_corr = corr_matrix.to_csv()
            st.download_button(
                label="📊 Correlation Matrix CSV",
                data=csv_corr,
                file_name=f"correlation_matrix_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_pairs = pairs_df.to_csv(index=False)
            st.download_button(
                label="📈 Pairwise Analysis CSV",
                data=csv_pairs,
                file_name=f"pairwise_correlations_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_returns = returns.to_csv()
            st.download_button(
                label="💹 Returns Data CSV",
                data=csv_returns,
                file_name=f"returns_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Correlation Finder", layout="wide")

    import yfinance as yf
    from datetime import date

    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "GLD", "TLT"]
    df_test = yf.download(
        test_tickers,
        start="2020-01-01",
        end=date.today(),
        auto_adjust=True
    )["Close"]

    render_correlation_finder(df_test, test_tickers, method="pearson")
