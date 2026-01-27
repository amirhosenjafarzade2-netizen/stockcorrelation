import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy import stats
from typing import List, Optional, Dict, Tuple
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """Advanced correlation analysis engine with statistical testing and regime detection."""
    
    def __init__(self, df: pd.DataFrame, tickers: List[str], method: str = "pearson"):
        self.df = df
        self.tickers = tickers
        self.method = method
        self.returns = np.log(df / df.shift(1)).dropna(how="all")
        self.simple_returns = df.pct_change().dropna(how="all")
        
    def calculate_correlation_matrix(self, min_periods: int = 20) -> pd.DataFrame:
        """Calculate correlation matrix with specified method."""
        return self.returns.corr(method=self.method, min_periods=min_periods)
    
    def calculate_rolling_correlation(
        self, 
        ticker1: str, 
        ticker2: str, 
        window: int
    ) -> pd.Series:
        """Calculate rolling correlation between two tickers."""
        return self.returns[ticker1].rolling(window).corr(self.returns[ticker2])
    
    def test_correlation_significance(
        self, 
        ticker1: str, 
        ticker2: str
    ) -> Dict[str, float]:
        """
        Test statistical significance of correlation using t-test.
        Returns correlation, p-value, and confidence interval.
        """
        r = self.returns[[ticker1, ticker2]].corr().iloc[0, 1]
        n = len(self.returns[[ticker1, ticker2]].dropna())
        
        if n < 3:
            return {"correlation": r, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se_z = 1 / np.sqrt(n - 3)
        
        # 95% confidence interval
        z_critical = 1.96
        ci_lower_z = z - z_critical * se_z
        ci_upper_z = z + z_critical * se_z
        
        # Transform back to correlation scale
        ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        
        # T-test for significance
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return {
            "correlation": r,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_observations": n
        }
    
    def detect_correlation_regimes(
        self, 
        ticker1: str, 
        ticker2: str, 
        window: int,
        threshold_high: float = 0.7,
        threshold_low: float = 0.3
    ) -> pd.DataFrame:
        """
        Detect different correlation regimes over time.
        Returns DataFrame with regime labels.
        """
        rolling_corr = self.calculate_rolling_correlation(ticker1, ticker2, window)
        
        regimes = pd.DataFrame(index=rolling_corr.index)
        regimes['correlation'] = rolling_corr
        
        # Classify regimes
        regimes['regime'] = 'Medium'
        regimes.loc[rolling_corr > threshold_high, 'regime'] = 'High'
        regimes.loc[rolling_corr < -threshold_high, 'regime'] = 'High Negative'
        regimes.loc[(rolling_corr >= -threshold_low) & (rolling_corr <= threshold_low), 'regime'] = 'Low'
        
        return regimes
    
    def calculate_partial_correlation(
        self, 
        ticker1: str, 
        ticker2: str, 
        control_tickers: List[str]
    ) -> float:
        """
        Calculate partial correlation controlling for other variables.
        """
        try:
            # Get relevant returns
            all_tickers = [ticker1, ticker2] + control_tickers
            data = self.returns[all_tickers].dropna()
            
            if len(data) < len(all_tickers) + 2:
                return np.nan
            
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Get precision matrix (inverse of correlation matrix)
            precision_matrix = np.linalg.inv(corr_matrix.values)
            
            # Partial correlation formula
            i = 0  # ticker1 index
            j = 1  # ticker2 index
            
            partial_corr = -precision_matrix[i, j] / np.sqrt(
                precision_matrix[i, i] * precision_matrix[j, j]
            )
            
            return partial_corr
        except:
            return np.nan
    
    def calculate_tail_correlation(
        self, 
        ticker1: str, 
        ticker2: str,
        tail_percentile: float = 5
    ) -> Dict[str, float]:
        """
        Calculate correlation in tail events (extreme moves).
        """
        data = self.returns[[ticker1, ticker2]].dropna()
        
        # Lower tail (both negative)
        threshold_lower = np.percentile(data, tail_percentile)
        lower_tail_mask = (data[ticker1] <= threshold_lower) & (data[ticker2] <= threshold_lower)
        
        # Upper tail (both positive)
        threshold_upper = np.percentile(data, 100 - tail_percentile)
        upper_tail_mask = (data[ticker1] >= threshold_upper) & (data[ticker2] >= threshold_upper)
        
        # Calculate correlations
        overall_corr = data.corr().iloc[0, 1]
        
        lower_tail_data = data[lower_tail_mask]
        lower_tail_corr = lower_tail_data.corr().iloc[0, 1] if len(lower_tail_data) > 2 else np.nan
        
        upper_tail_data = data[upper_tail_mask]
        upper_tail_corr = upper_tail_data.corr().iloc[0, 1] if len(upper_tail_data) > 2 else np.nan
        
        return {
            "overall": overall_corr,
            "lower_tail": lower_tail_corr,
            "upper_tail": upper_tail_corr,
            "lower_tail_n": len(lower_tail_data),
            "upper_tail_n": len(upper_tail_data)
        }
    
    def find_correlation_clusters(
        self, 
        corr_matrix: pd.DataFrame, 
        n_clusters: int = 3
    ) -> Dict[int, List[str]]:
        """
        Find clusters of highly correlated assets using hierarchical clustering.
        """
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix.abs()
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Perform clustering
        linkage_matrix = hierarchy.linkage(condensed_dist, method='ward')
        cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group tickers by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.tickers[i])
        
        return clusters


def create_advanced_heatmap(
    corr_matrix: pd.DataFrame, 
    title: str = "Correlation Heatmap",
    show_significance: bool = False,
    p_values: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Create an advanced interactive heatmap with optional significance markers."""
    
    # Create hover text
    hover_text = []
    for i in range(len(corr_matrix)):
        hover_row = []
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            text = f"{corr_matrix.index[i]} vs {corr_matrix.columns[j]}<br>"
            text += f"Correlation: {corr_val:.3f}"
            
            if show_significance and p_values is not None:
                p_val = p_values.iloc[i, j]
                text += f"<br>P-value: {p_val:.4f}"
                if p_val < 0.05:
                    text += " ✓ Significant"
            
            hover_row.append(text)
        hover_text.append(hover_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title="Correlation", tickformat=".2f")
    ))
    
    # Add annotations for correlation values
    annotations = []
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(
                        size=9,
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                    )
                )
            )
    
    fig.update_layout(
        title=title,
        annotations=annotations,
        height=max(400, len(corr_matrix) * 50),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_correlation_timeline(
    rolling_corr_dict: Dict[str, pd.Series],
    window: int,
    title: str = "Rolling Correlation Timeline"
) -> go.Figure:
    """Create timeline visualization of multiple rolling correlations."""
    
    fig = go.Figure()
    
    for pair_name, rolling_corr in rolling_corr_dict.items():
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            name=pair_name,
            mode='lines',
            line=dict(width=2),
            hovertemplate='%{y:.3f}<extra></extra>'
        ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.3, 
                  annotation_text="High +")
    fig.add_hline(y=-0.7, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="High -")
    
    fig.update_layout(
        title=title,
        yaxis_title="Correlation",
        yaxis_range=[-1, 1],
        hovermode='x unified',
        height=500,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_regime_visualization(
    regimes_df: pd.DataFrame,
    ticker1: str,
    ticker2: str
) -> go.Figure:
    """Create visualization of correlation regimes over time."""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Rolling Correlation: {ticker1} vs {ticker2}",
            "Correlation Regime"
        ),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15
    )
    
    # Correlation line
    fig.add_trace(
        go.Scatter(
            x=regimes_df.index,
            y=regimes_df['correlation'],
            name='Correlation',
            mode='lines',
            line=dict(width=2, color='blue')
        ),
        row=1, col=1
    )
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.3, row=1, col=1)
    fig.add_hline(y=-0.7, line_dash="dot", line_color="red", opacity=0.3, row=1, col=1)
    
    # Regime visualization
    regime_colors = {
        'High': 'green',
        'Medium': 'yellow',
        'Low': 'orange',
        'High Negative': 'red'
    }
    
    for regime, color in regime_colors.items():
        regime_data = regimes_df[regimes_df['regime'] == regime]
        if not regime_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=regime_data.index,
                    y=[regime] * len(regime_data),
                    mode='markers',
                    marker=dict(size=8, color=color),
                    name=regime,
                    showlegend=True
                ),
                row=2, col=1
            )
    
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Regime", row=2, col=1)
    fig.update_layout(height=600, hovermode='x unified')
    
    return fig


def create_network_graph_advanced(
    corr_matrix: pd.DataFrame,
    tickers: List[str],
    threshold: float = 0.5,
    layout_type: str = "spring"
) -> go.Figure:
    """Create advanced network graph with community detection."""
    
    G = nx.Graph()
    
    # Add nodes with attributes
    for ticker in tickers:
        G.add_node(ticker)
    
    # Add weighted edges
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                G.add_edge(
                    tickers[i],
                    tickers[j],
                    weight=abs(corr_val),
                    sign=1 if corr_val > 0 else -1,
                    correlation=corr_val
                )
    
    # Calculate layout
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Detect communities
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        node_colors = {}
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = i
    except:
        node_colors = {node: 0 for node in G.nodes()}
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        corr_val = edge[2]['correlation']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge[2]['weight'] * 4,
                color='green' if edge[2]['sign'] > 0 else 'red'
            ),
            hoverinfo='text',
            text=f"{edge[0]} ↔ {edge[1]}: {corr_val:.3f}",
            showlegend=False,
            opacity=0.6
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_colors_list = [node_colors[node] for node in G.nodes()]
    
    # Calculate node degrees for sizing
    node_degrees = dict(G.degree())
    node_sizes = [30 + node_degrees[node] * 5 for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_colors_list,
            colorscale='Viridis',
            line=dict(width=2, color='darkblue'),
            showscale=True,
            colorbar=dict(title="Community")
        ),
        hoverinfo='text',
        hovertext=[
            f"<b>{node}</b><br>"
            f"Connections: {node_degrees[node]}<br>"
            f"Community: {node_colors[node]}"
            for node in G.nodes()
        ]
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=f"Correlation Network (threshold: {threshold:.2f}, layout: {layout_type})",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    return fig


def render_correlation_finder(
    df: pd.DataFrame,
    tickers: List[str],
    method: str = "pearson",
    min_periods: int = 20
) -> None:
    """
    Advanced correlation analysis with statistical testing, regime detection, and network analysis.
    
    Args:
        df: DataFrame with dates as index and tickers as columns (adjusted Close prices)
        tickers: List of ticker symbols (must match df columns)
        method: Correlation method ('pearson', 'kendall', 'spearman')
        min_periods: Minimum number of observations required per pair
    """
    # ══════════════════════════════════════════════════════════════════════════
    # VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    if df.empty:
        st.warning("⚠️ No price data available.")
        return

    if len(tickers) < 2:
        st.info("ℹ️ Correlation analysis requires at least two tickers.")
        return

    # Validate tickers
    available = [t for t in tickers if t in df.columns]
    if len(available) < 2:
        st.error(f"❌ Only {len(available)} valid ticker(s) found in data.")
        return

    df = df[available].copy()
    tickers = available

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("# 🔍 Advanced Correlation Analysis")
    st.markdown(f"**Analysis Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} "
                f"({len(df)} trading days)")

    # ══════════════════════════════════════════════════════════════════════════
    # SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Analysis Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            method = st.selectbox(
                "Correlation Method",
                ["pearson", "spearman", "kendall"],
                help="**Pearson**: Linear relationships | **Spearman**: Monotonic relationships | **Kendall**: Rank-based"
            )
        
        with col2:
            window_size = st.slider(
                "Rolling Window (days)",
                min_value=20,
                max_value=252,
                value=60,
                step=10,
                help="Window size for rolling correlation calculations"
            )
        
        with col3:
            cluster_method = st.selectbox(
                "Clustering Method",
                ["ward", "average", "complete", "single"],
                help="**Ward**: Minimize variance | **Average**: Average linkage | **Complete**: Maximum distance | **Single**: Minimum distance"
            )
        
        with col4:
            network_layout = st.selectbox(
                "Network Layout",
                ["spring", "circular", "kamada_kawai"],
                help="Algorithm for positioning nodes in network graph"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZE ANALYZER
    # ══════════════════════════════════════════════════════════════════════════
    with st.spinner("🔄 Computing correlations and statistical tests..."):
        analyzer = CorrelationAnalyzer(df, tickers, method)
        
        if len(analyzer.returns) < min_periods:
            st.warning(f"⚠️ Too few overlapping days ({len(analyzer.returns)}) for reliable correlations.")
            return

        if len(analyzer.returns) < 10:
            st.error("❌ Not enough data points for meaningful correlation analysis.")
            return

        corr_matrix = analyzer.calculate_correlation_matrix(min_periods)

        # ══════════════════════════════════════════════════════════════════════
        # TAB LAYOUT
        # ══════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Correlation Matrix",
            "📈 Rolling Correlations",
            "🎯 Statistical Tests",
            "🌳 Hierarchical Clustering",
            "🔗 Network Analysis",
            "📉 Regime Detection",
            "🔬 Advanced Analysis"
        ])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: CORRELATION MATRIX
        # ══════════════════════════════════════════════════════════════════════
        with tab1:
            st.markdown("### Correlation Matrix Overview")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Interactive heatmap
                fig_heatmap = create_advanced_heatmap(
                    corr_matrix,
                    title=f"Correlation Heatmap ({method.title()} • {len(analyzer.returns)} days)"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                st.markdown("**Distribution Statistics**")
                
                # Get off-diagonal values
                mask = ~np.eye(len(corr_matrix), dtype=bool)
                off_diag_vals = corr_matrix.where(mask).stack().values
                
                # Distribution plot
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=off_diag_vals,
                    nbinsx=30,
                    marker_color='steelblue',
                    opacity=0.7,
                    name='Correlation'
                ))
                
                fig_hist.add_vline(
                    x=off_diag_vals.mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {off_diag_vals.mean():.3f}"
                )
                
                fig_hist.update_layout(
                    title="Correlation Distribution",
                    xaxis_title="Correlation",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Summary statistics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Mean", f"{off_diag_vals.mean():.3f}")
                    st.metric("Median", f"{np.median(off_diag_vals):.3f}")
                with col_b:
                    st.metric("Std Dev", f"{off_diag_vals.std():.3f}")
                    st.metric("Range", f"{off_diag_vals.max() - off_diag_vals.min():.3f}")

            # Key insights
            st.markdown("---")
            st.markdown("### 📊 Key Insights")
            
            off_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
            
            max_pair = off_diag.max().max()
            min_pair = off_diag.min().min()
            max_loc = off_diag.stack().idxmax()
            min_loc = off_diag.stack().idxmin()
            
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            with col4:
                positive_pct = (off_diag_vals > 0).sum() / len(off_diag_vals) * 100
                st.metric("Positive Correlations", f"{positive_pct:.1f}%")
            
            # Correlation strength breakdown
            st.markdown("---")
            st.markdown("### Correlation Strength Breakdown")
            
            strong_pos = (off_diag_vals > 0.7).sum()
            moderate_pos = ((off_diag_vals > 0.3) & (off_diag_vals <= 0.7)).sum()
            weak = ((off_diag_vals >= -0.3) & (off_diag_vals <= 0.3)).sum()
            moderate_neg = ((off_diag_vals >= -0.7) & (off_diag_vals < -0.3)).sum()
            strong_neg = (off_diag_vals < -0.7).sum()
            
            breakdown_df = pd.DataFrame({
                "Category": ["Strong Negative\n(< -0.7)", "Moderate Negative\n(-0.7 to -0.3)", 
                           "Weak\n(-0.3 to 0.3)", "Moderate Positive\n(0.3 to 0.7)", 
                           "Strong Positive\n(> 0.7)"],
                "Count": [strong_neg, moderate_neg, weak, moderate_pos, strong_pos],
                "Percentage": [
                    strong_neg/len(off_diag_vals)*100,
                    moderate_neg/len(off_diag_vals)*100,
                    weak/len(off_diag_vals)*100,
                    moderate_pos/len(off_diag_vals)*100,
                    strong_pos/len(off_diag_vals)*100
                ]
            })
            
            fig_breakdown = px.bar(
                breakdown_df,
                x="Category",
                y="Count",
                text="Percentage",
                color="Count",
                color_continuous_scale="RdBu_r"
            )
            
            fig_breakdown.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_breakdown.update_layout(height=350, showlegend=False)
            
            st.plotly_chart(fig_breakdown, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: ROLLING CORRELATIONS
        # ══════════════════════════════════════════════════════════════════════
        with tab2:
            st.markdown("### Rolling Correlation Analysis")
            
            # Pair selection
            if len(tickers) > 2:
                st.markdown("**Select pairs to analyze**")
                
                ticker_pairs = []
                for i in range(len(tickers)):
                    for j in range(i + 1, len(tickers)):
                        ticker_pairs.append(f"{tickers[i]} ↔ {tickers[j]}")
                
                # Default to top 5 by absolute correlation
                default_selections = []
                pair_corrs = []
                for pair in ticker_pairs:
                    t1, t2 = pair.split(" ↔ ")
                    pair_corrs.append((pair, abs(corr_matrix.loc[t1, t2])))
                
                pair_corrs.sort(key=lambda x: x[1], reverse=True)
                default_selections = [p[0] for p in pair_corrs[:min(5, len(pair_corrs))]]
                
                selected_pairs = st.multiselect(
                    "Select pairs to visualize (pre-selected by strongest correlation)",
                    ticker_pairs,
                    default=default_selections
                )
            else:
                selected_pairs = [f"{tickers[0]} ↔ {tickers[1]}"]
            
            if selected_pairs:
                # Calculate rolling correlations
                rolling_corr_dict = {}
                for pair_str in selected_pairs:
                    t1, t2 = pair_str.split(" ↔ ")
                    if t1 in analyzer.returns.columns and t2 in analyzer.returns.columns:
                        rolling_corr_dict[pair_str] = analyzer.calculate_rolling_correlation(
                            t1, t2, window_size
                        )
                
                # Timeline visualization
                fig_timeline = create_correlation_timeline(
                    rolling_corr_dict,
                    window_size,
                    f"Rolling {window_size}-Day Correlations"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Stability metrics
                st.markdown("---")
                st.markdown("### Correlation Stability Analysis")
                
                stability_data = []
                for pair_str in selected_pairs:
                    t1, t2 = pair_str.split(" ↔ ")
                    
                    if t1 in analyzer.returns.columns and t2 in analyzer.returns.columns:
                        rolling_corr = rolling_corr_dict[pair_str].dropna()
                        static_corr = corr_matrix.loc[t1, t2]
                        
                        # Calculate stability metrics
                        changes = rolling_corr.diff().abs()
                        sign_changes = ((rolling_corr > 0) != (rolling_corr.shift(1) > 0)).sum()
                        
                        stability_data.append({
                            "Pair": pair_str,
                            "Static Corr": static_corr,
                            "Rolling Mean": rolling_corr.mean(),
                            "Rolling Std": rolling_corr.std(),
                            "Min": rolling_corr.min(),
                            "Max": rolling_corr.max(),
                            "Range": rolling_corr.max() - rolling_corr.min(),
                            "Avg |Change|": changes.mean(),
                            "Sign Changes": sign_changes
                        })
                
                if stability_data:
                    stability_df = pd.DataFrame(stability_data).set_index("Pair")
                    
                    st.dataframe(
                        stability_df.style.format({
                            "Static Corr": "{:.3f}",
                            "Rolling Mean": "{:.3f}",
                            "Rolling Std": "{:.3f}",
                            "Min": "{:.3f}",
                            "Max": "{:.3f}",
                            "Range": "{:.3f}",
                            "Avg |Change|": "{:.4f}",
                            "Sign Changes": "{:.0f}"
                        }).background_gradient(cmap="YlOrRd", subset=["Rolling Std", "Range", "Avg |Change|"]),
                        use_container_width=True
                    )
                    
                    st.caption("""
                    **Rolling Std**: Higher values indicate unstable correlation |
                    **Range**: Total variation in correlation over time |
                    **Avg |Change|**: Average absolute daily change in rolling correlation |
                    **Sign Changes**: Number of times correlation switched from positive to negative (or vice versa)
                    """)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: STATISTICAL TESTS
        # ══════════════════════════════════════════════════════════════════════
        with tab3:
            st.markdown("### Statistical Significance Testing")
            st.caption("Tests whether observed correlations are statistically significant (p < 0.05)")
            
            # Calculate significance for all pairs
            significance_data = []
            
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    sig_test = analyzer.test_correlation_significance(t1, t2)
                    
                    significance_data.append({
                        "Pair": f"{t1} ↔ {t2}",
                        "Correlation": sig_test['correlation'],
                        "P-value": sig_test['p_value'],
                        "CI Lower": sig_test['ci_lower'],
                        "CI Upper": sig_test['ci_upper'],
                        "N Obs": sig_test['n_observations'],
                        "Significant": "✓" if sig_test['p_value'] < 0.05 else "✗"
                    })
            
            sig_df = pd.DataFrame(significance_data).sort_values("Correlation", key=abs, ascending=False)
            
            # Display table
            st.dataframe(
                sig_df.style.format({
                    "Correlation": "{:.4f}",
                    "P-value": "{:.4f}",
                    "CI Lower": "{:.4f}",
                    "CI Upper": "{:.4f}",
                    "N Obs": "{:.0f}"
                }).background_gradient(cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1)
                  .applymap(lambda x: 'background-color: lightgreen' if x == "✓" else 'background-color: lightcoral', 
                           subset=["Significant"]),
                use_container_width=True
            )
            
            st.caption("""
            **P-value < 0.05**: Correlation is statistically significant at 95% confidence level |
            **CI**: 95% Confidence interval for the correlation coefficient
            """)
            
            # Summary statistics
            st.markdown("---")
            st.markdown("### Significance Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                significant_count = (sig_df['P-value'] < 0.05).sum()
                total_pairs = len(sig_df)
                st.metric(
                    "Significant Pairs",
                    f"{significant_count}/{total_pairs}",
                    delta=f"{significant_count/total_pairs*100:.1f}%"
                )
            
            with col2:
                highly_sig = (sig_df['P-value'] < 0.01).sum()
                st.metric(
                    "Highly Significant (p<0.01)",
                    highly_sig,
                    delta=f"{highly_sig/total_pairs*100:.1f}%"
                )
            
            with col3:
                avg_conf_width = (sig_df['CI Upper'] - sig_df['CI Lower']).mean()
                st.metric(
                    "Avg CI Width",
                    f"{avg_conf_width:.4f}",
                    delta="Narrower = More precise"
                )
            
            # Visualization of confidence intervals
            st.markdown("---")
            st.markdown("### Confidence Intervals Visualization")
            
            # Select top 15 by absolute correlation
            top_pairs = sig_df.nlargest(min(15, len(sig_df)), 'Correlation', key=abs)
            
            fig_ci = go.Figure()
            
            for idx, row in top_pairs.iterrows():
                fig_ci.add_trace(go.Scatter(
                    x=[row['CI Lower'], row['Correlation'], row['CI Upper']],
                    y=[row['Pair'], row['Pair'], row['Pair']],
                    mode='markers+lines',
                    marker=dict(
                        size=[8, 12, 8],
                        color='green' if row['Significant'] == "✓" else 'red'
                    ),
                    line=dict(color='gray', width=2),
                    name=row['Pair'],
                    showlegend=False,
                    hovertemplate=f"<b>{row['Pair']}</b><br>"
                                f"Correlation: {row['Correlation']:.3f}<br>"
                                f"95% CI: [{row['CI Lower']:.3f}, {row['CI Upper']:.3f}]<br>"
                                f"P-value: {row['P-value']:.4f}<extra></extra>"
                ))
            
            fig_ci.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
            
            fig_ci.update_layout(
                title="95% Confidence Intervals for Correlations",
                xaxis_title="Correlation",
                yaxis_title="",
                height=max(400, len(top_pairs) * 30),
                xaxis_range=[-1, 1]
            )
            
            st.plotly_chart(fig_ci, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4: HIERARCHICAL CLUSTERING
        # ══════════════════════════════════════════════════════════════════════
        with tab4:
            st.markdown("### Hierarchical Clustering Analysis")
            st.caption("Groups assets by similarity in returns behavior")
            
            # Convert correlation to distance
            distance_matrix = 1 - corr_matrix.abs()
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = hierarchy.linkage(condensed_dist, method=cluster_method)
            
            # Dendrogram
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Dendrogram**")
                
                fig_dend, ax = plt.subplots(figsize=(14, 7))
                
                dendrogram = hierarchy.dendrogram(
                    linkage_matrix,
                    labels=tickers,
                    ax=ax,
                    color_threshold=0.7 * max(linkage_matrix[:, 2]),
                    above_threshold_color='gray'
                )
                
                ax.set_title(f"Hierarchical Clustering Dendrogram ({cluster_method} linkage)", 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel("Ticker", fontsize=12)
                ax.set_ylabel("Distance (1 - |correlation|)", fontsize=12)
                ax.axhline(y=0.7 * max(linkage_matrix[:, 2]), color='red', 
                          linestyle='--', label='Cut threshold')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig_dend)
                plt.close()
            
            with col2:
                st.markdown("**Cluster Information**")
                
                # Determine optimal number of clusters
                n_clusters_auto = min(5, len(tickers) // 2)
                
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=min(len(tickers), 10),
                    value=n_clusters_auto,
                    help="Divide assets into this many groups"
                )
                
                # Get clusters
                clusters = analyzer.find_correlation_clusters(corr_matrix, n_clusters)
                
                # Display clusters
                for cluster_id, members in sorted(clusters.items()):
                    st.markdown(f"**Cluster {cluster_id}**")
                    st.write(", ".join(members))
                    
                    # Calculate avg intra-cluster correlation
                    if len(members) > 1:
                        cluster_corrs = []
                        for i in range(len(members)):
                            for j in range(i + 1, len(members)):
                                cluster_corrs.append(corr_matrix.loc[members[i], members[j]])
                        
                        avg_corr = np.mean(cluster_corrs)
                        st.caption(f"Avg correlation: {avg_corr:.3f}")
                    
                    st.markdown("---")
            
            # Clustered heatmap
            st.markdown("### Clustered Correlation Matrix")
            
            cluster_order = dendrogram['leaves']
            reordered_tickers = [tickers[i] for i in cluster_order]
            clustered_corr = corr_matrix.loc[reordered_tickers, reordered_tickers]
            
            fig_clustered = create_advanced_heatmap(
                clustered_corr,
                title="Clustered Heatmap (assets reordered by similarity)"
            )
            
            st.plotly_chart(fig_clustered, use_container_width=True)
            
            st.info("💡 Assets grouped together in the dendrogram and heatmap have similar price movements")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5: NETWORK ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        with tab5:
            st.markdown("### Correlation Network Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Threshold slider
                threshold = st.slider(
                    "Correlation threshold (only show edges above this)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Higher values show only strongest relationships"
                )
            
            with col2:
                show_negative = st.checkbox(
                    "Show negative correlations",
                    value=True,
                    help="Include negatively correlated pairs"
                )
            
            # Create network graph
            fig_network = create_network_graph_advanced(
                corr_matrix,
                tickers,
                threshold,
                network_layout
            )
            
            st.plotly_chart(fig_network, use_container_width=True)
            
            st.caption("""
            🟢 **Green edges**: Positive correlation | 🔴 **Red edges**: Negative correlation | 
            **Edge thickness**: Correlation strength | **Node size**: Number of connections |
            **Node color**: Community/cluster membership
            """)
            
            # Network metrics
            st.markdown("---")
            st.markdown("### Network Metrics")
            
            # Build network for analysis
            G = nx.Graph()
            for ticker in tickers:
                G.add_node(ticker)
            
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        if show_negative or corr_val > 0:
                            G.add_edge(tickers[i], tickers[j], weight=abs(corr_val))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes (Assets)", len(G.nodes()))
            
            with col2:
                st.metric("Edges (Connections)", len(G.edges()))
            
            with col3:
                if len(G.nodes()) > 0:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
                else:
                    st.metric("Avg Connections", "N/A")
            
            with col4:
                if len(G.edges()) > 0:
                    density = nx.density(G)
                    st.metric("Network Density", f"{density:.2%}")
                else:
                    st.metric("Network Density", "0%")
            
            # Most connected assets
            if len(G.nodes()) > 0:
                st.markdown("---")
                st.markdown("### Most Connected Assets")
                
                degree_dict = dict(G.degree())
                degree_df = pd.DataFrame([
                    {"Ticker": ticker, "Connections": degree}
                    for ticker, degree in degree_dict.items()
                ]).sort_values("Connections", ascending=False).head(10)
                
                fig_degree = px.bar(
                    degree_df,
                    x="Ticker",
                    y="Connections",
                    title="Top 10 Most Connected Assets",
                    color="Connections",
                    color_continuous_scale="Blues"
                )
                
                fig_degree.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_degree, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6: REGIME DETECTION
        # ══════════════════════════════════════════════════════════════════════
        with tab6:
            st.markdown("### Correlation Regime Analysis")
            st.caption("Identify periods of high, medium, and low correlation")
            
            # Select pair for regime analysis
            if len(tickers) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    regime_t1 = st.selectbox("Select first ticker", tickers, key="regime_t1")
                
                with col2:
                    remaining = [t for t in tickers if t != regime_t1]
                    regime_t2 = st.selectbox("Select second ticker", remaining, key="regime_t2")
                
                # Regime thresholds
                col1, col2 = st.columns(2)
                
                with col1:
                    threshold_high = st.slider(
                        "High correlation threshold",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.7,
                        step=0.05
                    )
                
                with col2:
                    threshold_low = st.slider(
                        "Low correlation threshold",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.3,
                        step=0.05
                    )
                
                # Calculate regimes
                regimes_df = analyzer.detect_correlation_regimes(
                    regime_t1,
                    regime_t2,
                    window_size,
                    threshold_high,
                    threshold_low
                )
                
                # Visualize regimes
                fig_regimes = create_regime_visualization(regimes_df, regime_t1, regime_t2)
                st.plotly_chart(fig_regimes, use_container_width=True)
                
                # Regime statistics
                st.markdown("---")
                st.markdown("### Regime Statistics")
                
                regime_stats = regimes_df['regime'].value_counts()
                regime_pct = (regime_stats / len(regimes_df) * 100).round(1)
                
                stats_df = pd.DataFrame({
                    "Regime": regime_stats.index,
                    "Days": regime_stats.values,
                    "Percentage": regime_pct.values
                })
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_regime_dist = px.pie(
                        stats_df,
                        values='Days',
                        names='Regime',
                        title='Time Spent in Each Regime',
                        color='Regime',
                        color_discrete_map={
                            'High': 'green',
                            'Medium': 'yellow',
                            'Low': 'orange',
                            'High Negative': 'red'
                        }
                    )
                    
                    st.plotly_chart(fig_regime_dist, use_container_width=True)
                
                with col2:
                    st.dataframe(
                        stats_df.style.format({"Days": "{:.0f}", "Percentage": "{:.1f}%"}),
                        use_container_width=True
                    )
                
                # Average correlation by regime
                avg_by_regime = regimes_df.groupby('regime')['correlation'].agg(['mean', 'std', 'min', 'max'])
                
                st.markdown("### Average Correlation by Regime")
                st.dataframe(
                    avg_by_regime.style.format("{:.3f}"),
                    use_container_width=True
                )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 7: ADVANCED ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        with tab7:
            st.markdown("### Advanced Correlation Analysis")
            
            # Tail correlation analysis
            st.markdown("#### Tail Correlation Analysis")
            st.caption("How do correlations behave during extreme market moves?")
            
            # Select pair
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tail_t1 = st.selectbox("Select first ticker", tickers, key="tail_t1")
            
            with col2:
                remaining_tail = [t for t in tickers if t != tail_t1]
                tail_t2 = st.selectbox("Select second ticker", remaining_tail, key="tail_t2")
            
            with col3:
                tail_percentile = st.slider(
                    "Tail percentile (%)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Define extreme events as this percentile"
                )
            
            # Calculate tail correlations
            tail_results = analyzer.calculate_tail_correlation(
                tail_t1,
                tail_t2,
                tail_percentile
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Overall Correlation",
                    f"{tail_results['overall']:.3f}"
                )
            
            with col2:
                delta_lower = tail_results['lower_tail'] - tail_results['overall']
                st.metric(
                    f"Lower Tail Correlation",
                    f"{tail_results['lower_tail']:.3f}",
                    delta=f"{delta_lower:+.3f} vs overall",
                    help=f"Based on {tail_results['lower_tail_n']} observations"
                )
            
            with col3:
                delta_upper = tail_results['upper_tail'] - tail_results['overall']
                st.metric(
                    f"Upper Tail Correlation",
                    f"{tail_results['upper_tail']:.3f}",
                    delta=f"{delta_upper:+.3f} vs overall",
                    help=f"Based on {tail_results['upper_tail_n']} observations"
                )
            
            st.caption("""
            **Lower Tail**: Correlation during joint downward moves (bear markets) |
            **Upper Tail**: Correlation during joint upward moves (bull markets) |
            Higher tail correlation = Increased co-movement during extremes
            """)
            
            # Partial correlation analysis
            st.markdown("---")
            st.markdown("#### Partial Correlation Analysis")
            st.caption("Correlation after controlling for other variables")
            
            if len(tickers) >= 3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    partial_t1 = st.selectbox("Ticker 1", tickers, key="partial_t1")
                
                with col2:
                    remaining_p = [t for t in tickers if t != partial_t1]
                    partial_t2 = st.selectbox("Ticker 2", remaining_p, key="partial_t2")
                
                with col3:
                    control_options = [t for t in tickers if t not in [partial_t1, partial_t2]]
                    control_tickers = st.multiselect(
                        "Control for",
                        control_options,
                        default=control_options[:min(2, len(control_options))],
                        help="Variables to control for in partial correlation"
                    )
                
                if control_tickers:
                    # Calculate correlations
                    direct_corr = corr_matrix.loc[partial_t1, partial_t2]
                    partial_corr = analyzer.calculate_partial_correlation(
                        partial_t1,
                        partial_t2,
                        control_tickers
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Direct Correlation", f"{direct_corr:.3f}")
                    
                    with col2:
                        st.metric("Partial Correlation", f"{partial_corr:.3f}")
                    
                    with col3:
                        if not np.isnan(partial_corr):
                            change = partial_corr - direct_corr
                            st.metric("Change", f"{change:+.3f}")
                    
                    st.caption(f"Controlling for: {', '.join(control_tickers)}")
                    
                    if not np.isnan(partial_corr):
                        if abs(partial_corr) < abs(direct_corr):
                            st.info("🔍 Partial correlation is weaker, suggesting the relationship is partly explained by the control variables.")
                        elif abs(partial_corr) > abs(direct_corr):
                            st.warning("⚠️ Partial correlation is stronger, suggesting a suppression effect.")
                        else:
                            st.success("✓ Partial correlation similar to direct correlation, suggesting an independent relationship.")
            
            else:
                st.info("Partial correlation analysis requires at least 3 tickers")
            
            # Detailed pairwise breakdown
            st.markdown("---")
            st.markdown("#### Complete Pairwise Analysis")
            
            pairs_data = []
            
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    corr_val = corr_matrix.loc[t1, t2]
                    
                    # Rolling stats
                    rolling_corr = analyzer.calculate_rolling_correlation(t1, t2, window_size).dropna()
                    
                    # Significance test
                    sig_test = analyzer.test_correlation_significance(t1, t2)
                    
                    pairs_data.append({
                        "Pair": f"{t1} ↔ {t2}",
                        "Correlation": corr_val,
                        "Abs Corr": abs(corr_val),
                        "P-value": sig_test['p_value'],
                        "Significant": "✓" if sig_test['p_value'] < 0.05 else "✗",
                        "Rolling Mean": rolling_corr.mean(),
                        "Rolling Std": rolling_corr.std(),
                        "Type": "Positive" if corr_val > 0 else "Negative"
                    })
            
            pairs_df = pd.DataFrame(pairs_data).sort_values("Abs Corr", ascending=False)
            
            st.dataframe(
                pairs_df.style.format({
                    "Correlation": "{:.3f}",
                    "Abs Corr": "{:.3f}",
                    "P-value": "{:.4f}",
                    "Rolling Mean": "{:.3f}",
                    "Rolling Std": "{:.3f}"
                }).background_gradient(cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1)
                  .background_gradient(cmap="YlOrRd", subset=["Rolling Std"])
                  .applymap(lambda x: 'background-color: lightgreen' if x == "✓" else 'background-color: lightcoral', 
                           subset=["Significant"]),
                use_container_width=True,
                height=400
            )
            
            # Top correlations visualization
            st.markdown("### Top 10 Strongest Correlations (by absolute value)")
            
            top_10 = pairs_df.nlargest(min(10, len(pairs_df)), "Abs Corr")
            
            fig_bar = px.bar(
                top_10,
                x="Pair",
                y="Correlation",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
                title="Strongest Correlations",
                text="Correlation"
            )
            
            fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # DOWNLOAD SECTION
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📥 Export Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv_corr = corr_matrix.to_csv()
            st.download_button(
                label="📊 Correlation Matrix",
                data=csv_corr,
                file_name=f"correlation_matrix_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_pairs = pairs_df.to_csv(index=False)
            st.download_button(
                label="📈 Pairwise Analysis",
                data=csv_pairs,
                file_name=f"pairwise_analysis_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_sig = sig_df.to_csv(index=False)
            st.download_button(
                label="🎯 Significance Tests",
                data=csv_sig,
                file_name=f"significance_tests_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )
        
        with col4:
            csv_returns = analyzer.returns.to_csv()
            st.download_button(
                label="💹 Returns Data",
                data=csv_returns,
                file_name=f"returns_{'-'.join(tickers[:4])}.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION (for testing)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Correlation Analysis",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    import yfinance as yf
    from datetime import date, timedelta

    st.sidebar.header("Configuration")
    
    # Ticker selection
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "GLD", "TLT"]
    ticker_input = st.sidebar.text_input(
        "Enter tickers (comma-separated):",
        value=", ".join(default_tickers)
    )
    test_tickers = [t.strip().upper() for t in ticker_input.split(",")]
    
    # Date range
    end_date = date.today()
    start_date = end_date - timedelta(days=3*365)
    
    start_input = st.sidebar.date_input("Start Date", value=start_date)
    end_input = st.sidebar.date_input("End Date", value=end_date)
    
    # Correlation method
    method_input = st.sidebar.selectbox(
        "Correlation Method",
        ["pearson", "spearman", "kendall"]
    )
    
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Downloading data..."):
            try:
                df_test = yf.download(
                    test_tickers,
                    start=start_input,
                    end=end_input,
                    auto_adjust=True,
                    progress=False
                )["Close"]
                
                if isinstance(df_test, pd.Series):
                    df_test = df_test.to_frame(name=test_tickers[0])
                
                render_correlation_finder(df_test, test_tickers, method=method_input)
                
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
    else:
        st.info("👈 Configure settings and click 'Run Analysis' to start")
