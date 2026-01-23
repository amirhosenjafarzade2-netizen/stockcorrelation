import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
from datetime import date


def render_fundamental_comparison(tickers: List[str] = None) -> None:
    """
    Compare fundamental metrics across multiple stocks.
    
    Args:
        tickers: List of ticker symbols to compare (optional, uses UI input if None)
    """
    st.subheader("Fundamental Comparison • Multi-Stock Analysis")
    
    # ── Input Section ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_tickers = ", ".join(tickers) if tickers else "AAPL, MSFT, GOOGL"
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            value=default_tickers,
            help="Enter 2-10 ticker symbols separated by commas"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    with col2:
        frequency = st.selectbox("Frequency", ["Annual", "Quarterly"], index=0)
    
    if len(tickers) < 2:
        st.info("Enter at least 2 tickers to compare.")
        return
    
    if len(tickers) > 10:
        st.warning("Maximum 10 tickers supported. Using first 10.")
        tickers = tickers[:10]
    
    if st.button("Load & Compare", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
            try:
                # ── Fetch all data ────────────────────────────────────────────
                is_annual = (frequency == "Annual")
                all_data = {}
                failed_tickers = []
                
                for ticker in tickers:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        
                        if is_annual:
                            income = ticker_obj.get_income_stmt(freq="yearly")
                            balance = ticker_obj.get_balance_sheet(freq="yearly")
                            cashflow = ticker_obj.get_cashflow(freq="yearly")
                        else:
                            income = ticker_obj.get_income_stmt(freq="quarterly")
                            balance = ticker_obj.get_balance_sheet(freq="quarterly")
                            cashflow = ticker_obj.get_cashflow(freq="quarterly")
                        
                        # Get current price and market cap
                        info = ticker_obj.info
                        
                        all_data[ticker] = {
                            "income": income,
                            "balance": balance,
                            "cashflow": cashflow,
                            "info": info
                        }
                        
                    except Exception as e:
                        failed_tickers.append(ticker)
                        st.warning(f"⚠️ Failed to fetch {ticker}: {str(e)}")
                
                if not all_data:
                    st.error("No data retrieved. Check ticker symbols.")
                    return
                
                tickers = list(all_data.keys())  # Update to successful tickers only
                st.success(f"✓ Loaded data for {len(tickers)} stocks: {', '.join(tickers)}")
                
                if failed_tickers:
                    st.caption(f"Failed: {', '.join(failed_tickers)}")
                
                # ── Helper Functions ──────────────────────────────────────────
                def safe_get(df, key, default=None):
                    """Get row from DataFrame with fallback"""
                    if df.empty:
                        return pd.Series(dtype=float)
                    
                    if key in df.index:
                        return df.loc[key]
                    
                    alternatives = {
                        "Total Revenue": ["TotalRevenue", "Total Revenues"],
                        "Gross Profit": ["GrossProfit", "Gross Income"],
                        "Operating Income": ["OperatingIncome", "EBIT"],
                        "Net Income": ["NetIncome", "Net Income Common Stockholders"],
                        "Operating Cash Flow": ["OperatingCashFlow", "Total Cash From Operating Activities"],
                        "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures"],
                        "Free Cash Flow": ["FreeCashFlow", "Free Cash Flow"],
                        "Total Assets": ["TotalAssets", "Total Assets"],
                        "Total Debt": ["TotalDebt", "Long Term Debt"],
                        "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash"],
                        "Total Stockholder Equity": ["TotalStockholderEquity", "Stockholders Equity"],
                        "Diluted Average Shares": ["DilutedAverageShares"],
                        "Basic Average Shares": ["BasicAverageShares", "Ordinary Shares Number"]
                    }
                    
                    for alt in alternatives.get(key, []):
                        if alt in df.index:
                            return df.loc[alt]
                    
                    if default is not None:
                        return pd.Series(default, index=df.columns if not df.empty else [])
                    return pd.Series(dtype=float)
                
                def get_latest_value(series):
                    """Get most recent non-null value"""
                    valid = series.dropna()
                    return valid.iloc[-1] if len(valid) > 0 else np.nan
                
                def calculate_metrics(ticker_data: Dict) -> Dict:
                    """Calculate all metrics for a ticker"""
                    income = ticker_data["income"]
                    balance = ticker_data["balance"]
                    cashflow = ticker_data["cashflow"]
                    info = ticker_data["info"]
                    
                    # Get latest values
                    revenue = safe_get(income, "Total Revenue")
                    gross_profit = safe_get(income, "Gross Profit")
                    operating_income = safe_get(income, "Operating Income")
                    net_income = safe_get(income, "Net Income")
                    
                    ocf = safe_get(cashflow, "Operating Cash Flow")
                    capex = safe_get(cashflow, "Capital Expenditure", 0)
                    fcf_calc = ocf + capex  # CapEx is negative
                    
                    total_assets = safe_get(balance, "Total Assets")
                    total_debt = safe_get(balance, "Total Debt", 0)
                    cash = safe_get(balance, "Cash And Cash Equivalents", 0)
                    equity = safe_get(balance, "Total Stockholder Equity")
                    
                    shares = safe_get(income, "Diluted Average Shares")
                    if shares.empty:
                        shares = safe_get(income, "Basic Average Shares")
                    
                    # Latest values
                    rev_latest = get_latest_value(revenue)
                    gp_latest = get_latest_value(gross_profit)
                    oi_latest = get_latest_value(operating_income)
                    ni_latest = get_latest_value(net_income)
                    ocf_latest = get_latest_value(ocf)
                    fcf_latest = get_latest_value(fcf_calc)
                    assets_latest = get_latest_value(total_assets)
                    debt_latest = get_latest_value(total_debt)
                    cash_latest = get_latest_value(cash)
                    equity_latest = get_latest_value(equity)
                    shares_latest = get_latest_value(shares)
                    
                    # Market data
                    market_cap = info.get("marketCap", np.nan)
                    current_price = info.get("currentPrice", np.nan)
                    
                    # Calculate ratios
                    gross_margin = (gp_latest / rev_latest * 100) if rev_latest else np.nan
                    operating_margin = (oi_latest / rev_latest * 100) if rev_latest else np.nan
                    net_margin = (ni_latest / rev_latest * 100) if rev_latest else np.nan
                    
                    roe = (ni_latest / equity_latest * 100) if equity_latest else np.nan
                    roa = (ni_latest / assets_latest * 100) if assets_latest else np.nan
                    
                    debt_to_equity = (debt_latest / equity_latest) if equity_latest else np.nan
                    current_ratio = info.get("currentRatio", np.nan)
                    
                    # Per share metrics
                    eps = (ni_latest / shares_latest) if shares_latest else np.nan
                    revenue_per_share = (rev_latest / shares_latest) if shares_latest else np.nan
                    fcf_per_share = (fcf_latest / shares_latest) if shares_latest else np.nan
                    
                    # Valuation
                    pe_ratio = (current_price / eps) if eps and eps > 0 else np.nan
                    pb_ratio = (market_cap / equity_latest) if equity_latest else np.nan
                    ps_ratio = (market_cap / rev_latest) if rev_latest else np.nan
                    ev = market_cap + debt_latest - cash_latest if market_cap else np.nan
                    ev_to_ebitda = info.get("enterpriseToEbitda", np.nan)
                    
                    # Growth (if multiple periods available)
                    revenue_growth = revenue.pct_change().mean() * 100 if len(revenue.dropna()) > 1 else np.nan
                    ni_growth = net_income.pct_change().mean() * 100 if len(net_income.dropna()) > 1 else np.nan
                    
                    return {
                        # Size metrics
                        "Revenue": rev_latest,
                        "Net Income": ni_latest,
                        "Operating CF": ocf_latest,
                        "Free Cash Flow": fcf_latest,
                        "Total Assets": assets_latest,
                        "Market Cap": market_cap,
                        
                        # Profitability
                        "Gross Margin %": gross_margin,
                        "Operating Margin %": operating_margin,
                        "Net Margin %": net_margin,
                        "ROE %": roe,
                        "ROA %": roa,
                        
                        # Efficiency
                        "Revenue/Share": revenue_per_share,
                        "EPS": eps,
                        "FCF/Share": fcf_per_share,
                        
                        # Financial Health
                        "Debt/Equity": debt_to_equity,
                        "Current Ratio": current_ratio,
                        "Cash": cash_latest,
                        "Total Debt": debt_latest,
                        
                        # Valuation
                        "P/E": pe_ratio,
                        "P/B": pb_ratio,
                        "P/S": ps_ratio,
                        "EV/EBITDA": ev_to_ebitda,
                        
                        # Growth
                        "Revenue Growth %": revenue_growth,
                        "NI Growth %": ni_growth,
                        
                        # Time series for charts
                        "_revenue_ts": revenue,
                        "_net_income_ts": net_income,
                        "_ocf_ts": ocf,
                        "_fcf_ts": fcf_calc,
                        "_gross_margin_ts": (gross_profit / revenue * 100) if not revenue.empty else pd.Series(),
                        "_operating_margin_ts": (operating_income / revenue * 100) if not revenue.empty else pd.Series(),
                        "_net_margin_ts": (net_income / revenue * 100) if not revenue.empty else pd.Series(),
                    }
                
                # ── Calculate metrics for all tickers ─────────────────────────
                metrics_dict = {}
                for ticker in tickers:
                    metrics_dict[ticker] = calculate_metrics(all_data[ticker])
                
                # Convert to DataFrame
                comparison_df = pd.DataFrame(metrics_dict).T
                
                # ── TAB LAYOUT ────────────────────────────────────────────────
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Overview",
                    "💰 Profitability", 
                    "💵 Cash Flow",
                    "🏦 Financial Health",
                    "📈 Valuation",
                    "📉 Trends"
                ])
                
                # ══════════════════════════════════════════════════════════════
                # TAB 1: OVERVIEW
                # ══════════════════════════════════════════════════════════════
                with tab1:
                    st.markdown("**Key Metrics Comparison**")
                    
                    overview_cols = [
                        "Market Cap", "Revenue", "Net Income", "EPS",
                        "P/E", "P/B", "ROE %", "Net Margin %"
                    ]
                    
                    overview_df = comparison_df[overview_cols].copy()
                    
                    st.dataframe(
                        overview_df.style.format({
                            "Market Cap": "${:,.0f}",
                            "Revenue": "${:,.0f}",
                            "Net Income": "${:,.0f}",
                            "EPS": "${:.2f}",
                            "P/E": "{:.2f}",
                            "P/B": "{:.2f}",
                            "ROE %": "{:.2f}",
                            "Net Margin %": "{:.2f}"
                        }).background_gradient(cmap="RdYlGn", subset=["ROE %", "Net Margin %"]),
                        use_container_width=True
                    )
                    
                    # Bar chart comparison
                    st.markdown("**Market Cap Comparison**")
                    fig = go.Figure(data=[
                        go.Bar(x=tickers, y=overview_df["Market Cap"], 
                               marker_color='lightblue')
                    ])
                    fig.update_layout(yaxis_title="Market Cap ($)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════════════════════════════════════════════
                # TAB 2: PROFITABILITY
                # ══════════════════════════════════════════════════════════════
                with tab2:
                    st.markdown("**Profitability Metrics**")
                    
                    profit_cols = [
                        "Gross Margin %", "Operating Margin %", "Net Margin %",
                        "ROE %", "ROA %", "Revenue/Share", "EPS"
                    ]
                    
                    profit_df = comparison_df[profit_cols].copy()
                    
                    st.dataframe(
                        profit_df.style.format({
                            "Gross Margin %": "{:.2f}",
                            "Operating Margin %": "{:.2f}",
                            "Net Margin %": "{:.2f}",
                            "ROE %": "{:.2f}",
                            "ROA %": "{:.2f}",
                            "Revenue/Share": "${:.2f}",
                            "EPS": "${:.2f}"
                        }).background_gradient(cmap="RdYlGn"),
                        use_container_width=True
                    )
                    
                    # Margin comparison chart
                    st.markdown("**Margin Comparison**")
                    margin_data = profit_df[["Gross Margin %", "Operating Margin %", "Net Margin %"]]
                    
                    fig = go.Figure()
                    for col in margin_data.columns:
                        fig.add_trace(go.Bar(name=col, x=tickers, y=margin_data[col]))
                    
                    fig.update_layout(barmode='group', yaxis_title="Margin %", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROE vs ROA scatter
                    st.markdown("**ROE vs ROA**")
                    fig = px.scatter(
                        profit_df, x="ROA %", y="ROE %", 
                        text=profit_df.index,
                        size=[100]*len(profit_df),
                        height=350
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════════════════════════════════════════════
                # TAB 3: CASH FLOW
                # ══════════════════════════════════════════════════════════════
                with tab3:
                    st.markdown("**Cash Flow Metrics**")
                    
                    cf_cols = [
                        "Operating CF", "Free Cash Flow", "FCF/Share",
                        "Net Income", "Revenue"
                    ]
                    
                    cf_df = comparison_df[cf_cols].copy()
                    
                    st.dataframe(
                        cf_df.style.format({
                            "Operating CF": "${:,.0f}",
                            "Free Cash Flow": "${:,.0f}",
                            "FCF/Share": "${:.2f}",
                            "Net Income": "${:,.0f}",
                            "Revenue": "${:,.0f}"
                        }),
                        use_container_width=True
                    )
                    
                    # FCF vs Net Income comparison
                    st.markdown("**Free Cash Flow vs Net Income**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Net Income", x=tickers, 
                                        y=cf_df["Net Income"], marker_color='lightcoral'))
                    fig.add_trace(go.Bar(name="Free Cash Flow", x=tickers, 
                                        y=cf_df["Free Cash Flow"], marker_color='lightgreen'))
                    
                    fig.update_layout(barmode='group', yaxis_title="Amount ($)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # FCF Margin
                    st.markdown("**FCF Margin %**")
                    fcf_margin = (cf_df["Free Cash Flow"] / cf_df["Revenue"] * 100).dropna()
                    
                    if not fcf_margin.empty:
                        fig = go.Figure(data=[
                            go.Bar(x=fcf_margin.index, y=fcf_margin.values,
                                   marker_color='steelblue')
                        ])
                        fig.update_layout(yaxis_title="FCF Margin %", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════════════════════════════════════════════
                # TAB 4: FINANCIAL HEALTH
                # ══════════════════════════════════════════════════════════════
                with tab4:
                    st.markdown("**Financial Health Metrics**")
                    
                    health_cols = [
                        "Total Assets", "Total Debt", "Cash",
                        "Debt/Equity", "Current Ratio"
                    ]
                    
                    health_df = comparison_df[health_cols].copy()
                    
                    st.dataframe(
                        health_df.style.format({
                            "Total Assets": "${:,.0f}",
                            "Total Debt": "${:,.0f}",
                            "Cash": "${:,.0f}",
                            "Debt/Equity": "{:.2f}",
                            "Current Ratio": "{:.2f}"
                        }).background_gradient(cmap="RdYlGn_r", subset=["Debt/Equity"]),
                        use_container_width=True
                    )
                    
                    # Debt vs Cash
                    st.markdown("**Debt vs Cash Position**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Cash", x=tickers, 
                                        y=health_df["Cash"], marker_color='lightgreen'))
                    fig.add_trace(go.Bar(name="Total Debt", x=tickers, 
                                        y=health_df["Total Debt"], marker_color='lightcoral'))
                    
                    fig.update_layout(barmode='group', yaxis_title="Amount ($)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Debt/Equity comparison
                    st.markdown("**Debt/Equity Ratio (Lower is Better)**")
                    de_ratio = health_df["Debt/Equity"].dropna()
                    
                    if not de_ratio.empty:
                        fig = go.Figure(data=[
                            go.Bar(x=de_ratio.index, y=de_ratio.values,
                                   marker_color='indianred')
                        ])
                        fig.update_layout(yaxis_title="Debt/Equity", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════════════════════════════════════════════
                # TAB 5: VALUATION
                # ══════════════════════════════════════════════════════════════
                with tab5:
                    st.markdown("**Valuation Metrics**")
                    
                    val_cols = ["P/E", "P/B", "P/S", "EV/EBITDA", "Market Cap"]
                    val_df = comparison_df[val_cols].copy()
                    
                    st.dataframe(
                        val_df.style.format({
                            "P/E": "{:.2f}",
                            "P/B": "{:.2f}",
                            "P/S": "{:.2f}",
                            "EV/EBITDA": "{:.2f}",
                            "Market Cap": "${:,.0f}"
                        }),
                        use_container_width=True
                    )
                    
                    st.caption("Lower multiples generally indicate cheaper valuation (relative to fundamentals)")
                    
                    # Valuation multiples comparison
                    st.markdown("**Valuation Multiples**")
                    
                    fig = go.Figure()
                    for col in ["P/E", "P/B", "P/S", "EV/EBITDA"]:
                        if col in val_df.columns:
                            fig.add_trace(go.Bar(name=col, x=tickers, y=val_df[col]))
                    
                    fig.update_layout(barmode='group', yaxis_title="Multiple", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # P/E vs Growth scatter (if growth data available)
                    if "Revenue Growth %" in comparison_df.columns:
                        st.markdown("**P/E vs Revenue Growth (PEG Analysis)**")
                        peg_data = comparison_df[["P/E", "Revenue Growth %"]].dropna()
                        
                        if len(peg_data) > 1:
                            fig = px.scatter(
                                peg_data, x="Revenue Growth %", y="P/E",
                                text=peg_data.index,
                                size=[100]*len(peg_data),
                                height=350
                            )
                            fig.update_traces(textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════════════════════════════════════════════
                # TAB 6: TRENDS
                # ══════════════════════════════════════════════════════════════
                with tab6:
                    st.markdown("**Historical Trends (Last 4 Periods)**")
                    st.caption("Note: Yahoo Finance API provides only 4 periods of fundamental data")
                    
                    # Revenue trend
                    st.markdown("**Revenue Trend**")
                    revenue_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_revenue_ts"]
                        if not ts.empty:
                            revenue_trends[ticker] = ts
                    
                    if not revenue_trends.empty:
                        fig = go.Figure()
                        for ticker in revenue_trends.columns:
                            fig.add_trace(go.Scatter(
                                x=revenue_trends.index, y=revenue_trends[ticker],
                                name=ticker, mode='lines+markers'
                            ))
                        fig.update_layout(yaxis_title="Revenue ($)", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Net Income trend
                    st.markdown("**Net Income Trend**")
                    ni_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_net_income_ts"]
                        if not ts.empty:
                            ni_trends[ticker] = ts
                    
                    if not ni_trends.empty:
                        fig = go.Figure()
                        for ticker in ni_trends.columns:
                            fig.add_trace(go.Scatter(
                                x=ni_trends.index, y=ni_trends[ticker],
                                name=ticker, mode='lines+markers'
                            ))
                        fig.update_layout(yaxis_title="Net Income ($)", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Margin trends
                    st.markdown("**Net Margin % Trend**")
                    margin_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_net_margin_ts"]
                        if not ts.empty:
                            margin_trends[ticker] = ts
                    
                    if not margin_trends.empty:
                        fig = go.Figure()
                        for ticker in margin_trends.columns:
                            fig.add_trace(go.Scatter(
                                x=margin_trends.index, y=margin_trends[ticker],
                                name=ticker, mode='lines+markers'
                            ))
                        fig.update_layout(yaxis_title="Net Margin %", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # FCF trend
                    st.markdown("**Free Cash Flow Trend**")
                    fcf_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_fcf_ts"]
                        if not ts.empty:
                            fcf_trends[ticker] = ts
                    
                    if not fcf_trends.empty:
                        fig = go.Figure()
                        for ticker in fcf_trends.columns:
                            fig.add_trace(go.Scatter(
                                x=fcf_trends.index, y=fcf_trends[ticker],
                                name=ticker, mode='lines+markers'
                            ))
                        fig.update_layout(yaxis_title="Free Cash Flow ($)", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════════════════════════════════════════════
                # DOWNLOAD OPTIONS
                # ══════════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("### 📥 Export Comparison Data")
                
                # Remove time series columns for export
                export_df = comparison_df[[col for col in comparison_df.columns if not col.startswith("_")]]
                
                csv_data = export_df.to_csv()
                st.download_button(
                    label="📊 Download Full Comparison CSV",
                    data=csv_data,
                    file_name=f"fundamental_comparison_{'-'.join(tickers[:3])}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    st.set_page_config(page_title="Fundamental Comparison", layout="wide")
    render_fundamental_comparison()
