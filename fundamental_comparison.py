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
                        
                        # Use direct properties instead of get methods (more reliable)
                        if is_annual:
                            income = ticker_obj.income_stmt
                            balance = ticker_obj.balance_sheet
                            cashflow = ticker_obj.cashflow
                        else:
                            income = ticker_obj.quarterly_income_stmt
                            balance = ticker_obj.quarterly_balance_sheet
                            cashflow = ticker_obj.quarterly_cashflow
                        
                        # Check for empty data
                        if income.empty or balance.empty or cashflow.empty:
                            failed_tickers.append(ticker)
                            st.warning(f"⚠️ No data available for {ticker}")
                            continue
                        
                        # Get info with fallback
                        try:
                            info = ticker_obj.info
                            if not info or not isinstance(info, dict):
                                info = {}
                        except Exception:
                            info = {}
                        
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
                    if series.empty:
                        return np.nan
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
                    
                    # Market data - with safe fallbacks
                    market_cap = info.get("marketCap", np.nan)
                    current_price = info.get("currentPrice", info.get("regularMarketPrice", np.nan))
                    
                    # Calculate ratios
                    gross_margin = (gp_latest / rev_latest * 100) if rev_latest and not np.isnan(rev_latest) and rev_latest != 0 else np.nan
                    operating_margin = (oi_latest / rev_latest * 100) if rev_latest and not np.isnan(rev_latest) and rev_latest != 0 else np.nan
                    net_margin = (ni_latest / rev_latest * 100) if rev_latest and not np.isnan(rev_latest) and rev_latest != 0 else np.nan
                    
                    roe = (ni_latest / equity_latest * 100) if equity_latest and not np.isnan(equity_latest) and equity_latest != 0 else np.nan
                    roa = (ni_latest / assets_latest * 100) if assets_latest and not np.isnan(assets_latest) and assets_latest != 0 else np.nan
                    
                    debt_to_equity = (debt_latest / equity_latest) if equity_latest and not np.isnan(equity_latest) and equity_latest != 0 else np.nan
                    current_ratio = info.get("currentRatio", np.nan)
                    
                    # Per share metrics
                    eps = (ni_latest / shares_latest) if shares_latest and not np.isnan(shares_latest) and shares_latest != 0 else np.nan
                    revenue_per_share = (rev_latest / shares_latest) if shares_latest and not np.isnan(shares_latest) and shares_latest != 0 else np.nan
                    fcf_per_share = (fcf_latest / shares_latest) if shares_latest and not np.isnan(shares_latest) and shares_latest != 0 else np.nan
                    
                    # Valuation
                    pe_ratio = (current_price / eps) if eps and not np.isnan(eps) and eps > 0 and not np.isnan(current_price) else np.nan
                    pb_ratio = (market_cap / equity_latest) if equity_latest and not np.isnan(equity_latest) and equity_latest != 0 and not np.isnan(market_cap) else np.nan
                    ps_ratio = (market_cap / rev_latest) if rev_latest and not np.isnan(rev_latest) and rev_latest != 0 and not np.isnan(market_cap) else np.nan
                    ev = market_cap + debt_latest - cash_latest if not np.isnan(market_cap) else np.nan
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
                        }, na_rep="-").background_gradient(cmap="RdYlGn", subset=["ROE %", "Net Margin %"]),
                        use_container_width=True
                    )
                    
                    # Bar chart comparison
                    st.markdown("**Market Cap Comparison**")
                    valid_mcap = overview_df["Market Cap"].dropna()
                    if not valid_mcap.empty:
                        fig = go.Figure(data=[
                            go.Bar(x=valid_mcap.index, y=valid_mcap.values, 
                                   marker_color='lightblue')
                        ])
                        fig.update_layout(yaxis_title="Market Cap ($)", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No market cap data available")
                
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
                        }, na_rep="-").background_gradient(cmap="RdYlGn"),
                        use_container_width=True
                    )
                    
                    # Margin comparison chart
                    st.markdown("**Margin Comparison**")
                    margin_data = profit_df[["Gross Margin %", "Operating Margin %", "Net Margin %"]].dropna(how='all')
                    
                    if not margin_data.empty:
                        fig = go.Figure()
                        for col in margin_data.columns:
                            valid_data = margin_data[col].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                        
                        fig.update_layout(barmode='group', yaxis_title="Margin %", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ROE vs ROA scatter
                    st.markdown("**ROE vs ROA**")
                    scatter_data = profit_df[["ROA %", "ROE %"]].dropna()
                    if len(scatter_data) > 0:
                        fig = px.scatter(
                            scatter_data, x="ROA %", y="ROE %", 
                            text=scatter_data.index,
                            size=[100]*len(scatter_data),
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
                        }, na_rep="-"),
                        use_container_width=True
                    )
                    
                    # FCF vs Net Income comparison
                    st.markdown("**Free Cash Flow vs Net Income**")
                    
                    fcf_ni_data = cf_df[["Net Income", "Free Cash Flow"]].dropna(how='all')
                    if not fcf_ni_data.empty:
                        fig = go.Figure()
                        
                        ni_valid = fcf_ni_data["Net Income"].dropna()
                        if not ni_valid.empty:
                            fig.add_trace(go.Bar(name="Net Income", x=ni_valid.index, 
                                                y=ni_valid.values, marker_color='lightcoral'))
                        
                        fcf_valid = fcf_ni_data["Free Cash Flow"].dropna()
                        if not fcf_valid.empty:
                            fig.add_trace(go.Bar(name="Free Cash Flow", x=fcf_valid.index, 
                                                y=fcf_valid.values, marker_color='lightgreen'))
                        
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
                        }, na_rep="-").background_gradient(cmap="RdYlGn_r", subset=["Debt/Equity"]),
                        use_container_width=True
                    )
                    
                    # Debt vs Cash
                    st.markdown("**Debt vs Cash Position**")
                    
                    debt_cash_data = health_df[["Cash", "Total Debt"]].dropna(how='all')
                    if not debt_cash_data.empty:
                        fig = go.Figure()
                        
                        cash_valid = debt_cash_data["Cash"].dropna()
                        if not cash_valid.empty:
                            fig.add_trace(go.Bar(name="Cash", x=cash_valid.index, 
                                                y=cash_valid.values, marker_color='lightgreen'))
                        
                        debt_valid = debt_cash_data["Total Debt"].dropna()
                        if not debt_valid.empty:
                            fig.add_trace(go.Bar(name="Total Debt", x=debt_valid.index, 
                                                y=debt_valid.values, marker_color='lightcoral'))
                        
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
                        }, na_rep="-"),
                        use_container_width=True
                    )
                    
                    st.caption("Lower multiples generally indicate cheaper valuation (relative to fundamentals)")
                    
                    # Valuation multiples comparison
                    st.markdown("**Valuation Multiples**")
                    
                    fig = go.Figure()
                    for col in ["P/E", "P/B", "P/S", "EV/EBITDA"]:
                        if col in val_df.columns:
                            valid_data = val_df[col].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                    
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
                    st.caption("Note: Yahoo Finance API provides limited historical fundamental data")
                    
                    # Revenue trend
                    st.markdown("**Revenue Trend**")
                    revenue_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_revenue_ts"]
                        if not ts.empty and len(ts.dropna()) > 0:
                            revenue_trends[ticker] = ts
                    
                    if not revenue_trends.empty:
                        fig = go.Figure()
                        for ticker in revenue_trends.columns:
                            valid_data = revenue_trends[ticker].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=valid_data.index, y=valid_data.values,
                                    name=ticker, mode='lines+markers'
                                ))
                        fig.update_layout(yaxis_title="Revenue ($)", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No revenue trend data available")
                    
                    # Net Income trend
                    st.markdown("**Net Income Trend**")
                    ni_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_net_income_ts"]
                        if not ts.empty and len(ts.dropna()) > 0:
                            ni_trends[ticker] = ts
                    
                    if not ni_trends.empty:
                        fig = go.Figure()
                        for ticker in ni_trends.columns:
                            valid_data = ni_trends[ticker].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=valid_data.index, y=valid_data.values,
                                    name=ticker, mode='lines+markers'
                                ))
                        fig.update_layout(yaxis_title="Net Income ($)", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Margin trends
                    st.markdown("**Net Margin % Trend**")
                    margin_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_net_margin_ts"]
                        if not ts.empty and len(ts.dropna()) > 0:
                            margin_trends[ticker] = ts
                    
                    if not margin_trends.empty:
                        fig = go.Figure()
                        for ticker in margin_trends.columns:
                            valid_data = margin_trends[ticker].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=valid_data.index, y=valid_data.values,
                                    name=ticker, mode='lines+markers'
                                ))
                        fig.update_layout(yaxis_title="Net Margin %", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # FCF trend
                    st.markdown("**Free Cash Flow Trend**")
                    fcf_trends = pd.DataFrame()
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_fcf_ts"]
                        if not ts.empty and len(ts.dropna()) > 0:
                            fcf_trends[ticker] = ts
                    
                    if not fcf_trends.empty:
                        fig = go.Figure()
                        for ticker in fcf_trends.columns:
                            valid_data = fcf_trends[ticker].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=valid_data.index, y=valid_data.values,
                                    name=ticker, mode='lines+markers'
                                ))
                        fig.update_layout(yaxis_title="Free Cash Flow ($)", height=350, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                
                # ══════════════════════
