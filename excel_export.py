# excel_export.py
# Enhanced Excel Export Module with Multi-Ticker, Analysis Sheets, and Formatting

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, datetime
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExcelExporter:
    """Handles Excel export with multiple sheets and advanced formatting"""
    
    @staticmethod
    def create_workbook(ticker_data: Dict[str, pd.DataFrame], 
                       analysis_data: Optional[Dict[str, pd.DataFrame]] = None,
                       metadata: Optional[Dict] = None) -> BytesIO:
        """
        Create formatted Excel workbook with multiple sheets
        
        Args:
            ticker_data: {ticker: price_df}
            analysis_data: {sheet_name: analysis_df}
            metadata: Export metadata (tickers, dates, etc.)
        """
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1,
                'align': 'center'
            })
            
            number_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1
            })
            
            percent_format = workbook.add_format({
                'num_format': '0.00%',
                'border': 1
            })
            
            date_format = workbook.add_format({
                'num_format': 'yyyy-mm-dd',
                'border': 1
            })
            
            # 1. Summary/Overview Sheet
            if metadata:
                summary_data = {
                    'Parameter': ['Export Date', 'Tickers', 'Start Date', 'End Date', 
                                 'Frequency', 'Total Records'],
                    'Value': [
                        metadata.get('export_date', datetime.now().strftime('%Y-%m-%d')),
                        ', '.join(metadata.get('tickers', [])),
                        str(metadata.get('start_date', '')),
                        str(metadata.get('end_date', '')),
                        metadata.get('frequency', 'Daily'),
                        metadata.get('total_records', 0)
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                worksheet = writer.sheets['Summary']
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:B', 40)
            
            # 2. Price Data Sheets (one per ticker)
            for ticker, df in ticker_data.items():
                sheet_name = f"{ticker}_Prices"[:31]  # Excel limit
                df.to_excel(writer, sheet_name=sheet_name, index=True)
                
                worksheet = writer.sheets[sheet_name]
                
                # Format header row
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num + 1, value, header_format)
                
                # Auto-adjust column widths
                for idx, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    worksheet.set_column(idx + 1, idx + 1, min(max_len + 3, 20))
                
                # Format date column
                worksheet.set_column(0, 0, 12, date_format)
            
            # 3. Analysis Sheets
            if analysis_data:
                for sheet_name, df in analysis_data.items():
                    clean_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=clean_name, index=True)
                    
                    worksheet = writer.sheets[clean_name]
                    
                    # Format headers
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num + 1, value, header_format)
                    
                    # Auto-adjust columns
                    for idx, col in enumerate(df.columns):
                        max_len = max(
                            df[col].astype(str).map(len).max() if len(df) > 0 else 10,
                            len(str(col))
                        )
                        worksheet.set_column(idx + 1, idx + 1, min(max_len + 2, 25))
        
        output.seek(0)
        return output


def calculate_analytics(df: pd.DataFrame, ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Calculate various analytics from price data
    
    Returns dict of DataFrames for different analysis sheets
    """
    analytics = {}
    
    # 1. Returns Analysis
    returns_data = {
        'Metric': [],
        'Value': []
    }
    
    if 'Close' in df.columns and len(df) > 1:
        returns = df['Close'].pct_change()
        
        returns_data['Metric'].extend([
            'Total Return',
            'Annualized Return',
            'Volatility (Annual)',
            'Best Day',
            'Worst Day',
            'Positive Days %',
            'Avg Daily Return',
            'Max Drawdown'
        ])
        
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        days = len(df)
        ann_return = ((1 + total_return/100) ** (252/days) - 1) * 100
        volatility = returns.std() * (252 ** 0.5) * 100
        best_day = returns.max() * 100
        worst_day = returns.min() * 100
        positive_pct = (returns > 0).sum() / len(returns.dropna()) * 100
        avg_return = returns.mean() * 100
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = ((cum_returns - running_max) / running_max * 100).min()
        
        returns_data['Value'].extend([
            f"{total_return:.2f}%",
            f"{ann_return:.2f}%",
            f"{volatility:.2f}%",
            f"{best_day:.2f}%",
            f"{worst_day:.2f}%",
            f"{positive_pct:.1f}%",
            f"{avg_return:.4f}%",
            f"{drawdown:.2f}%"
        ])
        
        analytics['Returns_Analysis'] = pd.DataFrame(returns_data)
    
    # 2. Monthly Statistics
    if len(df) > 30:
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.to_period('M')
        
        monthly_stats = []
        for month, group in df_monthly.groupby('Month'):
            if len(group) > 0 and 'Close' in group.columns:
                month_return = (group['Close'].iloc[-1] / group['Close'].iloc[0] - 1) * 100
                monthly_stats.append({
                    'Month': str(month),
                    'Return (%)': round(month_return, 2),
                    'High': round(group['High'].max(), 2) if 'High' in group.columns else None,
                    'Low': round(group['Low'].min(), 2) if 'Low' in group.columns else None,
                    'Avg Volume': int(group['Volume'].mean()) if 'Volume' in group.columns else None
                })
        
        if monthly_stats:
            analytics['Monthly_Stats'] = pd.DataFrame(monthly_stats)
    
    # 3. Price Levels
    if 'Close' in df.columns:
        price_levels = {
            'Metric': [
                'Current Price',
                '52-Week High',
                '52-Week Low',
                'Distance from High',
                'Distance from Low',
                'Average Price (Period)',
                'Median Price (Period)'
            ],
            'Value': []
        }
        
        current = df['Close'].iloc[-1]
        high_52w = df['Close'].tail(252).max() if len(df) >= 252 else df['Close'].max()
        low_52w = df['Close'].tail(252).min() if len(df) >= 252 else df['Close'].min()
        
        price_levels['Value'].extend([
            f"${current:.2f}",
            f"${high_52w:.2f}",
            f"${low_52w:.2f}",
            f"{((current - high_52w) / high_52w * 100):.2f}%",
            f"{((current - low_52w) / low_52w * 100):.2f}%",
            f"${df['Close'].mean():.2f}",
            f"${df['Close'].median():.2f}"
        ])
        
        analytics['Price_Levels'] = pd.DataFrame(price_levels)
    
    return analytics


def render_excel_export() -> None:
    """
    Enhanced Excel Export module with multi-ticker support and analytics
    """
    st.subheader("📥 Excel Export • Advanced Historical Data")
    
    # ── Export Mode Selection ──
    export_mode = st.radio(
        "Export Mode",
        options=["Single Ticker", "Multiple Tickers", "Portfolio Analysis"],
        horizontal=True,
        help="Choose between single ticker, batch export, or portfolio analysis"
    )
    
    st.markdown("---")
    
    # ── Configuration Section ──
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if export_mode == "Single Ticker":
            ticker_input = st.text_input(
                "Ticker Symbol",
                value="AAPL",
                help="Examples: AAPL, GC=F, BTC-USD, EURUSD=X, THYAO.IS"
            ).strip().upper()
            tickers = [ticker_input] if ticker_input else []
        else:
            ticker_input = st.text_area(
                "Ticker Symbols (one per line or comma-separated)",
                value="AAPL\nMSFT\nGOOGL\nAMZN",
                height=100,
                help="Enter multiple tickers"
            )
            # Parse both newline and comma separated
            tickers = [t.strip().upper() for t in ticker_input.replace(',', '\n').split('\n') if t.strip()]
            
            if len(tickers) > 20:
                st.warning(f"⚠️ Limiting to first 20 tickers (you entered {len(tickers)})")
                tickers = tickers[:20]
            
            st.caption(f"✅ {len(tickers)} ticker(s) ready")
    
    with col2:
        frequency = st.selectbox(
            "Data Frequency",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            help="Daily = business days, Weekly/Monthly = aggregated"
        )
        
        include_analytics = st.checkbox(
            "Include Analytics Sheets",
            value=True,
            help="Add returns, monthly stats, and price level analysis"
        )
    
    # ── Date Range ──
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start Date",
            value=date(2020, 1, 1),
            max_value=date.today()
        )
    with col_end:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=start_date
        )
    
    # Validation
    if not tickers:
        st.info("👆 Please enter at least one ticker symbol above.")
        return
    
    if start_date >= end_date:
        st.warning("⚠️ Start date must be before end date.")
        return
    
    # Map frequency
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    selected_interval = interval_map[frequency]
    
    st.markdown("---")
    
    # ── Fetch & Export ──
    if st.button("📊 Fetch Data & Generate Excel", type="primary", use_container_width=True):
        
        with st.spinner(f"Downloading {frequency.lower()} data for {len(tickers)} ticker(s)..."):
            ticker_data = {}
            analysis_data = {}
            failed_tickers = []
            total_records = 0
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, ticker in enumerate(tickers):
                status_text.text(f"Fetching {ticker}... ({idx + 1}/{len(tickers)})")
                
                try:
                    df = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        interval=selected_interval,
                        progress=False,
                        auto_adjust=False,
                        actions=False
                    )
                    
                    if df.empty:
                        failed_tickers.append(ticker)
                        continue
                    
                    # Clean and prepare data
                    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                    df.index.name = "Date"
                    
                    # Round numeric columns
                    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close"]
                    df[numeric_cols] = df[numeric_cols].round(4)
                    
                    ticker_data[ticker] = df
                    total_records += len(df)
                    
                    # Calculate analytics if requested
                    if include_analytics and export_mode != "Multiple Tickers":
                        ticker_analytics = calculate_analytics(df, ticker)
                        for key, value in ticker_analytics.items():
                            analysis_data[f"{ticker}_{key}"] = value
                    
                except Exception as e:
                    failed_tickers.append(ticker)
                    st.warning(f"⚠️ Failed to fetch {ticker}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(tickers))
            
            progress_bar.empty()
            status_text.empty()
            
            # Results summary
            if not ticker_data:
                st.error("❌ No data could be fetched for any ticker.")
                if failed_tickers:
                    st.error(f"Failed tickers: {', '.join(failed_tickers)}")
                return
            
            # Success metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Successful", len(ticker_data))
            with col2:
                st.metric("❌ Failed", len(failed_tickers))
            with col3:
                st.metric("📊 Total Records", f"{total_records:,}")
            
            if failed_tickers:
                with st.expander("⚠️ Failed Tickers"):
                    st.write(", ".join(failed_tickers))
            
            # ── Portfolio Analytics (for multiple tickers) ──
            if export_mode == "Portfolio Analysis" and len(ticker_data) > 1:
                st.subheader("📊 Portfolio Overview")
                
                # Combine all close prices
                combined_prices = pd.DataFrame({
                    ticker: df['Close'] for ticker, df in ticker_data.items()
                })
                
                # Calculate correlation matrix
                corr_matrix = combined_prices.pct_change().corr()
                analysis_data['Correlation_Matrix'] = corr_matrix.round(3)
                
                # Portfolio returns
                returns = combined_prices.pct_change()
                portfolio_stats = pd.DataFrame({
                    'Ticker': list(ticker_data.keys()),
                    'Total Return (%)': [(combined_prices[t].iloc[-1] / combined_prices[t].iloc[0] - 1) * 100 
                                        for t in ticker_data.keys()],
                    'Volatility (%)': [returns[t].std() * (252**0.5) * 100 for t in ticker_data.keys()],
                    'Sharpe Ratio': [(returns[t].mean() / returns[t].std()) * (252**0.5) 
                                    for t in ticker_data.keys()]
                }).round(2)
                
                analysis_data['Portfolio_Stats'] = portfolio_stats
                
                # Display preview
                st.dataframe(portfolio_stats, use_container_width=True)
            
            # ── Preview ──
            st.markdown("---")
            st.subheader("📋 Data Preview")
            
            preview_ticker = st.selectbox(
                "Select ticker to preview",
                options=list(ticker_data.keys())
            )
            
            if preview_ticker:
                with st.expander(f"{preview_ticker} - First 10 rows", expanded=True):
                    st.dataframe(ticker_data[preview_ticker].head(10), use_container_width=True)
            
            # ── Generate Excel ──
            st.markdown("---")
            
            metadata = {
                'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tickers': list(ticker_data.keys()),
                'start_date': start_date,
                'end_date': end_date,
                'frequency': frequency,
                'total_records': total_records
            }
            
            excel_buffer = ExcelExporter.create_workbook(
                ticker_data=ticker_data,
                analysis_data=analysis_data if include_analytics else None,
                metadata=metadata
            )
            
            # Generate filename
            if len(ticker_data) == 1:
                file_name = f"{list(ticker_data.keys())[0]}_{frequency.lower()}_{start_date}_to_{end_date}.xlsx"
            else:
                file_name = f"portfolio_{len(ticker_data)}tickers_{frequency.lower()}_{start_date}_to_{end_date}.xlsx"
            
            st.download_button(
                label=f"📥 Download Excel • {len(ticker_data)} ticker(s) • {total_records:,} records",
                data=excel_buffer,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
            
            st.success(f"✅ Excel file ready! Includes {len(ticker_data) + len(analysis_data) + 1} sheets")
            
            # Sheet breakdown
            with st.expander("📑 Sheet Breakdown"):
                sheets = ['Summary'] + [f"{t}_Prices" for t in ticker_data.keys()] + list(analysis_data.keys())
                st.write(f"**Total sheets:** {len(sheets)}")
                for i, sheet in enumerate(sheets, 1):
                    st.caption(f"{i}. {sheet}")


# For quick local testing
if __name__ == "__main__":
    st.set_page_config(page_title="Excel Export Test", layout="wide")
    render_excel_export()
