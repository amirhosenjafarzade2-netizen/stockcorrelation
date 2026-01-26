# excel_export_improved.py
# Enhanced Excel Export Module with Multi-Ticker, Advanced Analytics, and Professional Formatting

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check for openpyxl (required for Excel operations)
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.chart import LineChart, BarChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    st.warning("⚠️ openpyxl not available. Install with: pip install openpyxl")


class ExcelExporter:
    """Handles Excel export with multiple sheets, advanced formatting, and charts"""
    
    def __init__(self):
        """Initialize the exporter with color schemes and formats"""
        self.colors = {
            'primary': 'FF4472C4',
            'secondary': 'FF70AD47',
            'accent': 'FFFFC000',
            'danger': 'FFE74C3C',
            'success': 'FF2ECC71',
            'light_gray': 'FFF2F2F2',
            'dark_gray': 'FF7F7F7F'
        }
    
    def create_workbook(self, 
                       ticker_data: Dict[str, pd.DataFrame], 
                       analysis_data: Optional[Dict[str, pd.DataFrame]] = None,
                       metadata: Optional[Dict] = None,
                       include_charts: bool = True) -> BytesIO:
        """
        Create formatted Excel workbook with multiple sheets
        
        Args:
            ticker_data: {ticker: price_df}
            analysis_data: {sheet_name: analysis_df}
            metadata: Export metadata (tickers, dates, etc.)
            include_charts: Whether to include charts
        
        Returns:
            BytesIO buffer with Excel file
        """
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 1. Summary Sheet
                if metadata:
                    self._create_summary_sheet(writer, metadata, ticker_data)
                
                # 2. Price Data Sheets (one per ticker)
                for ticker, df in ticker_data.items():
                    self._create_price_sheet(writer, ticker, df, include_charts)
                
                # 3. Analysis Sheets
                if analysis_data:
                    for sheet_name, df in analysis_data.items():
                        self._create_analysis_sheet(writer, sheet_name, df)
                
                # 4. Comparison Sheet (if multiple tickers)
                if len(ticker_data) > 1:
                    self._create_comparison_sheet(writer, ticker_data, metadata)
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Error creating Excel workbook: {str(e)}")
            # Fallback to basic Excel export
            return self._create_basic_workbook(ticker_data, analysis_data, metadata)
    
    def _create_basic_workbook(self, ticker_data, analysis_data, metadata):
        """Fallback basic Excel creation without advanced formatting"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary
            if metadata:
                summary_df = pd.DataFrame({
                    'Parameter': ['Export Date', 'Tickers', 'Start Date', 'End Date', 'Frequency', 'Total Records'],
                    'Value': [
                        metadata.get('export_date', ''),
                        ', '.join(metadata.get('tickers', [])),
                        str(metadata.get('start_date', '')),
                        str(metadata.get('end_date', '')),
                        metadata.get('frequency', ''),
                        metadata.get('total_records', 0)
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Price data
            for ticker, df in ticker_data.items():
                sheet_name = f"{ticker}_Prices"[:31]
                df.to_excel(writer, sheet_name=sheet_name)
            
            # Analysis data
            if analysis_data:
                for sheet_name, df in analysis_data.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31])
        
        output.seek(0)
        return output
    
    def _create_summary_sheet(self, writer, metadata, ticker_data):
        """Create formatted summary sheet"""
        summary_data = {
            'Parameter': [
                'Export Date & Time',
                'Number of Tickers',
                'Tickers',
                'Start Date',
                'End Date',
                'Period (Days)',
                'Data Frequency',
                'Total Data Points',
                'Average Points per Ticker'
            ],
            'Value': [
                metadata.get('export_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                len(metadata.get('tickers', [])),
                ', '.join(metadata.get('tickers', [])),
                str(metadata.get('start_date', '')),
                str(metadata.get('end_date', '')),
                (metadata.get('end_date') - metadata.get('start_date')).days if metadata.get('end_date') and metadata.get('start_date') else 'N/A',
                metadata.get('frequency', 'Daily'),
                metadata.get('total_records', 0),
                metadata.get('total_records', 0) // max(len(metadata.get('tickers', [])), 1)
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False, startrow=2)
        
        # Format summary sheet
        ws = writer.sheets['Summary']
        
        # Title
        ws['A1'] = 'Historical Stock Data Export'
        ws['A1'].font = Font(size=16, bold=True, color='FFFFFF')
        ws['A1'].fill = PatternFill(start_color=self.colors['primary'], end_color=self.colors['primary'], fill_type='solid')
        ws.merge_cells('A1:B1')
        
        # Column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 50
        
        # Header formatting
        for cell in ws[3]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color=self.colors['secondary'], end_color=self.colors['secondary'], fill_type='solid')
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Freeze panes
        ws.freeze_panes = 'A4'
        
        # Add performance summary if available
        if ticker_data:
            row_offset = len(summary_data) + 6
            ws[f'A{row_offset}'] = 'Quick Performance Summary'
            ws[f'A{row_offset}'].font = Font(size=12, bold=True)
            
            perf_data = []
            for ticker, df in ticker_data.items():
                if len(df) > 1 and 'Close' in df.columns:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    total_return = ((end_price - start_price) / start_price) * 100
                    
                    perf_data.append({
                        'Ticker': ticker,
                        'Start Price': f"${start_price:.2f}",
                        'End Price': f"${end_price:.2f}",
                        'Return': f"{total_return:+.2f}%"
                    })
            
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                df_perf.to_excel(writer, sheet_name='Summary', index=False, startrow=row_offset + 1)
    
    def _create_price_sheet(self, writer, ticker, df, include_charts=True):
        """Create formatted price data sheet with optional chart"""
        sheet_name = f"{ticker}_Prices"[:31]
        df.to_excel(writer, sheet_name=sheet_name, startrow=2)
        
        ws = writer.sheets[sheet_name]
        
        # Title
        ws['A1'] = f'{ticker} Historical Price Data'
        ws['A1'].font = Font(size=14, bold=True, color='FFFFFF')
        ws['A1'].fill = PatternFill(start_color=self.colors['primary'], end_color=self.colors['primary'], fill_type='solid')
        ws.merge_cells(f'A1:{chr(65 + len(df.columns))}1')
        
        # Format headers
        for idx, col in enumerate(['Date'] + list(df.columns)):
            cell = ws.cell(row=3, column=idx + 1)
            cell.value = col
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color=self.colors['secondary'], end_color=self.colors['secondary'], fill_type='solid')
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Auto-adjust column widths
        for idx, col in enumerate(['Date'] + list(df.columns)):
            max_length = len(str(col))
            for row in ws.iter_rows(min_row=4, max_row=min(100, len(df) + 3), min_col=idx + 1, max_col=idx + 1):
                for cell in row:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
            ws.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 15)
        
        # Add conditional formatting for returns column if exists
        if 'Adj Close' in df.columns:
            # Calculate daily returns and add as hidden column
            returns = df['Adj Close'].pct_change() * 100
            
            # Add summary statistics below data
            last_row = len(df) + 5
            ws[f'A{last_row}'] = 'Summary Statistics'
            ws[f'A{last_row}'].font = Font(bold=True, size=11)
            
            stats = {
                'Current Price': f"${df['Close'].iloc[-1]:.2f}" if 'Close' in df.columns else 'N/A',
                'Period High': f"${df['High'].max():.2f}" if 'High' in df.columns else 'N/A',
                'Period Low': f"${df['Low'].min():.2f}" if 'Low' in df.columns else 'N/A',
                'Average Volume': f"{int(df['Volume'].mean()):,}" if 'Volume' in df.columns else 'N/A',
                'Total Return': f"{((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100):+.2f}%" if 'Close' in df.columns and len(df) > 1 else 'N/A'
            }
            
            for idx, (key, value) in enumerate(stats.items()):
                ws[f'A{last_row + idx + 1}'] = key
                ws[f'B{last_row + idx + 1}'] = value
                ws[f'A{last_row + idx + 1}'].font = Font(bold=True)
        
        # Add chart if requested and data is available
        if include_charts and len(df) > 1 and 'Close' in df.columns:
            try:
                chart = LineChart()
                chart.title = f"{ticker} Price Chart"
                chart.style = 13
                chart.y_axis.title = 'Price ($)'
                chart.x_axis.title = 'Date'
                chart.height = 10
                chart.width = 20
                
                # Data for chart
                data = Reference(ws, min_col=5, min_row=3, max_row=min(len(df) + 3, 500))  # Close price column
                dates = Reference(ws, min_col=1, min_row=4, max_row=min(len(df) + 3, 500))
                
                chart.add_data(data, titles_from_data=True)
                chart.set_categories(dates)
                
                # Place chart
                ws.add_chart(chart, f'H5')
            except Exception as e:
                pass  # Silently skip chart if error
        
        # Freeze panes
        ws.freeze_panes = 'B4'
    
    def _create_analysis_sheet(self, writer, sheet_name, df):
        """Create formatted analysis sheet"""
        clean_name = sheet_name[:31]
        df.to_excel(writer, sheet_name=clean_name, startrow=2)
        
        ws = writer.sheets[clean_name]
        
        # Title
        ws['A1'] = sheet_name.replace('_', ' ').title()
        ws['A1'].font = Font(size=14, bold=True, color='FFFFFF')
        ws['A1'].fill = PatternFill(start_color=self.colors['accent'], end_color=self.colors['accent'], fill_type='solid')
        ws.merge_cells(f'A1:{chr(65 + len(df.columns))}1')
        
        # Format headers
        for idx, col in enumerate(['Index'] + list(df.columns)):
            cell = ws.cell(row=3, column=idx + 1)
            cell.value = col if idx > 0 else ''
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color=self.colors['secondary'], end_color=self.colors['secondary'], fill_type='solid')
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Auto-adjust column widths
        for column_cells in ws.columns:
            length = max(len(str(cell.value or '')) for cell in column_cells)
            ws.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 25)
        
        # Freeze panes
        ws.freeze_panes = 'B4'
    
    def _create_comparison_sheet(self, writer, ticker_data, metadata):
        """Create comparison sheet for multiple tickers"""
        comparison_data = []
        
        for ticker, df in ticker_data.items():
            if len(df) > 1 and 'Close' in df.columns:
                returns = df['Close'].pct_change()
                
                stats = {
                    'Ticker': ticker,
                    'Start Date': df.index[0].strftime('%Y-%m-%d'),
                    'End Date': df.index[-1].strftime('%Y-%m-%d'),
                    'Days': len(df),
                    'Start Price': df['Close'].iloc[0],
                    'End Price': df['Close'].iloc[-1],
                    'Total Return (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),
                    'Annualized Return (%)': (((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252 / len(df))) - 1) * 100,
                    'Volatility (%)': returns.std() * np.sqrt(252) * 100,
                    'Max Drawdown (%)': self._calculate_max_drawdown(df['Close']),
                    'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                    'Best Day (%)': returns.max() * 100,
                    'Worst Day (%)': returns.min() * 100,
                    'Positive Days (%)': (returns > 0).sum() / len(returns.dropna()) * 100,
                    'Avg Volume': int(df['Volume'].mean()) if 'Volume' in df.columns else 0
                }
                comparison_data.append(stats)
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            df_comp.to_excel(writer, sheet_name='Comparison', index=False, startrow=2)
            
            ws = writer.sheets['Comparison']
            
            # Title
            ws['A1'] = 'Multi-Ticker Comparison Analysis'
            ws['A1'].font = Font(size=14, bold=True, color='FFFFFF')
            ws['A1'].fill = PatternFill(start_color=self.colors['primary'], end_color=self.colors['primary'], fill_type='solid')
            ws.merge_cells(f'A1:{chr(65 + len(df_comp.columns) - 1)}1')
            
            # Format headers
            for idx, col in enumerate(df_comp.columns):
                cell = ws.cell(row=3, column=idx + 1)
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill(start_color=self.colors['secondary'], end_color=self.colors['secondary'], fill_type='solid')
                cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # Format numeric columns
            for row in ws.iter_rows(min_row=4, max_row=len(df_comp) + 3):
                for idx, cell in enumerate(row):
                    if idx >= 4:  # Numeric columns
                        try:
                            if isinstance(cell.value, (int, float)):
                                if idx in [4, 5]:  # Price columns
                                    cell.number_format = '$#,##0.00'
                                elif 'Return' in df_comp.columns[idx] or '%' in df_comp.columns[idx]:
                                    cell.number_format = '0.00%'
                                    cell.value = cell.value / 100 if cell.value else 0
                                else:
                                    cell.number_format = '#,##0.00'
                        except:
                            pass
            
            # Auto-adjust column widths
            for column_cells in ws.columns:
                length = max(len(str(cell.value or '')) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 20)
            
            # Freeze panes
            ws.freeze_panes = 'B4'
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        return drawdown.min()


def calculate_advanced_analytics(df: pd.DataFrame, ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Calculate comprehensive analytics from price data
    
    Returns dict of DataFrames for different analysis sheets
    """
    analytics = {}
    
    # 1. Returns Analysis
    if 'Close' in df.columns and len(df) > 1:
        returns = df['Close'].pct_change()
        
        returns_data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio (Rf=2%)',
                'Sortino Ratio',
                'Max Drawdown',
                'Calmar Ratio',
                'Best Day',
                'Worst Day',
                'Best Month',
                'Worst Month',
                'Positive Days',
                'Average Daily Return',
                'Average Positive Day',
                'Average Negative Day',
                'Win Rate',
                'Profit Factor',
                'Current Price',
                'Period High',
                'Period Low',
                'Distance from High',
                'Distance from Low'
            ],
            'Value': []
        }
        
        # Calculate metrics
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        days = len(df)
        ann_return = ((1 + total_return/100) ** (252/days) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assuming 2% risk-free rate)
        excess_returns = returns - (0.02 / 252)
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (ann_return - 2) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = ((cum_returns - running_max) / running_max).min() * 100
        
        # Calmar Ratio
        calmar = ann_return / abs(drawdown) if drawdown != 0 else 0
        
        # Best/Worst days
        best_day = returns.max() * 100
        worst_day = returns.min() * 100
        
        # Monthly returns
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.to_period('M')
        monthly_returns = df_monthly.groupby('Month')['Close'].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        
        # Win statistics
        positive_days = (returns > 0).sum()
        total_days = len(returns.dropna())
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        
        avg_positive = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
        avg_negative = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
        
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else 0
        
        # Price levels
        current = df['Close'].iloc[-1]
        high = df['High'].max() if 'High' in df.columns else df['Close'].max()
        low = df['Low'].min() if 'Low' in df.columns else df['Close'].min()
        
        returns_data['Value'] = [
            f"{total_return:.2f}%",
            f"{ann_return:.2f}%",
            f"{volatility:.2f}%",
            f"{sharpe:.2f}",
            f"{sortino:.2f}",
            f"{drawdown:.2f}%",
            f"{calmar:.2f}",
            f"{best_day:.2f}%",
            f"{worst_day:.2f}%",
            f"{best_month:.2f}%",
            f"{worst_month:.2f}%",
            f"{positive_days} ({win_rate:.1f}%)",
            f"{returns.mean() * 100:.4f}%",
            f"{avg_positive:.4f}%",
            f"{avg_negative:.4f}%",
            f"{win_rate:.2f}%",
            f"{profit_factor:.2f}",
            f"${current:.2f}",
            f"${high:.2f}",
            f"${low:.2f}",
            f"{((current - high) / high * 100):.2f}%",
            f"{((current - low) / low * 100):.2f}%"
        ]
        
        analytics['Returns_Analysis'] = pd.DataFrame(returns_data)
    
    # 2. Monthly Performance
    if len(df) > 30:
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.to_period('M')
        
        monthly_stats = []
        for month, group in df_monthly.groupby('Month'):
            if len(group) > 0 and 'Close' in group.columns:
                month_return = (group['Close'].iloc[-1] / group['Close'].iloc[0] - 1) * 100
                monthly_stats.append({
                    'Month': str(month),
                    'Start': round(group['Close'].iloc[0], 2),
                    'End': round(group['Close'].iloc[-1], 2),
                    'Return (%)': round(month_return, 2),
                    'High': round(group['High'].max(), 2) if 'High' in group.columns else None,
                    'Low': round(group['Low'].min(), 2) if 'Low' in group.columns else None,
                    'Avg Volume': int(group['Volume'].mean()) if 'Volume' in group.columns else None,
                    'Volatility (%)': round(group['Close'].pct_change().std() * np.sqrt(21) * 100, 2)
                })
        
        if monthly_stats:
            analytics['Monthly_Performance'] = pd.DataFrame(monthly_stats)
    
    # 3. Drawdown Analysis
    if 'Close' in df.columns and len(df) > 1:
        returns = df['Close'].pct_change()
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown_series = ((cum_returns - running_max) / running_max) * 100
        
        # Find top 5 drawdowns
        drawdowns = []
        in_drawdown = False
        drawdown_start = None
        drawdown_peak = None
        
        for i, (date, dd) in enumerate(drawdown_series.items()):
            if dd < -0.1 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                drawdown_start = date
                drawdown_peak = cum_returns.iloc[:i+1].idxmax()
            elif dd >= -0.05 and in_drawdown:  # End of drawdown
                in_drawdown = False
                drawdown_trough = date
                max_dd = drawdown_series[drawdown_peak:drawdown_trough].min()
                
                drawdowns.append({
                    'Peak Date': drawdown_peak.strftime('%Y-%m-%d'),
                    'Trough Date': drawdown_trough.strftime('%Y-%m-%d'),
                    'Max Drawdown (%)': round(max_dd, 2),
                    'Duration (Days)': (drawdown_trough - drawdown_peak).days,
                    'Peak Price': round(df.loc[drawdown_peak, 'Close'], 2),
                    'Trough Price': round(df.loc[drawdown_trough, 'Close'], 2)
                })
        
        if drawdowns:
            # Sort by max drawdown and take top 5
            drawdowns_sorted = sorted(drawdowns, key=lambda x: x['Max Drawdown (%)'])[:5]
            analytics['Top_Drawdowns'] = pd.DataFrame(drawdowns_sorted)
    
    # 4. Volume Analysis
    if 'Volume' in df.columns:
        volume_data = {
            'Metric': [
                'Average Daily Volume',
                'Median Daily Volume',
                'Max Volume Day',
                'Min Volume Day',
                'Volume Std Dev',
                'Recent 20-Day Avg',
                'Volume vs 20-Day Avg',
                'High Volume Days (>1.5x avg)',
                'Low Volume Days (<0.5x avg)'
            ],
            'Value': []
        }
        
        avg_vol = df['Volume'].mean()
        recent_avg = df['Volume'].tail(20).mean()
        
        volume_data['Value'] = [
            f"{int(avg_vol):,}",
            f"{int(df['Volume'].median()):,}",
            f"{int(df['Volume'].max()):,}",
            f"{int(df['Volume'].min()):,}",
            f"{int(df['Volume'].std()):,}",
            f"{int(recent_avg):,}",
            f"{((recent_avg / avg_vol - 1) * 100):+.1f}%",
            f"{(df['Volume'] > avg_vol * 1.5).sum()}",
            f"{(df['Volume'] < avg_vol * 0.5).sum()}"
        ]
        
        analytics['Volume_Analysis'] = pd.DataFrame(volume_data)
    
    # 5. Technical Levels
    if 'Close' in df.columns:
        closes = df['Close']
        
        # Calculate moving averages
        sma_20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else None
        sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else None
        sma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None
        
        current = closes.iloc[-1]
        
        tech_data = {
            'Indicator': [
                'Current Price',
                'SMA 20',
                'Distance from SMA 20',
                'SMA 50',
                'Distance from SMA 50',
                'SMA 200',
                'Distance from SMA 200',
                '52-Week High',
                'Distance from 52W High',
                '52-Week Low',
                'Distance from 52W Low',
                'ATR (14-day)',
                'Trend (SMA 20 vs 50)',
                'Trend (SMA 50 vs 200)'
            ],
            'Value': []
        }
        
        # ATR calculation
        if all(col in df.columns for col in ['High', 'Low', 'Close']) and len(df) >= 14:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        else:
            atr = None
        
        # 52-week high/low
        high_52w = closes.tail(252).max() if len(closes) >= 252 else closes.max()
        low_52w = closes.tail(252).min() if len(closes) >= 252 else closes.min()
        
        tech_data['Value'] = [
            f"${current:.2f}",
            f"${sma_20:.2f}" if sma_20 else "N/A",
            f"{((current / sma_20 - 1) * 100):+.2f}%" if sma_20 else "N/A",
            f"${sma_50:.2f}" if sma_50 else "N/A",
            f"{((current / sma_50 - 1) * 100):+.2f}%" if sma_50 else "N/A",
            f"${sma_200:.2f}" if sma_200 else "N/A",
            f"{((current / sma_200 - 1) * 100):+.2f}%" if sma_200 else "N/A",
            f"${high_52w:.2f}",
            f"{((current / high_52w - 1) * 100):+.2f}%",
            f"${low_52w:.2f}",
            f"{((current / low_52w - 1) * 100):+.2f}%",
            f"${atr:.2f}" if atr else "N/A",
            "Bullish" if (sma_20 and sma_50 and sma_20 > sma_50) else "Bearish" if (sma_20 and sma_50) else "N/A",
            "Bullish" if (sma_50 and sma_200 and sma_50 > sma_200) else "Bearish" if (sma_50 and sma_200) else "N/A"
        ]
        
        analytics['Technical_Levels'] = pd.DataFrame(tech_data)
    
    return analytics


def render_excel_export() -> None:
    """
    Enhanced Excel Export module with multi-ticker support and advanced analytics
    """
    st.title("📥 Advanced Excel Export")
    st.markdown("Export historical stock data with professional formatting and comprehensive analytics")
    
    # Check dependencies
    if not OPENPYXL_AVAILABLE:
        st.error("❌ openpyxl is required for Excel export. Install with: `pip install openpyxl`")
        st.info("You can still use basic export functionality, but advanced formatting will be limited.")
    
    st.divider()
    
    # ── Export Mode Selection ──
    col1, col2 = st.columns([3, 2])
    
    with col1:
        export_mode = st.radio(
            "📊 Export Mode",
            options=["Single Ticker", "Multiple Tickers", "Portfolio Analysis"],
            horizontal=True,
            help="Single: One ticker with full analytics | Multiple: Batch export | Portfolio: Comparison analysis"
        )
    
    with col2:
        include_charts = st.checkbox("Include Charts", value=True, help="Add price charts to sheets (requires openpyxl)")
    
    st.divider()
    
    # ── Configuration Section ──
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if export_mode == "Single Ticker":
            ticker_input = st.text_input(
                "🎯 Ticker Symbol",
                value="AAPL",
                placeholder="Enter ticker (e.g., AAPL, MSFT, TSLA)",
                help="Examples: AAPL (stock), GC=F (gold), BTC-USD (crypto), EURUSD=X (forex), THYAO.IS (international)"
            ).strip().upper()
            tickers = [ticker_input] if ticker_input else []
        else:
            ticker_input = st.text_area(
                "📝 Ticker Symbols (one per line or comma-separated)",
                value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
                height=120,
                help="Enter multiple tickers separated by newlines or commas"
            )
            # Parse both newline and comma separated
            raw_tickers = ticker_input.replace(',', '\n').replace(';', '\n').split('\n')
            tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
            
            if len(tickers) > 50:
                st.warning(f"⚠️ Limiting to first 50 tickers (you entered {len(tickers)})")
                tickers = tickers[:50]
            
            if tickers:
                st.success(f"✅ {len(tickers)} ticker(s) ready for export")
    
    with col2:
        frequency = st.selectbox(
            "📈 Data Frequency",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            help="Daily = business days only | Weekly/Monthly = aggregated periods"
        )
        
        include_analytics = st.checkbox(
            "📊 Include Analytics",
            value=True,
            help="Add returns analysis, monthly stats, technical levels, and more"
        )
        
        advanced_analytics = st.checkbox(
            "🔬 Advanced Analytics",
            value=False,
            help="Include drawdown analysis, volume patterns, and technical indicators (slower)"
        )
    
    # ── Date Range ──
    st.subheader("📅 Date Range")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preset = st.selectbox(
            "Quick Select",
            ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "10 Years", "Max"]
        )
    
    if preset != "Custom":
        end_date = date.today()
        if preset == "1 Month":
            start_date = end_date - timedelta(days=30)
        elif preset == "3 Months":
            start_date = end_date - timedelta(days=90)
        elif preset == "6 Months":
            start_date = end_date - timedelta(days=180)
        elif preset == "1 Year":
            start_date = end_date - timedelta(days=365)
        elif preset == "2 Years":
            start_date = end_date - timedelta(days=730)
        elif preset == "5 Years":
            start_date = end_date - timedelta(days=1825)
        elif preset == "10 Years":
            start_date = end_date - timedelta(days=3650)
        else:  # Max
            start_date = date(2000, 1, 1)
        
        with col2:
            start_date = st.date_input("Start Date", value=start_date, max_value=date.today())
        with col3:
            end_date = st.date_input("End Date", value=end_date, min_value=start_date, max_value=date.today())
    else:
        with col2:
            start_date = st.date_input("Start Date", value=date(2020, 1, 1), max_value=date.today())
        with col3:
            end_date = st.date_input("End Date", value=date.today(), min_value=start_date, max_value=date.today())
    
    # Validation
    if not tickers:
        st.info("👆 Please enter at least one ticker symbol above to continue.")
        return
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return
    
    # Map frequency
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    selected_interval = interval_map[frequency]
    
    # Calculate expected data points
    days_diff = (end_date - start_date).days
    if frequency == "Daily":
        expected_points = days_diff * 0.7  # Approximate business days
    elif frequency == "Weekly":
        expected_points = days_diff / 7
    else:  # Monthly
        expected_points = days_diff / 30
    
    st.info(f"📊 Period: {days_diff} days | Expected data points: ~{int(expected_points)} per ticker")
    
    st.divider()
    
    # ── Fetch & Export Button ──
    export_button = st.button(
        f"🚀 Fetch Data & Generate Excel Report",
        type="primary",
        use_container_width=True
    )
    
    if export_button:
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Processing Your Request")
            
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            
            ticker_data = {}
            analysis_data = {}
            failed_tickers = []
            total_records = 0
            
            # Fetch data for each ticker
            for idx, ticker in enumerate(tickers):
                status_text.info(f"📡 Fetching {ticker}... ({idx + 1}/{len(tickers)})")
                progress_bar.progress((idx / len(tickers)), text=f"Downloading {ticker}...")
                
                try:
                    # Download data
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
                        failed_tickers.append((ticker, "No data returned"))
                        continue
                    
                    # Clean and prepare data
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                    available_cols = [col for col in required_cols if col in df.columns]
                    
                    if not available_cols:
                        failed_tickers.append((ticker, "Missing required columns"))
                        continue
                    
                    df = df[available_cols]
                    df.index.name = "Date"
                    
                    # Round numeric columns
                    numeric_cols = [col for col in df.columns if col != "Volume"]
                    df[numeric_cols] = df[numeric_cols].round(4)
                    
                    # Ensure Volume is integer
                    if "Volume" in df.columns:
                        df["Volume"] = df["Volume"].astype(int)
                    
                    ticker_data[ticker] = df
                    total_records += len(df)
                    
                    # Calculate analytics
                    if include_analytics and (export_mode == "Single Ticker" or advanced_analytics):
                        status_text.info(f"📊 Calculating analytics for {ticker}...")
                        ticker_analytics = calculate_advanced_analytics(df, ticker)
                        
                        for key, value in ticker_analytics.items():
                            analysis_data[f"{ticker}_{key}"] = value
                    
                except Exception as e:
                    failed_tickers.append((ticker, str(e)))
                    continue
                
                progress_bar.progress(((idx + 1) / len(tickers)), text=f"Completed {ticker}")
            
            # Clear progress indicators
            progress_bar.progress(1.0, text="✅ Data fetch complete!")
            status_text.empty()
        
        # ── Results Summary ──
        st.divider()
        st.markdown("### 📈 Fetch Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Successful", len(ticker_data), delta=None)
        with col2:
            st.metric("❌ Failed", len(failed_tickers), delta=None, delta_color="inverse")
        with col3:
            st.metric("📊 Total Records", f"{total_records:,}", delta=None)
        with col4:
            avg_records = total_records // max(len(ticker_data), 1)
            st.metric("📉 Avg per Ticker", f"{avg_records:,}", delta=None)
        
        if not ticker_data:
            st.error("❌ No data could be fetched for any ticker. Please check your ticker symbols and try again.")
            if failed_tickers:
                st.error("**Failed tickers:**")
                for ticker, reason in failed_tickers:
                    st.write(f"• {ticker}: {reason}")
            return
        
        if failed_tickers:
            with st.expander(f"⚠️ {len(failed_tickers)} Failed Ticker(s) - Click to expand"):
                for ticker, reason in failed_tickers:
                    st.write(f"• **{ticker}**: {reason}")
        
        # ── Portfolio Analytics (for multiple tickers) ──
        if export_mode == "Portfolio Analysis" and len(ticker_data) > 1:
            st.divider()
            st.markdown("### 📊 Portfolio Overview")
            
            with st.spinner("Calculating portfolio metrics..."):
                # Combine all close prices
                combined_prices = pd.DataFrame({
                    ticker: df['Close'] for ticker, df in ticker_data.items()
                })
                
                # Calculate returns
                returns = combined_prices.pct_change()
                
                # Correlation matrix
                corr_matrix = returns.corr()
                analysis_data['Correlation_Matrix'] = corr_matrix.round(3)
                
                # Portfolio statistics
                portfolio_stats = []
                for ticker in ticker_data.keys():
                    ticker_returns = returns[ticker].dropna()
                    
                    stats = {
                        'Ticker': ticker,
                        'Total Return (%)': ((combined_prices[ticker].iloc[-1] / combined_prices[ticker].iloc[0] - 1) * 100),
                        'Ann. Return (%)': ((1 + ticker_returns.mean()) ** 252 - 1) * 100,
                        'Volatility (%)': ticker_returns.std() * np.sqrt(252) * 100,
                        'Sharpe Ratio': (ticker_returns.mean() / ticker_returns.std()) * np.sqrt(252) if ticker_returns.std() > 0 else 0,
                        'Max Drawdown (%)': ((1 + ticker_returns).cumprod().div((1 + ticker_returns).cumprod().cummax()) - 1).min() * 100,
                        'Best Day (%)': ticker_returns.max() * 100,
                        'Worst Day (%)': ticker_returns.min() * 100
                    }
                    portfolio_stats.append(stats)
                
                portfolio_df = pd.DataFrame(portfolio_stats).round(2)
                analysis_data['Portfolio_Statistics'] = portfolio_df
                
                # Display preview
                st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
                
                # Best/worst performers
                col1, col2 = st.columns(2)
                with col1:
                    best_performer = portfolio_df.loc[portfolio_df['Total Return (%)'].idxmax()]
                    st.success(f"🏆 **Best Performer**: {best_performer['Ticker']} ({best_performer['Total Return (%)']:+.2f}%)")
                with col2:
                    worst_performer = portfolio_df.loc[portfolio_df['Total Return (%)'].idxmin()]
                    st.error(f"📉 **Worst Performer**: {worst_performer['Ticker']} ({worst_performer['Total Return (%)']:+.2f}%)")
        
        # ── Data Preview ──
        st.divider()
        st.markdown("### 📋 Data Preview")
        
        preview_ticker = st.selectbox(
            "Select ticker to preview:",
            options=list(ticker_data.keys()),
            format_func=lambda x: f"{x} ({len(ticker_data[x])} records)"
        )
        
        if preview_ticker:
            with st.expander(f"📊 {preview_ticker} - Preview", expanded=True):
                preview_df = ticker_data[preview_ticker]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(preview_df.head(5), use_container_width=True)
                with col2:
                    st.write("**Last 5 rows:**")
                    st.dataframe(preview_df.tail(5), use_container_width=True)
                
                # Quick stats
                if 'Close' in preview_df.columns:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Start Price", f"${preview_df['Close'].iloc[0]:.2f}")
                    with col2:
                        st.metric("End Price", f"${preview_df['Close'].iloc[-1]:.2f}")
                    with col3:
                        total_ret = ((preview_df['Close'].iloc[-1] / preview_df['Close'].iloc[0] - 1) * 100)
                        st.metric("Total Return", f"{total_ret:+.2f}%")
                    with col4:
                        st.metric("Data Points", len(preview_df))
        
        # ── Generate Excel ──
        st.divider()
        st.markdown("### 📥 Download Excel Report")
        
        with st.spinner("🔨 Generating Excel workbook with formatting..."):
            metadata = {
                'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tickers': list(ticker_data.keys()),
                'start_date': start_date,
                'end_date': end_date,
                'frequency': frequency,
                'total_records': total_records
            }
            
            exporter = ExcelExporter()
            excel_buffer = exporter.create_workbook(
                ticker_data=ticker_data,
                analysis_data=analysis_data if include_analytics else None,
                metadata=metadata,
                include_charts=include_charts and OPENPYXL_AVAILABLE
            )
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if len(ticker_data) == 1:
            file_name = f"{list(ticker_data.keys())[0]}_{frequency.lower()}_{start_date}_to_{end_date}.xlsx"
        else:
            file_name = f"stock_data_{len(ticker_data)}tickers_{frequency.lower()}_{timestamp}.xlsx"
        
        # Download button
        st.download_button(
            label=f"📥 Download Excel Report • {len(ticker_data)} ticker(s) • {total_records:,} records",
            data=excel_buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        
        # File details
        total_sheets = 1 + len(ticker_data) + len(analysis_data) + (1 if len(ticker_data) > 1 else 0)  # Summary + Prices + Analysis + Comparison
        
        st.success(f"✅ Excel file ready for download!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📑 Total Sheets", total_sheets)
        with col2:
            st.metric("💾 Estimated Size", f"~{(total_records * 50) / 1024:.1f} KB")
        with col3:
            st.metric("📊 Charts", "Yes" if include_charts and OPENPYXL_AVAILABLE else "No")
        
        # Sheet breakdown
        with st.expander("📑 Detailed Sheet Breakdown"):
            sheets_list = []
            sheets_list.append("1. **Summary** - Export metadata and quick performance overview")
            
            for i, ticker in enumerate(ticker_data.keys(), 2):
                sheets_list.append(f"{i}. **{ticker}_Prices** - Historical price data with {len(ticker_data[ticker])} records")
            
            sheet_num = len(ticker_data) + 2
            for sheet_name in analysis_data.keys():
                sheets_list.append(f"{sheet_num}. **{sheet_name}** - Analytics and calculations")
                sheet_num += 1
            
            if len(ticker_data) > 1:
                sheets_list.append(f"{sheet_num}. **Comparison** - Multi-ticker performance comparison")
            
            for sheet_desc in sheets_list:
                st.markdown(sheet_desc)


# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Excel Export", layout="wide", page_icon="📥")
    render_excel_export()
