import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date

st.title("Stock Correlation Calculator (using Adjusted Prices)")

ticker1 = st.text_input("First stock ticker (e.g., AAPL):", "AAPL").strip().upper()
ticker2 = st.text_input("Second stock ticker (e.g., MSFT):", "MSFT").strip().upper()

start_date = st.date_input("Start Date:", value=date(2020, 1, 1))
end_date   = st.date_input("End Date:",   value=date.today())

if st.button("Calculate Correlation"):
    if not ticker1 or not ticker2:
        st.warning("Please enter both tickers.")
    elif start_date >= end_date:
        st.warning("Start date must be before end date.")
    else:
        try:
            # Download data – we rely on default auto_adjust=True → 'Close' is adjusted
            data1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)['Close']
            data2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)['Close']
            
            # Combine and clean
            combined = pd.concat([data1, data2], axis=1).dropna()
            combined.columns = [ticker1, ticker2]
            
            if combined.empty:
                st.error(f"No overlapping data found for {ticker1} and {ticker2} in the selected period.")
            else:
                corr = combined[ticker1].corr(combined[ticker2])
                st.success(f"**Correlation** between **{ticker1}** and **{ticker2}** ({start_date} to {end_date}): **{corr:.4f}**")
                st.info(f"(Based on adjusted closing prices – the 'Close' column from yfinance.)")
                
                # Optional: show a small preview
                with st.expander("Data preview (last 5 rows)"):
                    st.dataframe(combined.tail())

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.info("Common fixes:\n"
                    "- Double-check ticker symbols (case-insensitive, e.g. 'aapl' → 'AAPL')\n"
                    "- Try a wider/recent date range\n"
                    "- Update yfinance: `pip install --upgrade yfinance`\n"
                    "- Yahoo may have temporary issues – try again later")
