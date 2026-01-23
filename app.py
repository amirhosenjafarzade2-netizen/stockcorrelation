import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date

# Streamlit app title
st.title("Stock Correlation Calculator")

# User inputs for stock tickers
ticker1 = st.text_input("Enter the first stock ticker (e.g., AAPL):", "AAPL")
ticker2 = st.text_input("Enter the second stock ticker (e.g., MSFT):", "MSFT")

# User inputs for time span
start_date = st.date_input("Start Date:", value=date(2020, 1, 1))
end_date = st.date_input("End Date:", value=date.today())

# Button to calculate correlation
if st.button("Calculate Correlation"):
    try:
        # Fetch historical data using yfinance
        data1 = yf.download(ticker1, start=start_date, end=end_date)['Adj Close']
        data2 = yf.download(ticker2, start=start_date, end=end_date)['Adj Close']
        
        # Combine into a single DataFrame
        combined_data = pd.concat([data1, data2], axis=1)
        combined_data.columns = [ticker1, ticker2]
        
        # Drop any missing values
        combined_data = combined_data.dropna()
        
        # Calculate correlation
        correlation = combined_data[ticker1].corr(combined_data[ticker2])
        
        # Display the result
        st.success(f"The correlation factor between {ticker1} and {ticker2} from {start_date} to {end_date} is: {correlation:.4f}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please check the tickers or date range.")
