# excel_export.py

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from io import BytesIO
from typing import Optional


def render_excel_export() -> None:
    """
    Standalone module: Let user input one ticker + date range + frequency,
    then download historical data as Excel (.xlsx)
    """
    st.subheader("Excel Export • Historical Prices")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        ticker = st.text_input(
            "Ticker symbol",
            value="AAPL",
            help="Examples: AAPL, GC=F, BTC-USD, EURUSD=X, BIST30.IS, THYAO.IS"
        ).strip().upper()

    with col2:
        frequency = st.selectbox(
            "Data frequency",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            help="Daily = business days, Weekly/Monthly = aggregated"
        )

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start date",
            value=date(2020, 1, 1),
            max_value=date.today()
        )
    with col_end:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            min_value=start_date
        )

    if not ticker:
        st.info("Please enter a ticker symbol.")
        return

    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        return

    # Map friendly names to yfinance intervals
    interval_map = {
        "Daily": "1d",
        "Weekly": "1wk",
        "Monthly": "1mo"
    }
    selected_interval = interval_map[frequency]

    # ── Fetch & Process ───────────────────────────────────────────────────────
    if st.button("Fetch & Prepare Excel", type="primary", use_container_width=True):
        with st.spinner(f"Downloading {frequency.lower()} data for {ticker} ..."):
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=selected_interval,
                    progress=False,
                    auto_adjust=False,   # keep both Close and Adj Close
                    actions=False        # skip dividends/splits for simplicity
                )

                if df.empty:
                    st.error(f"No data returned for {ticker} in the selected period.")
                    return

                # Clean column names & order
                df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                df.index.name = "Date"

                # Round numeric columns for nicer Excel view
                numeric_cols = ["Open", "High", "Low", "Close", "Adj Close"]
                df[numeric_cols] = df[numeric_cols].round(4)

                st.success(f"Data ready: {len(df)} rows ({frequency.lower()})")

                # ── Preview ───────────────────────────────────────────────────────
                with st.expander("Data Preview (first 10 rows)", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)

                # ── Excel download ────────────────────────────────────────────────
                def to_excel_buffer(dataframe: pd.DataFrame) -> BytesIO:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        dataframe.to_excel(
                            writer,
                            sheet_name="Price History",
                            index=True,
                            float_format="%.4f"
                        )
                        # Optional: auto-adjust columns
                        worksheet = writer.sheets["Price History"]
                        for idx, col in enumerate(dataframe.columns):
                            max_len = max(
                                dataframe[col].astype(str).map(len).max(),
                                len(str(col))
                            )
                            worksheet.set_column(idx + 1, idx + 1, max_len + 2)

                    output.seek(0)
                    return output

                excel_buffer = to_excel_buffer(df)

                file_name = f"{ticker}_{frequency.lower()}_{start_date}_to_{end_date}.xlsx"

                st.download_button(
                    label="📥 Download Excel (.xlsx)",
                    data=excel_buffer,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info(
                    "Tips:\n"
                    "• Check ticker spelling (case insensitive)\n"
                    "• Try a shorter/more recent date range\n"
                    "• Some tickers (exotics) may have limited history"
                )


# For quick local testing / development
if __name__ == "__main__":
    st.set_page_config(page_title="Excel Export Test", layout="wide")
    render_excel_export()
