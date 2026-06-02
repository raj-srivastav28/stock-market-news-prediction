import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

#Supress the yfinance warnings that are not usefull and for clean output
warnings.filterwarnings('ignore')

TICKERS = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'AVGO', 'LLY']

def fetch_panel_data(tickers = TICKERS , start_date ="2020-01-01" , end_date ="2025-01-01"):

    print(f"Fetching the 5 year data for top {len(TICKERS)} equities based on market cap")
    raw_data_list = []

    for ticker in tickers:
        print(f"Downloading historical data for {ticker}")

        #fetch daily data which include OHLCV

        ticker_df = yf.download(ticker , start = start_date , end = end_date , interval= "1d" , progress = False)

        #sometimes yf returns columns as multiIndex to fix this 
        if isinstance(ticker_df.columns , pd.MultiIndex):
            ticker_df.columns = ticker_df.columns.get_level_values(0)

        # Reset the index so the date is an actionable column rather than a row index
        ticker_df = ticker_df.reset_index()    
        
        # Add an explicit Ticker identifier column to tag these rows
        ticker_df['Ticker'] = ticker
        
        # Ensure standard formatting for the Date column
        ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])

        raw_data_list.append(ticker_df)
        print(f"-> Successfully collected {len(ticker_df)} rows for {ticker}.")

    combined_raw_df = pd.concat(raw_data_list,ignore_index=True)

    # Ensure the target directory exists
    os.makedirs('../data/raw', exist_ok=True)
    
    # Save the raw data to disk
    save_path = '../Stock-MARKET-NEWS-PREDICTION/data/raw/raw_panel_data.csv'
    combined_raw_df.to_csv(save_path, index=False)
    print(os.path.abspath(save_path))
    print(f"Raw data saved to {save_path}")


if __name__ == "__main__":
    fetch_panel_data()



