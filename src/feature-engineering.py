import pandas as pd
import numpy as np
import os

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_features(input_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/cleaned_panel_data.csv', 
                      output_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/features_panel_data.csv'):
    
    
    print("STARTING ...")
    
    
    # 1. Load Cleaned Price Data
    print("Loading cleaned panel data...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    
    
    # 2. MERGE MACRO SENTIMENT DATA

    print("Merging Macro NLP Sentiment Data...")
    try:
        sentiment_df = pd.read_csv('../STOCK-MARKET-NEWS-PREDICTION/data/processed/macro_sentiment.csv')
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.normalize()
        
        rows_before = len(df)
        df = pd.merge(df, sentiment_df, on='Date', how='left')
        
        # Merge Sanity Check
        rows_after = len(df)
        missing_sentiment = df['Macro_Sentiment'].isna().sum()
        match_rate = 100 - ((missing_sentiment / rows_after) * 100)
        
        print(f"Sentiment Match Rate: {match_rate:.2f}%")
        
        # Fill missing days safely
        df['Macro_Sentiment'] = df['Macro_Sentiment'].fillna(0.0)
        df['Macro_Sentiment_3d'] = df['Macro_Sentiment_3d'].fillna(0.0)
        df['Macro_News_Volume'] = df['Macro_News_Volume'].fillna(0)
        
    except FileNotFoundError:
        print("WARNING: macro_sentiment.csv not found. Proceeding without NLP.")
    
    
    # 3. ENFORCE SORTING (CRITICAL)
    
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

        
    # 4. BASE RETURNS & TARGET FORMULATION
    
    print("Calculating Returns and Cross-Sectional Targets...")
    df['Log_Return'] = np.log(df['Close'] / df.groupby('Ticker')['Close'].shift(1))
    
    # Target: Rank of Tomorrow's Return (-0.5 to +0.5)
    df['Target_Fwd_Return'] = df.groupby('Ticker')['Log_Return'].shift(-1)
    df['Target'] = df.groupby('Date')['Target_Fwd_Return'].transform(lambda x: x.rank(pct=True)) - 0.5

    
    # 5. STANDARD ALPHA FEATURES
    
    print("Engineering Technical Features...")
    df['SMA_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df['Price_to_SMA_20'] = (df['Close'] / df['SMA_20']) - 1 
    df['Volatility_14d'] = df.groupby('Ticker')['Log_Return'].transform(lambda x: x.rolling(14).std())
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x, 14))
    
    df['Return_Lag1'] = df.groupby('Ticker')['Log_Return'].shift(1)
    df['Return_Lag2'] = df.groupby('Ticker')['Log_Return'].shift(2)
    df['Return_Lag3'] = df.groupby('Ticker')['Log_Return'].shift(3)

    
    # 6. ADVANCED ALPHA FEATURES
    
    # On-Balance Volume (OBV)
    df['Price_Dir'] = np.sign(df.groupby('Ticker')['Close'].diff())
    df['OBV'] = df.groupby('Ticker').apply(lambda x: (x['Volume'] * x['Price_Dir']).cumsum()).reset_index(level=0, drop=True)
    df['OBV_SMA_20'] = df.groupby('Ticker')['OBV'].transform(lambda x: x.rolling(20).mean())
    df['OBV_Signal'] = (df['OBV'] - df['OBV_SMA_20']) / df.groupby('Ticker')['OBV'].transform(lambda x: x.rolling(20).std())

    # MACD
    ema_12 = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD_Line'] = ema_12 - ema_26
    df['MACD_Signal'] = df.groupby('Ticker')['MACD_Line'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']

    # Bollinger Band Width
    std_20 = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).std())
    upper_band = df['SMA_20'] + (std_20 * 2)
    lower_band = df['SMA_20'] - (std_20 * 2)
    df['BB_Width'] = (upper_band - lower_band) / df['SMA_20']

    # Cross-Sectional Z-Score
    daily_mean = df.groupby('Date')['Log_Return'].transform('mean')
    daily_std = df.groupby('Date')['Log_Return'].transform('std')
    df['Cross_Sectional_Z'] = (df['Log_Return'] - daily_mean) / daily_std


    # extra features added 

    # ==========================================
    # 6.5 SENTIMENT FEATURE INTERACTIONS (NEW)
    # ==========================================
    # We multiply the Global Sentiment by Local Technicals to give the AI a localized text signal
    
    # 1. Sentiment x Volatility: Does good news make highly volatile stocks explode upward?
    df['Sentiment_x_Vol'] = df['Macro_Sentiment'] * df['Volatility_14d']
    
    # 2. Sentiment x Momentum: Does good news act as fuel for stocks already breaking out?
    df['Sentiment_x_ZScore'] = df['Macro_Sentiment'] * df['Cross_Sectional_Z']
    
    # 3. Sentiment x Price Trend: How does news impact stocks trading below their moving average?
    df['Sentiment_x_SMA'] = df['Macro_Sentiment'] * df['Price_to_SMA_20']
    print("added")

    
    # 7. CLEANUP & SAVE
    
    print("Dropping intermediate calculation columns and incomplete rows...")
    
    cols_to_drop = ['SMA_20', 'Price_Dir', 'OBV', 'OBV_SMA_20', 'MACD_Line', 'MACD_Signal']
    df = df.drop(columns=cols_to_drop)
    
    # Drop rows that don't have enough history to calculate rolling features (e.g., first 26 days)
    df = df.dropna()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSUCCESS: Final ML Matrix saved with {len(df.columns)} columns.")
    print("Columns included in output:")
    print(list(df.columns))
    
    print("==================================================")

   
if __name__ == "__main__":
    engineer_features()