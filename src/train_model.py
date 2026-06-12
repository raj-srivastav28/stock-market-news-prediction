print("execution starts")

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

def train_walk_forward(input_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/features_panel_data.csv',
                       output_predictions='../STOCK-MARKET-NEWS-PREDICTION/data/processed/test_predictions.csv'):
    print("==================================================")
    print("STARTING PHASE 5: Walk-Forward Rolling Window...")
    print("==================================================")

    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERROR: {input_path} not found.")
        return
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)

    # 2. Define Features (Ensure Macro_Sentiment is included, absolute prices are excluded)
    exclude_cols = ['Date', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume', 'Target', 'Target_Fwd_Return', 'Log_Return']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Model will train on {len(features)} stationary features: \n{features}\n")

    # 3. Walk-Forward Parameters
    unique_dates = np.sort(df['Date'].unique())
    train_size = 504  # Train on trailing 2 years of trading days
    step_size = 60    # Predict for the next 30 trading days (Rolling 1 Month)
    
    all_predictions = []
    feature_importances_list = []
    
    print(f"Executing Walk-Forward Validation (Train Window: {train_size} days | Step: {step_size} days)\n")

    # 4. The Rolling Loop
    for i in range(train_size, len(unique_dates), step_size):
        
        # --- THE EMBARGO BOUNDARIES (ZERO LEAKAGE) ---
        train_start = unique_dates[max(0, i - train_size)]
        
        # Explicit 2-Day Purge: We drop the immediate days before the test set 
        # to ensure no overlapping target returns bleed into the training data.
        train_end = unique_dates[i - 2] 
        
        test_start = unique_dates[i]
        test_end = unique_dates[min(i + step_size - 1, len(unique_dates) - 1)]
        
        # --- SLICE THE DATA ---
        train_df = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)].copy()
        test_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()
        
        if len(test_df) == 0:
            break
            
        X_train, y_train = train_df[features], train_df['Target']
        X_test = test_df[features]
        
        # --- TRAIN THE ADAPTIVE MODEL ---
        # We re-instantiate the model every 30 days so it learns fresh rules
        model = RandomForestRegressor(
            n_estimators=300, 
            max_depth=8,         
            min_samples_leaf=20, 
            max_features='sqrt', # Forces exploration of sentiment & alternative features
            random_state=42, 
            n_jobs=-1            
        )
        
        # Print progress to terminal
        start_str = pd.to_datetime(test_start).strftime('%Y-%m-%d')
        end_str = pd.to_datetime(test_end).strftime('%Y-%m-%d')
        print(f"Retraining Model... | Predicting Window: {start_str} to {end_str} ({len(test_df)} rows)")
        
        model.fit(X_train, y_train)
        
        # --- INFERENCE ---
        test_df['Prediction'] = model.predict(X_test)
        all_predictions.append(test_df)
        
        # Track what features the model prioritized during this specific month
        importances = pd.Series(model.feature_importances_, index=features)
        feature_importances_list.append(importances)

    # 5. Aggregate Results
    final_predictions_df = pd.concat(all_predictions).sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    
    # 6. Quantitative Evaluation
    print("\n==================================================")
    print("WALK-FORWARD OOS EVALUATION (Aggregated)")
    print("==================================================")
    
    daily_ic = final_predictions_df.groupby('Date').apply(
        lambda x: spearmanr(x['Prediction'], x['Target'])[0] if len(x) > 1 else np.nan
    ).dropna()
    
    mean_ic = daily_ic.mean()
    ic_hit_rate = (daily_ic > 0).mean() 
    
    print(f"Mean Information Coefficient (IC): {mean_ic:.4f}")
    print(f"IC Hit Rate (Days > 0):            {ic_hit_rate:.2%}")
    
    if mean_ic > 0.02:
        print("->ADAPTIVE EDGE: The rolling model successfully captured changing regimes.")
    elif mean_ic > 0.00:
        print("-> ⚖️ STABLE: The model is slightly positive but filtering heavy noise.")
    else:
        print("->REGIME INVERSION: The rolling window is capturing a persistent anti-momentum effect.")

    # 7. Average Feature Importance Across All Windows
    print("\nAverage Top 5 Features Across All Time Windows:")
    avg_importances = pd.concat(feature_importances_list, axis=1).mean(axis=1).sort_values(ascending=False)
    for feat, imp in avg_importances.head(5).items():
        print(f" - {feat}: {imp:.4f}")

    # 8. Save Predictions for Backtester
    os.makedirs(os.path.dirname(output_predictions), exist_ok=True)
    final_predictions_df.to_csv(output_predictions, index=False)
    print(f"\nRolling predictions saved to: {output_predictions}")
    print("==================================================")

if __name__ == "__main__":
    train_walk_forward()