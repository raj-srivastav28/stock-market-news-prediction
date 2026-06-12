import pandas as pd
import numpy as np
import os

def run_backtest(predictions_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/test_predictions.csv',
                 output_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/backtest_results.csv',
                 top_n=5):
    
    print("STARTING PHASE 6: Institutional Risk Parity Simulation...")

    print(f"Loading predictions from {predictions_path}...")
    try:
        df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print("ERROR: test_predictions.csv not found. Run train_model.py first.")
        return
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Simulating daily Risk Parity allocation to Top {top_n} predicted stocks...")
    
    # 2. Rank the predictions every day
    df['Daily_Rank'] = df.groupby('Date')['Prediction'].rank(method='first', ascending=False)
    
    # ==========================================
    # 3. POSITION SIZING: INVERSE VOLATILITY
    # ==========================================
    # Identify the Top N picks
    df['Is_Top_Pick'] = df['Daily_Rank'] <= top_n
    
    # Calculate inverse volatility (adding 1e-6 to avoid dividing by zero)
    df['Inv_Vol'] = np.where(df['Is_Top_Pick'], 1.0 / (df['Volatility_14d'] + 1e-6), 0.0)
    
    # Sum the inverse volatilities for the daily portfolio
    df['Daily_Inv_Vol_Sum'] = df.groupby('Date')['Inv_Vol'].transform('sum')
    
    # Final Position Weight: (Stock Inv Vol) / (Total Portfolio Inv Vol)
    df['Position'] = np.where(df['Daily_Inv_Vol_Sum'] > 0, df['Inv_Vol'] / df['Daily_Inv_Vol_Sum'], 0.0)
    
    # ==========================================
    # 4. PORTFOLIO AGGREGATION & METRICS
    # ==========================================
    df['Strategy_Return'] = df['Position'] * df['Target_Fwd_Return']
    
    daily_portfolio = df.groupby('Date').agg(
        Strategy_Return=('Strategy_Return', 'sum'),
        Benchmark_Return=('Target_Fwd_Return', 'mean') 
    ).reset_index()
    
    daily_portfolio['Strategy_Equity'] = np.exp(daily_portfolio['Strategy_Return'].cumsum())
    daily_portfolio['Benchmark_Equity'] = np.exp(daily_portfolio['Benchmark_Return'].cumsum())
    
    total_strat_ret = daily_portfolio['Strategy_Equity'].iloc[-1] - 1
    total_bench_ret = daily_portfolio['Benchmark_Equity'].iloc[-1] - 1
    
    trading_days = len(daily_portfolio)
    strat_cagr = (daily_portfolio['Strategy_Equity'].iloc[-1]) ** (252 / trading_days) - 1
    bench_cagr = (daily_portfolio['Benchmark_Equity'].iloc[-1]) ** (252 / trading_days) - 1
    
    strat_vol = daily_portfolio['Strategy_Return'].std() * np.sqrt(252)
    bench_vol = daily_portfolio['Benchmark_Return'].std() * np.sqrt(252)
    
    sharpe_ratio = (daily_portfolio['Strategy_Return'].mean() * 252) / strat_vol
    bench_sharpe = (daily_portfolio['Benchmark_Return'].mean() * 252) / bench_vol
    
    strat_neg_ret = daily_portfolio[daily_portfolio['Strategy_Return'] < 0]['Strategy_Return']
    bench_neg_ret = daily_portfolio[daily_portfolio['Benchmark_Return'] < 0]['Benchmark_Return']
    
    strat_downside_vol = strat_neg_ret.std() * np.sqrt(252) if len(strat_neg_ret) > 0 else 1e-9
    bench_downside_vol = bench_neg_ret.std() * np.sqrt(252) if len(bench_neg_ret) > 0 else 1e-9
    
    sortino_ratio = (daily_portfolio['Strategy_Return'].mean() * 252) / strat_downside_vol
    bench_sortino = (daily_portfolio['Benchmark_Return'].mean() * 252) / bench_downside_vol

    print("\n--- OUT-OF-SAMPLE BACKTEST RESULTS (RISK PARITY) ---")
    print(f"Benchmark -> Return: {total_bench_ret:.2%} | CAGR: {bench_cagr:.2%} | Sharpe: {bench_sharpe:.2f} | Sortino: {bench_sortino:.2f}")
    print(f"Strategy  -> Return: {total_strat_ret:.2%} | CAGR: {strat_cagr:.2%} | Sharpe: {sharpe_ratio:.2f} | Sortino: {sortino_ratio:.2f}")
    
    if sortino_ratio > bench_sortino:
        print("\nINSTITUTIONAL ALPHA: Strategy beat the benchmark on a downside risk-adjusted basis!")
    else:
        print("\nUNDERPERFORMANCE: Strategy failed to beat the benchmark Sortino ratio.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_portfolio.to_csv(output_path, index=False)
    print(f"\nBacktest equity curve saved to {output_path}")
    
    print("==================================================")

if __name__ == "__main__":
    run_backtest()