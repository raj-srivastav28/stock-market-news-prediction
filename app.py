import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Systematic Alpha Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. DATA LOADING & CACHING

@st.cache_data
def load_backtest_data():
    file_path = "/STOCK-MARKET-NEWS-PREDICTION/data/processed/backtest_results.csv" 
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Could not find {file_path}. Please run src/backtest.py first.")
        return None

df = load_backtest_data()

if df is not None:
    
    # SIDEBAR: RISK & CAPITAL CONTROLS
    
    st.sidebar.header("Portfolio Configuration")
    initial_capital = st.sidebar.number_input(
        label="Initial Capital ($)",
        min_value=1000,
        max_value=100000000,
        value=100000,
        step=10000,
        format="%d"
    )


    # 3. METRICS CALCULATION

    trading_days = len(df)
    
    # Final Equity Multipliers
    strat_final = df['Strategy_Equity'].iloc[-1]
    bench_final = df['Benchmark_Equity'].iloc[-1]
    
    # Total Return Percentages
    strat_ret = strat_final - 1
    bench_ret = bench_final - 1
    
    # Absolute Dollar PnL Calculations
    strat_pnl = initial_capital * strat_ret
    bench_pnl = initial_capital * bench_ret
    pnl_alpha = strat_pnl - bench_pnl
    
    # CAGR
    strat_cagr = (strat_final) ** (252 / trading_days) - 1
    bench_cagr = (bench_final) ** (252 / trading_days) - 1
    
    # Volatility
    strat_vol = df['Strategy_Return'].std() * np.sqrt(252)
    bench_vol = df['Benchmark_Return'].std() * np.sqrt(252)
    
    # Sharpe Ratio
    strat_sharpe = (df['Strategy_Return'].mean() * 252) / strat_vol
    bench_sharpe = (df['Benchmark_Return'].mean() * 252) / bench_vol
    
    # Sortino Ratio
    strat_neg_ret = df[df['Strategy_Return'] < 0]['Strategy_Return']
    bench_neg_ret = df[df['Benchmark_Return'] < 0]['Benchmark_Return']
    strat_sortino = (df['Strategy_Return'].mean() * 252) / (strat_neg_ret.std() * np.sqrt(252))
    bench_sortino = (df['Benchmark_Return'].mean() * 252) / (bench_neg_ret.std() * np.sqrt(252))

    # Add dollar value paths for plotting
    df['Strategy_Dollar_Value'] = df['Strategy_Equity'] * initial_capital
    df['Benchmark_Dollar_Value'] = df['Benchmark_Equity'] * initial_capital

    
    # 4. DASHBOARD UI - HEADER & KPIs

    st.title("Systematic Macro-Momentum Strategy")
    st.markdown("### Institutional Risk Parity Backtest Results (2024 Out-of-Sample)")
    st.markdown("---")
    
    # Row 1: Dollar PnL Metrics
    st.markdown("#### Absolute Financial Performance")
    pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
    with pnl_col1:
        st.metric(
            label="Strategy Net PnL", 
            value=f"${strat_pnl:,.2f}", 
            delta=f"{strat_ret:.2%} Total Return"
        )
    with pnl_col2:
        st.metric(
            label="Benchmark Net PnL", 
            value=f"${bench_pnl:,.2f}", 
            delta=f"{bench_ret:.2%} Total Return",
            delta_color="inverse"
        )
    with pnl_col3:
        st.metric(
            label="Net Dollar Alpha Generated", 
            value=f"${pnl_alpha:,.2f}", 
            delta=f"+{(strat_ret - bench_ret):.2%} Outperformance"
        )

    st.markdown("---")
    
    # Row 2: Risk-Adjusted Ratios
    st.markdown("#### Risk-Adjusted Efficiency Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Strategy CAGR", value=f"{strat_cagr:.2%}", delta=f"{(strat_cagr - bench_cagr):.2%} vs Bench")
    with col2:
        st.metric(label="Annualized Volatility", value=f"{strat_vol:.2%}", delta=f"{(strat_vol - bench_vol):.2%} vs Bench", delta_color="inverse")
    with col3:
        st.metric(label="Sharpe Ratio", value=f"{strat_sharpe:.2f}", delta=f"{(strat_sharpe - bench_sharpe):.2f}")
    with col4:
        st.metric(label="Sortino Ratio", value=f"{strat_sortino:.2f}", delta=f"{(strat_sortino - bench_sortino):.2f}")

    st.markdown("---")

    # ==========================================
    # 5. INTERACTIVE PLOTLY CHART
    # ==========================================
    st.markdown("#### Portfolio Growth Trajectory ($)")
    
    fig = go.Figure()
    
    # Add Strategy Line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Strategy_Dollar_Value'],
        mode='lines',
        name='AI Strategy (Risk Parity)',
        line=dict(color='#00FF00', width=3)
    ))
    
    # Add Benchmark Line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Benchmark_Dollar_Value'],
        mode='lines',
        name='Benchmark (Equal Weight)',
        line=dict(color='#888888', width=2, dash='dash')
    ))
    
    # Formatting
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600
    )
    
    st.plotly_chart(fig, width="stretch")

    # ==========================================
    # 6. RAW DATA EXPLORER
    # ==========================================
    with st.expander("View Raw Daily Returns & Valuation Data"):
        display_df = df[['Date', 'Strategy_Return', 'Benchmark_Return', 'Strategy_Dollar_Value', 'Benchmark_Dollar_Value']].copy()
        
        # Format columns for professional look
        display_df['Strategy_Return'] = (display_df['Strategy_Return'] * 100).round(2).astype(str) + '%'
        display_df['Benchmark_Return'] = (display_df['Benchmark_Return'] * 100).round(2).astype(str) + '%'
        display_df['Strategy_Dollar_Value'] = display_df['Strategy_Dollar_Value'].map('${:,.2f}'.format)
        display_df['Benchmark_Dollar_Value'] = display_df['Benchmark_Dollar_Value'].map('${:,.2f}'.format)
        
        st.dataframe(display_df.set_index('Date'))