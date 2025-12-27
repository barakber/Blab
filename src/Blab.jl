"""
# Blab

A minimal backtesting library for trading strategies.

Users implement: Strategy, Model, signal()
Library handles: data loading, splitting, backtest loop (no look-ahead)
"""
module Blab

using DataFrames
using UUIDs
using Dates
using Statistics

# Include the backtesting engine
include("BacktestEngine.jl")

# Export main types and functions
export Dataset, Model, Strategy, Metrics, BacktestResult, BacktestJob
export Train, Validation, Test, Untagged

# Export dataset functions
export assets, n_assets, getdf, timestamps, prices, price, prices_at
export ohlcv_at, returns, split, subset, nrow

# Export backtesting functions
export backtest, backtest_parallel, backtest_distributed, backtest_sweep

# Define generic functions that strategies will implement
function signal end
function train__ end

# Include built-in strategy modules
include("strategies/BuyHold.jl")
include("strategies/MA.jl")
include("strategies/Momentum.jl")
include("strategies/HMM.jl")
include("strategies/MACD.jl")
include("strategies/RSI.jl")
include("strategies/EMA.jl")
include("strategies/XGBoostML.jl")
include("strategies/RegimeSwitch.jl")
include("strategies/GeneticPortfolio.jl")
include("strategies/TDAStrategy.jl")
include("strategies/LeverageQQQ.jl")
include("strategies/AdaptiveRegime.jl")
include("strategies/GeneticMarkowitz.jl")
include("strategies/GeneticRegime.jl")
include("strategies/Markowitz.jl")
include("strategies/InstitutionalGrade.jl")

# Re-export strategy modules so users can access them as Blab.MAStrategy, etc.
using .BuyHoldStrategy
using .MAStrategy
using .MomentumStrategy
using .HMMStrategy
using .MACDStrategy
using .RSIStrategy
using .EMAStrategy
using .XGBoostMLStrategy
using .RegimeSwitchStrategy
using .GeneticPortfolioStrategy
using .TDAStrategy
using .LeverageQQQStrategy
using .AdaptiveRegimeStrategy
using .GeneticMarkowitzStrategy
using .GeneticRegimeStrategy
using .MarkowitzStrategy
using .InstitutionalGradeStrategy

# Include utilities
include("DataLoader.jl")
include("CompareStrategies.jl")
include("TimePeriodsAnalysis.jl")

# Use TimePeriodsAnalysis
using .TimePeriodsAnalysis

# Export utility functions
export load_stock, load_stocks, get_top_sp500_symbols, compare_all_strategies
export analyze_time_periods, get_market_periods

end # module
