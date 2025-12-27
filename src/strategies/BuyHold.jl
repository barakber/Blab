"""
Buy & Hold SPY Strategy
=======================
Passive investment strategy: buy SPY once at the beginning and hold.
No trading, no rebalancing - just stay 100% invested in SPY.

This serves as a baseline benchmark for comparing active strategies.
Represents pure passive S&P 500 index investing.
"""

module BuyHoldStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct BuyHoldParams
    allocation::Float64  # Portfolio allocation (1.0 = 100% invested)
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{BuyHoldParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{BuyHoldParams}
    @assert ds_train.uid == ds_val.uid

    # No training needed for buy & hold - just set allocation
    # Default: 100% invested (full buy & hold)
    Model(ds_train, ds_val, BuyHoldParams(1.0))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{BuyHoldParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Only allocate to SPY - if SPY is not available, return 0% for all assets
    if :SPY in assets(ds)
        # 100% allocation to SPY, 0% to everything else
        Dict(asset => (asset == :SPY ? params.allocation : 0.0) for asset in assets(ds))
    else
        # SPY not available - return 0% allocation for all assets
        Dict(asset => 0.0 for asset in assets(ds))
    end
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Buy & Hold Strategy Demo")
    println("="^60)

    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Simulate market with trend + volatility
    close_prices = Float64[100.0]
    for i in 2:n
        ret = 0.0005 + 0.015 * randn()  # Positive drift + noise
        push!(close_prices, close_prices[end] * (1 + ret))
    end

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(BuyHoldParams, train, val)
    s = Strategy(m)

    println("\nBacktesting Buy & Hold strategy...")
    println(backtest(test, s))
end

# demo()

export BuyHoldParams, train__, signal, demo

end # module
