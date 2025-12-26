"""
RSI Mean Reversion Strategy
============================
Uses RSI indicator from Indicators.jl for mean reversion.
Signal: Long when RSI < oversold threshold, flat when RSI > overbought threshold.
"""

module RSIStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, backtest_sweep, assets, nrow, prices, split
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames
using Indicators

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct RSIParams
    period::Int
    oversold::Float64
    overbought::Float64
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{RSIParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{RSIParams}
    @assert ds_train.uid == ds_val.uid
    # Standard RSI: 14 period, 30/70 thresholds
    Model(ds_train, ds_val, RSIParams(14, 30.0, 70.0))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{RSIParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    n = nrow(ds)
    asset = first(assets(ds))

    # Need enough data for RSI (at least 2x the period for stability)
    min_periods = max(params.period * 2, 30)
    if n < min_periods
        return Dict(asset => 0.0)
    end

    p = prices(ds, asset)

    # Calculate RSI using Indicators.jl
    rsi_vals = rsi(p; n=params.period)

    # Get current RSI
    current_rsi = rsi_vals[end]

    # Mean reversion: long when oversold, flat when overbought
    weight = if current_rsi < params.oversold
        1.0  # Buy oversold
    elseif current_rsi > params.overbought
        0.0  # Sell overbought
    else
        # Hold previous position - for simplicity, use 0.5 as neutral
        0.0
    end

    Dict(asset => weight)
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Generate mean-reverting price series
    close_prices = Float64[100.0]
    for _ in 2:n
        # Mean reversion to 100
        drift = 0.1 * (100.0 - close_prices[end])
        push!(close_prices, close_prices[end] + drift + 2.0 * randn())
    end
    close_prices = max.(close_prices, 1.0)

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(RSIParams, train, val)
    s = Strategy(m)

    println(backtest(test, s))
end

function demo_parallel()
    Random.seed!(42)
    println("Running on $(Threads.nthreads()) threads\n")

    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    close_prices = Float64[100.0]
    for _ in 2:n
        drift = 0.1 * (100.0 - close_prices[end])
        push!(close_prices, close_prices[end] + drift + 2.0 * randn())
    end
    close_prices = max.(close_prices, 1.0)

    ds = Dataset(:SPY => DataFrame(timestamp=dates, close=close_prices))
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    # Test different RSI parameters
    param_grid = [
        (period=14, oversold=30.0, overbought=70.0),
        (period=14, oversold=20.0, overbought=80.0),
        (period=21, oversold=30.0, overbought=70.0),
        (period=7, oversold=30.0, overbought=70.0)
    ]

    make_strategy(p, tr, va) = begin
        m = Model(tr, va, RSIParams(p.period, p.oversold, p.overbought))
        Strategy(m)
    end

    results = backtest_sweep(test, train, val, param_grid, make_strategy)
    println(results)
end

# demo()
# demo_parallel()

export RSIParams, train__, signal, demo, demo_parallel

end # module
