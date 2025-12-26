"""
MACD Crossover Strategy
========================
Uses MACD indicator from Indicators.jl for trend following.
Signal: Long when MACD crosses above signal line, flat otherwise.
"""

module MACDStrategy

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

struct MACDParams
    fast::Int
    slow::Int
    signal_period::Int
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{MACDParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{MACDParams}
    @assert ds_train.uid == ds_val.uid
    # MACD standard params: 12, 26, 9
    Model(ds_train, ds_val, MACDParams(12, 26, 9))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{MACDParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    n = nrow(ds)
    asset = first(assets(ds))

    # Need enough data for MACD (at least 2x the slow period + signal period for stability)
    min_periods = max((params.slow + params.signal_period) * 2, 60)
    if n < min_periods
        return Dict(asset => 0.0)
    end

    p = prices(ds, asset)

    # Calculate MACD using Indicators.jl (returns matrix with columns: [macd, signal, histogram])
    macd_result = macd(p; nfast=params.fast, nslow=params.slow, nsig=params.signal_period)

    # Get current values (last row)
    macd_line = macd_result[end, 1]     # MACD line
    signal_line = macd_result[end, 2]   # Signal line

    # Long when MACD > signal, flat otherwise
    weight = macd_line > signal_line ? 1.0 : 0.0

    Dict(asset => weight)
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)
    close_prices = cumsum(randn(n)) .+ 100
    close_prices = max.(close_prices, 1.0)

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(MACDParams, train, val)
    s = Strategy(m)

    println(backtest(test, s))
end

function demo_parallel()
    Random.seed!(42)
    println("Running on $(Threads.nthreads()) threads\n")

    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)
    close_prices = cumsum(randn(n)) .+ 100
    close_prices = max.(close_prices, 1.0)

    ds = Dataset(:SPY => DataFrame(timestamp=dates, close=close_prices))
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    # Test different MACD parameters
    param_grid = [
        (fast=8, slow=17, sig=9),
        (fast=12, slow=26, sig=9),
        (fast=5, slow=35, sig=5)
    ]

    make_strategy(p, tr, va) = begin
        m = Model(tr, va, MACDParams(p.fast, p.slow, p.sig))
        Strategy(m)
    end

    results = backtest_sweep(test, train, val, param_grid, make_strategy)
    println(results)
end

# demo()
# demo_parallel()

export MACDParams, train__, signal, demo, demo_parallel

end # module
