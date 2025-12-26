"""
EMA Crossover Strategy
======================
Uses Exponential Moving Averages from Indicators.jl for trend following.
Signal: Long when fast EMA > slow EMA, flat otherwise.
"""

module EMAStrategy

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

struct EMAParams
    fast_period::Int
    slow_period::Int
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{EMAParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{EMAParams}
    @assert ds_train.uid == ds_val.uid
    Model(ds_train, ds_val, EMAParams(12, 26))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{EMAParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    n = nrow(ds)
    asset = first(assets(ds))

    # Need enough data for EMA calculation (at least 2x the slow period for stability)
    min_periods = max(params.slow_period * 2, 50)
    if n < min_periods
        return Dict(asset => 0.0)
    end

    p = prices(ds, asset)

    # Calculate EMAs using Indicators.jl
    fast_ema = ema(p; n=params.fast_period)
    slow_ema = ema(p; n=params.slow_period)

    # Long when fast > slow
    weight = fast_ema[end] > slow_ema[end] ? 1.0 : 0.0

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

    m = train__(EMAParams, train, val)
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

    param_grid = [(fast=8, slow=21), (fast=12, slow=26), (fast=5, slow=20), (fast=20, slow=50)]
    make_strategy(p, tr, va) = begin
        m = Model(tr, va, EMAParams(p.fast, p.slow))
        Strategy(m)
    end

    results = backtest_sweep(test, train, val, param_grid, make_strategy)
    println(results)
end

# demo()
# demo_parallel()

export EMAParams, train__, signal, demo, demo_parallel

end # module
