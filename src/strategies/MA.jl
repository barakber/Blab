"""
Example: Moving Average Crossover Strategy
==========================================
Single-asset strategy using unified Dataset API.
"""

module MAStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, backtest_sweep, assets, nrow, prices, split
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct MAParams
    fast_period::Int
    slow_period::Int
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{MAParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{MAParams}
    @assert ds_train.uid == ds_val.uid
    Model(ds_train, ds_val, MAParams(10, 30))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{MAParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    n = nrow(ds)
    asset = first(assets(ds))
    
    if n < params.slow_period
        return Dict(asset => 0.0)
    end
    
    p = prices(ds, asset)
    fast_ma = mean(p[end-params.fast_period+1:end])
    slow_ma = mean(p[end-params.slow_period+1:end])
    
    weight = fast_ma > slow_ma ? 1.0 : 0.0
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
    ds = Dataset(:SPY => df)  # Single asset
    
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))
    
    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(MAParams, train, val)
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
    
    param_grid = [(fast=5, slow=20), (fast=10, slow=30), (fast=10, slow=50), (fast=20, slow=100)]
    make_strategy(p, tr, va) = begin
        m = Model(tr, va, MAParams(p.fast, p.slow))
        Strategy(m)
    end
    
    results = backtest_sweep(test, train, val, param_grid, make_strategy)
    println(results)
end

# demo()
# demo_parallel()

export MAParams, train__, signal, demo, demo_parallel

end # module
