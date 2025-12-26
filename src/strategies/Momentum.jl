"""
Example: Portfolio Momentum Rotation Strategy
=============================================
Multi-asset momentum rotation using unified Dataset API.
"""

module MomentumStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, backtest_sweep, assets, nrow, prices, split, getdf
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct MomentumParams
    lookback::Int
    top_n::Int
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{MomentumParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{MomentumParams}
    @assert ds_train.uid == ds_val.uid
    println("Assets: $(assets(ds_train))")
    Model(ds_train, ds_val, MomentumParams(20, 2))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{MomentumParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    n = nrow(ds)
    asset_list = assets(ds)
    
    if n < params.lookback + 1
        w = 1.0 / length(asset_list)
        return Dict(a => w for a in asset_list)
    end
    
    # Calculate momentum for each asset
    momentum = Dict{Symbol, Float64}()
    for asset in asset_list
        p = prices(ds, asset)
        momentum[asset] = (p[end] - p[end-params.lookback]) / p[end-params.lookback]
    end
    
    # Select top N by momentum
    sorted = sort(collect(momentum), by=x->x[2], rev=true)
    top = Set(x[1] for x in sorted[1:min(params.top_n, length(sorted))])
    
    w = 1.0 / params.top_n
    Dict(a => (a in top ? w : 0.0) for a in asset_list)
end

# =============================================================================
# HELPERS
# =============================================================================

function generate_ohlcv(dates, drift, vol)
    n = length(dates)
    closes = Float64[100.0]
    for _ in 2:n
        push!(closes, closes[end] * (1 + drift + vol*randn()))
    end
    opens = [closes[1]; closes[1:end-1]]
    highs = max.(opens, closes) .* (1 .+ 0.005*abs.(randn(n)))
    lows = min.(opens, closes) .* (1 .- 0.005*abs.(randn(n)))
    DataFrame(timestamp=dates, open=opens, high=highs, low=lows, close=closes, volume=rand(1_000_000:10_000_000, n))
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    
    data = Dict(
        :AAPL => generate_ohlcv(dates, 0.0004, 0.018),
        :GOOG => generate_ohlcv(dates, 0.0003, 0.020),
        :MSFT => generate_ohlcv(dates, 0.0005, 0.017),
        :AMZN => generate_ohlcv(dates, 0.0006, 0.022),
        :META => generate_ohlcv(dates, 0.0002, 0.025),
    )
    
    ds = Dataset(data)
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))
    
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")
    
    # Show sample OHLCV
    println("\nSample AAPL OHLCV:")
    println(first(getdf(ds, :AAPL), 3))

    m = train__(MomentumParams, train, val)
    s = Strategy(m)
    println(backtest(test, s))
end

function demo_parallel()
    Random.seed!(42)
    println("Running on $(Threads.nthreads()) threads\n")
    
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    
    data = Dict(
        :AAPL => generate_ohlcv(dates, 0.0004, 0.018),
        :GOOG => generate_ohlcv(dates, 0.0003, 0.020),
        :MSFT => generate_ohlcv(dates, 0.0005, 0.017),
        :AMZN => generate_ohlcv(dates, 0.0006, 0.022),
        :META => generate_ohlcv(dates, 0.0002, 0.025),
        :NVDA => generate_ohlcv(dates, 0.0007, 0.028),
    )
    
    ds = Dataset(data)
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))
    
    param_grid = [
        (lookback=10, top_n=2), (lookback=20, top_n=2), (lookback=40, top_n=2),
        (lookback=10, top_n=3), (lookback=20, top_n=3), (lookback=40, top_n=3),
    ]
    
    make_strategy(p, tr, va) = Strategy(Model(tr, va, MomentumParams(p.lookback, p.top_n)))
    
    results = backtest_sweep(test, train, val, param_grid, make_strategy)
    println(results)
end

# demo()
# demo_parallel()

export MomentumParams, train__, signal, demo, demo_parallel, generate_ohlcv

end # module
