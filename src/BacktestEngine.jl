"""
Minimal Backtesting Library
============================
Users implement: Strategy, Model, signal()
Library handles: data loading, splitting, backtest loop (no look-ahead)

Dataset always holds Dict{Symbol, DataFrame} - single asset is just one entry.
"""

using DataFrames
using UUIDs
using Dates
using Statistics

# =============================================================================
# TYPES
# =============================================================================

struct Untagged end
struct Train end
struct Validation end
struct Test end

"""
Dataset holds OHLCV data as Dict{Symbol, DataFrame}.
Single-asset is just a Dict with one key.
All DataFrames aligned on same timestamps.
"""
struct Dataset{T}
    data::Dict{Symbol, DataFrame}
    timestamps::Vector{DateTime}
    uid::UUID
end

# =============================================================================
# DATASET ACCESSORS
# =============================================================================

assets(ds::Dataset) = collect(keys(ds.data))
n_assets(ds::Dataset) = length(ds.data)
nrow(ds::Dataset) = length(ds.timestamps)

"""Get full OHLCV DataFrame for an asset."""
function getdf(ds::Dataset, asset::Symbol)::DataFrame
    @assert haskey(ds.data, asset) "Unknown asset: $asset"
    ds.data[asset]
end

"""Get timestamps."""
timestamps(ds::Dataset) = ds.timestamps

"""Get close prices for an asset."""
prices(ds::Dataset, asset::Symbol)::Vector{Float64} = getdf(ds, asset).close

"""Get close price at row i for an asset."""
price(ds::Dataset, i::Int, asset::Symbol)::Float64 = getdf(ds, asset).close[i]

"""Get all close prices at row i as Dict."""
prices_at(ds::Dataset, i::Int)::Dict{Symbol, Float64} = 
    Dict(asset => df.close[i] for (asset, df) in ds.data)

"""Get OHLCV row at index i for an asset."""
function ohlcv_at(ds::Dataset, i::Int, asset::Symbol)
    df = getdf(ds, asset)
    (
        timestamp = df.timestamp[i],
        open = :open in propertynames(df) ? df.open[i] : missing,
        high = :high in propertynames(df) ? df.high[i] : missing,
        low = :low in propertynames(df) ? df.low[i] : missing,
        close = df.close[i],
        volume = :volume in propertynames(df) ? df.volume[i] : missing
    )
end

"""Get log returns for an asset."""
returns(ds::Dataset, asset::Symbol)::Vector{Float64} = diff(log.(prices(ds, asset)))

"""Get returns for all assets as Dict."""
returns(ds::Dataset)::Dict{Symbol, Vector{Float64}} = 
    Dict(asset => returns(ds, asset) for asset in assets(ds))

# =============================================================================
# METRICS
# =============================================================================

struct Metrics
    pnl::Float64
    sharpe::Float64
    max_drawdown::Float64
    trades::Vector{NamedTuple}
    asset_pnl::Dict{Symbol, Float64}
end
Metrics() = Metrics(0.0, 0.0, 0.0, NamedTuple[], Dict{Symbol,Float64}())

# =============================================================================
# MODEL & STRATEGY
# =============================================================================

struct Model{T}
    ds_train::Dataset{Train}
    ds_val::Dataset{Validation}
    params::T
    
    function Model(ds_train::Dataset{Train}, ds_val::Dataset{Validation}, params::T) where T
        @assert ds_train.uid == ds_val.uid "Train/Val UID mismatch"
        new{T}(ds_train, ds_val, params)
    end
end

struct Strategy{M, T}
    ds_train::Dataset{Train}
    ds_val::Dataset{Validation}
    model::Model{M}
    params::T
    
    function Strategy(model::Model{M}, params::T) where {M, T}
        new{M, T}(model.ds_train, model.ds_val, model, params)
    end
end

Strategy(model::Model{M}) where M = Strategy(model, nothing)

# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================

"""Create Dataset from Dict of DataFrames.
Each DataFrame must have :timestamp and :close (other OHLCV optional).
Auto-aligns on common timestamps.

Example:
    data = Dict(
        :AAPL => aapl_df,
        :SPY  => spy_df,
    )
    ds = Dataset(data)
    
Single-asset:
    ds = Dataset(:SPY => spy_df)
"""
function Dataset(data::Dict{Symbol, DataFrame})::Dataset{Untagged}
    @assert length(data) > 0 "Must provide at least one asset"
    
    # Validate and sort each DataFrame
    for (asset, df) in data
        @assert :timestamp in propertynames(df) "$asset: missing :timestamp"
        @assert :close in propertynames(df) "$asset: missing :close"
        sort!(df, :timestamp)
    end
    
    # Get common timestamps
    all_ts = [Set(df.timestamp) for df in values(data)]
    common_ts = sort(collect(intersect(all_ts...)))
    
    @assert length(common_ts) > 0 "No overlapping timestamps"
    
    # Filter to common timestamps
    aligned = Dict{Symbol, DataFrame}()
    for (asset, df) in data
        mask = [t in Set(common_ts) for t in df.timestamp]
        aligned[asset] = df[mask, :]
    end
    
    Dataset{Untagged}(aligned, common_ts, uuid4())
end

# Convenience: single Pair
Dataset(p::Pair{Symbol, DataFrame}) = Dataset(Dict(p))

# Convenience: symbol and df
Dataset(asset::Symbol, df::DataFrame) = Dataset(Dict(asset => df))

"""Split dataset into train/val/test."""
function split(ds::Dataset{Untagged}, cutoff1::DateTime, cutoff2::DateTime)::Tuple{Dataset{Train}, Dataset{Validation}, Dataset{Test}}
    @assert cutoff1 < cutoff2 "cutoff1 must be before cutoff2"
    
    ts = ds.timestamps
    train_mask = ts .< cutoff1
    val_mask = (ts .>= cutoff1) .& (ts .< cutoff2)
    test_mask = ts .>= cutoff2
    
    train_data = Dict(a => df[train_mask, :] for (a, df) in ds.data)
    val_data = Dict(a => df[val_mask, :] for (a, df) in ds.data)
    test_data = Dict(a => df[test_mask, :] for (a, df) in ds.data)
    
    (
        Dataset{Train}(train_data, ts[train_mask], ds.uid),
        Dataset{Validation}(val_data, ts[val_mask], ds.uid),
        Dataset{Test}(test_data, ts[test_mask], ds.uid)
    )
end

"""Subset dataset to rows 1:i."""
function subset(ds::Dataset{T}, i::Int)::Dataset{T} where T
    sub_data = Dict(a => df[1:i, :] for (a, df) in ds.data)
    Dataset{T}(sub_data, ds.timestamps[1:i], ds.uid)
end

# =============================================================================
# BACKTEST
# =============================================================================

"""
Main backtest loop. Iterates through test data, passing only data up to current row.

signal() should return Dict{Symbol, Float64} of weights (even for single asset).
"""
function backtest(ds::Dataset{Test}, strategy::Strategy)::Metrics
    @assert ds.uid == strategy.ds_train.uid "Test/Train UID mismatch"
    
    signals = Vector{Dict{Symbol, Float64}}()
    n = nrow(ds)
    
    for i in 1:n
        ds_subset = subset(ds, i)
        sig = signal(ds_subset, strategy)
        push!(signals, sig)
    end
    
    compute_metrics(ds, signals)
end

# =============================================================================
# PARALLEL BACKTEST
# =============================================================================

struct BacktestJob
    name::String
    ds::Dataset{Test}
    strategy::Strategy
end

struct BacktestResult
    name::String
    metrics::Metrics
    elapsed::Float64
end

function backtest_parallel(jobs::Vector{BacktestJob})::Vector{BacktestResult}
    n = length(jobs)
    results = Vector{BacktestResult}(undef, n)
    
    Threads.@threads for i in 1:n
        job = jobs[i]
        t0 = time()
        metrics = backtest(job.ds, job.strategy)
        elapsed = time() - t0
        results[i] = BacktestResult(job.name, metrics, elapsed)
    end
    
    results
end

function backtest_distributed(jobs::Vector{BacktestJob})::Vector{BacktestResult}
    @eval using Distributed
    pmap(jobs) do job
        t0 = time()
        metrics = backtest(job.ds, job.strategy)
        BacktestResult(job.name, metrics, time() - t0)
    end
end

function backtest_sweep(
    ds_test::Dataset{Test},
    ds_train::Dataset{Train},
    ds_val::Dataset{Validation},
    param_grid::Vector,
    make_strategy::Function
)::Vector{BacktestResult}
    jobs = [BacktestJob("run_$i:$p", ds_test, make_strategy(p, ds_train, ds_val)) 
            for (i, p) in enumerate(param_grid)]
    backtest_parallel(jobs)
end

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

function compute_metrics(ds::Dataset, signals::Vector{Dict{Symbol, Float64}})::Metrics
    n = length(signals)
    n < 2 && return Metrics()
    
    returns_vec = Float64[]
    asset_returns = Dict(a => Float64[] for a in assets(ds))
    
    for i in 2:n
        weights = signals[i-1]
        total_ret = 0.0
        
        for asset in assets(ds)
            df = getdf(ds, asset)
            ret = (df.close[i] - df.close[i-1]) / df.close[i-1]
            w = get(weights, asset, 0.0)
            weighted_ret = w * ret
            total_ret += weighted_ret
            push!(asset_returns[asset], weighted_ret)
        end
        
        push!(returns_vec, total_ret)
    end
    
    # PnL
    cum_ret = cumprod(1 .+ returns_vec)
    pnl = cum_ret[end] - 1
    
    # Sharpe
    sharpe = length(returns_vec) > 1 && std(returns_vec) > 0 ? 
        mean(returns_vec) / std(returns_vec) * sqrt(252) : 0.0
    
    # Max drawdown
    peak, max_dd = 1.0, 0.0
    for r in cum_ret
        peak = max(peak, r)
        max_dd = max(max_dd, (peak - r) / peak)
    end
    
    # Per-asset PnL
    asset_pnl = Dict(a => prod(1 .+ r) - 1 for (a, r) in asset_returns if !isempty(r))
    
    Metrics(pnl, sharpe, max_dd, NamedTuple[], asset_pnl)
end

# =============================================================================
# DISPLAY
# =============================================================================

function Base.show(io::IO, m::Metrics)
    println(io, "Metrics:")
    println(io, "  PnL:          $(round(m.pnl * 100, digits=2))%")
    println(io, "  Sharpe:       $(round(m.sharpe, digits=2))")
    println(io, "  Max Drawdown: $(round(m.max_drawdown * 100, digits=2))%")
    if !isempty(m.asset_pnl)
        println(io, "  Per-asset PnL:")
        for (a, p) in sort(collect(m.asset_pnl), by=x->x[2], rev=true)
            println(io, "    $a: $(round(p * 100, digits=2))%")
        end
    end
end

function Base.show(io::IO, r::BacktestResult)
    println(io, "$(r.name) ($(round(r.elapsed, digits=2))s): PnL=$(round(r.metrics.pnl*100,digits=2))% Sharpe=$(round(r.metrics.sharpe,digits=2))")
end

function Base.show(io::IO, results::Vector{BacktestResult})
    println(io, "="^50)
    println(io, "BACKTEST RESULTS ($(length(results)) runs)")
    println(io, "="^50)
    for (i, r) in enumerate(sort(results, by=x->x.metrics.sharpe, rev=true))
        println(io, "#$i: $(r.name)")
        println(io, "    PnL=$(round(r.metrics.pnl*100,digits=2))%  Sharpe=$(round(r.metrics.sharpe,digits=2))  MaxDD=$(round(r.metrics.max_drawdown*100,digits=2))%")
    end
end
