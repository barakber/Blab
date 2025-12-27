"""
Markowitz Mean-Variance Portfolio Optimization
===============================================
Classic Modern Portfolio Theory approach that optimizes portfolio weights
to maximize Sharpe ratio given expected returns and covariance matrix.

Key features:
1. Estimates expected returns from training data
2. Computes covariance matrix
3. Optimizes weights to maximize Sharpe ratio
4. Supports constraints: long-only, max allocation per asset
5. Rebalances periodically based on updated estimates

Harry Markowitz won the Nobel Prize for this framework in 1990.
"""

module MarkowitzStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns
import ..Blab: signal, train__
using Statistics
using LinearAlgebra
using Random
using Dates
using DataFrames

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct MarkowitzParams
    expected_returns::Dict{Symbol, Float64}  # Estimated from training data
    cov_matrix::Matrix{Float64}              # Covariance matrix
    asset_list::Vector{Symbol}               # Ordered list of assets
    lookback::Int                            # Days to estimate returns/cov
    max_weight::Float64                      # Max allocation per asset
    min_weight::Float64                      # Min allocation per asset (for diversification)
    rebalance_period::Int                    # Days between rebalancing
    last_rebalance::Ref{Int}                 # Track last rebalance day
    current_weights::Ref{Dict{Symbol, Float64}}  # Current portfolio weights
end

# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

"""
Compute expected returns (annualized) from historical data.
"""
function compute_expected_returns(ds::Dataset, assets_list::Vector{Symbol}, lookback::Int)::Dict{Symbol, Float64}
    expected = Dict{Symbol, Float64}()

    for asset in assets_list
        rets = returns(ds, asset)
        # Use last 'lookback' days
        window = rets[max(1, end-lookback+1):end]
        # Annualize (assume 252 trading days)
        expected[asset] = mean(window) * 252
    end

    return expected
end

"""
Compute covariance matrix (annualized) from historical returns.
"""
function compute_covariance_matrix(ds::Dataset, assets_list::Vector{Symbol}, lookback::Int)::Matrix{Float64}
    n_assets = length(assets_list)

    # Collect returns for all assets
    returns_matrix = zeros(lookback, n_assets)

    for (i, asset) in enumerate(assets_list)
        rets = returns(ds, asset)
        window = rets[max(1, end-lookback+1):end]
        # Pad if needed
        if length(window) < lookback
            window = vcat(zeros(lookback - length(window)), window)
        end
        returns_matrix[:, i] = window
    end

    # Compute covariance and annualize (252 trading days)
    cov_mat = cov(returns_matrix) * 252

    return cov_mat
end

"""
Optimize portfolio weights to maximize Sharpe ratio.
Uses analytical solution for minimum variance portfolio with constraints.
"""
function optimize_weights(
    expected_returns::Dict{Symbol, Float64},
    cov_matrix::Matrix{Float64},
    assets_list::Vector{Symbol};
    max_weight::Float64=0.30,
    min_weight::Float64=0.00,
    risk_free_rate::Float64=0.02
)::Dict{Symbol, Float64}

    n_assets = length(assets_list)

    # Convert expected returns to vector (in same order as assets_list)
    mu = [expected_returns[a] for a in assets_list]

    # Simple analytical solution for max Sharpe ratio (unconstrained)
    # w = Σ^(-1) * (μ - rf) / 1'Σ^(-1)(μ - rf)

    try
        # Regularize covariance matrix to avoid singularity
        cov_reg = cov_matrix + 1e-5 * I

        # Inverse covariance
        inv_cov = inv(cov_reg)

        # Excess returns
        excess = mu .- risk_free_rate

        # Optimal weights (unconstrained)
        w_unconstrained = inv_cov * excess
        w_unconstrained = w_unconstrained / sum(w_unconstrained)

        # Apply constraints
        w = copy(w_unconstrained)

        # Clip to [min_weight, max_weight]
        w = clamp.(w, min_weight, max_weight)

        # Renormalize to sum to 1
        w = w / sum(w)

        # Handle any negative weights (force to 0 for long-only)
        w = max.(w, 0.0)
        w = w / sum(w)

        # Convert back to dictionary
        weights = Dict{Symbol, Float64}()
        for (i, asset) in enumerate(assets_list)
            weights[asset] = w[i]
        end

        return weights

    catch e
        # If optimization fails, use equal weights
        println("  Warning: Optimization failed ($e), using equal weights")
        return Dict(a => 1.0 / n_assets for a in assets_list)
    end
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{MarkowitzParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{MarkowitzParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Markowitz Mean-Variance Portfolio Optimizer...")

    assets_list = collect(assets(ds_train))
    n_assets = length(assets_list)

    println("  Optimizing portfolio with $(n_assets) assets")

    # Use 60-day lookback for estimation
    lookback = min(60, nrow(ds_train))

    # Estimate expected returns from training data
    println("  Estimating expected returns ($(lookback)-day lookback)...")
    expected_rets = compute_expected_returns(ds_train, assets_list, lookback)

    println("  Expected annual returns:")
    for (asset, ret) in sort(collect(expected_rets), by=x->x[2], rev=true)
        println("    $(asset): $(round(100*ret, digits=1))%")
    end

    # Compute covariance matrix
    println("  Computing covariance matrix...")
    cov_mat = compute_covariance_matrix(ds_train, assets_list, lookback)

    # Optimize weights
    println("  Optimizing portfolio weights (max Sharpe ratio)...")
    max_weight = 0.30  # Max 30% per asset for diversification
    min_weight = 0.00  # Long-only

    optimal_weights = optimize_weights(expected_rets, cov_mat, assets_list;
                                       max_weight=max_weight, min_weight=min_weight)

    # Display top holdings
    println("  Optimal portfolio weights:")
    sorted_weights = sort(collect(optimal_weights), by=x->x[2], rev=true)
    for (i, (asset, weight)) in enumerate(sorted_weights)
        if weight > 0.01  # Only show weights > 1%
            println("    $(asset): $(round(100*weight, digits=1))%")
        end
        if i >= 10  # Show top 10
            break
        end
    end

    # Estimate portfolio statistics on validation set
    println("  Validating portfolio...")
    val_expected = compute_expected_returns(ds_val, assets_list, min(60, nrow(ds_val)))
    portfolio_return = sum(optimal_weights[a] * val_expected[a] for a in assets_list)
    println("  Expected portfolio return: $(round(100*portfolio_return, digits=1))%")

    # Initialize with quarterly rebalancing
    rebalance_period = 63  # Quarterly (approximately 3 months)
    initial_weights = optimal_weights

    params = MarkowitzParams(
        expected_rets, cov_mat, assets_list, lookback,
        max_weight, min_weight, rebalance_period,
        Ref(0), Ref(initial_weights)
    )

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{MarkowitzParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Need enough data to compute statistics
    if nrow(ds) < params.lookback
        # Return equal weights if not enough data
        n = length(params.asset_list)
        return Dict(a => 1.0/n for a in assets(ds))
    end

    current_day = nrow(ds)

    # Check if it's time to rebalance (quarterly = every 63 trading days)
    if current_day - params.last_rebalance[] >= params.rebalance_period
        # Recompute expected returns and covariance from recent data
        expected_rets = compute_expected_returns(ds, params.asset_list, params.lookback)
        cov_mat = compute_covariance_matrix(ds, params.asset_list, params.lookback)

        # Optimize weights
        new_weights = optimize_weights(expected_rets, cov_mat, params.asset_list;
                                      max_weight=params.max_weight, min_weight=params.min_weight)

        # Update stored weights and rebalance day
        params.current_weights[] = new_weights
        params.last_rebalance[] = current_day
    end

    # Return current weights (either newly optimized or previous weights)
    return params.current_weights[]
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Markowitz Mean-Variance Optimization Demo")
    println("="^60)

    # Create synthetic data for 3 assets with different risk/return profiles
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Asset 1: High return, high volatility
    prices1 = Float64[100.0]
    for i in 2:n
        ret = 0.001 + 0.025 * randn()
        push!(prices1, prices1[end] * (1 + ret))
    end

    # Asset 2: Moderate return, moderate volatility
    prices2 = Float64[100.0]
    for i in 2:n
        ret = 0.0007 + 0.015 * randn()
        push!(prices2, prices2[end] * (1 + ret))
    end

    # Asset 3: Low return, low volatility
    prices3 = Float64[100.0]
    for i in 2:n
        ret = 0.0003 + 0.008 * randn()
        push!(prices3, prices3[end] * (1 + ret))
    end

    df1 = DataFrame(timestamp=dates, close=prices1)
    df2 = DataFrame(timestamp=dates, close=prices2)
    df3 = DataFrame(timestamp=dates, close=prices3)

    ds = Dataset(Dict(:TECH => df1, :CONSUMER => df2, :BOND => df3))

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(MarkowitzParams, train, val)
    s = Strategy(m)

    println("\nBacktesting Markowitz portfolio...")
    println(backtest(test, s))
end

# demo()

export MarkowitzParams, train__, signal, demo

end # module
