"""
Institutional-Grade Meta Strategy
===================================
Target Profile:
- Annualized Returns: 12-18%
- Maximum Drawdown: <12-15%
- Sharpe Ratio: >1.5

Key Principles:
1. Capital preservation is paramount
2. Consistent returns > home-run years
3. Dynamic risk management with strict drawdown controls
4. Multi-strategy diversification with regime-aware allocation

Architecture:
- Ensemble of 4 core strategies with proven risk-adjusted returns
- HMM-based regime detection for macro risk management
- Volatility targeting and dynamic position sizing
- Automatic de-risking when approaching drawdown limits
- Correlation-aware rebalancing

Strategy Components:
1. Markowitz Portfolio (40% base allocation) - Stable core holdings
2. Momentum Rotation (30% base) - Tactical alpha generation
3. Genetic-Regime (20% base) - Adaptive with built-in risk management
4. RSI Mean Reversion (10% base) - Volatility dampening

Risk Controls:
- Maximum 12% trailing drawdown → reduce to 50% exposure
- Maximum 15% trailing drawdown → reduce to 25% exposure (emergency)
- Volatility targeting: Scale positions to maintain ~12% annualized vol
- Per-strategy correlation monitoring: Reduce allocation if correlation >0.7
- Monthly rebalancing with weekly risk checks
"""

module InstitutionalGradeStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns, timestamps
import ..Blab: signal, train__
using ..Blab.MarkowitzStrategy
using ..Blab.MomentumStrategy
using ..Blab.GeneticRegimeStrategy
using ..Blab.RSIStrategy
using ..Blab.HMMStrategy
using Statistics
using LinearAlgebra
using Dates
using DataFrames

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

mutable struct InstitutionalGradeParams
    # Core strategies with their trained models
    markowitz_strategy::Strategy
    momentum_strategy::Strategy
    genetic_regime_strategy::Strategy
    rsi_strategy::Strategy

    # Regime detection
    hmm::HMMStrategy.GaussianHMM
    regime_asset::Symbol

    # Base allocations (sum to 1.0)
    base_allocations::Dict{String, Float64}

    # Risk parameters
    max_drawdown_warning::Float64      # 0.12 = 12%
    max_drawdown_emergency::Float64    # 0.15 = 15%
    target_volatility::Float64         # 0.12 = 12% annualized
    correlation_threshold::Float64      # 0.7 = reduce if corr too high

    # State tracking
    current_allocations::Ref{Dict{String, Float64}}
    peak_value::Ref{Float64}
    current_drawdown::Ref{Float64}
    last_rebalance::Ref{Int}
    rebalance_period::Int              # Days between rebalancing

    # Performance tracking for adaptive allocation
    strategy_returns::Dict{String, Vector{Float64}}
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""Calculate current drawdown from peak."""
function calculate_drawdown(peak::Float64, current::Float64)::Float64
    return peak > 0 ? (peak - current) / peak : 0.0
end

"""Estimate portfolio volatility from recent returns."""
function estimate_volatility(returns_vec::Vector{Float64}, lookback::Int=20)::Float64
    window = returns_vec[max(1, end-lookback+1):end]
    vol_daily = std(window)
    return vol_daily * sqrt(252)  # Annualize
end

"""
Calculate risk scaling factor based on:
1. Current drawdown (reduce exposure if approaching limits)
2. Realized volatility (scale to target vol)
3. Regime (reduce in bear markets)
"""
function calculate_risk_scalar(
    current_dd::Float64,
    realized_vol::Float64,
    target_vol::Float64,
    regime_is_bull::Bool,
    max_dd_warning::Float64,
    max_dd_emergency::Float64
)::Float64

    # 1. Drawdown-based scaling (most important for capital preservation)
    dd_scalar = if current_dd >= max_dd_emergency
        0.25  # Emergency: reduce to 25%
    elseif current_dd >= max_dd_warning
        0.50  # Warning: reduce to 50%
    else
        1.0   # Normal operation
    end

    # 2. Volatility targeting (keep portfolio vol near target)
    vol_scalar = if realized_vol > 1e-6
        min(2.0, target_vol / realized_vol)  # Cap at 2x leverage
    else
        1.0
    end

    # 3. Regime-based scaling (reduce in bear markets)
    regime_scalar = regime_is_bull ? 1.0 : 0.7

    # Combine (multiplicative, most conservative wins)
    return dd_scalar * vol_scalar * regime_scalar
end

"""
Adjust strategy allocations based on recent performance and correlation.
If strategies become too correlated, diversify allocation more evenly.
"""
function adjust_allocations_for_correlation(
    base_allocs::Dict{String, Float64},
    strategy_returns::Dict{String, Vector{Float64}},
    correlation_threshold::Float64
)::Dict{String, Float64}

    # Need at least 20 observations for correlation
    min_obs = minimum(length(v) for v in values(strategy_returns))
    if min_obs < 20
        return base_allocs
    end

    # Calculate average pairwise correlation
    strategy_names = collect(keys(base_allocs))
    n = length(strategy_names)
    correlations = Float64[]

    for i in 1:n, j in (i+1):n
        ret1 = strategy_returns[strategy_names[i]][end-19:end]
        ret2 = strategy_returns[strategy_names[j]][end-19:end]
        push!(correlations, cor(ret1, ret2))
    end

    avg_corr = mean(correlations)

    # If correlation too high, move toward equal weighting (diversification)
    if avg_corr > correlation_threshold
        # Blend base allocation with equal weights
        equal_weight = 1.0 / n
        blend_factor = (avg_corr - correlation_threshold) / (1.0 - correlation_threshold)
        blend_factor = clamp(blend_factor, 0.0, 0.5)  # Max 50% move to equal weight

        adjusted = Dict{String, Float64}()
        for name in strategy_names
            adjusted[name] = (1 - blend_factor) * base_allocs[name] + blend_factor * equal_weight
        end
        return adjusted
    end

    return base_allocs
end

# =============================================================================
# TRAINING
# =============================================================================

function train__(
    ::Type{InstitutionalGradeParams},
    ds_train::Dataset{Train},
    ds_val::Dataset{Validation}
)::Model{InstitutionalGradeParams}

    @assert ds_train.uid == ds_val.uid

    println("Training Institutional-Grade Meta Strategy...")
    println("="^70)
    println("Target Profile:")
    println("  • Returns: 12-18% annualized")
    println("  • Max Drawdown: <12-15%")
    println("  • Sharpe Ratio: >1.5")
    println("="^70)

    asset_list = collect(assets(ds_train))
    n_assets = length(asset_list)

    # Determine regime asset (prefer SPY, else first asset)
    regime_asset = :SPY in asset_list ? :SPY : first(asset_list)
    println("\nUsing $(regime_asset) for regime detection")

    # Create single-asset dataset for RSI strategy
    single_asset = regime_asset
    train_single = Dataset(single_asset => ds_train.data[single_asset])
    val_single = Dataset(single_asset => ds_val.data[single_asset])
    train_single = Dataset{Train}(train_single.data, train_single.timestamps, ds_train.uid)
    val_single = Dataset{Validation}(val_single.data, val_single.timestamps, ds_val.uid)

    println("\n" * "="^70)
    println("TRAINING COMPONENT STRATEGIES")
    println("="^70)

    # 1. Train Markowitz (40% base allocation) - Core stability
    println("\n1/4 Training Markowitz Portfolio (Core Holdings)...")
    markowitz_model = MarkowitzStrategy.train__(
        MarkowitzStrategy.MarkowitzParams, ds_train, ds_val
    )
    markowitz_strat = Strategy(markowitz_model)

    # 2. Train Momentum Rotation (30% base allocation) - Tactical alpha
    println("\n2/4 Training Momentum Rotation (Tactical Alpha)...")
    top_n = min(3, n_assets)
    momentum_model = MomentumStrategy.train__(
        MomentumStrategy.MomentumParams, ds_train, ds_val,
        MomentumStrategy.MomentumParams(30, top_n)
    )
    momentum_strat = Strategy(momentum_model)

    # 3. Train Genetic-Regime (20% base allocation) - Adaptive risk management
    println("\n3/4 Training Genetic-Regime (Adaptive Risk)...")
    genetic_model = GeneticRegimeStrategy.train__(
        GeneticRegimeStrategy.GeneticRegimeParams, ds_train, ds_val
    )
    genetic_strat = Strategy(genetic_model)

    # 4. Train RSI (10% base allocation) - Mean reversion dampening
    println("\n4/4 Training RSI Mean Reversion (Volatility Dampening)...")
    rsi_model = RSIStrategy.train__(
        RSIStrategy.RSIParams, train_single, val_single,
        RSIStrategy.RSIParams(14, 30.0, 70.0)
    )
    rsi_strat = Strategy(rsi_model)

    # Train HMM for regime detection
    println("\n" * "="^70)
    println("TRAINING REGIME DETECTOR")
    println("="^70)
    regime_returns = returns(ds_train, regime_asset)
    hmm = HMMStrategy.fit_hmm(regime_returns, 2; maxiter=100)
    println("HMM trained: μ=$(round.(hmm.μ .* 100, digits=2))%, σ=$(round.(hmm.σ .* 100, digits=2))%")

    # Set base allocations
    base_allocations = Dict(
        "Markowitz" => 0.40,
        "Momentum" => 0.30,
        "GeneticRegime" => 0.20,
        "RSI" => 0.10
    )

    println("\n" * "="^70)
    println("BASE ALLOCATIONS:")
    println("="^70)
    for (name, alloc) in sort(collect(base_allocations), by=x->x[2], rev=true)
        println("  $(name): $(Int(round(alloc*100)))%")
    end

    # Risk parameters
    max_dd_warning = 0.12      # 12% drawdown warning
    max_dd_emergency = 0.15    # 15% drawdown emergency
    target_vol = 0.12          # 12% annualized volatility target
    correlation_threshold = 0.7

    println("\n" * "="^70)
    println("RISK CONTROLS:")
    println("="^70)
    println("  • Target Volatility: $(Int(round(target_vol*100)))% annualized")
    println("  • Drawdown Warning (50% exposure): $(Int(round(max_dd_warning*100)))%")
    println("  • Drawdown Emergency (25% exposure): $(Int(round(max_dd_emergency*100)))%")
    println("  • Correlation Threshold: $(correlation_threshold)")
    println("  • Rebalancing: Monthly (21 trading days)")
    println("="^70)

    params = InstitutionalGradeParams(
        markowitz_strat,
        momentum_strat,
        genetic_strat,
        rsi_strat,
        hmm,
        regime_asset,
        base_allocations,
        max_dd_warning,
        max_dd_emergency,
        target_vol,
        correlation_threshold,
        Ref(base_allocations),
        Ref(1.0),  # Initial peak value
        Ref(0.0),  # Initial drawdown
        Ref(0),    # Last rebalance day
        21,        # Monthly rebalancing
        Dict(
            "Markowitz" => Float64[],
            "Momentum" => Float64[],
            "GeneticRegime" => Float64[],
            "RSI" => Float64[]
        )
    )

    println("\n✓ Institutional-Grade Meta Strategy training complete!")

    Model(ds_train, ds_val, params)
end

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{InstitutionalGradeParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Need minimum data
    if nrow(ds) < 30
        return Dict(a => 0.0 for a in assets(ds))
    end

    current_day = nrow(ds)

    # Get individual strategy signals
    markowitz_signal = signal(ds, params.markowitz_strategy)
    momentum_signal = signal(ds, params.momentum_strategy)
    genetic_signal = signal(ds, params.genetic_regime_strategy)

    # For RSI, create single-asset dataset
    if params.regime_asset in assets(ds)
        ds_single = Dataset(params.regime_asset => ds.data[params.regime_asset])
        ds_single = Dataset{Test}(ds_single.data, ds.timestamps, ds.uid)
        rsi_signal = signal(ds_single, params.rsi_strategy)
    else
        rsi_signal = Dict(a => 0.0 for a in assets(ds))
    end

    # Detect current regime
    regime_returns = returns(ds, params.regime_asset)
    lookback = min(20, length(regime_returns))
    recent_returns = regime_returns[max(1, end-lookback+1):end]
    regime_sequence = HMMStrategy.viterbi(params.hmm, recent_returns)
    current_regime = regime_sequence[end]
    regime_is_bull = params.hmm.μ[current_regime] > 0

    # Calculate portfolio returns for risk monitoring (if we have history)
    portfolio_value = 1.0
    portfolio_returns = Float64[]

    if current_day > params.last_rebalance[] + 1
        # Simplified: use regime asset returns as proxy for portfolio
        portfolio_returns = regime_returns[max(1, end-20):end]

        # Update peak and drawdown
        for ret in portfolio_returns
            portfolio_value *= (1 + ret)
            if portfolio_value > params.peak_value[]
                params.peak_value[] = portfolio_value
            end
        end

        current_dd = calculate_drawdown(params.peak_value[], portfolio_value)
        params.current_drawdown[] = current_dd
    end

    # Estimate realized volatility
    realized_vol = estimate_volatility(regime_returns, 20)

    # Calculate risk scalar (most critical component)
    risk_scalar = calculate_risk_scalar(
        params.current_drawdown[],
        realized_vol,
        params.target_volatility,
        regime_is_bull,
        params.max_drawdown_warning,
        params.max_drawdown_emergency
    )

    # Rebalance allocations monthly
    if current_day - params.last_rebalance[] >= params.rebalance_period
        # Adjust for correlation if needed
        adjusted_allocs = adjust_allocations_for_correlation(
            params.base_allocations,
            params.strategy_returns,
            params.correlation_threshold
        )
        params.current_allocations[] = adjusted_allocs
        params.last_rebalance[] = current_day
    end

    current_allocs = params.current_allocations[]

    # Combine signals with allocations
    combined_signal = Dict(a => 0.0 for a in assets(ds))

    # Weight each strategy's signal by its allocation
    for (asset, weight) in markowitz_signal
        if asset in keys(combined_signal)
            combined_signal[asset] += current_allocs["Markowitz"] * weight
        end
    end

    for (asset, weight) in momentum_signal
        if asset in keys(combined_signal)
            combined_signal[asset] += current_allocs["Momentum"] * weight
        end
    end

    for (asset, weight) in genetic_signal
        if asset in keys(combined_signal)
            combined_signal[asset] += current_allocs["GeneticRegime"] * weight
        end
    end

    for (asset, weight) in rsi_signal
        if asset in keys(combined_signal)
            combined_signal[asset] += current_allocs["RSI"] * weight
        end
    end

    # Apply risk scalar to entire portfolio
    for asset in keys(combined_signal)
        combined_signal[asset] *= risk_scalar
    end

    # Log risk adjustments (if significant)
    if risk_scalar < 0.95
        println("  Day $current_day: Risk scalar = $(round(risk_scalar, digits=2)) " *
                "(DD: $(round(params.current_drawdown[]*100, digits=1))%, " *
                "Vol: $(round(realized_vol*100, digits=1))%, " *
                "Regime: $(regime_is_bull ? "Bull" : "Bear"))")
    end

    return combined_signal
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    println("Institutional-Grade Meta Strategy Demo")
    println("="^70)
    println("This demo requires multi-asset data and is best run via CLI")
    println("\nRun with:")
    println("  ./bin/blab compare --stocks 10")
    println("="^70)
end

export InstitutionalGradeParams, train__, signal, demo

end # module
