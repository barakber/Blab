"""
Regime Switching Strategy
=========================
Meta-strategy that combines multiple strategies based on market regime identification.
Uses HMM to detect regimes and switches between strategies optimized for each regime.

Key insight from backtests:
- RSI strategies excel in choppy/bear markets (low volatility, mean-reverting)
- Momentum strategies dominate bull markets (high trend strength)
- EMA strategies work well in moderate trending markets
"""

module RegimeSwitchStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns, split
import ..Blab: signal, train__
using ..Blab.RSIStrategy
using ..Blab.MomentumStrategy
using ..Blab.EMAStrategy
using Statistics
using Random
using Dates
using DataFrames

# =============================================================================
# REGIME DETECTION
# =============================================================================

"""
Market regimes detected from data.
"""
@enum Regime begin
    BULL        # Strong uptrend, low volatility
    BEAR        # Downtrend or high volatility
    SIDEWAYS    # Low trend, low volatility
end

"""
Simple regime detection based on volatility and trend.
More robust than HMM for real-time decisions.
"""
function detect_regime(price_series::Vector{Float64}, lookback::Int=60)::Regime
    if length(price_series) < lookback
        return SIDEWAYS
    end

    # Get recent window
    recent = price_series[end-lookback+1:end]

    # Calculate returns
    rets = diff(log.(recent))

    # Metrics
    volatility = std(rets)
    mean_return = mean(rets)
    trend_strength = abs(mean_return) / (volatility + 1e-10)

    # Regime classification
    # High vol OR negative trend with any vol = BEAR
    # Positive trend with low/medium vol = BULL
    # Low trend, low vol = SIDEWAYS

    if volatility > 0.02  # High volatility (>2% daily std)
        return BEAR
    elseif mean_return < -0.0005  # Negative trend
        return BEAR
    elseif mean_return > 0.001 && trend_strength > 0.5  # Strong positive trend
        return BULL
    else
        return SIDEWAYS
    end
end

"""
Alternative: HMM-based regime detection (more sophisticated).
Identifies regimes from return distribution characteristics.
"""
function detect_regime_hmm(price_series::Vector{Float64}, n_states::Int=3)::Regime
    # For simplicity, use volatility-based clustering
    # In practice, could use full HMM with Viterbi algorithm

    if length(price_series) < 30
        return SIDEWAYS
    end

    rets = diff(log.(price_series))
    recent_vol = std(rets[max(1, end-19):end])  # Recent 20-period vol

    # Simple threshold-based classification
    if recent_vol > 0.025
        return BEAR  # High volatility regime
    elseif mean(rets[max(1, end-19):end]) > 0.001
        return BULL  # Positive return regime
    else
        return SIDEWAYS  # Low volatility, neutral return regime
    end
end

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct RegimeSwitchParams
    # Sub-strategies for each regime
    bull_strategy::Strategy
    bear_strategy::Strategy
    sideways_strategy::Strategy

    # Regime detection params
    lookback::Int
    use_hmm::Bool
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{RegimeSwitchParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{RegimeSwitchParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Regime Switching Strategy...")

    # Train sub-strategies for each regime
    # BULL regime: Use Momentum (excels in bull markets)
    bull_model = Model(ds_train, ds_val, MomentumStrategy.MomentumParams(30, 3))
    bull_strategy = Strategy(bull_model)

    # BEAR regime: Use RSI mean reversion (protects capital, low drawdown)
    bear_model = Model(ds_train, ds_val, RSIStrategy.RSIParams(14, 30.0, 70.0))
    bear_strategy = Strategy(bear_model)

    # SIDEWAYS regime: Use EMA (moderate trending)
    sideways_model = Model(ds_train, ds_val, EMAStrategy.EMAParams(8, 21))
    sideways_strategy = Strategy(sideways_model)

    println("  Bull regime: Momentum_L30_T3")
    println("  Bear regime: RSI_14_30_70")
    println("  Sideways regime: EMA_8_21")

    params = RegimeSwitchParams(
        bull_strategy,
        bear_strategy,
        sideways_strategy,
        60,      # lookback for regime detection
        false    # use simple regime detection (not HMM)
    )

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{RegimeSwitchParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    asset = first(assets(ds))

    # Need enough data for regime detection
    if nrow(ds) < params.lookback
        return Dict(asset => 0.0)
    end

    # Get price series for regime detection
    price_series = prices(ds, asset)

    # Detect current regime
    current_regime = if params.use_hmm
        detect_regime_hmm(price_series)
    else
        detect_regime(price_series, params.lookback)
    end

    # Select strategy based on regime
    active_strategy = if current_regime == BULL
        params.bull_strategy
    elseif current_regime == BEAR
        params.bear_strategy
    else  # SIDEWAYS
        params.sideways_strategy
    end

    # Delegate to the selected strategy
    signal(ds, active_strategy)
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Regime Switching Strategy Demo")
    println("="^60)

    # Create mixed regime data
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Simulate different regimes
    close_prices = Float64[100.0]
    for i in 2:n
        if i < n÷3  # Bull market
            ret = 0.001 + 0.01 * randn()
        elseif i < 2n÷3  # Bear market
            ret = -0.0005 + 0.025 * randn()
        else  # Sideways
            ret = 0.0 + 0.008 * randn()
        end
        push!(close_prices, close_prices[end] * (1 + ret))
    end

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(RegimeSwitchParams, train, val)
    s = Strategy(m)

    println("\nBacktesting regime-switching strategy...")
    println(backtest(test, s))
end

# demo()

export RegimeSwitchParams, train__, signal, demo

end # module
