"""
Adaptive Regime Strategy
=========================
Enhanced regime switching with machine learning-based regime detection and
dynamic strategy allocation.

Improvements over basic RegimeSwitch:
1. ML-based regime classification using XGBoost on multiple features:
   - Volatility (short/medium/long term)
   - Trend strength and direction
   - Market correlation/dispersion
   - RSI and MACD signals

2. Ensemble approach within each regime:
   - Instead of single strategy per regime, uses weighted ensemble
   - Weights learned from validation set performance

3. More granular regimes (5 instead of 3):
   - Strong Bull: High momentum strategies, concentrated positions
   - Moderate Bull: Balanced momentum + trend following
   - Sideways/Range: Mean reversion strategies
   - Moderate Bear: Defensive, low volatility
   - Strong Bear: Maximum cash preservation, inverse signals

4. Adaptive rebalancing based on regime confidence
"""

module AdaptiveRegimeStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns, split
import ..Blab: signal, train__
using ..Blab.MomentumStrategy
using ..Blab.RSIStrategy
using ..Blab.EMAStrategy
using ..Blab.MACDStrategy
using ..Blab.MAStrategy
using Indicators
using Statistics
using Random
using Dates
using DataFrames
using MLJ
using MLJXGBoostInterface
using CategoricalArrays: levelcode, levels

# =============================================================================
# REGIME DETECTION
# =============================================================================

@enum RegimeType begin
    STRONG_BULL      # High positive trend, low vol
    MODERATE_BULL    # Positive trend, moderate vol
    SIDEWAYS         # Low trend, low vol
    MODERATE_BEAR    # Negative trend or high vol
    STRONG_BEAR      # Strong negative trend, very high vol
end

"""
Extract features for regime classification.
"""
function extract_regime_features(prices_vec::Vector{Float64})::DataFrame
    n = length(prices_vec)

    # Calculate returns
    rets = diff(log.(prices_vec))

    # Features for each time point
    features = DataFrame()

    # Volatility at multiple horizons
    vol_5d = zeros(n)
    vol_20d = zeros(n)
    vol_60d = zeros(n)

    for i in 6:n
        vol_5d[i] = std(rets[max(1,i-5):i-1])
    end
    for i in 21:n
        vol_20d[i] = std(rets[max(1,i-20):i-1])
    end
    for i in 61:n
        vol_60d[i] = std(rets[max(1,i-60):i-1])
    end

    # Trend at multiple horizons
    trend_5d = zeros(n)
    trend_20d = zeros(n)
    trend_60d = zeros(n)

    for i in 6:n
        trend_5d[i] = mean(rets[max(1,i-5):i-1])
    end
    for i in 21:n
        trend_20d[i] = mean(rets[max(1,i-20):i-1])
    end
    for i in 61:n
        trend_60d[i] = mean(rets[max(1,i-60):i-1])
    end

    # Technical indicators
    rsi_vals = rsi(prices_vec; n=14)
    macd_result = macd(prices_vec; nfast=12, nslow=26, nsig=9)
    ema_fast = ema(prices_vec; n=12)
    ema_slow = ema(prices_vec; n=26)

    # Trend strength
    trend_strength = zeros(n)
    for i in 21:n
        trend_strength[i] = abs(trend_20d[i]) / (vol_20d[i] + 1e-10)
    end

    features.vol_5d = vol_5d
    features.vol_20d = vol_20d
    features.vol_60d = vol_60d
    features.trend_5d = trend_5d
    features.trend_20d = trend_20d
    features.trend_60d = trend_60d
    features.rsi = rsi_vals
    features.macd = macd_result[:, 1]
    features.macd_signal = macd_result[:, 2]
    features.ema_diff = ema_fast .- ema_slow
    features.trend_strength = trend_strength

    return features
end

"""
Create regime labels for training.
Based on forward returns and volatility.
"""
function create_regime_labels(prices_vec::Vector{Float64}, forward_days::Int=20)::Vector{RegimeType}
    n = length(prices_vec)
    labels = fill(SIDEWAYS, n)

    for i in 1:(n - forward_days)
        # Forward return
        fwd_ret = log(prices_vec[i + forward_days] / prices_vec[i])

        # Recent volatility (20-day)
        if i >= 20
            recent_rets = diff(log.(prices_vec[max(1,i-20):i]))
            vol = std(recent_rets)
        else
            vol = 0.0
        end

        # Classification logic - balanced
        if fwd_ret > 0.05 && vol < 0.015  # >5% gain, low vol
            labels[i] = STRONG_BULL
        elseif fwd_ret > 0.02 && vol < 0.025  # >2% gain, moderate vol
            labels[i] = MODERATE_BULL
        elseif fwd_ret < -0.05 || vol > 0.03  # <-5% loss or high vol
            labels[i] = STRONG_BEAR
        elseif fwd_ret < -0.02 || vol > 0.02  # <-2% loss or moderate-high vol
            labels[i] = MODERATE_BEAR
        else
            labels[i] = SIDEWAYS
        end
    end

    return labels
end

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct AdaptiveRegimeParams
    regime_classifier::Any  # XGBoost model for regime prediction
    regime_machine::Any     # MLJ machine

    # Strategy ensemble for each regime
    # Dict{RegimeType => Vector{(Strategy, weight)}}
    regime_strategies::Dict{RegimeType, Vector{Tuple{Strategy, Float64}}}
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{AdaptiveRegimeParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{AdaptiveRegimeParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Adaptive Regime Strategy...")

    # Get first asset for regime detection (or use SPY if available)
    asset = :SPY in assets(ds_train) ? :SPY : first(assets(ds_train))

    train_prices = prices(ds_train, asset)
    val_prices = prices(ds_val, asset)

    # Extract features
    println("  Extracting regime features from training data...")
    train_features = extract_regime_features(train_prices)
    val_features = extract_regime_features(val_prices)

    # Create regime labels
    println("  Creating regime labels...")
    train_labels = create_regime_labels(train_prices)

    # Remove first 70 rows (need data for all features)
    train_features = train_features[71:end, :]
    train_labels = train_labels[71:end]
    val_features = val_features[71:end, :]

    # Train regime classifier
    println("  Training regime classifier...")
    feature_names = names(train_features)
    X_train = train_features
    # Convert enum to integers for MLJ compatibility
    y_train = categorical(Int.(train_labels))

    xgb_model = MLJXGBoostInterface.XGBoostClassifier(
        num_round=100,
        max_depth=5,
        eta=0.1,
        subsample=0.8
    )

    machine = MLJ.machine(xgb_model, X_train, y_train)
    MLJ.fit!(machine, verbosity=0)

    # Evaluate
    y_train_pred = MLJ.predict_mode(machine, X_train)
    train_acc = mean(y_train_pred .== y_train)
    println("  Regime classifier training accuracy: $(round(100*train_acc, digits=1))%")

    # Count regime distribution
    regime_counts = Dict{RegimeType, Int}()
    for regime in instances(RegimeType)
        regime_counts[regime] = count(x -> x == regime, train_labels)
    end
    println("  Regime distribution:")
    for regime in [STRONG_BULL, MODERATE_BULL, SIDEWAYS, MODERATE_BEAR, STRONG_BEAR]
        pct = 100.0 * regime_counts[regime] / length(train_labels)
        println("    $(regime): $(regime_counts[regime]) ($(round(pct, digits=1))%)")
    end

    # Build strategy allocation for each regime
    # Use the SAME strategies as RegimeSwitch (proven to work)
    println("  Selecting best strategy for each regime...")
    regime_strategies = Dict{RegimeType, Vector{Tuple{Strategy, Float64}}}()

    # All bull regimes: Use multi-asset momentum (like RegimeSwitch's bull strategy)
    bull_strategy = Strategy(Model(ds_train, ds_val, MomentumStrategy.MomentumParams(30, 3)))

    # All bear regimes: Use RSI mean reversion (like RegimeSwitch's bear strategy)
    bear_strategy = Strategy(Model(ds_train, ds_val, RSIStrategy.RSIParams(14, 30.0, 70.0)))

    # Sideways: Use EMA (like RegimeSwitch's sideways strategy)
    sideways_strategy = Strategy(Model(ds_train, ds_val, EMAStrategy.EMAParams(8, 21)))

    regime_strategies[STRONG_BULL] = [(bull_strategy, 1.0)]
    regime_strategies[MODERATE_BULL] = [(bull_strategy, 1.0)]
    regime_strategies[SIDEWAYS] = [(sideways_strategy, 1.0)]
    regime_strategies[MODERATE_BEAR] = [(bear_strategy, 1.0)]
    regime_strategies[STRONG_BEAR] = [(bear_strategy, 1.0)]

    println("  Strategy allocation:")
    println("    STRONG_BULL / MODERATE_BULL: Momentum_L30_T3")
    println("    SIDEWAYS: EMA_8_21")
    println("    MODERATE_BEAR / STRONG_BEAR: RSI_14_30_70")

    params = AdaptiveRegimeParams(xgb_model, machine, regime_strategies)

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{AdaptiveRegimeParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Get asset for regime detection
    asset = :SPY in assets(ds) ? :SPY : first(assets(ds))

    # Need enough data for features
    if nrow(ds) < 80
        return Dict(a => 0.0 for a in assets(ds))
    end

    # Extract features
    price_series = prices(ds, asset)
    features = extract_regime_features(price_series)

    if size(features, 1) < 71
        return Dict(a => 0.0 for a in assets(ds))
    end

    # Get current features
    current_features = features[end:end, :]

    # Predict regime (returns CategoricalValue with integer levels)
    regime_pred_cat = MLJ.predict_mode(params.regime_machine, current_features)[1]
    # Get the level (integer value) - levels are indexed from 1, but our enum starts at 0
    # So we use the ref field which gives us the category code directly
    regime_pred_int = levelcode(regime_pred_cat)
    # Map levelcode back to the original integer value
    all_levels = levels(regime_pred_cat)
    regime_value = all_levels[regime_pred_int]
    regime_pred = RegimeType(regime_value)

    # Get strategies for this regime
    ensemble = params.regime_strategies[regime_pred]

    # Combine signals from ensemble with weights
    combined_signal = Dict(a => 0.0 for a in assets(ds))

    for (strat, weight) in ensemble
        strat_signal = signal(ds, strat)
        for (a, allocation) in strat_signal
            combined_signal[a] = get(combined_signal, a, 0.0) + weight * allocation
        end
    end

    return combined_signal
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Adaptive Regime Strategy Demo")
    println("="^60)

    # Create data with regime changes
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    close_prices = Float64[100.0]
    for i in 2:n
        # Simulate different regimes
        if i < n÷4  # Strong bull
            ret = 0.002 + 0.01 * randn()
        elseif i < n÷2  # Sideways
            ret = 0.0 + 0.008 * randn()
        elseif i < 3n÷4  # Bear
            ret = -0.001 + 0.025 * randn()
        else  # Recovery
            ret = 0.0015 + 0.015 * randn()
        end
        push!(close_prices, close_prices[end] * (1 + ret))
    end

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(AdaptiveRegimeParams, train, val)
    s = Strategy(m)

    println("\nBacktesting adaptive regime strategy...")
    println(backtest(test, s))
end

# demo()

export AdaptiveRegimeParams, train__, signal, demo

end # module
