"""
Leveraged QQQ Strategy (TQQQ/SQQQ)
===================================
Predicts QQQ direction using technical analysis and ML, then trades:
- TQQQ (3x leveraged long) when bullish with high confidence
- SQQQ (3x leveraged short) when bearish with high confidence
- Cash when uncertain or low conviction

Key insights from research:
- Volatility decay erodes leveraged ETF returns in choppy markets
- Best for short-term tactical trades (3-5 days)
- Requires strong directional conviction
- Avoid holding through uncertain periods

Strategy combines:
- QQQ technical indicators (MACD, RSI, trend strength, volatility)
- XGBoost prediction of QQQ direction
- Confidence-based allocation to TQQQ/SQQQ/Cash
"""

module LeverageQQQStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns
import ..Blab: signal, train__
using Indicators
using Statistics
using Random
using Dates
using DataFrames
using MLJ
using MLJXGBoostInterface

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct LeverageQQQParams
    xgb_model::Any      # Trained XGBoost model
    machine::Any        # MLJ machine
    confidence_threshold::Float64  # Minimum confidence for leveraged position
end

# =============================================================================
# FEATURE ENGINEERING FOR QQQ
# =============================================================================

"""
Extract technical features from QQQ price data.
"""
function extract_qqq_features(prices_vec::Vector{Float64})::DataFrame
    n = length(prices_vec)

    # Calculate returns
    rets = diff(log.(prices_vec))
    ret_current = vcat([0.0], rets)  # Pad to match length

    # Lagged returns
    ret_lag1 = vcat([0.0, 0.0], rets[1:end-1])
    ret_lag2 = vcat([0.0, 0.0, 0.0], rets[1:end-2])
    ret_lag3 = vcat([0.0, 0.0, 0.0, 0.0], rets[1:end-3])

    # Technical indicators using Indicators.jl
    rsi_vals = rsi(prices_vec; n=14)

    # MACD (returns matrix with columns: [macd, signal, histogram])
    macd_result = macd(prices_vec; nfast=12, nslow=26, nsig=9)
    macd_vals = macd_result[:, 1]
    macd_signal_vals = macd_result[:, 2]
    macd_hist = macd_result[:, 3]

    # EMAs
    ema_fast = ema(prices_vec; n=12)
    ema_slow = ema(prices_vec; n=26)
    ema_diff = ema_fast .- ema_slow
    ema_ratio = ema_fast ./ ema_slow

    # Trend strength: ratio of mean return to volatility
    vol = zeros(n)
    trend_strength = zeros(n)
    for i in 21:n
        window_rets = rets[i-20:i-1]
        vol[i] = std(window_rets)
        trend_strength[i] = abs(mean(window_rets)) / (vol[i] + 1e-10)
    end

    # Price momentum (% from 20-day low/high)
    mom_20 = zeros(n)
    for i in 21:n
        window_prices = prices_vec[i-20:i]
        price_min = minimum(window_prices)
        price_max = maximum(window_prices)
        price_range = price_max - price_min
        if price_range > 0
            mom_20[i] = (prices_vec[i] - price_min) / price_range
        end
    end

    # Bollinger Band position
    bb_position = zeros(n)
    for i in 21:n
        window = prices_vec[i-20:i]
        bb_mid = mean(window)
        bb_std = std(window)
        if bb_std > 0
            bb_position[i] = (prices_vec[i] - bb_mid) / (2 * bb_std)
        end
    end

    # Create targets: predict next return direction
    target = vcat(rets .> 0, [false])  # Pad last value

    df = DataFrame(
        ret_current = ret_current,
        ret_lag1 = ret_lag1,
        ret_lag2 = ret_lag2,
        ret_lag3 = ret_lag3,
        rsi = rsi_vals,
        macd = macd_vals,
        macd_signal = macd_signal_vals,
        macd_hist = macd_hist,
        ema_fast = ema_fast,
        ema_slow = ema_slow,
        ema_diff = ema_diff,
        ema_ratio = ema_ratio,
        volatility = vol,
        trend_strength = trend_strength,
        momentum_20d = mom_20,
        bb_position = bb_position,
        target = target
    )

    return df
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{LeverageQQQParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{LeverageQQQParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Leveraged QQQ Strategy (TQQQ/SQQQ)...")

    # Check if we have QQQ data
    if !(:QQQ in assets(ds_train))
        error("QQQ is required for LeverageQQQ strategy")
    end

    println("  Learning from QQQ to trade TQQQ/SQQQ...")

    # Get QQQ prices
    train_prices = prices(ds_train, :QQQ)
    val_prices = prices(ds_val, :QQQ)

    # Extract features
    println("  Extracting QQQ technical features...")
    train_features_df = extract_qqq_features(train_prices)
    val_features_df = extract_qqq_features(val_prices)

    # Remove NaN/Inf values (from first 30 rows due to indicators)
    train_features_df = train_features_df[31:end, :]
    val_features_df = val_features_df[31:end, :]

    # Separate features and target
    feature_names = ["ret_current", "ret_lag1", "ret_lag2", "ret_lag3",
                     "rsi", "macd", "macd_signal", "macd_hist",
                     "ema_fast", "ema_slow", "ema_diff", "ema_ratio",
                     "volatility", "trend_strength", "momentum_20d", "bb_position"]

    X_train = select(train_features_df, feature_names)
    y_train = categorical(train_features_df.target)

    X_val = select(val_features_df, feature_names)
    y_val = categorical(val_features_df.target)

    println("  Training samples: $(size(X_train, 1))")
    println("  Positive class ratio: $(round(100*mean(train_features_df.target), digits=1))%")

    # Train XGBoost classifier
    println("  Training XGBoost to predict QQQ direction...")

    xgb_model = MLJXGBoostInterface.XGBoostClassifier(
        num_round=150,
        max_depth=5,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3
    )

    machine = MLJ.machine(xgb_model, X_train, y_train)
    MLJ.fit!(machine, verbosity=0)

    # Evaluate on validation set
    y_val_pred = MLJ.predict_mode(machine, X_val)
    val_accuracy = mean(y_val_pred .== y_val)
    println("  Validation accuracy: $(round(100*val_accuracy, digits=1))%")

    # Get prediction probabilities to assess confidence
    y_val_probs = MLJ.predict(machine, X_val)
    # Extract probability of positive class
    probs = [pdf(p, true) for p in y_val_probs]
    avg_confidence = mean(abs.(probs .- 0.5))
    println("  Average prediction confidence: $(round(100*avg_confidence, digits=1))%")

    # Set confidence threshold (60% = 0.1 away from 0.5)
    confidence_threshold = 0.10

    params = LeverageQQQParams(xgb_model, machine, confidence_threshold)

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{LeverageQQQParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Need QQQ and TQQQ data
    if !(:QQQ in assets(ds))
        error("QQQ is required for signal generation")
    end

    # Check which leveraged ETFs are available
    has_tqqq = :TQQQ in assets(ds)
    has_sqqq = :SQQQ in assets(ds)

    if !has_tqqq && !has_sqqq
        error("At least one of TQQQ or SQQQ is required")
    end

    # Need enough data for features
    if nrow(ds) < 35
        # Return 0 allocation
        result = Dict{Symbol, Float64}()
        for asset in assets(ds)
            result[asset] = 0.0
        end
        return result
    end

    # Get QQQ prices
    qqq_prices = prices(ds, :QQQ)

    # Extract features
    features_df = extract_qqq_features(qqq_prices)

    # Get most recent features (skip first 30 rows)
    if size(features_df, 1) < 31
        result = Dict{Symbol, Float64}()
        for asset in assets(ds)
            result[asset] = 0.0
        end
        return result
    end

    latest_features = features_df[end, :]

    # Prepare for prediction
    feature_names = ["ret_current", "ret_lag1", "ret_lag2", "ret_lag3",
                     "rsi", "macd", "macd_signal", "macd_hist",
                     "ema_fast", "ema_slow", "ema_diff", "ema_ratio",
                     "volatility", "trend_strength", "momentum_20d", "bb_position"]

    X = select(DataFrame([latest_features]), feature_names)

    # Predict direction and get confidence
    y_prob = MLJ.predict(params.machine, X)[1]
    prob_positive = pdf(y_prob, true)

    # Calculate confidence (distance from 0.5)
    confidence = abs(prob_positive - 0.5)

    # Determine position
    # - High confidence bullish (>60%): 100% TQQQ
    # - High confidence bearish (<40%): 100% SQQQ (if available)
    # - Low confidence: 0% (stay in cash)

    result = Dict{Symbol, Float64}()

    # Initialize all assets to 0
    for asset in assets(ds)
        result[asset] = 0.0
    end

    if confidence >= params.confidence_threshold
        if prob_positive > 0.5
            # Bullish: allocate to TQQQ
            if has_tqqq
                result[:TQQQ] = 1.0
            end
        else
            # Bearish: allocate to SQQQ
            if has_sqqq
                result[:SQQQ] = 1.0
            end
        end
    end
    # Else: stay in cash (all weights = 0)

    return result
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Leveraged QQQ Strategy Demo")
    println("="^60)

    # Create synthetic QQQ-like data
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Simulate QQQ with trend + volatility
    qqq_prices = Float64[300.0]
    for i in 2:n
        ret = 0.0008 + 0.018 * randn()  # Positive drift, high vol
        push!(qqq_prices, qqq_prices[end] * (1 + ret))
    end

    # Simulate TQQQ (3x leveraged, simplified)
    tqqq_prices = Float64[100.0]
    for i in 2:n
        qqq_ret = log(qqq_prices[i] / qqq_prices[i-1])
        tqqq_ret = 3.0 * qqq_ret  # 3x daily leverage
        push!(tqqq_prices, tqqq_prices[end] * exp(tqqq_ret))
    end

    qqq_df = DataFrame(timestamp=dates, close=qqq_prices)
    tqqq_df = DataFrame(timestamp=dates, close=tqqq_prices)

    ds = Dataset(Dict(:QQQ => qqq_df, :TQQQ => tqqq_df))

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(LeverageQQQParams, train, val)
    s = Strategy(m)

    println("\nBacktesting leveraged QQQ strategy...")
    println(backtest(test, s))
end

# demo()

export LeverageQQQParams, train__, signal, demo

end # module
