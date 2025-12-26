"""
XGBoost Machine Learning Strategy
===================================
Uses XGBoost via MLJ for price direction prediction.
Features: Returns, RSI, MACD, EMAs, volatility.
"""

module XGBoostMLStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns, split
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames
using Indicators
using MLJ
using MLJXGBoostInterface

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

"""
Extract features from price data using technical indicators.
Returns DataFrame with features, removing NaN rows.
"""
function extract_features(prices_vec::Vector{Float64})::DataFrame
    n = length(prices_vec)

    # Returns
    rets = vcat([0.0], diff(log.(prices_vec)))

    # Technical indicators using Indicators.jl
    rsi_vals = rsi(prices_vec; n=14)
    macd_result = macd(prices_vec; nfast=12, nslow=26, nsig=9)  # Returns [macd, signal, histogram]
    ema_fast = ema(prices_vec; n=12)
    ema_slow = ema(prices_vec; n=26)

    # Extract MACD values (columns from matrix)
    macd_vals = macd_result[:, 1]        # MACD line
    macd_signal_vals = macd_result[:, 2] # Signal line

    # Rolling volatility (20-period)
    vol = zeros(n)
    for i in 20:n
        vol[i] = std(rets[i-19:i])
    end

    # Lag features
    ret_lag1 = vcat([0.0], rets[1:end-1])
    ret_lag2 = vcat([0.0, 0.0], rets[1:end-2])

    # Create target: 1 if next return > 0, 0 otherwise
    target = vcat(rets[2:end] .> 0, [false])

    df = DataFrame(
        ret = rets,
        ret_lag1 = ret_lag1,
        ret_lag2 = ret_lag2,
        rsi = rsi_vals,
        macd = macd_vals,
        macd_signal = macd_signal_vals,
        ema_fast = ema_fast,
        ema_slow = ema_slow,
        ema_diff = ema_fast .- ema_slow,
        vol = vol,
        target = target
    )

    # Remove rows with NaN (from indicators)
    dropmissing!(df)
    df[50:end, :]  # Keep only after sufficient warmup period
end

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct XGBoostParams
    model::Any  # Trained XGBoost model
    threshold::Float64  # Probability threshold for going long
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{XGBoostParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{XGBoostParams}
    @assert ds_train.uid == ds_val.uid

    println("Training XGBoost model...")

    asset = first(assets(ds_train))
    train_prices = prices(ds_train, asset)

    # Extract features
    train_df = extract_features(train_prices)

    println("  Training samples: $(size(train_df, 1))")
    println("  Positive class ratio: $(round(mean(train_df.target) * 100, digits=1))%")

    # Prepare data for MLJ
    X = select(train_df, Not(:target))
    y = train_df.target

    # Create XGBoost model directly
    xgb_model = MLJXGBoostInterface.XGBoostClassifier(
        num_round=100,
        max_depth=4,
        eta=0.1
    )

    # Create machine and train
    mach = machine(xgb_model, X, categorical(y))
    MLJ.fit!(mach, verbosity=0)

    # Validate on validation set
    val_prices = prices(ds_val, asset)
    val_df = extract_features(val_prices)
    X_val = select(val_df, Not(:target))
    y_val = val_df.target

    val_pred = MLJ.predict(mach, X_val)
    val_proba = pdf.(val_pred, true)  # Probability of positive class
    val_accuracy = mean((val_proba .> 0.5) .== y_val)

    println("  Validation accuracy: $(round(val_accuracy * 100, digits=1))%")

    Model(ds_train, ds_val, XGBoostParams(mach, 0.5))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{XGBoostParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    asset = first(assets(ds))

    # Need minimum data for feature extraction
    if nrow(ds) < 100
        return Dict(asset => 0.0)
    end

    test_prices = prices(ds, asset)

    # Extract features for current data
    test_df = extract_features(test_prices)

    if size(test_df, 1) == 0
        return Dict(asset => 0.0)
    end

    # Get features for last row (current state)
    X_current = select(test_df[end:end, :], Not(:target))

    # Predict
    pred = MLJ.predict(params.model, X_current)
    proba = pdf(pred[1], true)

    # Go long if probability > threshold
    weight = proba > params.threshold ? 1.0 : 0.0

    Dict(asset => weight)
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Generate trending price series
    close_prices = Float64[100.0]
    trend = 0.0005
    for _ in 2:n
        ret = trend + 0.015 * randn()
        push!(close_prices, close_prices[end] * (1 + ret))
    end

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(XGBoostParams, train, val)
    s = Strategy(m)

    println("\nBacktesting...")
    println(backtest(test, s))
end

# demo()

export XGBoostParams, train__, signal, demo

end # module
