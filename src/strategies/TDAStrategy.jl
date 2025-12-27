"""
Topological Data Analysis Strategy
===================================
Uses persistent homology and topological features to analyze market structure
and predict future returns.

Key concepts:
- Constructs point clouds from price time series using delay embeddings
- Computes persistent homology (H0, H1, H2) using Ripserer.jl
- Extracts topological features: Betti numbers, persistence statistics, lifetimes
- Uses XGBoost to predict market direction from TDA features

Topological features capture:
- H0 (connected components): Market fragmentation/clustering
- H1 (loops/cycles): Cyclical patterns and mean reversion
- H2 (voids): Higher-order structure and regime changes
"""

module TDAStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns
import ..Blab: signal, train__
using Ripserer
using PersistenceDiagrams
using Statistics
using Random
using Dates
using DataFrames
using LinearAlgebra

# Import XGBoost for classification
using MLJ
using MLJXGBoostInterface

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct TDAParams
    xgb_model::Any  # Trained XGBoost model
    machine::Any    # MLJ machine
    window::Int     # Window size for point cloud construction
    embedding_dim::Int  # Dimension for delay embedding
    delay::Int      # Delay for embedding
end

# =============================================================================
# TOPOLOGICAL DATA ANALYSIS
# =============================================================================

"""
Create delay embedding (Takens' theorem) for time series.
Constructs a point cloud in R^d from 1D time series.
"""
function delay_embedding(x::Vector{Float64}, dim::Int, delay::Int)::Matrix{Float64}
    n = length(x)
    m = n - (dim - 1) * delay

    if m <= 0
        error("Not enough data for embedding: need at least $((dim-1)*delay + 1) points")
    end

    # Each row is a point in R^dim
    embedding = zeros(m, dim)

    for i in 1:m
        for j in 1:dim
            embedding[i, j] = x[i + (j-1) * delay]
        end
    end

    return embedding
end

"""
Compute persistent homology and extract topological features.
"""
function extract_tda_features(prices_window::Vector{Float64};
                              embedding_dim::Int=3,
                              delay::Int=1,
                              max_dim::Int=2)::Vector{Float64}

    features = Float64[]

    try
        # Normalize prices to [0, 1] for better numerical stability
        prices_norm = (prices_window .- minimum(prices_window)) ./ (maximum(prices_window) - minimum(prices_window) + 1e-10)

        # Create delay embedding
        embedding = delay_embedding(prices_norm, embedding_dim, delay)

        # Compute persistent homology using Vietoris-Rips complex
        result = ripserer(embedding, dim_max=max_dim, threshold=2.0)

        # Extract features for each homological dimension
        for dim in 0:max_dim
            if dim < length(result)
                diagram = result[dim + 1]  # Julia 1-indexed

                # Extract persistence intervals
                births = [interval.birth for interval in diagram]
                deaths = [interval.death for interval in diagram]
                lifetimes = deaths .- births

                # Filter out infinite persistence (for H0)
                finite_mask = .!isinf.(deaths)
                finite_lifetimes = lifetimes[finite_mask]

                # Betti number: number of topological features
                betti = length(diagram)
                push!(features, Float64(betti))

                # Statistics of finite lifetimes
                if !isempty(finite_lifetimes)
                    push!(features, mean(finite_lifetimes))    # Mean persistence
                    push!(features, std(finite_lifetimes))     # Std persistence
                    push!(features, maximum(finite_lifetimes)) # Max persistence
                    push!(features, sum(finite_lifetimes))     # Total persistence
                else
                    push!(features, 0.0, 0.0, 0.0, 0.0)
                end

                # Birth/death statistics
                finite_births = births[finite_mask]
                if !isempty(finite_births)
                    push!(features, mean(finite_births))
                    push!(features, mean(deaths[finite_mask]))
                else
                    push!(features, 0.0, 0.0)
                end
            else
                # No features for this dimension
                push!(features, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            end
        end

    catch e
        # If TDA computation fails, return zeros
        # 7 features per dimension × (max_dim + 1) dimensions
        n_features = 7 * (max_dim + 1)
        features = zeros(n_features)
    end

    return features
end

"""
Build feature matrix for entire time series.
"""
function build_feature_matrix(prices::Vector{Float64};
                              window::Int=50,
                              embedding_dim::Int=3,
                              delay::Int=1)::Matrix{Float64}

    n = length(prices)
    n_samples = n - window + 1

    if n_samples <= 0
        error("Not enough data: need at least $window prices")
    end

    # Extract features for first window to determine feature count
    first_features = extract_tda_features(
        prices[1:window],
        embedding_dim=embedding_dim,
        delay=delay
    )
    n_features = length(first_features)

    # Preallocate feature matrix
    features_matrix = zeros(n_samples, n_features)
    features_matrix[1, :] = first_features

    # Compute features for all windows
    println("  Computing TDA features for $n_samples windows...")
    for i in 2:n_samples
        if i % 100 == 0
            println("    Progress: $(round(100*i/n_samples, digits=1))%")
        end

        window_prices = prices[i:i+window-1]
        features_matrix[i, :] = extract_tda_features(
            window_prices,
            embedding_dim=embedding_dim,
            delay=delay
        )
    end

    return features_matrix
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{TDAParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{TDAParams}
    @assert ds_train.uid == ds_val.uid

    println("Training TDA Strategy with Persistent Homology...")

    asset = first(assets(ds_train))

    # Parameters
    window = 50
    embedding_dim = 3
    delay = 1

    println("  TDA parameters:")
    println("    Window: $window")
    println("    Embedding dimension: $embedding_dim")
    println("    Delay: $delay")

    # Get prices
    train_prices = prices(ds_train, asset)
    val_prices = prices(ds_val, asset)

    # Build TDA features
    println("  Building TDA features for training set...")
    train_features = build_feature_matrix(
        train_prices,
        window=window,
        embedding_dim=embedding_dim,
        delay=delay
    )

    println("  Building TDA features for validation set...")
    val_features = build_feature_matrix(
        val_prices,
        window=window,
        embedding_dim=embedding_dim,
        delay=delay
    )

    # Create targets: predict if next return is positive
    # Features: row i uses prices[i:i+window-1]
    # Target: row i should predict return after window, which is returns[i+window-1]
    # But we need returns to exist, so we need i+window-1 <= length(returns)
    # Since returns has length n-1, we need i+window-1 <= n-1, so i <= n-window
    # But features has n-window+1 rows, so we drop the last row of features

    all_train_returns = returns(ds_train, asset)
    all_val_returns = returns(ds_val, asset)

    # Align features and targets
    n_train_features = size(train_features, 1)
    n_val_features = size(val_features, 1)

    # We can only use rows where we have a next return
    # features[i] → predict returns[i+window-1]
    max_train_idx = min(n_train_features, length(all_train_returns) - window + 1)
    max_val_idx = min(n_val_features, length(all_val_returns) - window + 1)

    train_features = train_features[1:max_train_idx, :]
    val_features = val_features[1:max_val_idx, :]

    train_target = all_train_returns[window:window+max_train_idx-1] .> 0
    val_target = all_val_returns[window:window+max_val_idx-1] .> 0

    # Create DataFrames
    feature_names = ["tda_$i" for i in 1:size(train_features, 2)]
    train_df = DataFrame(train_features, feature_names)
    train_df[!, :target] = train_target

    val_df = DataFrame(val_features, feature_names)
    val_df[!, :target] = val_target

    println("  Training samples: $(size(train_df, 1))")
    println("  Positive class ratio: $(round(100*mean(train_target), digits=1))%")

    # Train XGBoost model
    println("  Training XGBoost classifier on TDA features...")

    xgb_model = MLJXGBoostInterface.XGBoostClassifier(
        num_round=100,
        max_depth=4,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Prepare data for MLJ
    X_train = select(train_df, feature_names)
    y_train = categorical(train_target)

    machine = MLJ.machine(xgb_model, X_train, y_train)
    MLJ.fit!(machine, verbosity=0)

    # Evaluate on validation set
    X_val = select(val_df, feature_names)
    y_val_pred = MLJ.predict_mode(machine, X_val)
    val_accuracy = mean(y_val_pred .== categorical(val_target))
    println("  Validation accuracy: $(round(100*val_accuracy, digits=1))%")

    params = TDAParams(xgb_model, machine, window, embedding_dim, delay)

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{TDAParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    asset = first(assets(ds))

    # Need enough data for TDA features
    if nrow(ds) < params.window
        return Dict(asset => 0.0)
    end

    # Get recent prices
    price_series = prices(ds, asset)
    recent_prices = price_series[end-params.window+1:end]

    # Extract TDA features for current window
    features = extract_tda_features(
        recent_prices,
        embedding_dim=params.embedding_dim,
        delay=params.delay
    )

    # Create DataFrame for prediction
    # Reshape features to 1×n matrix
    feature_names = ["tda_$i" for i in 1:length(features)]
    features_matrix = reshape(features, 1, length(features))
    X = DataFrame(features_matrix, feature_names)

    # Predict
    prediction = MLJ.predict_mode(params.machine, X)[1]

    # Long if predicted positive, flat if predicted negative
    weight = prediction == true ? 1.0 : 0.0

    Dict(asset => weight)
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("TDA Strategy Demo")
    println("="^60)

    # Create synthetic data with regime changes
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Create price series with embedded cycles and structure
    close_prices = Float64[100.0]
    for i in 2:n
        # Add cyclical component + trend + noise
        cycle = 0.001 * sin(2π * i / 60)  # 60-day cycle
        trend = 0.0005
        noise = 0.015 * randn()
        ret = trend + cycle + noise
        push!(close_prices, close_prices[end] * (1 + ret))
    end

    df = DataFrame(timestamp=dates, close=close_prices)
    ds = Dataset(:SPY => df)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(TDAParams, train, val)
    s = Strategy(m)

    println("\nBacktesting TDA strategy...")
    println(backtest(test, s))
end

# demo()

export TDAParams, train__, signal, demo

end # module
