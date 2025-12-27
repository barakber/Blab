"""
Genetic-Markowitz Hybrid Strategy
==================================
Combines genetic algorithms with Markowitz mean-variance optimization.

Key innovation:
- Uses GA for global optimization (avoids local optima)
- Fitness function based on Markowitz Sharpe ratio (expected return / volatility)
- Incorporates forward-looking estimates (not just historical backtest)
- Handles non-convex constraints naturally with GA

Advantages over pure Markowitz:
- GA can handle non-linear constraints and objectives
- More robust to estimation errors
- Can optimize non-standard risk metrics

Advantages over pure Genetic:
- Uses sound financial theory (mean-variance framework)
- More stable than pure historical optimization
- Incorporates covariance structure explicitly
"""

module GeneticMarkowitzStrategy

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

struct GeneticMarkowitzParams
    optimal_weights::Dict{Symbol, Float64}
    expected_returns::Dict{Symbol, Float64}
    cov_matrix::Matrix{Float64}
    asset_list::Vector{Symbol}
    lookback::Int
    rebalance_period::Int
    last_rebalance::Ref{Int}
    current_weights::Ref{Dict{Symbol, Float64}}
end

# =============================================================================
# MARKOWITZ UTILITIES
# =============================================================================

"""
Compute expected returns (annualized) from historical data.
"""
function compute_expected_returns(ds::Dataset, assets_list::Vector{Symbol}, lookback::Int)::Dict{Symbol, Float64}
    expected = Dict{Symbol, Float64}()

    for asset in assets_list
        rets = returns(ds, asset)
        window = rets[max(1, end-lookback+1):end]
        expected[asset] = mean(window) * 252  # Annualize
    end

    return expected
end

"""
Compute covariance matrix (annualized) from historical returns.
"""
function compute_covariance_matrix(ds::Dataset, assets_list::Vector{Symbol}, lookback::Int)::Matrix{Float64}
    n_assets = length(assets_list)
    returns_matrix = zeros(lookback, n_assets)

    for (i, asset) in enumerate(assets_list)
        rets = returns(ds, asset)
        window = rets[max(1, end-lookback+1):end]
        if length(window) < lookback
            window = vcat(zeros(lookback - length(window)), window)
        end
        returns_matrix[:, i] = window
    end

    cov_mat = cov(returns_matrix) * 252  # Annualize
    return cov_mat
end

"""
Calculate portfolio Sharpe ratio using Markowitz framework.
"""
function markowitz_sharpe(
    weights::Vector{Float64},
    expected_returns::Vector{Float64},
    cov_matrix::Matrix{Float64};
    risk_free_rate::Float64=0.02
)::Float64
    # Portfolio expected return
    port_return = dot(weights, expected_returns)

    # Portfolio variance
    port_variance = dot(weights, cov_matrix * weights)
    port_std = sqrt(max(port_variance, 1e-10))

    # Sharpe ratio
    sharpe = (port_return - risk_free_rate) / port_std

    return sharpe
end

# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

"""
Normalize weights to sum to 1.
"""
function normalize_weights(w::Vector{Float64})::Vector{Float64}
    # Force non-negative (long-only)
    w = max.(w, 0.0)
    s = sum(w)
    return s > 0 ? w / s : ones(length(w)) / length(w)
end

"""
Genetic algorithm to optimize portfolio weights using Markowitz fitness.
"""
function genetic_algorithm_markowitz(
    expected_returns::Vector{Float64},
    cov_matrix::Matrix{Float64},
    n_assets::Int;
    population_size::Int=80,
    n_generations::Int=150,
    mutation_rate::Float64=0.15,
    crossover_rate::Float64=0.7,
    elite_fraction::Float64=0.1,
    max_weight::Float64=0.25,
    min_weight::Float64=0.02
)::Vector{Float64}

    # Initialize population
    population = Vector{Vector{Float64}}(undef, population_size)
    for i in 1:population_size
        # Random weights with preference for diversification
        w = rand(n_assets)
        w = normalize_weights(w)
        # Apply constraints
        w = clamp.(w, min_weight, max_weight)
        w = normalize_weights(w)
        population[i] = w
    end

    # Fitness function: Markowitz Sharpe ratio
    fitness_fn(w) = markowitz_sharpe(w, expected_returns, cov_matrix)

    n_elite = max(1, Int(round(elite_fraction * population_size)))

    for gen in 1:n_generations
        # Evaluate fitness
        fitnesses = [fitness_fn(individual) for individual in population]

        # Sort by fitness (descending)
        sorted_indices = sortperm(fitnesses, rev=true)
        population = population[sorted_indices]
        fitnesses = fitnesses[sorted_indices]

        # Elitism: keep top performers
        new_population = population[1:n_elite]

        # Generate offspring
        while length(new_population) < population_size
            # Tournament selection
            parent1 = population[rand(1:min(20, population_size))]
            parent2 = population[rand(1:min(20, population_size))]

            # Crossover
            if rand() < crossover_rate
                # Uniform crossover
                mask = rand(Bool, n_assets)
                child = [mask[i] ? parent1[i] : parent2[i] for i in 1:n_assets]
            else
                child = copy(parent1)
            end

            # Mutation
            if rand() < mutation_rate
                # Gaussian mutation
                mutation = randn(n_assets) * 0.1
                child = child + mutation
            end

            # Apply constraints and normalize
            child = clamp.(child, min_weight, max_weight)
            child = normalize_weights(child)

            push!(new_population, child)
        end

        population = new_population[1:population_size]

        # Print progress every 30 generations
        if gen % 30 == 0
            best_sharpe = fitnesses[1]
            println("  Gen $gen: Best Sharpe = $(round(best_sharpe, digits=3))")
        end
    end

    # Return best individual
    fitnesses = [fitness_fn(individual) for individual in population]
    best_idx = argmax(fitnesses)
    best_weights = population[best_idx]

    println("  Optimization complete!")
    println("  Best Sharpe: $(round(fitnesses[best_idx], digits=3))")

    return best_weights
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{GeneticMarkowitzParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{GeneticMarkowitzParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Genetic-Markowitz Hybrid Optimizer...")

    assets_list = collect(assets(ds_train))
    n_assets = length(assets_list)

    println("  Optimizing portfolio with $(n_assets) assets")
    println("  Using Genetic Algorithm with Markowitz fitness function")

    # Use 60-day lookback
    lookback = min(60, nrow(ds_train))

    # Estimate expected returns and covariance from training data
    println("  Estimating expected returns & covariance ($(lookback)-day lookback)...")
    expected_rets = compute_expected_returns(ds_train, assets_list, lookback)
    cov_mat = compute_covariance_matrix(ds_train, assets_list, lookback)

    # Convert to vectors for optimization
    mu = [expected_rets[a] for a in assets_list]

    println("  Running genetic algorithm (80 population × 150 generations)...")

    # Run GA with Markowitz-based fitness
    optimal_w = genetic_algorithm_markowitz(
        mu, cov_mat, n_assets;
        population_size=80,
        n_generations=150,
        max_weight=0.25,  # Max 25% per asset
        min_weight=0.02   # Min 2% for diversification
    )

    # Convert to dictionary
    optimal_weights = Dict{Symbol, Float64}()
    for (i, asset) in enumerate(assets_list)
        optimal_weights[asset] = optimal_w[i]
    end

    # Display top holdings
    println("  Optimal portfolio weights:")
    sorted_weights = sort(collect(optimal_weights), by=x->x[2], rev=true)
    for (i, (asset, weight)) in enumerate(sorted_weights)
        if weight > 0.02  # Only show weights > 2%
            println("    $(asset): $(round(100*weight, digits=1))%")
        end
        if i >= 10  # Show top 10
            break
        end
    end

    # Validation metrics
    println("  Validating on validation set...")
    val_expected = compute_expected_returns(ds_val, assets_list, min(60, nrow(ds_val)))
    portfolio_return = sum(optimal_weights[a] * val_expected[a] for a in assets_list)
    println("  Expected portfolio return: $(round(100*portfolio_return, digits=1))%")

    # Initialize with quarterly rebalancing
    rebalance_period = 63  # Quarterly

    params = GeneticMarkowitzParams(
        optimal_weights, expected_rets, cov_mat, assets_list, lookback,
        rebalance_period, Ref(0), Ref(optimal_weights)
    )

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{GeneticMarkowitzParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Need enough data
    if nrow(ds) < params.lookback
        n = length(params.asset_list)
        return Dict(a => 1.0/n for a in assets(ds))
    end

    current_day = nrow(ds)

    # Only rebalance quarterly (every 63 trading days)
    if current_day - params.last_rebalance[] >= params.rebalance_period
        # Recompute expected returns and covariance from recent data
        expected_rets = compute_expected_returns(ds, params.asset_list, params.lookback)
        cov_mat = compute_covariance_matrix(ds, params.asset_list, params.lookback)

        # Convert to vectors
        mu = [expected_rets[a] for a in params.asset_list]

        # Reoptimize with GA (lighter version for quarterly rebalance)
        optimal_w = genetic_algorithm_markowitz(
            mu, cov_mat, length(params.asset_list);
            population_size=40,
            n_generations=50,  # Fewer generations for speed
            max_weight=0.25,
            min_weight=0.02
        )

        # Convert to dictionary
        new_weights = Dict{Symbol, Float64}()
        for (i, asset) in enumerate(params.asset_list)
            new_weights[asset] = optimal_w[i]
        end

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

    println("Genetic-Markowitz Hybrid Demo")
    println("="^60)

    # Create synthetic data for 4 assets with different profiles
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    prices_data = []
    for i in 1:4
        prices = Float64[100.0]
        for j in 2:n
            # Different risk/return profiles
            drift = 0.0005 * i
            vol = 0.01 + 0.005 * i
            ret = drift + vol * randn()
            push!(prices, prices[end] * (1 + ret))
        end
        push!(prices_data, prices)
    end

    dfs = [DataFrame(timestamp=dates, close=p) for p in prices_data]
    ds = Dataset(Dict(:A1 => dfs[1], :A2 => dfs[2], :A3 => dfs[3], :A4 => dfs[4]))

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(GeneticMarkowitzParams, train, val)
    s = Strategy(m)

    println("\nBacktesting Genetic-Markowitz portfolio...")
    println(backtest(test, s))
end

# demo()

export GeneticMarkowitzParams, train__, signal, demo

end # module
