"""
Genetic Algorithm Portfolio Optimization
=========================================
Uses genetic algorithms to find optimal portfolio weights that maximize
risk-adjusted returns (Sharpe ratio) on the validation set.

The GA evolves a population of candidate portfolios, selecting those with
higher Sharpe ratios and combining them to find the optimal allocation.

Constraints:
- Long-only (weights >= 0)
- Fully invested (weights sum to 1)
- Rebalances monthly to maintain target weights
"""

module GeneticPortfolioStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns, timestamps
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct GeneticPortfolioParams
    weights::Dict{Symbol, Float64}  # Optimized weights for each asset
    rebalance_period::Int            # Days between rebalancing
end

# =============================================================================
# GENETIC ALGORITHM IMPLEMENTATION
# =============================================================================

"""
Simple genetic algorithm for portfolio optimization.
"""
function genetic_algorithm(
    fitness_fn::Function,
    n_assets::Int;
    population_size::Int=50,
    n_generations::Int=100,
    mutation_rate::Float64=0.1,
    crossover_rate::Float64=0.7,
    elite_fraction::Float64=0.1
)
    # Initialize population with random normalized weights
    population = [normalize_weights(rand(n_assets)) for _ in 1:population_size]

    best_individual = nothing
    best_fitness = -Inf

    for gen in 1:n_generations
        # Evaluate fitness for all individuals
        fitnesses = [fitness_fn(individual) for individual in population]

        # Track best individual
        max_idx = argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness
            best_fitness = fitnesses[max_idx]
            best_individual = copy(population[max_idx])
        end

        # Selection: tournament selection
        n_elite = max(1, Int(round(elite_fraction * population_size)))
        sorted_indices = sortperm(fitnesses, rev=true)
        elite = [population[i] for i in sorted_indices[1:n_elite]]

        # Create next generation
        next_population = copy(elite)

        while length(next_population) < population_size
            # Select parents
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)

            # Crossover
            if rand() < crossover_rate
                child = crossover(parent1, parent2)
            else
                child = copy(parent1)
            end

            # Mutation
            if rand() < mutation_rate
                child = mutate(child)
            end

            # Normalize to ensure valid weights
            child = normalize_weights(child)

            push!(next_population, child)
        end

        population = next_population

        if gen % 20 == 0
            println("  Gen $gen: Best Sharpe = $(round(best_fitness, digits=3))")
        end
    end

    return best_individual, best_fitness
end

"""Normalize weights to sum to 1 and be non-negative."""
function normalize_weights(weights::Vector{Float64})::Vector{Float64}
    # Make non-negative
    weights = max.(weights, 0.0)

    # Normalize to sum to 1
    total = sum(weights)
    if total > 0
        return weights ./ total
    else
        # If all zero, return equal weights
        return fill(1.0 / length(weights), length(weights))
    end
end

"""Tournament selection: pick best of k random individuals."""
function tournament_select(population::Vector{Vector{Float64}}, fitnesses::Vector{Float64}; k::Int=3)
    indices = rand(1:length(population), k)
    tournament_fitnesses = fitnesses[indices]
    best_idx = indices[argmax(tournament_fitnesses)]
    return population[best_idx]
end

"""Single-point crossover."""
function crossover(parent1::Vector{Float64}, parent2::Vector{Float64})::Vector{Float64}
    n = length(parent1)
    point = rand(1:n-1)
    child = vcat(parent1[1:point], parent2[point+1:end])
    return child
end

"""Gaussian mutation."""
function mutate(individual::Vector{Float64}; mutation_strength::Float64=0.1)::Vector{Float64}
    n = length(individual)
    # Mutate random subset of genes
    for i in 1:n
        if rand() < 0.3  # 30% chance to mutate each gene
            individual[i] += mutation_strength * randn()
        end
    end
    return individual
end

# =============================================================================
# FITNESS FUNCTION
# =============================================================================

"""
Calculate Sharpe ratio for a portfolio with given weights.
"""
function portfolio_sharpe(
    weights::Vector{Float64},
    returns_matrix::Matrix{Float64}  # rows = time, cols = assets
)::Float64
    if size(returns_matrix, 1) < 2
        return -Inf
    end

    # Calculate portfolio returns
    portfolio_returns = returns_matrix * weights

    # Calculate Sharpe ratio (annualized, assuming daily data)
    mean_return = mean(portfolio_returns)
    std_return = std(portfolio_returns)

    if std_return < 1e-10
        return -Inf
    end

    # Annualize (252 trading days)
    sharpe = (mean_return * 252) / (std_return * sqrt(252))

    return sharpe
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{GeneticPortfolioParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{GeneticPortfolioParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Genetic Algorithm Portfolio Optimizer...")

    asset_list = collect(assets(ds_val))
    n_assets = length(asset_list)

    if n_assets < 2
        error("Genetic portfolio requires at least 2 assets")
    end

    # Build returns matrix for validation set (rows = time, cols = assets)
    returns_matrix = zeros(nrow(ds_val) - 1, n_assets)

    for (i, asset) in enumerate(asset_list)
        asset_returns = returns(ds_val, asset)
        returns_matrix[:, i] = asset_returns
    end

    # Define fitness function
    fitness_fn = (weights::Vector{Float64}) -> portfolio_sharpe(weights, returns_matrix)

    # Run genetic algorithm
    println("  Optimizing portfolio with $(n_assets) assets...")
    println("  Running genetic algorithm (50 population × 100 generations)...")

    best_weights, best_sharpe = genetic_algorithm(
        fitness_fn,
        n_assets;
        population_size=50,
        n_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elite_fraction=0.1
    )

    println("  Optimization complete!")
    println("  Best validation Sharpe: $(round(best_sharpe, digits=3))")

    # Convert to Dict
    weights_dict = Dict(asset_list[i] => best_weights[i] for i in 1:n_assets)

    # Print top holdings
    println("  Top holdings:")
    sorted_weights = sort(collect(weights_dict), by=x->x[2], rev=true)
    for (asset, weight) in sorted_weights[1:min(5, length(sorted_weights))]
        if weight > 0.01  # Only show weights > 1%
            println("    $(asset): $(round(weight * 100, digits=1))%")
        end
    end

    params = GeneticPortfolioParams(weights_dict, 21)  # Rebalance monthly (21 trading days)

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{GeneticPortfolioParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Return the optimized weights
    # In a real implementation, could add rebalancing logic based on rebalance_period
    return params.weights
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Genetic Algorithm Portfolio Optimization Demo")
    println("="^60)

    # Create synthetic multi-asset data
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Simulate 5 assets with different characteristics
    assets_data = Dict{Symbol, DataFrame}()

    for (i, asset) in enumerate([:AAPL, :MSFT, :GOOGL, :AMZN, :NVDA])
        close_prices = Float64[100.0]
        drift = 0.0003 + 0.0002 * i  # Different expected returns
        vol = 0.01 + 0.005 * (i % 3)  # Different volatilities

        for j in 2:n
            ret = drift + vol * randn()
            push!(close_prices, close_prices[end] * (1 + ret))
        end

        assets_data[asset] = DataFrame(timestamp=dates, close=close_prices)
    end

    ds = Dataset(assets_data)

    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(GeneticPortfolioParams, train, val)
    s = Strategy(m)

    println("\nBacktesting optimized portfolio...")
    println(backtest(test, s))
end

# demo()

export GeneticPortfolioParams, train__, signal, demo

end # module
