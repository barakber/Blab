"""
Genetic Portfolio with Regime-Based Risk Management
===================================================
Combines genetic algorithm portfolio optimization with HMM regime detection
to reduce drawdowns while maintaining strong returns.

Strategy:
1. Use GA to find optimal portfolio weights (maximize Sharpe on validation)
2. Use HMM to detect market regime (bull/bear) from market index (SPY)
3. Modulate exposure based on regime:
   - Bull regime: 100% of optimized portfolio
   - Bear regime: 20% of optimized portfolio (defensive)

This approach aims to capture upside in favorable markets while reducing
exposure during unfavorable regimes to minimize drawdowns.
"""

module GeneticRegimeStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, prices, returns, timestamps
import ..Blab: signal, train__
using Statistics
using Random
using Dates
using DataFrames

# Import HMM utilities from HMM strategy module (already loaded in Blab)
using ..HMMStrategy: GaussianHMM, fit_hmm, viterbi

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct GeneticRegimeParams
    weights::Dict{Symbol, Float64}      # Optimized portfolio weights from GA
    hmm::GaussianHMM                     # HMM for regime detection
    regime_asset::Symbol                 # Asset to use for regime detection (e.g., :SPY)
    bull_exposure::Float64               # Exposure in bull regime (e.g., 1.0 = 100%)
    bear_exposure::Float64               # Exposure in bear regime (e.g., 0.2 = 20%)
    rebalance_period::Int                # Days between portfolio rebalancing
    last_rebalance::Ref{Int}
    current_weights::Ref{Dict{Symbol, Float64}}
end

# =============================================================================
# GENETIC ALGORITHM (same as GeneticPortfolio)
# =============================================================================

"""Normalize weights to sum to 1 and be non-negative."""
function normalize_weights(weights::Vector{Float64})::Vector{Float64}
    weights = max.(weights, 0.0)
    total = sum(weights)
    if total > 0
        return weights ./ total
    else
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
    for i in 1:n
        if rand() < 0.3  # 30% chance to mutate each gene
            individual[i] += mutation_strength * randn()
        end
    end
    return individual
end

"""Calculate Sharpe ratio for a portfolio."""
function portfolio_sharpe(
    weights::Vector{Float64},
    returns_matrix::Matrix{Float64}  # rows = time, cols = assets
)::Float64
    if size(returns_matrix, 1) < 2
        return -Inf
    end

    portfolio_returns = returns_matrix * weights
    mean_return = mean(portfolio_returns)
    std_return = std(portfolio_returns)

    if std_return < 1e-10
        return -Inf
    end

    # Annualize (252 trading days)
    sharpe = (mean_return * 252) / (std_return * sqrt(252))
    return sharpe
end

"""Simple genetic algorithm for portfolio optimization."""
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

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{GeneticRegimeParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{GeneticRegimeParams}
    @assert ds_train.uid == ds_val.uid

    println("Training Genetic-Regime Strategy (GA Portfolio + HMM Risk Management)...")

    asset_list = collect(assets(ds_val))
    n_assets = length(asset_list)

    if n_assets < 2
        error("Genetic-Regime requires at least 2 assets")
    end

    # Determine regime indicator asset (prefer SPY if available, else use first asset)
    regime_asset = :SPY in asset_list ? :SPY : first(asset_list)
    println("  Using $(regime_asset) for regime detection")

    # Step 1: Fit HMM on training data for regime detection
    println("  Fitting HMM (2 regimes) on training data...")
    regime_returns = returns(ds_train, regime_asset)
    hmm = fit_hmm(regime_returns, 2; maxiter=100)

    # Identify bull vs bear regime (higher mean = bull)
    if hmm.μ[1] > hmm.μ[2]
        bull_regime = 1
        bear_regime = 2
    else
        bull_regime = 2
        bear_regime = 1
    end

    println("  HMM trained: Bull regime ($(bull_regime)) μ=$(round(hmm.μ[bull_regime]*100, digits=2))%, Bear regime ($(bear_regime)) μ=$(round(hmm.μ[bear_regime]*100, digits=2))%")

    # Step 2: Build returns matrix for GA optimization on validation set
    println("  Building returns matrix for portfolio optimization...")
    returns_matrix = zeros(nrow(ds_val) - 1, n_assets)

    for (i, asset) in enumerate(asset_list)
        asset_returns = returns(ds_val, asset)
        returns_matrix[:, i] = asset_returns
    end

    # Step 3: Run genetic algorithm to find optimal portfolio weights
    fitness_fn = (weights::Vector{Float64}) -> portfolio_sharpe(weights, returns_matrix)

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

    println("  GA optimization complete!")
    println("  Best validation Sharpe: $(round(best_sharpe, digits=3))")

    # Convert to Dict
    weights_dict = Dict(asset_list[i] => best_weights[i] for i in 1:n_assets)

    # Print top holdings
    println("  Top holdings (full exposure):")
    sorted_weights = sort(collect(weights_dict), by=x->x[2], rev=true)
    for (asset, weight) in sorted_weights[1:min(5, length(sorted_weights))]
        if weight > 0.01
            println("    $(asset): $(round(weight * 100, digits=1))%")
        end
    end

    # Set exposure levels based on regime
    bull_exposure = 1.0   # 100% exposure in bull regime
    bear_exposure = 0.2   # 20% exposure in bear regime (defensive)

    println("  Regime-based exposure:")
    println("    Bull regime: $(Int(bull_exposure*100))%")
    println("    Bear regime: $(Int(bear_exposure*100))%")

    params = GeneticRegimeParams(
        weights_dict, hmm, regime_asset,
        bull_exposure, bear_exposure,
        63,  # Quarterly rebalancing
        Ref(0), Ref(weights_dict)
    )

    Model(ds_train, ds_val, params)
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{GeneticRegimeParams})::Dict{Symbol, Float64}
    params = strategy.model.params

    # Need enough data for regime detection
    if nrow(ds) < 20
        n = length(params.current_weights[])
        return Dict(a => 1.0/n for a in assets(ds))
    end

    current_day = nrow(ds)

    # Rebalance portfolio weights quarterly (optional - could keep static)
    if current_day - params.last_rebalance[] >= params.rebalance_period
        # For now, keep weights static (don't reoptimize during test)
        # Could add re-optimization here if desired
        params.last_rebalance[] = current_day
    end

    # Detect current regime using recent returns
    regime_returns = returns(ds, params.regime_asset)
    lookback = min(20, length(regime_returns))  # Use last 20 days for regime detection
    recent_returns = regime_returns[max(1, end-lookback+1):end]

    # Predict regime using Viterbi (returns most likely state sequence)
    regime_sequence = viterbi(params.hmm, recent_returns)
    current_regime = regime_sequence[end]  # Get current (last) regime

    # Determine exposure based on regime
    # Identify which regime is bull (higher mean)
    bull_regime = params.hmm.μ[1] > params.hmm.μ[2] ? 1 : 2
    exposure = (current_regime == bull_regime) ? params.bull_exposure : params.bear_exposure

    # Scale weights by exposure
    scaled_weights = Dict{Symbol, Float64}()
    for (asset, weight) in params.current_weights[]
        if asset in assets(ds)
            scaled_weights[asset] = weight * exposure
        end
    end

    # If reduced exposure, remainder goes to cash (represented by reducing all weights proportionally)
    # The backtest engine will interpret weights < 1.0 total as partial cash position

    return scaled_weights
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)

    println("Genetic-Regime Strategy Demo")
    println("="^60)

    # Create synthetic multi-asset data with regime changes
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)

    # Simulate regime changes in SPY
    spy_prices = Float64[100.0]
    regime = 1  # Start in bull
    for j in 2:n
        # Switch regime occasionally
        if rand() < 0.01
            regime = 3 - regime  # Toggle between 1 and 2
        end

        drift = regime == 1 ? 0.0008 : -0.0003  # Bull vs bear
        vol = 0.015
        ret = drift + vol * randn()
        push!(spy_prices, spy_prices[end] * (1 + ret))
    end

    # Other assets correlated with SPY
    assets_data = Dict{Symbol, DataFrame}(:SPY => DataFrame(timestamp=dates, close=spy_prices))

    for (i, asset) in enumerate([:AAPL, :MSFT, :GOOGL, :AMZN])
        close_prices = Float64[100.0]
        for j in 2:n
            spy_ret = (spy_prices[j] - spy_prices[j-1]) / spy_prices[j-1]
            own_noise = 0.01 * randn()
            ret = 0.7 * spy_ret + own_noise  # Correlated with SPY
            push!(close_prices, close_prices[end] * (1 + ret))
        end
        assets_data[asset] = DataFrame(timestamp=dates, close=close_prices)
    end

    ds = Dataset(assets_data)
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

    println("Assets: $(assets(train))")
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(GeneticRegimeParams, train, val)
    s = Strategy(m)

    println("\nBacktesting Genetic-Regime portfolio...")
    println(backtest(test, s))
end

# demo()

export GeneticRegimeParams, train__, signal, demo

end # module
