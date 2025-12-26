"""
Example: HMM Regime Strategy
============================
Single-asset HMM regime detection using unified Dataset API.
Includes from-scratch HMM implementation (no external deps).
"""

module HMMStrategy

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: backtest, assets, nrow, returns, split
import ..Blab: signal, train__
using Statistics
using LinearAlgebra
using Random
using Dates
using DataFrames

# =============================================================================
# HMM IMPLEMENTATION
# =============================================================================

gaussian_pdf(x, μ, σ) = exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))
gaussian_logpdf(x, μ, σ) = -0.5 * ((x - μ) / σ)^2 - log(σ) - 0.5 * log(2π)

struct GaussianHMM
    K::Int
    π::Vector{Float64}
    A::Matrix{Float64}
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function forward(hmm::GaussianHMM, obs::Vector{Float64})
    T, K = length(obs), hmm.K
    α = zeros(T, K)
    for k in 1:K
        α[1, k] = hmm.π[k] * gaussian_pdf(obs[1], hmm.μ[k], hmm.σ[k])
    end
    α[1, :] ./= sum(α[1, :])
    for t in 2:T
        for j in 1:K
            α[t, j] = sum(α[t-1, i] * hmm.A[i, j] for i in 1:K) * gaussian_pdf(obs[t], hmm.μ[j], hmm.σ[j])
        end
        α[t, :] ./= sum(α[t, :])
    end
    α
end

function backward(hmm::GaussianHMM, obs::Vector{Float64})
    T, K = length(obs), hmm.K
    β = zeros(T, K)
    β[T, :] .= 1.0
    for t in (T-1):-1:1
        for i in 1:K
            β[t, i] = sum(hmm.A[i, j] * gaussian_pdf(obs[t+1], hmm.μ[j], hmm.σ[j]) * β[t+1, j] for j in 1:K)
        end
        β[t, :] ./= sum(β[t, :])
    end
    β
end

function viterbi(hmm::GaussianHMM, obs::Vector{Float64})
    T, K = length(obs), hmm.K
    log_δ = zeros(T, K)
    ψ = zeros(Int, T, K)
    for k in 1:K
        log_δ[1, k] = log(hmm.π[k]) + gaussian_logpdf(obs[1], hmm.μ[k], hmm.σ[k])
    end
    for t in 2:T
        for j in 1:K
            candidates = [log_δ[t-1, i] + log(hmm.A[i, j]) for i in 1:K]
            log_δ[t, j] = maximum(candidates) + gaussian_logpdf(obs[t], hmm.μ[j], hmm.σ[j])
            ψ[t, j] = argmax(candidates)
        end
    end
    states = zeros(Int, T)
    states[T] = argmax(log_δ[T, :])
    for t in (T-1):-1:1
        states[t] = ψ[t+1, states[t+1]]
    end
    states
end

function fit_hmm(obs::Vector{Float64}, K::Int=2; maxiter::Int=100)
    T = length(obs)
    sorted_obs = sort(obs)
    chunk = T ÷ K
    
    μ = [mean(sorted_obs[(i-1)*chunk+1:i*chunk]) for i in 1:K]
    σ = max.([std(sorted_obs[(i-1)*chunk+1:i*chunk]) for i in 1:K], 1e-6)
    π = fill(1.0/K, K)
    A = [i == j ? 0.95 : 0.05/(K-1) for i in 1:K, j in 1:K]
    
    hmm = GaussianHMM(K, π, A, μ, σ)
    
    for _ in 1:maxiter
        α, β = forward(hmm, obs), backward(hmm, obs)
        γ = α .* β
        γ ./= sum(γ, dims=2)
        
        ξ = zeros(T-1, K, K)
        for t in 1:(T-1), i in 1:K, j in 1:K
            ξ[t, i, j] = α[t, i] * hmm.A[i, j] * gaussian_pdf(obs[t+1], hmm.μ[j], hmm.σ[j]) * β[t+1, j]
        end
        for t in 1:(T-1)
            ξ[t, :, :] ./= sum(ξ[t, :, :])
        end
        
        π_new = γ[1, :]
        A_new = [sum(ξ[:, i, j]) / sum(γ[1:T-1, i]) for i in 1:K, j in 1:K]
        A_new ./= sum(A_new, dims=2)
        μ_new = [sum(γ[:, k] .* obs) / sum(γ[:, k]) for k in 1:K]
        σ_new = max.([sqrt(sum(γ[:, k] .* (obs .- μ_new[k]).^2) / sum(γ[:, k])) for k in 1:K], 1e-6)
        
        hmm = GaussianHMM(K, π_new, A_new, μ_new, σ_new)
    end
    hmm
end

# =============================================================================
# USER DEFINES: Model params
# =============================================================================

struct HMMParams
    hmm::GaussianHMM
end

# =============================================================================
# USER IMPLEMENTS: train__()
# =============================================================================

function train__(::Type{HMMParams}, ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{HMMParams}
    @assert ds_train.uid == ds_val.uid

    asset = first(assets(ds_train))
    rets = returns(ds_train, asset)
    hmm = fit_hmm(rets, 2; maxiter=100)

    println("Fitted HMM:")
    println("  State 1: μ=$(round(hmm.μ[1]*100, digits=3))%, σ=$(round(hmm.σ[1]*100, digits=3))%")
    println("  State 2: μ=$(round(hmm.μ[2]*100, digits=3))%, σ=$(round(hmm.σ[2]*100, digits=3))%")

    Model(ds_train, ds_val, HMMParams(hmm))
end

# =============================================================================
# USER IMPLEMENTS: signal() -> Dict{Symbol, Float64}
# =============================================================================

function signal(ds::Dataset{Test}, strategy::Strategy{HMMParams})::Dict{Symbol, Float64}
    asset = first(assets(ds))
    n = nrow(ds)
    
    if n < 20
        return Dict(asset => 0.0)
    end
    
    rets = returns(ds, asset)
    hmm = strategy.model.params.hmm
    
    states = viterbi(hmm, rets)
    current_regime = states[end]
    bull_regime = argmax(hmm.μ)
    
    weight = current_regime == bull_regime ? 1.0 : 0.0
    Dict(asset => weight)
end

# =============================================================================
# DEMO
# =============================================================================

function demo()
    Random.seed!(42)
    
    dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
    n = length(dates)
    
    # Generate regime-switching prices
    prices = Float64[100.0]
    regime = 1
    for _ in 2:n
        rand() < 0.02 && (regime = 3 - regime)
        ret = regime == 1 ? 0.001 + 0.01*randn() : -0.0005 + 0.02*randn()
        push!(prices, prices[end] * (1 + ret))
    end
    
    ds = Dataset(:SPY => DataFrame(timestamp=dates, close=prices))
    train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))
    
    println("Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

    m = train__(HMMParams, train, val)
    s = Strategy(m)
    println(backtest(test, s))
end

# demo()

export HMMParams, GaussianHMM, train__, signal, demo
export fit_hmm, forward, backward, viterbi, gaussian_pdf, gaussian_logpdf

end # module
