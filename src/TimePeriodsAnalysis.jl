"""
Time Periods Analysis
=====================
Test strategies across multiple market regimes and time periods.
"""

module TimePeriodsAnalysis

using ..Blab: Dataset, Model, Strategy, Train, Validation, Test
using ..Blab: BacktestJob, BacktestResult, backtest_parallel
using ..Blab: assets, split, nrow, load_stocks, load_stock, get_top_sp500_symbols, getdf
using ..Blab.BuyHoldStrategy
using ..Blab.MAStrategy
using ..Blab.MomentumStrategy
using ..Blab.HMMStrategy
using ..Blab.MACDStrategy
using ..Blab.RSIStrategy
using ..Blab.EMAStrategy
using ..Blab.XGBoostMLStrategy
using ..Blab.RegimeSwitchStrategy
using ..Blab.GeneticPortfolioStrategy
using ..Blab.TDAStrategy
using ..Blab.LeverageQQQStrategy
using ..Blab.AdaptiveRegimeStrategy
using ..Blab.MarkowitzStrategy
using ..Blab.GeneticMarkowitzStrategy
using ..Blab.GeneticRegimeStrategy
using ..Blab.InstitutionalGradeStrategy
using Dates
using Printf
using Statistics

# =============================================================================
# MARKET PERIODS DEFINITION
# =============================================================================

"""
Represents a specific market period to test strategies on.
"""
struct MarketPeriod
    name::String
    train_end::DateTime
    val_end::DateTime
    test_end::DateTime
    description::String
end

"""
Define key market periods for backtesting.
"""
function get_market_periods()::Vector{MarketPeriod}
    [
        MarketPeriod(
            "Dotcom_Bubble_2000",
            DateTime("1997-01-01"),
            DateTime("1999-01-01"),
            DateTime("2002-12-01"),
            "Dot-com Bubble Crash (2000-2002)"
        ),
        MarketPeriod(
            "Financial_Crisis_2008",
            DateTime("2005-01-01"),
            DateTime("2007-01-01"),
            DateTime("2009-06-01"),
            "Financial Crisis (2007-2009)"
        ),
        MarketPeriod(
            "2018_Correction",
            DateTime("2017-06-01"),
            DateTime("2018-01-01"),
            DateTime("2019-01-01"),
            "2018 Market Correction (Q4 selloff)"
        ),
        MarketPeriod(
            "COVID_Crash",
            DateTime("2019-01-01"),
            DateTime("2019-10-01"),
            DateTime("2020-06-01"),
            "COVID-19 Market Crash (Feb-Mar 2020)"
        ),
        MarketPeriod(
            "Bull_2020_2021",
            DateTime("2019-07-01"),
            DateTime("2020-04-01"),
            DateTime("2021-12-01"),
            "Bull Market Recovery 2020-2021"
        ),
        MarketPeriod(
            "Rate_Hikes_2022",
            DateTime("2020-12-01"),
            DateTime("2021-09-01"),
            DateTime("2022-12-01"),
            "Fed Rate Hikes & Tech Selloff 2022"
        ),
        MarketPeriod(
            "AI_Boom_2023",
            DateTime("2021-12-01"),
            DateTime("2022-09-01"),
            DateTime("2023-12-01"),
            "AI Boom & Recovery 2023"
        ),
        MarketPeriod(
            "Recent_2024",
            DateTime("2022-12-01"),
            DateTime("2023-09-01"),
            DateTime("2024-12-01"),
            "Recent Market 2024"
        ),
    ]
end

# =============================================================================
# STRATEGY SETUP
# =============================================================================

"""
Create all strategy configurations to test.
Returns tuples of (name, strategy, test_dataset).
"""
function create_strategies(train::Dataset{Train}, val::Dataset{Validation}, test::Dataset{Test})
    strategies = Tuple{String, Strategy, Dataset{Test}}[]

    # Create SPY-only datasets for BuyHold baseline
    if !(:SPY in assets(train))
        error("SPY is required for Buy & Hold baseline strategy")
    end
    train_spy = Dataset(:SPY => train.data[:SPY])
    val_spy = Dataset(:SPY => val.data[:SPY])
    test_spy = Dataset(:SPY => test.data[:SPY])
    # Preserve phantom types
    train_spy = Dataset{Train}(train_spy.data, train_spy.timestamps, train.uid)
    val_spy = Dataset{Validation}(val_spy.data, val_spy.timestamps, val.uid)
    test_spy = Dataset{Test}(test_spy.data, test_spy.timestamps, test.uid)

    # Create single-asset datasets for single-asset strategies (use first asset)
    first_asset = first(assets(train))
    train_single = Dataset(first_asset => train.data[first_asset])
    val_single = Dataset(first_asset => val.data[first_asset])
    test_single = Dataset(first_asset => test.data[first_asset])
    # Preserve phantom types
    train_single = Dataset{Train}(train_single.data, train_single.timestamps, train.uid)
    val_single = Dataset{Validation}(val_single.data, val_single.timestamps, val.uid)
    test_single = Dataset{Test}(test_single.data, test_single.timestamps, test.uid)

    # Buy & Hold SPY baseline
    m = Model(train_spy, val_spy, BuyHoldStrategy.BuyHoldParams(1.0))
    push!(strategies, ("BuyHold_SPY", Strategy(m), test_spy))

    # Moving Average strategies (single-asset)
    for (fast, slow) in [(10, 30), (20, 50)]
        m = Model(train_single, val_single, MAStrategy.MAParams(fast, slow))
        push!(strategies, ("MA_$(fast)_$(slow)", Strategy(m), test_single))
    end

    # EMA strategies (single-asset)
    for (fast, slow) in [(8, 21), (12, 26)]
        m = Model(train_single, val_single, EMAStrategy.EMAParams(fast, slow))
        push!(strategies, ("EMA_$(fast)_$(slow)", Strategy(m), test_single))
    end

    # MACD strategies (single-asset)
    for (fast, slow, sig) in [(12, 26, 9), (8, 17, 9)]
        m = Model(train_single, val_single, MACDStrategy.MACDParams(fast, slow, sig))
        push!(strategies, ("MACD_$(fast)_$(slow)_$(sig)", Strategy(m), test_single))
    end

    # RSI strategies (single-asset)
    for (period, oversold, overbought) in [(14, 30.0, 70.0), (14, 20.0, 80.0)]
        m = Model(train_single, val_single, RSIStrategy.RSIParams(period, oversold, overbought))
        push!(strategies, ("RSI_$(period)_$(Int(oversold))_$(Int(overbought))", Strategy(m), test_single))
    end

    # Momentum Rotation strategies (multi-asset)
    for (lookback, top_n) in [(20, 3), (30, 3), (60, 4)]
        m = Model(train, val, MomentumStrategy.MomentumParams(lookback, top_n))
        push!(strategies, ("Momentum_L$(lookback)_T$(top_n)", Strategy(m), test))
    end

    # HMM Regime strategy (single-asset)
    try
        m = HMMStrategy.train__(HMMStrategy.HMMParams, train_single, val_single)
        push!(strategies, ("HMM_regime", Strategy(m), test_single))
    catch e
        println("  Warning: HMM failed to train: $(typeof(e))")
    end

    # XGBoost ML strategy (single-asset)
    try
        m = XGBoostMLStrategy.train__(XGBoostMLStrategy.XGBoostParams, train_single, val_single)
        push!(strategies, ("XGBoost_ML", Strategy(m), test_single))
    catch e
        println("  Warning: XGBoost failed to train: $(typeof(e))")
    end

    # Regime Switching strategy (single-asset)
    try
        m = RegimeSwitchStrategy.train__(RegimeSwitchStrategy.RegimeSwitchParams, train_single, val_single)
        push!(strategies, ("RegimeSwitch", Strategy(m), test_single))
    catch e
        println("  Warning: RegimeSwitch failed to train: $(typeof(e))")
    end

    # Genetic Algorithm Portfolio Optimization (multi-asset)
    try
        m = GeneticPortfolioStrategy.train__(GeneticPortfolioStrategy.GeneticPortfolioParams, train, val)
        push!(strategies, ("GeneticPortfolio", Strategy(m), test))
    catch e
        println("  Warning: GeneticPortfolio failed to train: $(typeof(e))")
    end

    # TDA (Topological Data Analysis) strategy (single-asset)
    try
        m = TDAStrategy.train__(TDAStrategy.TDAParams, train_single, val_single)
        push!(strategies, ("TDA", Strategy(m), test_single))
    catch e
        println("  Warning: TDA failed to train: $(typeof(e))")
    end

    # Leveraged QQQ (TQQQ) strategy - learns from QQQ, trades TQQQ
    if :QQQ in assets(train) && :TQQQ in assets(train)
        try
            # Create QQQ + TQQQ dataset
            train_qqq = Dataset(Dict(:QQQ => train.data[:QQQ], :TQQQ => train.data[:TQQQ]))
            val_qqq = Dataset(Dict(:QQQ => val.data[:QQQ], :TQQQ => val.data[:TQQQ]))
            test_qqq = Dataset(Dict(:QQQ => test.data[:QQQ], :TQQQ => test.data[:TQQQ]))
            # Preserve phantom types
            train_qqq = Dataset{Train}(train_qqq.data, train_qqq.timestamps, train.uid)
            val_qqq = Dataset{Validation}(val_qqq.data, val_qqq.timestamps, val.uid)
            test_qqq = Dataset{Test}(test_qqq.data, test_qqq.timestamps, test.uid)

            m = LeverageQQQStrategy.train__(LeverageQQQStrategy.LeverageQQQParams, train_qqq, val_qqq)
            push!(strategies, ("LeverageQQQ", Strategy(m), test_qqq))
        catch e
            println("  Warning: LeverageQQQ failed to train: $e")
        end
    else
        println("  Warning: QQQ or TQQQ not available for LeverageQQQ strategy")
    end

    # Adaptive Regime strategy (single-asset)
    try
        m = AdaptiveRegimeStrategy.train__(AdaptiveRegimeStrategy.AdaptiveRegimeParams, train_single, val_single)
        push!(strategies, ("AdaptiveRegime", Strategy(m), test_single))
    catch e
        println("  Warning: AdaptiveRegime failed to train: $(typeof(e))")
    end

    # Markowitz Mean-Variance Optimization (multi-asset)
    try
        m = MarkowitzStrategy.train__(MarkowitzStrategy.MarkowitzParams, train, val)
        push!(strategies, ("Markowitz", Strategy(m), test))
    catch e
        println("  Warning: Markowitz failed to train: $(typeof(e))")
    end

    # Genetic-Markowitz Hybrid (multi-asset)
    try
        m = GeneticMarkowitzStrategy.train__(GeneticMarkowitzStrategy.GeneticMarkowitzParams, train, val)
        push!(strategies, ("GeneticMarkowitz", Strategy(m), test))
    catch e
        println("  Warning: GeneticMarkowitz failed to train: $(typeof(e))")
    end

    # Genetic-Regime (GA Portfolio + HMM Risk Management) (multi-asset)
    try
        m = GeneticRegimeStrategy.train__(GeneticRegimeStrategy.GeneticRegimeParams, train, val)
        push!(strategies, ("GeneticRegime", Strategy(m), test))
    catch e
        println("  Warning: GeneticRegime failed to train: $(typeof(e))")
    end

    # Institutional-Grade Meta Strategy (multi-asset)
    try
        m = InstitutionalGradeStrategy.train__(InstitutionalGradeStrategy.InstitutionalGradeParams, train, val)
        push!(strategies, ("InstitutionalGrade", Strategy(m), test))
    catch e
        println("  Warning: InstitutionalGrade failed to train: $(typeof(e))")
    end

    strategies
end

# =============================================================================
# TIME PERIODS ANALYSIS
# =============================================================================

"""
Run all strategies across all market periods.
"""
function analyze_time_periods(;
    n_stocks::Int=6,
    datasets_dir::String="../datasets"
)
    println("\n" * "="^70)
    println("TIME PERIODS ANALYSIS - ALL STRATEGIES ACROSS MARKET REGIMES")
    println("="^70)
    println("Running on $(Threads.nthreads()) threads\n")

    # Load data - always include SPY for Buy & Hold baseline
    # Get market periods
    periods = get_market_periods()

    println("Testing across $(length(periods)) market periods:")
    for period in periods
        println("  • $(period.name): $(period.description)")
    end
    println()

    # Create jobs for all (strategy, period) combinations
    jobs = BacktestJob[]
    all_results = Dict{String, Vector{BacktestResult}}()

    for period in periods
        println("Setting up period: $(period.name)")

        # Get all requested symbols
        symbols = get_top_sp500_symbols(n_stocks)
        required_etfs = [:SPY, :QQQ, :TQQQ]
        for etf in required_etfs
            if !(etf in symbols)
                symbols = [etf; symbols]
            end
        end

        # Filter symbols to only those with data covering this period
        # Load each stock and check if it has data covering the full period
        valid_symbols = Symbol[]
        for symbol in symbols
            df = load_stock(symbol, datasets_dir)
            if !isnothing(df)
                min_date = minimum(df.timestamp)
                max_date = maximum(df.timestamp)
                # Stock must have:
                # 1. Data early enough for training (before train_end)
                # 2. Data late enough to cover test period (after test_end)
                if min_date <= period.train_end && max_date >= period.test_end
                    push!(valid_symbols, symbol)
                end
            end
        end

        if length(valid_symbols) < 10
            println("  Warning: Only $(length(valid_symbols)) stocks available for $(period.name), skipping...")
            continue
        end

        println("  Using $(length(valid_symbols)) stocks with data for this period")

        # Load only the valid stocks
        period_ds = load_stocks(valid_symbols, datasets_dir)

        # Use type-safe split with train_end and val_end (test goes from val_end to end)
        train, val, test = split(period_ds, period.train_end, period.val_end)

        # Skip if insufficient data
        if nrow(test) < 50
            println("  Warning: Insufficient test data for $(period.name), skipping...")
            continue
        end

        println("  Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")

        # Create strategies for this period (using type-safe train/val/test datasets)
        strategies = create_strategies(train, val, test)
        println("  Created $(length(strategies)) strategies")

        # Create jobs for this period
        for (name, strategy, test_ds) in strategies
            job_name = "$(period.name)__$(name)"
            push!(jobs, BacktestJob(job_name, test_ds, strategy))
        end

        all_results[period.name] = BacktestResult[]
    end

    println("\nRunning $(length(jobs)) backtests in parallel ($(length(periods)) periods × ~18 strategies)...\n")

    # Run all backtests in parallel
    results = backtest_parallel(jobs)

    # Group results by period
    for result in results
        parts = Base.split(result.name, "__")
        if length(parts) == 2
            period_name = parts[1]
            if haskey(all_results, period_name)
                push!(all_results[period_name], BacktestResult(
                    parts[2],  # strategy name
                    result.metrics,
                    result.elapsed
                ))
            end
        end
    end

    # Display results
    display_period_results(all_results, periods)

    # Display comprehensive matrix
    display_complete_matrix(all_results, periods)

    all_results
end

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

"""
Display results organized by time period and strategy.
"""
function display_period_results(all_results::Dict{String, Vector{BacktestResult}}, periods::Vector{MarketPeriod})
    println("\n" * "="^70)
    println("RESULTS BY MARKET PERIOD")
    println("="^70)

    # For each period, show top strategies
    for period in periods
        if !haskey(all_results, period.name) || isempty(all_results[period.name])
            continue
        end

        results = sort(all_results[period.name], by=r -> r.metrics.sharpe, rev=true)

        println("\n📅 $(period.name): $(period.description)")
        println("-"^70)
        @printf("%-25s %10s %10s %10s\n", "Strategy", "PnL%", "Sharpe", "MaxDD%")
        println("-"^70)

        for (i, result) in enumerate(results[1:min(5, length(results))])
            marker = i == 1 ? "🏆" : "  "
            @printf("%s %-23s %9.2f%% %10.2f %9.2f%%\n",
                marker, result.name, result.metrics.pnl * 100, result.metrics.sharpe, result.metrics.max_drawdown * 100)
        end
    end

    # Cross-period summary
    println("\n" * "="^70)
    println("STRATEGY PERFORMANCE ACROSS ALL PERIODS")
    println("="^70)

    # Aggregate by strategy
    strategy_stats = Dict{String, Vector{Float64}}()  # strategy => [sharpes...]

    for (period_name, results) in all_results
        for result in results
            if !haskey(strategy_stats, result.name)
                strategy_stats[result.name] = Float64[]
            end
            push!(strategy_stats[result.name], result.metrics.sharpe)
        end
    end

    # Calculate average Sharpe across periods
    strategy_avg = [(name, mean(sharpes), std(sharpes), length(sharpes))
                    for (name, sharpes) in strategy_stats]
    sort!(strategy_avg, by=x -> x[2], rev=true)

    println()
    @printf("%-25s %12s %12s %10s\n", "Strategy", "Avg Sharpe", "Std Dev", "Periods")
    println("-"^70)

    for (i, (name, avg_sharpe, std_sharpe, n_periods)) in enumerate(strategy_avg)
        marker = i <= 3 ? "⭐" : "  "
        @printf("%s %-23s %12.2f %12.2f %10d\n",
            marker, name, avg_sharpe, std_sharpe, n_periods)
    end

    println("\n" * "="^70)
end

"""
Display comprehensive matrix of all strategies across all periods.
"""
function display_complete_matrix(all_results::Dict{String, Vector{BacktestResult}}, periods::Vector{MarketPeriod})
    println("\n" * "="^140)
    println("COMPREHENSIVE RESULTS: ALL STRATEGIES × ALL PERIODS")
    println("="^140)

    # Collect all unique strategy names
    all_strategy_names = Set{String}()
    for (period_name, results) in all_results
        for result in results
            push!(all_strategy_names, result.name)
        end
    end
    strategy_names = sort(collect(all_strategy_names))

    # Build matrix: strategy -> period -> metrics
    matrix = Dict{String, Dict{String, Union{BacktestResult, Nothing}}}()
    for strategy in strategy_names
        matrix[strategy] = Dict{String, Union{BacktestResult, Nothing}}()
        for period in periods
            matrix[strategy][period.name] = nothing
        end
    end

    # Fill matrix
    for (period_name, results) in all_results
        for result in results
            if haskey(matrix, result.name)
                matrix[result.name][period_name] = result
            end
        end
    end

    # Display Sharpe Ratio Table
    println("\n📊 SHARPE RATIO BY PERIOD")
    println("-"^140)

    # Header
    @printf("%-20s", "Strategy")
    for period in periods
        @printf(" │ %12s", period.name[1:min(12, length(period.name))])
    end
    @printf(" │ %8s %8s\n", "Avg", "StdDev")
    println("-"^140)

    # Data rows
    for strategy in strategy_names
        @printf("%-20s", strategy[1:min(20, length(strategy))])

        sharpes = Float64[]
        for period in periods
            result = matrix[strategy][period.name]
            if !isnothing(result)
                sharpe = result.metrics.sharpe
                push!(sharpes, sharpe)

                # Color code: green if > 1.0, yellow if > 0.5, red if < 0
                if sharpe >= 1.0
                    @printf(" │ \033[32m%12.2f\033[0m", sharpe)
                elseif sharpe >= 0.5
                    @printf(" │ \033[33m%12.2f\033[0m", sharpe)
                elseif sharpe >= 0.0
                    @printf(" │ %12.2f", sharpe)
                else
                    @printf(" │ \033[31m%12.2f\033[0m", sharpe)
                end
            else
                @printf(" │ %12s", "-")
            end
        end

        if !isempty(sharpes)
            @printf(" │ %8.2f %8.2f\n", mean(sharpes), std(sharpes))
        else
            @printf(" │ %8s %8s\n", "-", "-")
        end
    end

    # Display PnL Table
    println("\n💰 PROFIT & LOSS (%) BY PERIOD")
    println("-"^140)

    # Header
    @printf("%-20s", "Strategy")
    for period in periods
        @printf(" │ %12s", period.name[1:min(12, length(period.name))])
    end
    @printf(" │ %8s %8s\n", "Avg", "StdDev")
    println("-"^140)

    # Data rows
    for strategy in strategy_names
        @printf("%-20s", strategy[1:min(20, length(strategy))])

        pnls = Float64[]
        for period in periods
            result = matrix[strategy][period.name]
            if !isnothing(result)
                pnl = result.metrics.pnl * 100
                push!(pnls, pnl)

                # Color code: green if > 20%, yellow if > 0%, red if < 0%
                if pnl >= 20.0
                    @printf(" │ \033[32m%11.1f%%\033[0m", pnl)
                elseif pnl >= 0.0
                    @printf(" │ \033[33m%11.1f%%\033[0m", pnl)
                else
                    @printf(" │ \033[31m%11.1f%%\033[0m", pnl)
                end
            else
                @printf(" │ %12s", "-")
            end
        end

        if !isempty(pnls)
            @printf(" │ %7.1f%% %7.1f%%\n", mean(pnls), std(pnls))
        else
            @printf(" │ %8s %8s\n", "-", "-")
        end
    end

    # Display Max Drawdown Table
    println("\n📉 MAX DRAWDOWN (%) BY PERIOD")
    println("-"^140)

    # Header
    @printf("%-20s", "Strategy")
    for period in periods
        @printf(" │ %12s", period.name[1:min(12, length(period.name))])
    end
    @printf(" │ %8s %8s\n", "Avg", "StdDev")
    println("-"^140)

    # Data rows
    for strategy in strategy_names
        @printf("%-20s", strategy[1:min(20, length(strategy))])

        drawdowns = Float64[]
        for period in periods
            result = matrix[strategy][period.name]
            if !isnothing(result)
                dd = result.metrics.max_drawdown * 100
                push!(drawdowns, dd)

                # Color code: green if < 10%, yellow if < 20%, red if >= 20%
                if dd < 10.0
                    @printf(" │ \033[32m%11.1f%%\033[0m", dd)
                elseif dd < 20.0
                    @printf(" │ \033[33m%11.1f%%\033[0m", dd)
                else
                    @printf(" │ \033[31m%11.1f%%\033[0m", dd)
                end
            else
                @printf(" │ %12s", "-")
            end
        end

        if !isempty(drawdowns)
            @printf(" │ %7.1f%% %7.1f%%\n", mean(drawdowns), std(drawdowns))
        else
            @printf(" │ %8s %8s\n", "-", "-")
        end
    end

    println("\n" * "="^140)
    println("Legend:")
    println("  Sharpe: \033[32mGreen\033[0m > 1.0 (excellent) | \033[33mYellow\033[0m > 0.5 (good) | White 0.0-0.5 | \033[31mRed\033[0m < 0.0 (poor)")
    println("  PnL:    \033[32mGreen\033[0m > 20% (strong) | \033[33mYellow\033[0m > 0% (positive) | \033[31mRed\033[0m < 0% (loss)")
    println("  MaxDD:  \033[32mGreen\033[0m < 10% (low) | \033[33mYellow\033[0m < 20% (moderate) | \033[31mRed\033[0m >= 20% (high)")
    println("="^140)
end

export analyze_time_periods, get_market_periods

end # module
