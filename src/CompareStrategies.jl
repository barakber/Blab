"""
Compare All Strategies Demo
============================
Runs all built-in strategies on real S&P 500 data and compares performance.
Uses the existing parallel backtest infrastructure.
"""

using Printf

function compare_all_strategies(;
    n_stocks::Int=10,
    train_end::DateTime=DateTime("2022-01-01"),
    val_end::DateTime=DateTime("2023-01-01"),
    datasets_dir::String=joinpath(dirname(@__DIR__), "..", "datasets")
)
    println("\n" * "="^70)
    println("COMPARING ALL STRATEGIES ON REAL S&P 500 DATA")
    println("="^70)
    println("Running on $(Threads.nthreads()) threads\n")

    # Load top S&P 500 stocks
    symbols = get_top_sp500_symbols(n_stocks)
    println("Loading stocks: $(join(symbols, ", "))")

    ds = load_stocks(symbols, datasets_dir)

    println("\nDataset Info:")
    println("  Assets loaded: $(assets(ds))")
    println("  Total rows: $(nrow(ds))")
    println("  Date range: $(ds.timestamps[1]) to $(ds.timestamps[end])")

    # Split data
    train, val, test = split(ds, train_end, val_end)
    println("  Train: $(nrow(train)) | Val: $(nrow(val)) | Test: $(nrow(test))")
    println()

    # Single asset dataset for MA and HMM (use SPY or first stock)
    single_symbol = :SPY in assets(ds) ? :SPY : first(assets(ds))
    ds_single = Dataset(single_symbol => getdf(ds, single_symbol))
    train_single, val_single, test_single = split(ds_single, train_end, val_end)

    # Create all strategy jobs
    jobs = BacktestJob[]

    # Strategy 1: Moving Average (single asset)
    println("Setting up Moving Average strategies...")
    for (fast, slow) in [(10, 30), (20, 50)]
        m = Model(train_single, val_single, MAStrategy.MAParams(fast, slow))
        s = Strategy(m)
        push!(jobs, BacktestJob("MA_$(fast)_$(slow)", test_single, s))
    end

    # Strategy 2: EMA Crossover (single asset)
    println("Setting up EMA strategies...")
    for (fast, slow) in [(12, 26), (8, 21)]
        m = Model(train_single, val_single, EMAStrategy.EMAParams(fast, slow))
        s = Strategy(m)
        push!(jobs, BacktestJob("EMA_$(fast)_$(slow)", test_single, s))
    end

    # Strategy 3: MACD (single asset)
    println("Setting up MACD strategies...")
    for (fast, slow, sig) in [(12, 26, 9), (8, 17, 9)]
        m = Model(train_single, val_single, MACDStrategy.MACDParams(fast, slow, sig))
        s = Strategy(m)
        push!(jobs, BacktestJob("MACD_$(fast)_$(slow)_$(sig)", test_single, s))
    end

    # Strategy 4: RSI Mean Reversion (single asset)
    println("Setting up RSI strategies...")
    for (period, oversold, overbought) in [(14, 30.0, 70.0), (14, 20.0, 80.0)]
        m = Model(train_single, val_single, RSIStrategy.RSIParams(period, oversold, overbought))
        s = Strategy(m)
        push!(jobs, BacktestJob("RSI_$(period)_$(Int(oversold))_$(Int(overbought))", test_single, s))
    end

    # Strategy 5: Momentum Rotation (multi-asset)
    println("Setting up Momentum Rotation strategies...")
    num_assets = n_assets(ds)
    for (lookback, top_n) in [(20, 3), (30, 3), (60, 4)]
        if top_n <= num_assets
            m = Model(train, val, MomentumStrategy.MomentumParams(lookback, top_n))
            s = Strategy(m)
            push!(jobs, BacktestJob("Momentum_L$(lookback)_T$(top_n)", test, s))
        end
    end

    # Strategy 6: HMM Regime (single asset)
    println("Setting up HMM Regime strategy...")
    asset = first(assets(train_single))
    rets = returns(train_single, asset)
    hmm = HMMStrategy.fit_hmm(rets, 2; maxiter=100)
    println("  Fitted HMM: μ=$(round.(hmm.μ .* 100, digits=2))%, σ=$(round.(hmm.σ .* 100, digits=2))%")
    m = Model(train_single, val_single, HMMStrategy.HMMParams(hmm))
    s = Strategy(m)
    push!(jobs, BacktestJob("HMM_regime", test_single, s))

    # Strategy 7: XGBoost ML (single asset)
    println("Setting up XGBoost ML strategy...")
    try
        m = train__(XGBoostMLStrategy.XGBoostParams, train_single, val_single)
        s = Strategy(m)
        push!(jobs, BacktestJob("XGBoost_ML", test_single, s))
    catch e
        println("  Warning: XGBoost failed to train: $e")
        println("  Skipping XGBoost strategy...")
    end

    # Strategy 8: Regime Switching (meta-strategy)
    println("Setting up Regime Switching strategy...")
    try
        m = train__(RegimeSwitchStrategy.RegimeSwitchParams, train_single, val_single)
        s = Strategy(m)
        push!(jobs, BacktestJob("RegimeSwitch", test_single, s))
    catch e
        println("  Warning: RegimeSwitch failed to train: $e")
        println("  Skipping RegimeSwitch strategy...")
    end

    println("\nRunning $(length(jobs)) strategies in parallel...\n")

    # Run all backtests in parallel using existing infrastructure
    results = backtest_parallel(jobs)

    # Display results
    println("\n" * "="^70)
    println("BACKTEST RESULTS (sorted by Sharpe ratio)")
    println("="^70)

    sorted_results = sort(results, by=x->x.metrics.sharpe, rev=true)

    @printf("%-25s %10s %10s %12s %10s\n",
        "Strategy", "PnL%", "Sharpe", "MaxDD%", "Time(s)")
    println("-"^70)

    for r in sorted_results
        @printf("%-25s %9.2f%% %10.2f %11.2f%% %10.2f\n",
            r.name,
            r.metrics.pnl * 100,
            r.metrics.sharpe,
            r.metrics.max_drawdown * 100,
            r.elapsed)
    end

    println("-"^70)

    # Show winner details
    println("\n🏆 TOP PERFORMER: $(sorted_results[1].name)")
    println(sorted_results[1].metrics)

    # Performance stats
    total_time = sum(r.elapsed for r in results)
    max_time = maximum(r.elapsed for r in results)
    println("\nParallel Performance:")
    println("  Sequential time: $(round(total_time, digits=2))s")
    println("  Wallclock time: $(round(max_time, digits=2))s")
    println("  Speedup: $(round(total_time / max_time, digits=1))x")

    # Strategy type comparison
    println("\nStrategy Type Summary:")

    strategy_types = [
        ("MA", "MA_"),
        ("EMA", "EMA_"),
        ("MACD", "MACD_"),
        ("RSI", "RSI_"),
        ("Momentum", "Momentum_"),
        ("HMM", "HMM_"),
        ("XGBoost", "XGBoost_")
    ]

    for (type_name, prefix) in strategy_types
        type_results = filter(r -> startswith(r.name, prefix), sorted_results)
        if !isempty(type_results)
            best = first(type_results)
            println("  Best $type_name: $(best.name) - Sharpe: $(round(best.metrics.sharpe, digits=2))")
        end
    end

    println("="^70)

    return results
end
