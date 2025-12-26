# Blab

A minimal backtesting library for trading strategies in Julia.

## Features

- **Clean API**: Users implement `Strategy`, `Model`, and `signal()` functions
- **No Look-Ahead Bias**: Proper temporal data splitting and subset handling
- **Multi-Asset Support**: Built-in support for portfolio strategies
- **Parallel Execution**: Backtest parameter sweeps using threads or distributed computing
- **Built-in Strategies**: Moving Average, Momentum Rotation, HMM Regime Detection

## Installation

### From Source

```bash
git clone <your-repo-url>
cd Blab
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Add to Julia Path (Optional)

Add the `bin` directory to your PATH to use the CLI globally:

```bash
export PATH="/path/to/Blab/bin:$PATH"
```

Or add it to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.).

## Quick Start

### Using the CLI

```bash
# List available strategies
./bin/blab list

# Compare all strategies on real S&P 500 data (RECOMMENDED)
julia --threads=4 ./bin/blab compare -n 10

# Compare with custom parameters
julia --threads=8 ./bin/blab compare \
  --stocks 15 \
  --train-end "2023-01-01" \
  --val-end "2024-01-01"

# Run all demos
./bin/blab demo all

# Run specific strategy demo
./bin/blab demo ma
./bin/blab demo momentum
./bin/blab demo hmm

# Run parallel demo (where available)
./bin/blab demo ma --parallel
```

### Using as a Library

```julia
using Pkg
Pkg.activate("path/to/Blab")
using Blab
using DataFrames
using Dates
using Random

# Generate sample data
Random.seed!(42)
dates = collect(DateTime(2018,1,1):Day(1):DateTime(2020,12,31))
prices = cumsum(randn(length(dates))) .+ 100

# Create dataset
df = DataFrame(timestamp=dates, close=prices)
ds = Dataset(:SPY => df)

# Split data
train, val, test = split(ds, DateTime("2019-06-01"), DateTime("2020-01-01"))

# Define and train a strategy (using MA as example)
m = train__(train, val)
s = Strategy(m)

# Backtest
results = backtest(test, s)
println(results)
```

## Project Structure

```
Blab/
├── Project.toml           # Package manifest
├── README.md              # This file
├── bin/
│   └── blab              # CLI executable
└── src/
    ├── Blab.jl           # Main module
    ├── BacktestEngine.jl # Core backtesting engine
    └── strategies/       # Example strategies
        ├── MA.jl         # Moving Average Crossover
        ├── Momentum.jl   # Portfolio Momentum Rotation
        └── HMM.jl        # HMM Regime Detection
```

## Creating a Custom Strategy

### 1. Define Model Parameters

```julia
struct MyParams
    window::Int
    threshold::Float64
end
```

### 2. Implement Training Function

```julia
function train__(ds_train::Dataset{Train}, ds_val::Dataset{Validation})::Model{MyParams}
    @assert ds_train.uid == ds_val.uid

    # Your training logic here
    params = MyParams(20, 0.5)

    Model(ds_train, ds_val, params)
end
```

### 3. Implement Signal Function

```julia
function signal(ds::Dataset{Test}, strategy::Strategy{MyParams})::Dict{Symbol, Float64}
    params = strategy.model.params
    asset = first(assets(ds))

    # Your signal logic here
    # Return Dict of asset => weight

    Dict(asset => 1.0)  # Example: always fully invested
end
```

### 4. Run Backtest

```julia
# Load your data
ds = Dataset(:ASSET => your_dataframe)
train, val, test = split(ds, cutoff1, cutoff2)

# Train and backtest
model = train__(train, val)
strategy = Strategy(model)
results = backtest(test, strategy)
```

## Dataset API

### Creating Datasets

```julia
# Single asset
ds = Dataset(:SPY => df)

# Multiple assets
data = Dict(
    :AAPL => aapl_df,
    :GOOG => goog_df,
    :MSFT => msft_df
)
ds = Dataset(data)
```

### DataFrame Requirements

Each DataFrame must have:
- `:timestamp` column (DateTime)
- `:close` column (Float64)
- Optional: `:open`, `:high`, `:low`, `:volume`

### Accessing Data

```julia
# Get assets
assets(ds)                    # Vector{Symbol}

# Get close prices
prices(ds, :AAPL)             # Vector{Float64}
price(ds, i, :AAPL)           # Float64

# Get OHLCV data
ohlcv_at(ds, i, :AAPL)        # NamedTuple

# Get returns
returns(ds, :AAPL)            # Vector{Float64}
returns(ds)                   # Dict{Symbol, Vector{Float64}}
```

## Parallel Backtesting

### Parameter Sweep

```julia
# Define parameter grid
param_grid = [
    (fast=5, slow=20),
    (fast=10, slow=30),
    (fast=20, slow=50)
]

# Define strategy factory
make_strategy(p, train, val) = Strategy(Model(train, val, MAParams(p.fast, p.slow)))

# Run sweep
results = backtest_sweep(test, train, val, param_grid, make_strategy)

# Results are sorted by Sharpe ratio
println(results)
```

### Threading

```julia
# Start Julia with multiple threads
julia --threads=8 --project=.

# Parallel execution happens automatically in backtest_sweep
```

## Metrics

Each backtest returns:
- **PnL**: Total profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Per-Asset PnL**: Individual asset contributions (for multi-asset strategies)

## Examples

### Moving Average Strategy

See `src/strategies/MA.jl` for a complete example of:
- Single-asset backtesting
- Parameter sweep
- Parallel execution

### Momentum Rotation Strategy

See `src/strategies/Momentum.jl` for:
- Multi-asset portfolio
- Top-N asset selection
- Rebalancing logic

### HMM Regime Strategy

See `src/strategies/HMM.jl` for:
- Custom model implementation (from-scratch HMM)
- Regime detection
- State-dependent positioning

## Development

### Running Tests

```julia
# In Julia REPL
using Pkg
Pkg.activate(".")
Pkg.test()
```

### Adding Dependencies

```julia
using Pkg
Pkg.activate(".")
Pkg.add("PackageName")
```

## Comparing Strategies (Parallel Backtest)

The `compare` command runs all strategies on real S&P 500 data in parallel:

```bash
# Run with 4 threads, 10 stocks
julia --threads=4 ./bin/blab compare -n 10
```

This will:
- Load historical data for top 10 S&P 500 stocks from `../datasets/`
- Test multiple parameter combinations for each strategy
- Run all backtests in parallel using available threads
- Display results ranked by Sharpe ratio
- Show performance comparison across strategy types

Example output:
```
======================================================================
BACKTEST RESULTS (sorted by Sharpe ratio)
======================================================================
Strategy                        PnL%     Sharpe       MaxDD%    Time(s)
----------------------------------------------------------------------
MA_10_50                     135.36%       1.94       15.33%       0.00
Momentum_L10_T2              173.61%       1.82       32.34%       0.03
MA_20_50                     115.69%       1.74       16.70%       0.00
...

🏆 TOP PERFORMER: MA_10_50
Parallel Performance:
  Sequential time: 0.15s
  Wallclock time: 0.03s
  Speedup: 4.5x
```

## CLI Reference

```
blab - Minimal Backtesting Library

Commands:
  compare             Compare all strategies on real S&P 500 data
  demo [strategy]     Run demo strategies
  list                List available strategies
  run strategy        Run a backtest (not yet implemented)

Options:
  -h, --help         Show help message
  -v, --version      Show version

Compare Options:
  -n, --stocks N     Number of top S&P 500 stocks (default: 10)
  --train-end DATE   Training end date YYYY-MM-DD (default: 2022-01-01)
  --val-end DATE     Validation end date YYYY-MM-DD (default: 2023-01-01)
  -d, --datasets DIR Path to datasets directory

Demo Options:
  -p, --parallel     Run parallel demo if available

Examples:
  julia --threads=8 blab compare -n 15         # Compare with 15 stocks, 8 threads
  blab demo all                                # Run all demos
  blab demo ma --parallel                      # Run MA demo with parallelism
  blab list                                    # List strategies
```

## License

MIT

## Author

Barak Bercovitz <barakber@gmail.com>
