"""
Data Loading Utilities
=======================
Load historical stock data from JSON files.
"""

using JSON
using DataFrames
using Dates

"""
Load a single stock's historical data from JSON file.
Returns DataFrame with timestamp, open, high, low, close, volume.
"""
function load_stock(symbol::Symbol, datasets_dir::String)::Union{DataFrame, Nothing}
    filepath = joinpath(datasets_dir, "$(symbol)_historical.json")

    if !isfile(filepath)
        @warn "File not found: $filepath"
        return nothing
    end

    data = JSON.parsefile(filepath)

    # Parse and convert to DataFrame
    timestamps = DateTime[]
    opens = Float64[]
    highs = Float64[]
    lows = Float64[]
    closes = Float64[]
    volumes = Int[]

    for entry in data
        push!(timestamps, DateTime(entry["date"][1:19]))  # Remove timezone
        push!(opens, entry["open"])
        push!(highs, entry["high"])
        push!(lows, entry["low"])
        push!(closes, entry["close"])
        push!(volumes, entry["volume"])
    end

    DataFrame(
        timestamp = timestamps,
        open = opens,
        high = highs,
        low = lows,
        close = closes,
        volume = volumes
    )
end

"""
Load multiple stocks and create a Dataset.
"""
function load_stocks(symbols::Vector{Symbol}, datasets_dir::String)::Dataset{Untagged}
    data = Dict{Symbol, DataFrame}()

    for symbol in symbols
        df = load_stock(symbol, datasets_dir)
        if !isnothing(df)
            data[symbol] = df
        else
            @warn "Skipping $symbol (file not found)"
        end
    end

    if isempty(data)
        error("No stocks loaded successfully")
    end

    Dataset(data)
end

"""
Get diversified S&P 500 stocks across major sectors.
Organized by sector for better cross-industry diversification.
"""
function get_top_sp500_symbols(n::Int=30)::Vector{Symbol}
    # Diversified across sectors (approx. by market cap within sector)
    #
    # Sector Breakdown (default n=30):
    # - Technology: 8 stocks (~27%)
    # - Healthcare: 5 stocks (~17%)
    # - Financials: 4 stocks (~13%)
    # - Consumer Discretionary: 4 stocks (~13%)
    # - Consumer Staples: 3 stocks (~10%)
    # - Industrials: 3 stocks (~10%)
    # - Energy: 2 stocks (~7%)
    # - Utilities: 1 stock (~3%)

    all_symbols = [
        # Technology (8) - Innovation & Growth
        :AAPL, :MSFT, :NVDA, :GOOGL, :META, :AVGO, :ADBE, :CRM,

        # Healthcare (5) - Defensive & Stable
        :UNH, :LLY, :JNJ, :ABBV, :MRK,

        # Financials (4) - Economic Cycle Exposure
        :JPM, :V, :MA, :BAC,

        # Consumer Discretionary (4) - Consumer Spending
        :AMZN, :TSLA, :HD, :MCD,

        # Consumer Staples (3) - Defensive & Recession-Resistant
        :WMT, :PG, :KO,

        # Industrials (3) - Economic Growth
        :CAT, :BA, :UNP,

        # Energy (2) - Inflation Hedge
        :XOM, :CVX,

        # Utilities (1) - Defensive & Low Volatility
        :NEE,

        # Additional Diversifiers (if n > 30):
        # Technology
        :AMD, :INTC, :QCOM, :ORCL, :CSCO, :NFLX,
        # Healthcare
        :PFE, :TMO, :ABT, :BMY,
        # Financials
        :GS, :MS, :SPGI,
        # Consumer
        :COST, :PEP, :NKE, :SBUX, :LOW, :DIS,
        # Industrials
        :HON, :RTX, :UPS,
        # Communication
        :T, :VZ,
        # Other
        Symbol("BRK-B"), :LIN
    ]

    all_symbols[1:min(n, length(all_symbols))]
end
