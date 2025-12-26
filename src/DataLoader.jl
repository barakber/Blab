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
Get top S&P 500 stocks (by market cap / common large caps).
"""
function get_top_sp500_symbols(n::Int=10)::Vector{Symbol}
    # Top S&P 500 by market cap (as of common knowledge)
    # Note: Using Symbol() for BRK-B due to hyphen
    all_symbols = [
        :AAPL, :MSFT, :GOOGL, :AMZN, :NVDA, :META, :TSLA, Symbol("BRK-B"),
        :JPM, :V, :UNH, :JNJ, :WMT, :MA, :PG, :HD, :CVX, :ABBV,
        :MRK, :KO, :PEP, :COST, :AVGO, :CSCO, :ADBE, :NFLX,
        :INTC, :AMD, :QCOM, :TXN, :CRM, :ORCL, :PYPL, :INTU
    ]

    all_symbols[1:min(n, length(all_symbols))]
end
