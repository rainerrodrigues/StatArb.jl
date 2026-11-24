# ─────────────────────────────────────────────────────────────────────────────
# Exporting results, and the plotting API implemented by the Makie extension.
# ─────────────────────────────────────────────────────────────────────────────

_csvcell(x::AbstractString) = occursin(r"[\",\n]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x
_csvcell(x::Union{Enum,Symbol}) = string(x)
_csvcell(x::DateTime) = string(x)
_csvcell(x::Bool) = x ? "true" : "false"
_csvcell(x::Integer) = string(x)
_csvcell(x::AbstractFloat) = isnan(x) ? "" : isinf(x) ? (x > 0 ? "Inf" : "-Inf") : repr(Float64(x))
_csvcell(x) = _csvcell(string(x))

"""
    write_csv(path, columns::NamedTuple) -> path

Write equally long column vectors as a CSV file with a header row (`NaN` is written as an
empty cell, enums and symbols as their names). Dependency-free.
"""
function write_csv(path::AbstractString, cols::NamedTuple)
    n = isempty(cols) ? 0 : length(first(cols))
    all(c -> length(c) == n, cols) || throw(DimensionMismatch("all columns must have the same length"))
    open(path, "w") do io
        println(io, join(string.(keys(cols)), ","))
        for i in 1:n
            println(io, join((_csvcell(c[i]) for c in cols), ","))
        end
    end
    path
end

_pairs_string(ps) = join(("$(a)-$(b)" for (a, b) in ps), ";")
_baskets_string(bs) = join((join(b, "-") for b in bs), ";")

"""
    result_tables(res) -> NamedTuple of column tables

The backtest output as plain column tables: `equity` (bar, timestamp, equity, return),
`bars`, `health`, `fills` and `formations` (selected pairs written as `y-x;…`, baskets as
`a-b-c;…`). Every table is a `NamedTuple` of vectors, so it can go straight to any
Tables.jl sink.
"""
function result_tables(res::BacktestResult)
    f = res.formations
    (equity = (t = res.t, timestamp = res.timestamps, equity = res.equity,
               ret = vcat(NaN, res.returns)),
     bars = StructArrays.components(res.bars),
     health = StructArrays.components(res.health),
     fills = StructArrays.components(res.fills),
     formations = (t = [x.t for x in f], n_candidates = [x.n_candidates for x in f],
                   n_cointegrated = [x.n_cointegrated for x in f], min_pvalue = [x.min_pvalue for x in f],
                   selected = [_pairs_string(x.selected) for x in f],
                   n_basket_candidates = [x.n_basket_candidates for x in f],
                   n_basket_significant = [x.n_basket_significant for x in f],
                   baskets = [_baskets_string(x.baskets) for x in f]))
end

# implemented by the Arrow extension (which adds the specific method)
_write_arrow_table(path, tab) =
    throw(ArgumentError("format = :arrow needs the Arrow extension: run `using Arrow` first"))

"""
    export_results(dir, res; format=:csv) -> Vector{String}

Write [`result_tables`](@ref) to `dir` (created if needed) as `equity`, `bars`, `health`,
`fills` and `formations` files, plus `summary.csv` with the performance metrics.
`format = :arrow` writes Arrow files instead (requires `using Arrow`). Returns the paths
written.
"""
function export_results(dir::AbstractString, res::BacktestResult; format::Symbol = :csv)
    format in (:csv, :arrow) || throw(ArgumentError("format must be :csv or :arrow"))
    if format === :arrow && Base.get_extension(@__MODULE__, :StatArbArrowExt) === nothing
        throw(ArgumentError("format = :arrow needs the Arrow extension: run `using Arrow` first"))
    end
    mkpath(dir)
    tabs = result_tables(res)
    p = performance(res)
    tabs = merge(tabs, (summary = (metric = collect(string.(keys(p))), value = collect(Float64.(values(p)))),))
    paths = String[]
    for (name, tab) in pairs(tabs)
        path = joinpath(dir, string(name, format === :csv ? ".csv" : ".arrow"))
        format === :csv ? write_csv(path, tab) : _write_arrow_table(path, tab)
        push!(paths, path)
    end
    paths
end

"""
    plot_backtest(res; size=(1400, 1000)) -> Makie.Figure

Dashboard of a backtest: equity and drawdown, gross exposure with the number of open
spreads, the Johansen trace-statistic trajectory, half-lives per spread, Kalman
innovation autocorrelation, and the execution-cost breakdown. Requires a Makie backend
(`using CairoMakie` or `using GLMakie`).
"""
function plot_backtest end

"""
    plot_health(res, spread_id) -> Makie.Figure

Health timeline of one spread: Johansen trace statistic against its critical value,
estimated half-lives (Engle-Granger or Johansen, and the filter's), innovation
autocorrelation, with quarantine periods shaded. Requires a Makie backend.
"""
function plot_health end

"""
    plot_sweep(sweep) -> Makie.Figure

Sharpe ratio for every perturbed parameter of a [`sensitivity_sweep`](@ref), one panel
per parameter. Requires a Makie backend.
"""
function plot_sweep end
