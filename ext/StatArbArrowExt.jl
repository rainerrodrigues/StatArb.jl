module StatArbArrowExt

using StatArb
using StatArb: PricePanel
using Arrow
using Dates

"""
    load_panel_arrow(path; bars_per_year=252, spread_bps=5, depth_frac=0.01, missing_policy=:error) -> PricePanel

Read a long-format Arrow file (`timestamp, symbol, mid` required; `spread, volume, depth,
sector` optional). The file is memory-mapped by `Arrow.Table`; it is pivoted once into the
contiguous `n_assets × T` panel layout used by the backtester. Absent (symbol, timestamp)
cells are an error unless `missing_policy = :ffill`, which forward-fills them as
non-tradable bars (see [`PricePanel`](@ref)).
"""
function StatArb.load_panel_arrow(path::AbstractString; bars_per_year::Real = 252.0,
                                            spread_bps::Real = 5.0, depth_frac::Real = 0.01,
                                            missing_policy::Symbol = :error)
    tbl = Arrow.Table(path)
    cols = propertynames(tbl)
    for c in (:timestamp, :symbol, :mid)
        c in cols || throw(ArgumentError("Arrow file must contain column `$c`"))
    end
    ts = DateTime.(tbl.timestamp)
    sy = Symbol.(tbl.symbol)
    syms = unique(sy)                       # first-appearance order
    times = sort!(unique(ts))
    si = Dict(s => i for (i, s) in enumerate(syms))
    ti = Dict(t => j for (j, t) in enumerate(times))
    n, T = length(syms), length(times)
    grab(c) = c in cols ? fill(NaN, n, T) : nothing
    M = fill(NaN, n, T); SP = grab(:spread); V = grab(:volume); D = grab(:depth)
    sector = ones(Int, n)
    for r in eachindex(sy)
        i = si[sy[r]]; j = ti[ts[r]]
        M[i, j] = tbl.mid[r]
        SP === nothing || (SP[i, j] = tbl.spread[r])
        V === nothing || (V[i, j] = tbl.volume[r])
        D === nothing || (D[i, j] = tbl.depth[r])
        :sector in cols && (sector[i] = tbl.sector[r])
    end
    kw = (; symbols = syms, timestamps = times, sector = sector, bars_per_year = bars_per_year,
          spread_bps = spread_bps, depth_frac = depth_frac, missing_policy = missing_policy)
    PricePanel(M; spread = SP, volume = V, depth = D, kw...)
end

"""
    save_panel_arrow(path, panel)

Write a panel in the long format understood by [`load_panel_arrow`](@ref).
"""
function StatArb.save_panel_arrow(path::AbstractString, p::PricePanel)
    n, T = size(p.mid)
    idx = [(i, j) for j in 1:T for i in 1:n]
    Arrow.write(path, (timestamp = [p.timestamps[j] for (_, j) in idx],
                       symbol = [String(p.symbols[i]) for (i, _) in idx],
                       mid = [p.mid[i, j] for (i, j) in idx],
                       spread = [p.spread[i, j] for (i, j) in idx],
                       volume = [p.volume[i, j] for (i, j) in idx],
                       depth = [p.depth[i, j] for (i, j) in idx],
                       sector = [p.sector[i] for (i, _) in idx]))
    path
end

# Arrow has no enum type: write enums (and symbols) by name, as the CSV export does
_arrow_col(c::AbstractVector{<:Union{Enum,Symbol}}) = string.(c)
_arrow_col(c::AbstractVector) = c
StatArb._write_arrow_table(path::String, tab::NamedTuple) = (Arrow.write(path, map(_arrow_col, tab)); path)

end # module
