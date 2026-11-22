# ─────────────────────────────────────────────────────────────────────────────
# Market data layer: contiguous panels, mirrored ring buffers, temporal guard.
# ─────────────────────────────────────────────────────────────────────────────

"""
    PricePanel

Synchronised multi-asset bar panel. Every field matrix is `n_assets × n_timestamps`
and stored column-major, so one bar (all assets at time `t`) is a contiguous column.

Fields
- `symbols`, `timestamps` (strictly increasing), `sector` (integer group labels)
- `mid`, `logmid`: mid prices and their logs (cointegration works on log prices)
- `spread`: absolute quoted spread (ask − bid)
- `volume`: bar traded volume in shares (drives participation caps and impact)
- `depth`: displayed size at the touch in shares (liquidity verification)
- `bars_per_year`: annualisation constant (252 for daily bars)
"""
struct PricePanel
    symbols::Vector{Symbol}
    timestamps::Vector{DateTime}
    mid::Matrix{Float64}
    logmid::Matrix{Float64}
    spread::Matrix{Float64}
    volume::Matrix{Float64}
    depth::Matrix{Float64}
    sector::Vector{Int}
    bars_per_year::Float64
end

"""
    PricePanel(mid; symbols, timestamps, spread, spread_bps=5, volume, default_volume=1e6,
               depth, depth_frac=0.01, sector, bars_per_year=252)

Build a validated panel from an `n_assets × T` mid-price matrix. Missing microstructure
fields are filled with conservative defaults (a constant relative spread, constant volume,
depth equal to `depth_frac` of bar volume).
"""
function PricePanel(mid::AbstractMatrix{<:Real};
                    symbols = [Symbol("A", i) for i in 1:size(mid, 1)],
                    timestamps = [DateTime(2000, 1, 1) + Day(t - 1) for t in 1:size(mid, 2)],
                    spread = nothing, spread_bps::Real = 5.0,
                    volume = nothing, default_volume::Real = 1e6,
                    depth = nothing, depth_frac::Real = 0.01,
                    sector = ones(Int, size(mid, 1)),
                    bars_per_year::Real = 252.0)
    n, T = size(mid)
    m = Matrix{Float64}(mid)
    all(x -> isfinite(x) && x > 0, m) || throw(ArgumentError("prices must be finite and strictly positive"))
    length(symbols) == n || throw(DimensionMismatch("symbols"))
    length(timestamps) == T || throw(DimensionMismatch("timestamps"))
    length(sector) == n || throw(DimensionMismatch("sector"))
    for i in 2:T
        timestamps[i] > timestamps[i-1] ||
            throw(ArgumentError("timestamps must be strictly increasing (violation at index $i)"))
    end
    sp = spread === nothing ? m .* (spread_bps / 1e4) : Matrix{Float64}(spread)
    vol = volume === nothing ? fill(Float64(default_volume), n, T) : Matrix{Float64}(volume)
    dep = depth === nothing ? vol .* depth_frac : Matrix{Float64}(depth)
    for (nm, A) in (("spread", sp), ("volume", vol), ("depth", dep))
        size(A) == (n, T) || throw(DimensionMismatch(nm))
        all(x -> isfinite(x) && x >= 0, A) || throw(ArgumentError("$nm must be finite and non-negative"))
    end
    PricePanel(collect(Symbol, symbols), collect(DateTime, timestamps), m, log.(m), sp, vol, dep,
               collect(Int, sector), Float64(bars_per_year))
end

nassets(p::PricePanel) = size(p.mid, 1)
ntimes(p::PricePanel) = size(p.mid, 2)
Base.show(io::IO, p::PricePanel) =
    print(io, "PricePanel(", nassets(p), " assets × ", ntimes(p), " bars, ",
          first(p.timestamps), " → ", last(p.timestamps), ")")

"""
    asset_bars(panel, i) -> StructArray

Columnar (`StructArrays.jl`) zero-copy view of one asset's bar history with fields
`timestamp, mid, spread, volume, depth`.
"""
asset_bars(p::PricePanel, i::Integer) =
    StructArray((timestamp = p.timestamps, mid = view(p.mid, i, :), spread = view(p.spread, i, :),
                 volume = view(p.volume, i, :), depth = view(p.depth, i, :)))

# ─── Temporal guardrails ─────────────────────────────────────────────────────

"""
    Clock(t)

Single monotonic clock for a simulation. The clock only moves forward (`tick!`), and all
data access for signal generation goes through `history`/`assert_past`, which throw a
[`LookaheadError`](@ref) for any index beyond `clock.t`.
"""
mutable struct Clock
    t::Int
end

struct LookaheadError <: Exception
    requested::Int
    now::Int
end
Base.showerror(io::IO, e::LookaheadError) =
    print(io, "LookaheadError: requested bar ", e.requested, " while the clock is at bar ", e.now)

"""Advance the clock to `t`; time can never go backwards."""
function tick!(c::Clock, t::Integer)
    t >= c.t || throw(ArgumentError("clock must be monotonic: $(c.t) → $t"))
    c.t = t
    c
end

@inline function assert_past(c::Clock, τ::Integer)
    τ <= c.t || throw(LookaheadError(τ, c.t))
    nothing
end

"""
    history(panel, clock, W; field=:logmid)        -> n × W view
    history(panel, clock, asset, W; field=:logmid) -> length-W view

Zero-copy view of the `W` bars ending at (and including) `clock.t`.
"""
function history(p::PricePanel, c::Clock, W::Integer; field::Symbol = :logmid)
    c.t - W + 1 >= 1 || throw(ArgumentError("not enough history: t=$(c.t), W=$W"))
    view(getfield(p, field), :, c.t-W+1:c.t)
end
function history(p::PricePanel, c::Clock, asset::Integer, W::Integer; field::Symbol = :logmid)
    c.t - W + 1 >= 1 || throw(ArgumentError("not enough history: t=$(c.t), W=$W"))
    view(getfield(p, field), asset, c.t-W+1:c.t)
end

# ─── Mirrored ring buffer ────────────────────────────────────────────────────

"""
    MirroredRing(k, W)

Fixed-capacity circular buffer of `k`-vectors with lookback `W`. Every sample is written
twice (positions `p` and `p+W` of a `k × 2W` matrix), so the most recent `W` samples are
*always* one contiguous, chronologically ordered block: `window(r)` returns a plain
`SubArray` with no copying, wrap-around logic or allocation.
"""
mutable struct MirroredRing
    buf::Matrix{Float64}
    W::Int
    head::Int
    count::Int
end
MirroredRing(k::Integer, W::Integer) = MirroredRing(zeros(Float64, k, 2W), W, 0, 0)

@inline function _advance!(r::MirroredRing)
    pos = r.head == r.W ? 1 : r.head + 1
    r.head = pos
    r.count += 1
    pos
end

@inline function Base.push!(r::MirroredRing, x::Real)
    pos = _advance!(r)
    @inbounds r.buf[1, pos] = x
    @inbounds r.buf[1, pos+r.W] = x
    r
end

@inline function Base.push!(r::MirroredRing, x::Union{Tuple,AbstractVector})
    pos = _advance!(r)
    W = r.W
    @inbounds for i in 1:size(r.buf, 1)
        v = Float64(x[i])
        r.buf[i, pos] = v
        r.buf[i, pos+W] = v
    end
    r
end

"""Push column `t` of `src` restricted to `rows` (e.g. asset indices of a pair/basket)."""
@inline function push_column!(r::MirroredRing, src::AbstractMatrix, rows, t::Integer)
    pos = _advance!(r)
    W = r.W
    @inbounds for (i, a) in enumerate(rows)
        v = src[a, t]
        r.buf[i, pos] = v
        r.buf[i, pos+W] = v
    end
    r
end

isfull(r::MirroredRing) = r.count >= r.W
Base.length(r::MirroredRing) = min(r.count, r.W)
reset!(r::MirroredRing) = (r.head = 0; r.count = 0; r)
@inline window(r::MirroredRing) = view(r.buf, :, r.head+1:r.head+r.W)
@inline window(r::MirroredRing, i::Integer) = view(r.buf, i, r.head+1:r.head+r.W)
"""Most recent `min(count, W)` samples of row `i` (valid before the buffer is full)."""
@inline function filled_window(r::MirroredRing, i::Integer)
    n = length(r)
    view(r.buf, i, r.head+r.W-n+1:r.head+r.W)
end
@inline latest(r::MirroredRing, i::Integer = 1) = @inbounds r.buf[i, r.head]
