# ─────────────────────────────────────────────────────────────────────────────
# Market data layer: contiguous panels, mirrored ring buffers, temporal guard.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CorporateAction(asset, t; split=1.0, dividend=0.0)

A corporate action on bar `t` (the ex-date) for `asset`:

- `split`: shares received per share held (2.0 for a 2-for-1 split, 0.1 for a 1-for-10
  reverse split)
- `dividend`: cash paid per *pre-split* share

Panel `mid` prices are raw (as traded), so they drop on the ex-date. The backtester
multiplies share positions by `split`, credits longs / debits shorts the dividend, and
rescales resting orders. Signals use `logmid`, which is adjusted *forward in time only*
(see [`PricePanel`](@ref)), so the adjustment never leaks future information.
"""
struct CorporateAction
    asset::Int
    t::Int
    split::Float64
    dividend::Float64
end
CorporateAction(asset::Integer, t::Integer; split::Real = 1.0, dividend::Real = 0.0) =
    CorporateAction(Int(asset), Int(t), Float64(split), Float64(dividend))

"""
    PricePanel

Synchronised multi-asset bar panel. Every field matrix is `n_assets × n_timestamps`
and stored column-major, so one bar (all assets at time `t`) is a contiguous column.

Fields
- `symbols`, `timestamps` (strictly increasing), `sector` (integer group labels)
- `mid`: raw mid prices, as traded (used for fills, valuation and financing)
- `logmid`: log prices adjusted for corporate actions *causally*: on and after an
  ex-date the log of the cumulative adjustment factor is added, so the series is
  continuous through splits and dividends but never depends on future actions. All
  statistics and signals use `logmid`.
- `spread`: absolute quoted spread (ask − bid)
- `volume`: bar traded volume in shares (drives participation caps and impact)
- `depth`: displayed size at the touch in shares (liquidity verification)
- `tradable`: `false` where the bar was missing in the source data, had zero volume, or
  was marked halted; no fills happen there and no new positions are opened
- `actions`: corporate actions sorted by bar
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
    tradable::BitMatrix
    actions::Vector{CorporateAction}
    sector::Vector{Int}
    bars_per_year::Float64
end

_tofloat(x) = x === missing ? NaN : Float64(x)

# one-pass conversion to a fresh Matrix{Float64} (`missing` → NaN), checking the shape
function _float_matrix(x::AbstractMatrix, n::Int, T::Int, name::AbstractString)
    size(x) == (n, T) || throw(DimensionMismatch("$name must be $n × $T, got $(size(x))"))
    x isa Matrix{Float64} && return copy(x)
    A = Matrix{Float64}(undef, n, T)
    @inbounds for j in 1:T, i in 1:n
        A[i, j] = _tofloat(x[i, j])
    end
    A
end

"""
    PricePanel(mid; symbols, timestamps, spread, spread_bps=5, volume, default_volume=1e6,
               depth, depth_frac=0.01, sector, bars_per_year=252,
               missing_policy=:error, tradable=nothing, actions=CorporateAction[])

Build a validated panel from an `n_assets × T` mid-price matrix. Missing microstructure
fields are filled with conservative defaults (a constant relative spread, constant volume,
depth equal to `depth_frac` of bar volume).

Missing prices (`missing`, `NaN`, non-positive values):
- `missing_policy = :error` (default) rejects the panel;
- `missing_policy = :ffill` carries the last valid price forward (a leading gap takes the
  first valid price) and marks those bars non-tradable with zero volume and depth.

`tradable` is an optional `n_assets × T` Bool mask of additional halts. Bars with zero
volume are always non-tradable.
"""
function PricePanel(mid::AbstractMatrix;
                    symbols = [Symbol("A", i) for i in 1:size(mid, 1)],
                    timestamps = [DateTime(2000, 1, 1) + Day(t - 1) for t in 1:size(mid, 2)],
                    spread = nothing, spread_bps::Real = 5.0,
                    volume = nothing, default_volume::Real = 1e6,
                    depth = nothing, depth_frac::Real = 0.01,
                    sector = ones(Int, size(mid, 1)),
                    bars_per_year::Real = 252.0,
                    missing_policy::Symbol = :error,
                    tradable = nothing,
                    actions::AbstractVector{CorporateAction} = CorporateAction[])
    n, T = size(mid)
    n >= 1 && T >= 1 || throw(ArgumentError("panel must have at least one asset and one bar"))
    length(symbols) == n || throw(DimensionMismatch("symbols"))
    allunique(symbols) || throw(ArgumentError("symbols must be unique"))
    length(timestamps) == T || throw(DimensionMismatch("timestamps"))
    length(sector) == n || throw(DimensionMismatch("sector"))
    bars_per_year > 0 || throw(ArgumentError("bars_per_year must be positive"))
    missing_policy in (:error, :ffill) || throw(ArgumentError("missing_policy must be :error or :ffill"))
    for i in 2:T
        timestamps[i] > timestamps[i-1] ||
            throw(ArgumentError("timestamps must be strictly increasing (violation at index $i)"))
    end
    m = _float_matrix(mid, n, T, "mid")
    bad = BitMatrix(undef, n, T)
    @inbounds for j in 1:T, i in 1:n
        x = m[i, j]
        bad[i, j] = !(isfinite(x) && x > 0)
    end
    if any(bad)
        if missing_policy === :error
            I = findfirst(bad)::CartesianIndex{2}       # non-empty: we are inside any(bad)
            throw(ArgumentError("$(count(bad)) missing or non-positive prices (first at asset $(I[1]), bar $(I[2])); " *
                                "pass missing_policy = :ffill to forward-fill them as non-tradable bars"))
        end
        for a in 1:n
            f = findfirst(t -> !bad[a, t], 1:T)
            f === nothing && throw(ArgumentError("asset $a ($(symbols[a])) has no valid price"))
            lastv = m[a, f]
            for t in 1:T
                bad[a, t] ? (m[a, t] = lastv) : (lastv = m[a, t])
            end
        end
    end
    # defaults are only materialised when a field isn't supplied
    sp = spread === nothing ? m .* (spread_bps / 1e4) : _float_matrix(spread, n, T, "spread")
    vol = volume === nothing ? fill(Float64(default_volume), n, T) : _float_matrix(volume, n, T, "volume")
    dep = depth === nothing ? vol .* depth_frac : _float_matrix(depth, n, T, "depth")
    for j in 1:T, i in 1:n          # gaps: keep the panel finite, make the bar untradable
        if bad[i, j]
            isfinite(sp[i, j]) && sp[i, j] >= 0 || (sp[i, j] = m[i, j] * spread_bps / 1e4)
            vol[i, j] = 0.0
            dep[i, j] = 0.0
        end
    end
    for (nm, A) in (("spread", sp), ("volume", vol), ("depth", dep))
        all(x -> isfinite(x) && x >= 0, A) || throw(ArgumentError("$nm must be finite and non-negative"))
    end
    tr = .!bad .& (vol .> 0)                    # fused broadcast: already a BitMatrix
    if tradable !== nothing
        size(tradable) == (n, T) || throw(DimensionMismatch("tradable"))
        tr .&= Bool.(tradable)
    end
    acts = sort(collect(actions); by = a -> a.t)
    logm = log.(m)
    for a in acts
        1 <= a.asset <= n || throw(ArgumentError("corporate action for unknown asset $(a.asset)"))
        2 <= a.t <= T || throw(ArgumentError("corporate action bar $(a.t) must be in 2:$T"))
        a.split > 0 && isfinite(a.split) || throw(ArgumentError("split ratio must be positive"))
        prev = m[a.asset, a.t-1]
        0 <= a.dividend < prev || throw(ArgumentError("dividend must be in [0, previous price)"))
        # forward (causal) adjustment: only bars on/after the ex-date move
        f = log(a.split) + log(prev / (prev - a.dividend))
        for t in a.t:T
            logm[a.asset, t] += f
        end
    end
    PricePanel(collect(Symbol, symbols), collect(DateTime, timestamps), m, logm, sp, vol, dep, tr, acts,
               collect(Int, sector), Float64(bars_per_year))
end

"""Number of assets (rows) in a panel."""
nassets(p::PricePanel) = size(p.mid, 1)
"""Number of bars (columns) in a panel."""
ntimes(p::PricePanel) = size(p.mid, 2)
Base.show(io::IO, p::PricePanel) =
    print(io, "PricePanel(", nassets(p), " assets × ", ntimes(p), " bars, ",
          first(p.timestamps), " → ", last(p.timestamps), ")")

"""
    asset_bars(panel, i) -> StructArray

Columnar (`StructArrays.jl`) zero-copy view of one asset's bar history with fields
`timestamp, mid, spread, volume, depth, tradable`.
"""
asset_bars(p::PricePanel, i::Integer) =
    StructArray((timestamp = p.timestamps, mid = view(p.mid, i, :), spread = view(p.spread, i, :),
                 volume = view(p.volume, i, :), depth = view(p.depth, i, :),
                 tradable = view(p.tradable, i, :)))

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

"""
    LookaheadError(requested, now)

Thrown when code asks for bar `requested` while the simulation clock is at bar `now`.
"""
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

"""Throw a [`LookaheadError`](@ref) unless bar `τ` is at or before the clock."""
@inline function assert_past(c::Clock, τ::Integer)
    τ <= c.t || throw(LookaheadError(τ, c.t))
    nothing
end

const PANEL_FIELDS = (:mid, :logmid, :spread, :volume, :depth)

# Type-stable access to one of the bar matrices (`getfield` with a runtime Symbol would
# infer as Union{Matrix, Float64, …} and accept non-matrix fields).
function panel_field(p::PricePanel, field::Symbol)::Matrix{Float64}
    field === :logmid && return p.logmid
    field === :mid && return p.mid
    field === :spread && return p.spread
    field === :volume && return p.volume
    field === :depth && return p.depth
    throw(ArgumentError("field must be one of $(PANEL_FIELDS), got :$field"))
end

@inline function _history_range(c::Clock, W::Integer)
    W >= 1 || throw(ArgumentError("W must be ≥ 1"))
    c.t - W + 1 >= 1 || throw(ArgumentError("not enough history: t=$(c.t), W=$W"))
    (c.t - W + 1):c.t
end

"""
    history(panel, clock, W; field=:logmid)        -> n × W view
    history(panel, clock, asset, W; field=:logmid) -> length-W view

Zero-copy view of the `W` bars ending at (and including) `clock.t`. `field` is one of
`:mid, :logmid, :spread, :volume, :depth`.
"""
history(p::PricePanel, c::Clock, W::Integer; field::Symbol = :logmid) =
    view(panel_field(p, field), :, _history_range(c, W))
history(p::PricePanel, c::Clock, asset::Integer, W::Integer; field::Symbol = :logmid) =
    view(panel_field(p, field), asset, _history_range(c, W))

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

"""`true` once at least `W` samples have been pushed (so `window` is fully populated)."""
isfull(r::MirroredRing) = r.count >= r.W
Base.length(r::MirroredRing) = min(r.count, r.W)
"""Forget all samples (the storage is kept)."""
reset!(r::MirroredRing) = (r.head = 0; r.count = 0; r)
"""
    window(r)    -> k × W view
    window(r, i) -> length-W view of row i

The last `W` samples in chronological order, as a contiguous zero-copy view. Only
meaningful once [`isfull`](@ref); use [`filled_window`](@ref) before that.
"""
@inline window(r::MirroredRing) = view(r.buf, :, r.head+1:r.head+r.W)
@inline window(r::MirroredRing, i::Integer) = view(r.buf, i, r.head+1:r.head+r.W)
"""Most recent `min(count, W)` samples of row `i` (valid before the buffer is full)."""
@inline function filled_window(r::MirroredRing, i::Integer)
    n = length(r)
    view(r.buf, i, r.head+r.W-n+1:r.head+r.W)
end
"""Most recent sample of row `i`."""
@inline function latest(r::MirroredRing, i::Integer = 1)
    r.count > 0 || throw(ArgumentError("ring buffer is empty"))
    @inbounds r.buf[i, r.head]
end
