# ─────────────────────────────────────────────────────────────────────────────
# Pair pre-screening, threaded cointegration scans, and the mathematical health gate.
# ─────────────────────────────────────────────────────────────────────────────

"""
    with_single_threaded_blas(f)

Run `f()` with `BLAS.set_num_threads(1)` so that BLAS does not oversubscribe cores that
Julia's own thread pool is using, then restore the previous setting.
"""
function with_single_threaded_blas(f)
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(old)
    end
end

"""Run `f(range)` over contiguous chunks of `1:n`, one task per chunk (each owns its workspaces)."""
function foreach_chunk(f, n::Integer; threaded::Bool = true)
    n == 0 && return nothing
    nch = threaded ? min(n, Threads.nthreads()) : 1
    if nch <= 1
        f(1:n)
        return nothing
    end
    bounds = round.(Int, range(0, n; length = nch + 1))
    chunks = [bounds[c]+1:bounds[c+1] for c in 1:nch]
    with_single_threaded_blas() do
        Threads.@threads :static for c in chunks
            f(c)
        end
    end
    nothing
end

"""
    distance_screen(logp, t, W; top=100, sector=nothing, same_sector=true) -> Vector{NTuple{2,Int}}

Gatev-style pre-screen: z-normalise each asset's log price over the window ending at `t`
and keep the `top` pairs with the smallest sum of squared differences. Optionally restrict
to pairs within the same sector. Cuts the O(N²) cointegration workload to `top` tests.
"""
function distance_screen(logp::AbstractMatrix, t::Integer, W::Integer; top::Integer = 100,
                         sector = nothing, same_sector::Bool = true)
    n = size(logp, 1)
    t - W + 1 >= 1 || throw(ArgumentError("not enough history"))
    Z = Matrix{Float64}(undef, W, n)            # column per asset: contiguous inner loop
    for a in 1:n
        μ = 0.0
        for s in 1:W
            μ += logp[a, t-W+s]
        end
        μ /= W
        v = 0.0
        for s in 1:W
            d = logp[a, t-W+s] - μ
            Z[s, a] = d
            v += d * d
        end
        sd = sqrt(v / (W - 1))
        sd = sd > 0 ? sd : 1.0
        @views Z[:, a] ./= sd
    end
    cands = NTuple{2,Int}[]
    dists = Float64[]
    for i in 1:n-1, j in i+1:n
        (same_sector && sector !== nothing && sector[i] != sector[j]) && continue
        d = 0.0
        @inbounds @simd for s in 1:W
            d += (Z[s, i] - Z[s, j])^2
        end
        push!(cands, (i, j)); push!(dists, d)
    end
    k = min(top, length(cands))
    idx = partialsortperm(dists, 1:k)
    cands[idx]
end

"""Result of testing one candidate pair over a formation window."""
struct PairTest
    y::Int
    x::Int
    eg::EGResult
    joh::JohansenSummary
end

"""
    test_pairs(logp, pairs, t, W; adf_lags=1, kdiff=1, level=0.05, threaded=true) -> Vector{PairTest}

Engle-Granger (both orientations; the more negative ADF wins) plus a k=2 Johansen test
for each candidate over the window `t-W+1:t`. Threaded with one EG/Johansen workspace per
task and single-threaded BLAS.
"""
function test_pairs(logp::AbstractMatrix, pairs::AbstractVector{NTuple{2,Int}}, t::Integer, W::Integer;
                    adf_lags::Int = 1, kdiff::Int = 1, level::Real = 0.05, threaded::Bool = true)
    t - W + 1 >= 1 || throw(ArgumentError("not enough history"))
    out = Vector{PairTest}(undef, length(pairs))
    rng = t-W+1:t
    foreach_chunk(length(pairs); threaded = threaded) do chunk
        eg = EGWorkspace(W, adf_lags)
        jw = JohansenWorkspace(2, W; kdiff = kdiff)
        X = zeros(2, W)
        for c in chunk
            i, j = pairs[c]
            yi = view(logp, i, rng); xj = view(logp, j, rng)
            r1 = engle_granger!(eg, yi, xj; lags = adf_lags)
            r2 = engle_granger!(eg, xj, yi; lags = adf_lags)
            yy, xx, r = r1.adf <= r2.adf ? (i, j, r1) : (j, i, r2)
            for s in 1:W
                X[1, s] = logp[yy, rng[s]]
                X[2, s] = logp[xx, rng[s]]
            end
            out[c] = PairTest(yy, xx, r, johansen!(jw, X; level = level))
        end
    end
    out
end

"""
    scan_johansen(logp, baskets, t, W; kdiff=1, level=0.05, det=:restricted, threaded=true)

Threaded Johansen test of many baskets (vectors of asset indices, any mix of sizes 2–5)
over the window ending at `t`. One workspace per (task, basket size).
"""
function scan_johansen(logp::AbstractMatrix, baskets::AbstractVector, t::Integer, W::Integer;
                       kdiff::Int = 1, level::Real = 0.05, det::Symbol = :restricted, threaded::Bool = true)
    t <= size(logp, 2) || throw(ArgumentError("t beyond data"))
    t - W + 1 >= 1 || throw(ArgumentError("not enough history"))
    out = Vector{JohansenResult}(undef, length(baskets))
    foreach_chunk(length(baskets); threaded = threaded) do chunk
        wss = Dict{Int,Tuple{JohansenWorkspace,Matrix{Float64}}}()
        for c in chunk
            b = baskets[c]
            k = length(b)
            ws, X = get!(() -> (JohansenWorkspace(k, W; kdiff = kdiff, det = det), zeros(k, W)), wss, k)
            for s in 1:W, (r, a) in enumerate(b)
                X[r, s] = logp[a, t-W+s]
            end
            out[c] = JohansenResult(ws, johansen!(ws, X; level = level), b, t)
        end
    end
    out
end
scan_johansen(p::PricePanel, baskets, t, W; kw...) = scan_johansen(p.logmid, baskets, t, W; kw...)

"""
    rolling_johansen(logp, basket, W; step=1, kdiff=1, level=0.05, threaded=true)

Rolling-window trace-statistic trajectory for one basket. Each window only uses data up
to its own end bar. Returns a NamedTuple of vectors `(t, trace_r0, maxeig_r0, rank, half_life)`.
"""
function rolling_johansen(logp::AbstractMatrix, basket::AbstractVector{<:Integer}, W::Integer;
                          step::Integer = 1, kdiff::Int = 1, level::Real = 0.05,
                          det::Symbol = :restricted, threaded::Bool = true)
    T = size(logp, 2)
    ts = collect(W:step:T)
    k = length(basket)
    tr = fill(NaN, length(ts)); me = fill(NaN, length(ts)); rk = zeros(Int, length(ts)); hl = fill(Inf, length(ts))
    foreach_chunk(length(ts); threaded = threaded) do chunk
        ws = JohansenWorkspace(k, W; kdiff = kdiff, det = det)
        ring = MirroredRing(k, W)
        tstart = ts[first(chunk)] - W + 1   # each chunk warms its own ring from the past
        ci = first(chunk)
        for τ in tstart:ts[last(chunk)]
            push_column!(ring, logp, basket, τ)
            if ci <= last(chunk) && τ == ts[ci]
                s = johansen!(ws, window(ring); level = level)
                tr[ci], me[ci], rk[ci], hl[ci] = s.trace_r0, s.maxeig_r0, s.rank, s.half_life
                ci += 1
            end
        end
    end
    (t = ts, trace_r0 = tr, maxeig_r0 = me, rank = rk, half_life = hl)
end

# ─── Health gate ─────────────────────────────────────────────────────────────

"""
    HealthState

Lifecycle of a traded pair: `ACTIVE` (may open positions), `QUARANTINE` (close-only until
the health checks recover) and `RETIRED` (dropped from the universe and flat).
"""
@enum HealthState ACTIVE QUARANTINE RETIRED
@doc "Pair passes its health checks and may open positions." ACTIVE
@doc "Pair failed a health check: close-only until it recovers." QUARANTINE
@doc "Pair left the universe and has been closed out." RETIRED

"""
    HealthGateConfig(; hl_bounds=(1, 60), gate_on=:johansen, recover_checks=2, level=0.05)

A pair is moved to `QUARANTINE` (close-only) when the rolling Johansen rank drops to 0
(`gate_on=:johansen`; `:eg` uses the Engle-Granger test, `:either` requires both to fail)
or when the OU half-life leaves `hl_bounds`. It returns to `ACTIVE` after
`recover_checks` consecutive healthy checks.
"""
Base.@kwdef struct HealthGateConfig
    hl_bounds::Tuple{Float64,Float64} = (1.0, 60.0)
    gate_on::Symbol = :johansen
    recover_checks::Int = 2
    level::Float64 = 0.05
end

"""Is one health check passing?"""
function healthy(cfg::HealthGateConfig, joh::JohansenSummary, eg::EGResult)
    hl = eg.half_life
    lo, hi = cfg.hl_bounds
    hl_ok = lo <= hl <= hi
    jo = joh.ok && joh.rank >= 1
    eo = iscointegrated(eg, cfg.level)
    coint_ok = cfg.gate_on === :johansen ? jo : cfg.gate_on === :eg ? eo : (jo || eo)
    hl_ok && coint_ok
end

"""Advance the gate state machine; returns `(new_state, new_good_count)`."""
function gate_transition(cfg::HealthGateConfig, state::HealthState, good::Int, ok::Bool)
    state == RETIRED && return (RETIRED, 0)
    if ok
        g = good + 1
        return state == QUARANTINE && g < cfg.recover_checks ? (QUARANTINE, g) : (ACTIVE, g)
    else
        return (QUARANTINE, 0)
    end
end
