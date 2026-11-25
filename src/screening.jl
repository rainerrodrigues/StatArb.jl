# ─────────────────────────────────────────────────────────────────────────────
# Pair pre-screening (memory-bounded distance screen) and threaded cointegration scans.
# ─────────────────────────────────────────────────────────────────────────────

const _BLAS_LOCK = ReentrantLock()
const _BLAS_DEPTH = Ref(0)
const _BLAS_SAVED = Ref(1)

"""
    with_single_threaded_blas(f)

Run `f()` with `BLAS.set_num_threads(1)` so that BLAS does not oversubscribe cores that
Julia's own thread pool is using, then restore the previous setting.
"""
function with_single_threaded_blas(f)
    # BLAS's thread count is global: nested or concurrent callers share one save/restore,
    # so the original setting is restored only when the last of them finishes.
    lock(_BLAS_LOCK) do
        _BLAS_DEPTH[] == 0 && (_BLAS_SAVED[] = BLAS.get_num_threads(); BLAS.set_num_threads(1))
        _BLAS_DEPTH[] += 1
    end
    try
        return f()
    finally
        lock(_BLAS_LOCK) do
            _BLAS_DEPTH[] -= 1
            _BLAS_DEPTH[] == 0 && BLAS.set_num_threads(_BLAS_SAVED[])
        end
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
    normalized_window!(Z, logp, t, W) -> Z

Fill the `W × n` matrix `Z` with each asset's log price over the window ending at `t`,
demeaned and scaled to unit sample standard deviation (one column per asset, so every
per-asset loop is contiguous). Constant series are left demeaned (all zeros).
"""
function normalized_window!(Z::AbstractMatrix{Float64}, logp::AbstractMatrix, t::Integer, W::Integer)
    n = size(logp, 1)
    size(Z) == (W, n) || throw(DimensionMismatch("Z must be $W × $n"))
    t - W + 1 >= 1 || throw(ArgumentError("not enough history"))
    @inbounds for a in 1:n
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
        for s in 1:W
            Z[s, a] /= sd
        end
    end
    Z
end

# exact sum of squared differences between two normalised columns
@inline function _ssd(Z::AbstractMatrix{Float64}, i::Int, j::Int)
    d = 0.0
    @inbounds @simd for s in 1:size(Z, 1)
        d += (Z[s, i] - Z[s, j])^2
    end
    d
end

"""
    ScreenWorkspace(n, W; block=256)

Reusable buffers for [`distance_screen!`](@ref) and basket-candidate generation on `n`
assets and windows of `W` bars: the normalised window, column norms, a gathered group of
columns, one `n × block` Gram block and a bounded heap. Memory is `O(W·n + n·block)`
however many pairs exist.
"""
mutable struct ScreenWorkspace
    Z::Matrix{Float64}
    Zg::Matrix{Float64}
    G::Matrix{Float64}
    nrm::Vector{Float64}
    hd::Vector{Float64}
    hi::Vector{Int}
    hj::Vector{Int}
    hn::Int
    block::Int
end
ScreenWorkspace(n::Integer, W::Integer; block::Integer = 256) =
    ScreenWorkspace(zeros(W, n), zeros(W, 0), zeros(0, 0), zeros(n), Float64[], Int[], Int[], 0, Int(block))

# bounded max-heap on d: keeps the `cap` smallest distances seen
function _heap_reset!(ws::ScreenWorkspace, cap::Int)
    resize!(ws.hd, cap); resize!(ws.hi, cap); resize!(ws.hj, cap)
    ws.hn = 0
end
@inline function _heap_swap!(ws, a, b)
    @inbounds begin
        ws.hd[a], ws.hd[b] = ws.hd[b], ws.hd[a]
        ws.hi[a], ws.hi[b] = ws.hi[b], ws.hi[a]
        ws.hj[a], ws.hj[b] = ws.hj[b], ws.hj[a]
    end
end
@inline function _heap_push!(ws::ScreenWorkspace, d::Float64, i::Int, j::Int)
    cap = length(ws.hd)
    cap == 0 && return
    hd = ws.hd
    @inbounds if ws.hn < cap
        ws.hn += 1
        c = ws.hn
        hd[c] = d; ws.hi[c] = i; ws.hj[c] = j
        while c > 1 && hd[c>>1] < hd[c]
            _heap_swap!(ws, c, c >> 1)
            c >>= 1
        end
    elseif d < hd[1]
        hd[1] = d; ws.hi[1] = i; ws.hj[1] = j
        c = 1
        while true
            l = 2c; r = l + 1; m = c
            l <= cap && hd[l] > hd[m] && (m = l)
            r <= cap && hd[r] > hd[m] && (m = r)
            m == c && break
            _heap_swap!(ws, c, m)
            c = m
        end
    end
end

function _screen_groups(n, sector, same_sector::Bool, eligible)
    ok(a) = eligible === nothing || eligible[a]
    if same_sector && sector !== nothing
        groups = Dict{eltype(sector),Vector{Int}}()
        for a in 1:n
            ok(a) && push!(get!(() -> Int[], groups, sector[a]), a)
        end
        return collect(values(groups))
    end
    [filter(ok, 1:n)]
end

"""
    distance_screen!(ws, logp, t, W; top=100, sector=nothing, same_sector=true, eligible=nothing)

In-place [`distance_screen`](@ref) on a reusable [`ScreenWorkspace`](@ref) (the normalised
window is left in `ws.Z` for basket-candidate generation).

Distances come from Gram blocks, `‖zᵢ − zⱼ‖² = ‖zᵢ‖² + ‖zⱼ‖² − 2 zᵢ·zⱼ`, computed with BLAS
on at most `block` columns at a time and streamed through a bounded max-heap, so neither
time nor memory carries an `n²/2`-element candidate list. The survivors are re-ranked by
their exact sum of squared differences (ties broken by asset index).
"""
function distance_screen!(ws::ScreenWorkspace, logp::AbstractMatrix, t::Integer, W::Integer; top::Integer = 100,
                          sector = nothing, same_sector::Bool = true, eligible = nothing)
    n = size(logp, 1)
    size(ws.Z) == (W, n) || (ws.Z = zeros(W, n); ws.nrm = zeros(n))
    Z = normalized_window!(ws.Z, logp, t, W)
    @inbounds for a in 1:n
        s = 0.0
        @simd for r in 1:W
            s += Z[r, a]^2
        end
        ws.nrm[a] = s
    end
    _heap_reset!(ws, Int(top))
    for g in _screen_groups(n, sector, same_sector, eligible)
        ng = length(g)
        ng < 2 && continue
        size(ws.Zg, 1) == W && size(ws.Zg, 2) >= ng || (ws.Zg = zeros(W, max(ng, size(ws.Zg, 2))))
        b = min(ws.block, ng)
        size(ws.G, 1) >= ng && size(ws.G, 2) >= b || (ws.G = zeros(max(ng, size(ws.G, 1)), max(b, size(ws.G, 2))))
        Zg = view(ws.Zg, :, 1:ng)
        @inbounds for (c, a) in enumerate(g), r in 1:W
            Zg[r, c] = Z[r, a]
        end
        for jb in 1:b:ng
            J = jb:min(jb + b - 1, ng)
            G = view(ws.G, 1:ng, 1:length(J))
            mul!(G, transpose(Zg), view(Zg, :, J))
            @inbounds for (cj, jj) in enumerate(J)
                aj = g[jj]
                for ii in 1:jj-1
                    ai = g[ii]
                    _heap_push!(ws, ws.nrm[ai] + ws.nrm[aj] - 2G[ii, cj], min(ai, aj), max(ai, aj))
                end
            end
        end
    end
    # exact re-rank of the survivors
    k = ws.hn
    out = Vector{NTuple{2,Int}}(undef, k)
    d = Vector{Float64}(undef, k)
    @inbounds for c in 1:k
        out[c] = (ws.hi[c], ws.hj[c])
        d[c] = _ssd(Z, ws.hi[c], ws.hj[c])
    end
    out[sortperm(collect(zip(d, out)))]
end

"""
    distance_screen(logp, t, W; top=100, sector=nothing, same_sector=true, eligible=nothing) -> Vector{NTuple{2,Int}}

Gatev-style pre-screen: z-normalise each asset's log price over the window ending at `t`
and keep the `top` pairs with the smallest sum of squared differences, sorted by
distance. Optionally restrict to pairs within the same sector, and to `eligible` assets
(a Bool vector). Cuts the O(N²) cointegration workload to `top` tests; see
[`distance_screen!`](@ref) for the reusable, memory-bounded version.
"""
distance_screen(logp::AbstractMatrix, t::Integer, W::Integer; kw...) =
    distance_screen!(ScreenWorkspace(size(logp, 1), W), logp, t, W; kw...)

"""
    basket_candidates(logp, pairs, t, W, k; top=50, sector=nothing, same_sector=true, eligible=nothing)

Grow each screened pair into a `k`-asset basket by repeatedly adding the eligible asset
with the smallest total normalised distance to the current members (within the sector
if `same_sector`). Returns up to `top` distinct baskets (sorted asset lists), in the order
of the pairs they grew from.
"""
function basket_candidates(logp::AbstractMatrix, pairs::AbstractVector{NTuple{2,Int}}, t::Integer, W::Integer,
                           k::Integer; kw...)
    Z = normalized_window!(zeros(W, size(logp, 1)), logp, t, W)
    basket_candidates(Z, pairs, k; kw...)
end

"""
    basket_candidates(Z, pairs, k; top=50, sector=nothing, same_sector=true, eligible=nothing)

Same, on an already normalised window `Z` (e.g. `ws.Z` after [`distance_screen!`](@ref)).
"""
function basket_candidates(Z::AbstractMatrix{Float64}, pairs::AbstractVector{NTuple{2,Int}}, k::Integer;
                           top::Integer = 50, sector = nothing, same_sector::Bool = true, eligible = nothing)
    n = size(Z, 2)
    seen = Set{Vector{Int}}()
    out = Vector{Int}[]
    members = Int[]
    for (i, j) in pairs
        length(out) >= top && break
        empty!(members); push!(members, i, j)
        while length(members) < k
            best, bestd = 0, Inf
            for m in 1:n
                m in members && continue
                (eligible !== nothing && !eligible[m]) && continue
                (same_sector && sector !== nothing && sector[m] != sector[i]) && continue
                d = 0.0
                for a in members
                    d += _ssd(Z, m, a)
                end
                d < bestd && ((best, bestd) = (m, d))
            end
            best == 0 && break
            push!(members, best)
        end
        length(members) == k || continue
        key = sort(members)
        key in seen && continue
        push!(seen, key)
        push!(out, key)
    end
    out
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
