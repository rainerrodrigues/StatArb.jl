# ─────────────────────────────────────────────────────────────────────────────
# Multiple-testing control for pair selection.
#
# Formation tests many candidate pairs at once. Testing each one at 5% lets roughly
# 5 % of the non-cointegrated candidates through. Instead, every candidate gets an exact
# finite-sample p-value from a simulated null distribution of the *same* statistic the
# formation step computes, and the Benjamini-Hochberg procedure keeps the expected
# share of false discoveries among selected pairs at or below `q`.
# ─────────────────────────────────────────────────────────────────────────────

"""
    NullDistribution(stats, tail)

Sorted Monte Carlo draws of a test statistic under the null hypothesis. `tail` is
`:lower` (reject for small values, e.g. ADF) or `:upper` (reject for large values, e.g.
the Johansen trace). See [`null_pvalue`](@ref).
"""
struct NullDistribution
    stats::Vector{Float64}
    tail::Symbol
    function NullDistribution(stats::Vector{Float64}, tail::Symbol)
        tail in (:lower, :upper) || throw(ArgumentError("tail must be :lower or :upper"))
        new(sort(stats), tail)
    end
end

"""
    null_pvalue(nd, stat) -> Float64

Monte Carlo p-value `(#{null draws at least as extreme} + 1) / (n + 1)`, which is valid
(never anti-conservative) for any number of draws. Non-finite statistics get `p = 1`.
"""
function null_pvalue(nd::NullDistribution, s::Real)
    isfinite(s) || return 1.0
    n = length(nd.stats)
    k = nd.tail === :lower ? searchsortedlast(nd.stats, s) : n - searchsortedfirst(nd.stats, s) + 1
    (k + 1) / (n + 1)
end

const _NULL_CACHE = Dict{NTuple{4,Int},NullDistribution}()
const _NULL_LOCK = ReentrantLock()

@inline function _random_walk!(rng::AbstractRNG, v::AbstractVector{Float64})
    s = 0.0
    @inbounds for t in eachindex(v)
        s += randn(rng)
        v[t] = s
    end
    v
end

"""
    eg_null_distribution(T; lags=1, nsim=10_000) -> NullDistribution

Null distribution (independent random walks) of `min(ADF(y on x), ADF(x on y))`, the
two-orientation Engle-Granger statistic used by [`test_pairs`](@ref), for windows of `T`
bars. Simulated once with a fixed seed and cached; thread-safe.
"""
function eg_null_distribution(T::Integer; lags::Integer = 1, nsim::Integer = 10_000)
    lock(_NULL_LOCK) do
        get!(_NULL_CACHE, (1, Int(T), Int(lags), Int(nsim))) do
            rng = Xoshiro(0x5eed_0001 + T)
            ws = EGWorkspace(T, lags)
            y = zeros(T); x = zeros(T)
            stats = Vector{Float64}(undef, nsim)
            for i in 1:nsim
                _random_walk!(rng, y); _random_walk!(rng, x)
                a = engle_granger!(ws, y, x; lags = Int(lags)).adf
                b = engle_granger!(ws, x, y; lags = Int(lags)).adf
                stats[i] = min(a, b)
            end
            NullDistribution(stats, :lower)
        end
    end
end

"""
    johansen_null_distribution(T; kdiff=1, nsim=10_000) -> NullDistribution

Null distribution (two independent random walks) of the Johansen trace statistic for
rank 0, restricted constant, for windows of `T` bars. Cached; thread-safe.
"""
function johansen_null_distribution(T::Integer; kdiff::Integer = 1, nsim::Integer = 10_000)
    lock(_NULL_LOCK) do
        get!(_NULL_CACHE, (2, Int(T), Int(kdiff), Int(nsim))) do
            rng = Xoshiro(0x5eed_0002 + T)
            ws = JohansenWorkspace(2, T; kdiff = kdiff)
            X = zeros(2, T)
            stats = Vector{Float64}(undef, nsim)
            for i in 1:nsim
                _random_walk!(rng, view(X, 1, :)); _random_walk!(rng, view(X, 2, :))
                stats[i] = johansen!(ws, X).trace_r0
            end
            NullDistribution(stats, :upper)
        end
    end
end

"""
    benjamini_hochberg(p, q) -> BitVector

Benjamini-Hochberg step-up procedure: rejects the hypotheses with the `k` smallest
p-values, where `k` is the largest index with `p₍ₖ₎ ≤ k q / m`. Controls the false
discovery rate at `q` for independent or positively dependent tests.
"""
function benjamini_hochberg(p::AbstractVector{<:Real}, q::Real)
    0 < q <= 1 || throw(ArgumentError("q must be in (0, 1]"))
    m = length(p)
    rej = falses(m)
    m == 0 && return rej
    ord = sortperm(p)
    kmax = 0
    for (k, i) in enumerate(ord)
        p[i] <= k * q / m && (kmax = k)
    end
    for k in 1:kmax
        rej[ord[k]] = true
    end
    rej
end

"""
    pair_pvalues(tests, T; lags=1, kdiff=1, nsim=10_000) -> Vector{Float64}

Intersection-union p-value per tested pair: `max(p_EG, p_Johansen)`. A pair counts as
significant only if *both* tests reject.
"""
function pair_pvalues(tests::AbstractVector{PairTest}, T::Integer; lags::Integer = 1,
                      kdiff::Integer = 1, nsim::Integer = 10_000)
    egn = eg_null_distribution(T; lags = lags, nsim = nsim)
    jn = johansen_null_distribution(T; kdiff = kdiff, nsim = nsim)
    [pt.joh.ok ? max(null_pvalue(egn, pt.eg.adf), null_pvalue(jn, pt.joh.trace_r0)) : 1.0 for pt in tests]
end
