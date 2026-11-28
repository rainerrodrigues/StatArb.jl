# ─────────────────────────────────────────────────────────────────────────────
# Overfitting statistics: is the best backtest real, or the best of many coin flips?
#
#   probabilistic_sharpe   PSR, Bailey & López de Prado (2012)
#   deflated_sharpe        DSR, Bailey & López de Prado (2014)
#   pbo                    Probability of Backtest Overfitting via CSCV,
#                          Bailey, Borwein, López de Prado & Zhu (2015)
#   bootstrap_ci           stationary block bootstrap, Politis & Romano (1994)
#   overfitting_report     all of the above for a grid of backtest configurations
# ─────────────────────────────────────────────────────────────────────────────

# ─── standard normal distribution (dependency-free) ──────────────────────────

const _SQRTPI = sqrt(π)

# erfc on x ≥ 0: Taylor series of erf below 2.5, Laplace continued fraction above
function _erfc_pos(x::Float64)
    if x < 2.5
        x2 = x * x
        term = x
        s = x
        for n in 1:200
            term *= -x2 / n
            add = term / (2n + 1)
            s += add
            abs(add) <= 1e-17 * abs(s) && break
        end
        return 1.0 - 2.0 / _SQRTPI * s
    end
    # erfc(x) = exp(-x²)/√π · 1/(x + ½/(x + 1/(x + (3/2)/(x + 2/(x + …))))), evaluated backwards
    k = x
    for n in 80:-1:1
        k = x + (n / 2) / k
    end
    exp(-x * x) / (_SQRTPI * k)
end
_erfc(x::Float64) = x >= 0 ? _erfc_pos(x) : 2.0 - _erfc_pos(-x)

"""
    normcdf(z) -> Float64

Standard normal CDF Φ(z), accurate to near machine precision (no external dependency).
"""
normcdf(z::Real) = 0.5 * _erfc(-Float64(z) / sqrt(2.0))

"""
    norminvcdf(p) -> Float64

Standard normal quantile Φ⁻¹(p) for `0 < p < 1`: Acklam's rational approximation followed
by one Halley step on [`normcdf`](@ref), accurate to near machine precision.
"""
function norminvcdf(p::Real)
    p = Float64(p)
    0 < p < 1 || throw(DomainError(p, "p must be in (0, 1)"))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    x = if p < plow
        q = sqrt(-2log(p))
        (((((c[1]q + c[2])q + c[3])q + c[4])q + c[5])q + c[6]) / ((((d[1]q + d[2])q + d[3])q + d[4])q + 1)
    elseif p <= 1 - plow
        q = p - 0.5
        r = q * q
        (((((a[1]r + a[2])r + a[3])r + a[4])r + a[5])r + a[6])q /
        (((((b[1]r + b[2])r + b[3])r + b[4])r + b[5])r + 1)
    else
        q = sqrt(-2log1p(-p))
        -(((((c[1]q + c[2])q + c[3])q + c[4])q + c[5])q + c[6]) / ((((d[1]q + d[2])q + d[3])q + d[4])q + 1)
    end
    # one Halley refinement step
    e = normcdf(x) - p
    u = e * sqrt(2π) * exp(x * x / 2)
    x - u / (1 + x * u / 2)
end

# ─── Sharpe-ratio inference ──────────────────────────────────────────────────

# per-period Sharpe ratio, skewness and (non-excess) kurtosis with population moments
function _sharpe_moments(r::AbstractVector{<:Real})
    n = length(r)
    n >= 3 || return (NaN, NaN, NaN, n)
    μ = mean(r)
    m2 = 0.0; m3 = 0.0; m4 = 0.0
    @inbounds for x in r
        d = x - μ
        d2 = d * d
        m2 += d2; m3 += d2 * d; m4 += d2 * d2
    end
    m2 /= n; m3 /= n; m4 /= n
    sd = sqrt(m2 * n / (n - 1))
    sd > _MIN_RETURN_SD || return (_riskless_sharpe(μ), 0.0, 3.0, n)
    (μ / sd, m3 / m2^1.5, m4 / m2^2, n)
end

"""
    probabilistic_sharpe(r; benchmark=0.0) -> Float64

Probabilistic Sharpe Ratio: the probability that the true per-period Sharpe ratio of the
return series `r` exceeds `benchmark` (per period), accounting for sample length,
skewness `γ₃` and kurtosis `γ₄` of the returns:

    PSR = Φ( (SR̂ − SR*) √(T − 1) / √(1 − γ₃ SR̂ + (γ₄ − 1)/4 · SR̂²) )
"""
function probabilistic_sharpe(r::AbstractVector{<:Real}; benchmark::Real = 0.0)
    sr, g3, g4, n = _sharpe_moments(r)
    isnan(sr) && return NaN
    isinf(sr) && return sr > benchmark ? 1.0 : 0.0          # riskless series
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr^2
    den > 0 || return NaN
    normcdf((sr - benchmark) * sqrt(n - 1) / sqrt(den))
end

const _EULER_GAMMA = 0.5772156649015329

"""
    expected_max_sharpe(n_trials, sr_variance) -> Float64

Expected maximum of `n_trials` per-period Sharpe ratios whose true value is zero and
whose estimates have variance `sr_variance` (the false-positive benchmark of the
Deflated Sharpe Ratio):

    E[max SR] ≈ √V · ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))

with `γ` the Euler-Mascheroni constant. Zero for a single trial.
"""
function expected_max_sharpe(n_trials::Integer, sr_variance::Real)
    n_trials >= 1 || throw(ArgumentError("n_trials must be ≥ 1"))
    sr_variance >= 0 || throw(ArgumentError("sr_variance must be ≥ 0"))
    n_trials == 1 && return 0.0
    sqrt(sr_variance) * ((1 - _EULER_GAMMA) * norminvcdf(1 - 1 / n_trials) +
                         _EULER_GAMMA * norminvcdf(1 - 1 / (n_trials * ℯ)))
end

"""
    deflated_sharpe(r, trial_sharpes) -> NamedTuple
    deflated_sharpe(r; n_trials, sr_variance) -> NamedTuple

Deflated Sharpe Ratio of the chosen return series `r` after trying several strategies.
`trial_sharpes` are the per-period Sharpe ratios of *all* trials (including the chosen
one); their count and variance give the benchmark [`expected_max_sharpe`](@ref), the
Sharpe ratio the best of that many skill-less trials would reach by luck. The DSR is the
[`probabilistic_sharpe`](@ref) against that benchmark.

Returns `(dsr, psr, sharpe, benchmark, n_trials)` (Sharpe ratios per period; `psr` is
against zero). A DSR above 0.95 means the selected strategy beats the best-of-`n`
luck benchmark at 95% confidence. If trials are strongly correlated, pass a smaller
effective `n_trials`.
"""
deflated_sharpe(r::AbstractVector{<:Real}, trial_sharpes::AbstractVector{<:Real}) =
    deflated_sharpe(r; n_trials = length(trial_sharpes), sr_variance = var(filter(isfinite, trial_sharpes)))

function deflated_sharpe(r::AbstractVector{<:Real}; n_trials::Integer, sr_variance::Real)
    sr = _sharpe_moments(r)[1]
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    (dsr = probabilistic_sharpe(r; benchmark = sr0), psr = probabilistic_sharpe(r), sharpe = sr,
     benchmark = sr0, n_trials = Int(n_trials))
end

# ─── Probability of Backtest Overfitting (CSCV) ──────────────────────────────

@inline function _block_sharpe(s::Float64, ss::Float64, c::Int)
    c >= 2 || return 0.0
    v = (ss - s * s / c) / (c - 1)
    v > _MIN_RETURN_SD^2 ? (s / c) / sqrt(v) : 0.0
end

"""
    pbo(R; n_blocks=16) -> NamedTuple

Probability of Backtest Overfitting by combinatorially symmetric cross-validation (CSCV).
`R` is a `T × N` matrix: per-period returns of `N` strategy configurations over the same
`T` periods.

The rows are cut into `n_blocks` contiguous blocks (even, ≤ 20). For each of the
`C(n_blocks, n_blocks/2)` ways to pick half of the blocks as in-sample, the configuration
with the best in-sample Sharpe ratio is found and its out-of-sample relative rank `ω` is
recorded as a logit `λ = log(ω / (1 − ω))`. PBO is the share of splits where that
configuration lands in the bottom half out of sample (`λ < 0`; ties count half).

Returns `(pbo, logits, is_sharpe, oos_sharpe, oos_is_ratio, degradation, prob_oos_loss, n_splits)`:
per split, the in- and out-of-sample per-period Sharpe of the in-sample winner;
`oos_is_ratio`, mean OOS over mean IS Sharpe of the winner (near 1 when in-sample
performance carries over, near 0 or negative when it was luck); `degradation`, the slope
of OOS on IS Sharpe across splits from the CSCV paper (because the halves partition the
same series, this slope is negative even for a genuinely skilled strategy, so read it
alongside the ratio); and how often the winner loses money out of sample. Block
aggregates make each split `O(n_blocks · N)`.
"""
function pbo(R::AbstractMatrix{<:Real}; n_blocks::Integer = 16)
    T, N = size(R)
    S = Int(n_blocks)
    (iseven(S) && 2 <= S <= 20) || throw(ArgumentError("n_blocks must be even and in 2:20"))
    N >= 2 || throw(ArgumentError("need at least 2 configurations"))
    T >= 2S || throw(ArgumentError("need at least 2 rows per block ($(2S) rows for $S blocks)"))
    bounds = round.(Int, range(0, T; length = S + 1))
    bs = zeros(S, N); bss = zeros(S, N); bc = zeros(Int, S)
    for b in 1:S
        rows = bounds[b]+1:bounds[b+1]
        bc[b] = length(rows)
        for n in 1:N, t in rows
            x = Float64(R[t, n])
            bs[b, n] += x
            bss[b, n] += x * x
        end
    end
    half = S ÷ 2
    is_sr = zeros(N); oos_sr = zeros(N)
    logits = Float64[]; ism = Float64[]; oosm = Float64[]
    for mask in 0:(1<<S)-1
        count_ones(mask) == half || continue
        for n in 1:N
            si = 0.0; ssi = 0.0; ci = 0; so = 0.0; sso = 0.0; co = 0
            for b in 1:S
                if (mask >> (b - 1)) & 1 == 1
                    si += bs[b, n]; ssi += bss[b, n]; ci += bc[b]
                else
                    so += bs[b, n]; sso += bss[b, n]; co += bc[b]
                end
            end
            is_sr[n] = _block_sharpe(si, ssi, ci)
            oos_sr[n] = _block_sharpe(so, sso, co)
        end
        best = argmax(is_sr)
        x = oos_sr[best]
        below = count(<(x), oos_sr)
        ties = count(==(x), oos_sr) - 1
        rank = below + 0.5ties + 1                 # 1 = worst … N = best
        ω = rank / (N + 1)
        push!(logits, log(ω / (1 - ω)))
        push!(ism, is_sr[best]); push!(oosm, x)
    end
    m_is = mean(ism)
    sxx = sum(abs2, ism .- m_is)
    slope = sxx > 0 ? sum((ism .- m_is) .* (oosm .- mean(oosm))) / sxx : NaN
    (pbo = (count(<(0), logits) + 0.5count(==(0), logits)) / length(logits), logits = logits,
     is_sharpe = ism, oos_sharpe = oosm, oos_is_ratio = m_is != 0 ? mean(oosm) / m_is : NaN, degradation = slope,
     prob_oos_loss = count(<(0), oosm) / length(oosm), n_splits = length(logits))
end

# ─── stationary block bootstrap ──────────────────────────────────────────────

"""
    stationary_bootstrap!(out, r, rng; block=20) -> out

Fill `out` with one stationary-bootstrap resample of `r` (Politis & Romano 1994): blocks
start at uniformly random positions, have geometric lengths with mean `block`, and wrap
around the end. Preserves serial dependence up to roughly the block length.
"""
function stationary_bootstrap!(out::AbstractVector, r::AbstractVector, rng::AbstractRNG; block::Real = 20)
    n = length(r)
    n >= 1 || throw(ArgumentError("empty series"))
    block >= 1 || throw(ArgumentError("block must be ≥ 1"))
    p = 1 / block
    i = rand(rng, 1:n)
    @inbounds for k in eachindex(out)
        out[k] = r[i]
        i = rand(rng) < p ? rand(rng, 1:n) : (i == n ? 1 : i + 1)
    end
    out
end

"""
    bootstrap_ci(stat, r; samples=2000, block=20, level=0.95, rng=Xoshiro(0)) -> NamedTuple

Percentile confidence interval of `stat(r)` (any function of a return vector) from
`samples` stationary-bootstrap resamples. Returns `(estimate, lo, hi, se)`.

    bootstrap_ci(x -> performance(x).sharpe, res.returns)
"""
function bootstrap_ci(stat, r::AbstractVector{<:Real}; samples::Integer = 2000, block::Real = 20,
                      level::Real = 0.95, rng::AbstractRNG = Xoshiro(0))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    samples >= 10 || throw(ArgumentError("samples must be ≥ 10"))
    buf = similar(r, Float64)
    vals = Vector{Float64}(undef, samples)
    for b in 1:samples
        vals[b] = Float64(stat(stationary_bootstrap!(buf, r, rng; block = block)))
    end
    v = filter(isfinite, vals)
    α = (1 - level) / 2
    (estimate = Float64(stat(r)), lo = quantile(v, α), hi = quantile(v, 1 - α), se = std(v))
end

# ─── grids and the overfitting report ────────────────────────────────────────

"""
    parameter_grid(base; field = values, ...) -> Vector{BacktestConfig}

Every combination of the given values for top-level [`BacktestConfig`](@ref) fields,
applied to `base` with [`reconfigure`](@ref). Nested settings are varied by passing whole
config objects:

    parameter_grid(BacktestConfig(); z_entry = [1.5, 2.0, 2.5], z_exit = [0.25, 0.5],
                   kalman = [KalmanConfig(q_scale = q) for q in (0.5, 1.0, 2.0)])
"""
function parameter_grid(base::BacktestConfig; kw...)
    isempty(kw) && return [base]
    names = collect(keys(kw))
    vec([reconfigure(base; (names .=> combo)...) for combo in Iterators.product(values(kw)...)])
end

"""
    OverfittingReport

Result of [`overfitting_report`](@ref): the `configs` tried, their per-bar excess
`returns` (`T × N`), annualised `sharpe` ratios, the index of the `best` configuration,
its deflated Sharpe ratio `dsr` and probabilistic Sharpe ratio `psr`, the annualised
luck benchmark `sharpe_benchmark` (the best of `n_trials` skill-less configurations), the
CSCV probability of backtest overfitting `pbo` with its split `logits`, `oos_is_ratio`
(mean out-of-sample over mean in-sample Sharpe of the CSCV winner), the probability that
the in-sample winner loses out of sample (`prob_oos_loss`), and a bootstrap confidence
interval for the best configuration's annualised Sharpe ratio (`sharpe_ci`, at `level`).
"""
struct OverfittingReport
    configs::Vector{BacktestConfig}
    returns::Matrix{Float64}
    sharpe::Vector{Float64}
    best::Int
    dsr::Float64
    psr::Float64
    sharpe_benchmark::Float64
    n_trials::Int
    pbo::Float64
    logits::Vector{Float64}
    oos_is_ratio::Float64
    prob_oos_loss::Float64
    sharpe_ci::Tuple{Float64,Float64}
    level::Float64
end

# top-level config fields whose values differ across the configurations
function _varying_fields(cfgs::AbstractVector{BacktestConfig})
    [f for f in fieldnames(BacktestConfig) if f !== :threaded_formation &&
         any(c -> getfield(c, f) != getfield(cfgs[1], f), cfgs)]
end

_short(x) = x
function _short(x::Union{KalmanConfig,ExecConfig,RiskConfig})
    d = typeof(x)()
    ch = [string(f, "=", getfield(x, f)) for f in fieldnames(typeof(x)) if getfield(x, f) != getfield(d, f)]
    string(nameof(typeof(x)), "(", join(ch, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::OverfittingReport)
    verdict(good) = good ? "✓" : "⚠"
    println(io, "OverfittingReport: ", length(r.configs), " configurations, ", size(r.returns, 1), " bars")
    if all(iszero, r.sharpe)
        println(io, "  ⚠ no configuration took any risk (nothing was traded): there is nothing to evaluate")
    end
    fs = _varying_fields(r.configs)
    settings = join((string(f, " = ", _short(getfield(r.configs[r.best], f))) for f in fs), ", ")
    @printf(io, "  best configuration       #%d, Sharpe %.2f  (%.0f%% CI %.2f to %.2f)\n", r.best,
            r.sharpe[r.best], 100r.level, r.sharpe_ci[1], r.sharpe_ci[2])
    isempty(settings) || println(io, "                           ", settings)
    @printf(io, "  luck benchmark           Sharpe %.2f  (expected best of %d skill-less trials)\n",
            r.sharpe_benchmark, r.n_trials)
    @printf(io, "  probabilistic Sharpe     %.3f  (P[true Sharpe > 0])\n", r.psr)
    @printf(io, "  deflated Sharpe          %.3f  %s (P[true Sharpe > luck benchmark]; want > 0.95)\n",
            r.dsr, verdict(r.dsr > 0.95))
    @printf(io, "  PBO (CSCV)               %.3f  %s (P[in-sample winner is below median out of sample]; want < 0.5, ideally < 0.2)\n",
            r.pbo, verdict(r.pbo < 0.5))
    @printf(io, "  OOS / IS Sharpe ratio    %.2f  (≈1: in-sample performance carries over; ≈0 or below: it was luck)\n",
            r.oos_is_ratio)
    @printf(io, "  P[OOS loss | IS winner]  %.3f\n", r.prob_oos_loss)
end
Base.show(io::IO, r::OverfittingReport) =
    print(io, "OverfittingReport(", length(r.configs), " configs, DSR ", round(r.dsr; digits = 3),
          ", PBO ", round(r.pbo; digits = 3), ")")

"""
    overfitting_report(panel, configs; n_blocks=16, samples=2000, block=20, level=0.95,
                       n_trials=length(configs), rng=Xoshiro(0), threaded=true) -> OverfittingReport

Run every configuration (in parallel, sharing formations through a [`FormationCache`](@ref)),
pick the one with the highest annualised Sharpe ratio on excess returns, and ask whether
that choice is real:

- [`deflated_sharpe`](@ref) against the best Sharpe ratio `n_trials` skill-less
  configurations would reach by luck (lower `n_trials` if the grid is highly correlated)
- [`pbo`](@ref) by CSCV over the `T × N` matrix of excess returns
- a stationary-bootstrap confidence interval ([`bootstrap_ci`](@ref)) for the best Sharpe

All configurations must share the formation window so their return series line up.
Build grids with [`parameter_grid`](@ref).
"""
function overfitting_report(panel::PricePanel, configs::AbstractVector{BacktestConfig}; n_blocks::Integer = 16,
                            samples::Integer = 2000, block::Real = 20, level::Real = 0.95,
                            n_trials::Integer = length(configs), rng::AbstractRNG = Xoshiro(0),
                            threaded::Bool = true)
    N = length(configs)
    N >= 2 || throw(ArgumentError("need at least 2 configurations"))
    F = configs[1].formation
    all(c -> c.formation == F, configs) ||
        throw(ArgumentError("all configurations must share the formation window so their returns align"))
    cfgs = [reconfigure(c; threaded_formation = false) for c in configs]
    cache = FormationCache(panel)
    rets = Vector{Vector{Float64}}(undef, N)
    foreach_chunk(N; threaded = threaded) do chunk
        for i in chunk
            res = run_backtest(panel, cfgs[i]; formation_cache = cache)
            rets[i] = res.returns .- cfgs[i].exec.cash_rate / res.bars_per_year
        end
    end
    T = length(rets[1])
    R = Matrix{Float64}(undef, T, N)
    for i in 1:N
        R[:, i] = rets[i]
    end
    bpy = panel.bars_per_year
    per_period = [_sharpe_moments(view(R, :, i))[1] for i in 1:N]
    ann = per_period .* sqrt(bpy)
    best = argmax(map(x -> isfinite(x) ? x : -Inf, ann))
    rb = view(R, :, best)
    d = deflated_sharpe(rb; n_trials = n_trials, sr_variance = var(filter(isfinite, per_period)))
    p = pbo(R; n_blocks = n_blocks)
    ci = bootstrap_ci(x -> _sharpe_moments(x)[1] * sqrt(bpy), rb; samples = samples, block = block,
                      level = level, rng = rng)
    OverfittingReport(collect(configs), R, ann, best, d.dsr, d.psr, d.benchmark * sqrt(bpy), Int(n_trials),
                      p.pbo, p.logits, p.oos_is_ratio, p.prob_oos_loss, (ci.lo, ci.hi), Float64(level))
end
