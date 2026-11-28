# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics, factor attribution, model-health telemetry, text dashboard.
# ─────────────────────────────────────────────────────────────────────────────

# Below this per-period standard deviation a return series carries no risk: a strategy that
# never trades has excess returns of pure floating-point dust, whose mean/σ is meaningless.
# Such a series has Sharpe 0, unless its mean is clearly non-zero (riskless excess return: ±Inf).
const _MIN_RETURN_SD = 1e-10
_riskless_sharpe(μ) = abs(μ) <= _MIN_RETURN_SD ? 0.0 : copysign(Inf, μ)

"""
    performance(returns; bars_per_year=252, rf=0.0) -> NamedTuple

Annualised return / volatility, Sharpe and Sortino on returns in excess of the annual
risk-free rate `rf`, max drawdown, Calmar, hit rate. (A market-neutral book earns interest
on idle cash; without `rf` that carry would masquerade as alpha.)
"""
function performance(r::AbstractVector{<:Real}; bars_per_year::Real = 252.0, rf::Real = 0.0)
    n = length(r)
    n == 0 && return (total_return = 0.0, cagr = 0.0, vol = 0.0, sharpe = NaN, sortino = NaN,
                      max_drawdown = 0.0, calmar = NaN, hit_rate = NaN, n = 0)
    rfb = rf / bars_per_year
    μ = mean(r) - rfb; σ = std(r)
    dd = sqrt(mean(x -> min(x - rfb, 0.0)^2, r))
    eq = cumprod(1 .+ r)
    peak = accumulate(max, eq)
    mdd = maximum(1 .- eq ./ peak)
    years = n / bars_per_year
    tot = eq[end] - 1
    cagr = eq[end] > 0 ? eq[end]^(1 / years) - 1 : -1.0
    nz = count(!iszero, r)
    (total_return = tot, cagr = cagr, vol = σ * sqrt(bars_per_year),
     sharpe = σ > _MIN_RETURN_SD ? μ / σ * sqrt(bars_per_year) : _riskless_sharpe(μ),
     sortino = dd > _MIN_RETURN_SD ? μ / dd * sqrt(bars_per_year) : _riskless_sharpe(μ),
     max_drawdown = mdd, calmar = mdd > 0 ? cagr / mdd : NaN,
     hit_rate = nz > 0 ? count(>(0), r) / nz : NaN, n = n)
end
performance(res::BacktestResult; rf::Real = res.config.exec.cash_rate) =
    performance(res.returns; bars_per_year = res.bars_per_year, rf = rf)

"""
    factor_attribution(returns, factors, names; bars_per_year=252) -> NamedTuple

Time-series regression `r_t = a + Σ b_k f_{k,t} + e_t` (e.g. Fama-French 5 + short-term
reversal). Returns annualised alpha, loadings, t-statistics and R².
"""
function factor_attribution(r::AbstractVector, F::AbstractMatrix, names::AbstractVector;
                            bars_per_year::Real = 252.0)
    n, K = size(F)
    length(r) == n || throw(DimensionMismatch("returns and factors must share the time axis"))
    length(names) == K || throw(DimensionMismatch("names"))
    X = hcat(ones(n), F)
    b = X \ r
    e = r - X * b
    σ2 = sum(abs2, e) / (n - K - 1)
    se = sqrt.(σ2 .* diag(inv(Symmetric(X' * X))))
    r2 = 1 - sum(abs2, e) / sum(abs2, r .- mean(r))
    (alpha_ann = b[1] * bars_per_year, alpha_t = b[1] / se[1],
     names = collect(String.(string.(names))), loadings = b[2:end], tstats = (b./se)[2:end], r2 = r2)
end

"""
    panel_factors(panel, t_range) -> (F, names)

Factors computable from the panel itself: equal-weight market (`MKT`) and a cross-sectional
short-term reversal factor (`STR`: long previous-bar losers, short winners). Merge with
externally sourced Fama-French factors for a full attribution.
"""
function panel_factors(p::PricePanel, trange::AbstractVector{<:Integer})
    n = nassets(p)
    F = zeros(length(trange), 2)
    r = zeros(n); rp = zeros(n); wts = zeros(n)          # reused across bars
    for (k, t) in enumerate(trange)
        t >= 3 || continue
        @inbounds for a in 1:n
            r[a] = p.mid[a, t] / p.mid[a, t-1] - 1
            rp[a] = p.mid[a, t-1] / p.mid[a, t-2] - 1
        end
        F[k, 1] = mean(r)
        μp = mean(rp)
        @. wts = -(rp - μp)
        s = sum(abs, wts)
        F[k, 2] = s > 0 ? sum(i -> wts[i] * r[i], 1:n) / (s / 2) : 0.0
    end
    F, ["MKT", "STR"]
end

# call f(key, indices) for every distinct key, in key order; indices keep their original order
function _foreach_group(f, keys::AbstractVector)
    ord = sortperm(keys; alg = Base.Sort.DEFAULT_STABLE)
    i = 1
    while i <= length(ord)
        j = i
        while j < length(ord) && keys[ord[j+1]] == keys[ord[i]]
            j += 1
        end
        f(keys[ord[i]], view(ord, i:j))
        i = j + 1
    end
end

"""
    health_telemetry(res) -> NamedTuple

Model-health metrics:
- `coint_survival`: fraction of health checks that passed the gate
- `rank_survival`: fraction of checks with Johansen rank ≥ 1
- `trace_trajectory`: (t, median trace statistic across live pairs) per check date
- `half_life_cv`: median across pairs of the coefficient of variation of the half-life
- `innov_acf1`: mean lag-1 autocorrelation of standardised Kalman innovations
- `quarantine_rate`: fraction of pair-checks in quarantine
"""
function health_telemetry(res::BacktestResult)
    h = res.health
    isempty(h) && return (coint_survival = NaN, rank_survival = NaN, trace_trajectory = (Int[], Float64[]),
                          half_life_cv = NaN, innov_acf1 = NaN, quarantine_rate = NaN, n_checks = 0)
    # group by bar / by spread with one stable sort each (linear scans per group would be
    # quadratic in the number of health checks)
    ts = Int[]; med = Float64[]
    _foreach_group(h.t) do t, idx
        push!(ts, t); push!(med, median(view(h.trace_r0, idx)))
    end
    cvs = Float64[]
    _foreach_group(h.pair) do _, idx
        hl = filter(isfinite, view(h.half_life, idx))
        length(hl) >= 3 && push!(cvs, std(hl) / mean(hl))
    end
    acf = filter(isfinite, h.innov_acf1)
    (coint_survival = mean(h.healthy), rank_survival = mean(h.rank .>= 1),
     trace_trajectory = (ts, med), half_life_cv = isempty(cvs) ? NaN : median(cvs),
     innov_acf1 = isempty(acf) ? NaN : mean(acf),
     quarantine_rate = mean(h.state .== QUARANTINE), n_checks = length(h))
end

"""Execution-cost breakdown in currency units and as bps of traded notional."""
function cost_breakdown(res::BacktestResult)
    b = res.book
    tn = max(b.traded_notional, eps())
    bps(x) = 1e4 * x / tn
    (traded_notional = b.traded_notional, spread = b.spread_paid, impact = b.impact_paid,
     commission = b.commission_paid, borrow = b.borrow_paid, interest_net = b.interest_net,
     dividends_net = b.dividends_net,
     spread_bps = bps(b.spread_paid), impact_bps = bps(b.impact_paid), commission_bps = bps(b.commission_paid),
     n_fills = b.n_fills, n_legged = b.n_legged, n_passive = b.n_passive)
end

"""
    report([io], res; factors=nothing, factor_names=nothing)

Dual dashboard: trading performance + execution costs + model-health telemetry.
`factors` is a `T−1 × K` matrix aligned with `res.returns`.
"""
report(res::BacktestResult; kw...) = report(stdout, res; kw...)
function report(io::IO, res::BacktestResult; factors = nothing, factor_names = nothing)
    p = performance(res)
    c = cost_breakdown(res)
    h = health_telemetry(res)
    line = "─"^64
    println(io, line)
    println(io, " StatArb — ", first(res.timestamps), " → ", last(res.timestamps),
            "  (", length(res.returns), " bars)")
    println(io, line)
    println(io, " PERFORMANCE")
    @printf(io, "   Total return   %9.2f %%     CAGR          %8.2f %%\n", 100p.total_return, 100p.cagr)
    @printf(io, "   Volatility     %9.2f %%     Max drawdown  %8.2f %%\n", 100p.vol, 100p.max_drawdown)
    @printf(io, "   Sharpe (xs rf) %9.2f       Sortino       %8.2f\n", p.sharpe, p.sortino)
    @printf(io, "   Calmar         %9.2f       Hit rate      %8.2f %%\n", p.calmar, 100p.hit_rate)
    if factors !== nothing
        names = factor_names === nothing ? ["F$k" for k in 1:size(factors, 2)] : factor_names
        fa = factor_attribution(res.returns .- res.config.exec.cash_rate / res.bars_per_year, factors, names;
                                bars_per_year = res.bars_per_year)
        println(io, " FACTOR ATTRIBUTION  (R² = ", round(fa.r2; digits = 3), ")")
        @printf(io, "   alpha (ann.)   %9.2f %%     t = %6.2f\n", 100fa.alpha_ann, fa.alpha_t)
        for (nm, b, tt) in zip(fa.names, fa.loadings, fa.tstats)
            @printf(io, "   %-12s   %9.4f       t = %6.2f\n", nm, b, tt)
        end
    end
    println(io, " EXECUTION")
    @printf(io, "   Traded notional %12.0f   fills %d  (passive %d, legged %d)\n",
            c.traded_notional, c.n_fills, c.n_passive, c.n_legged)
    @printf(io, "   Spread %10.0f (%5.2f bps)  Impact %10.0f (%5.2f bps)  Comm %8.0f (%4.2f bps)\n",
            c.spread, c.spread_bps, c.impact, c.impact_bps, c.commission, c.commission_bps)
    @printf(io, "   Borrow fees %10.0f   Net interest %10.0f   Net dividends %10.0f\n",
            c.borrow, c.interest_net, c.dividends_net)
    if !isempty(res.breaks)
        b = res.breaks
        @printf(io, "   Structural breaks: %d alarms on %d spreads (mean ↑ %d, mean ↓ %d, variance %d)\n",
                length(b), length(unique(b.pair)), count(==(:mean_up), b.kind), count(==(:mean_down), b.kind),
                count(==(:variance), b.kind))
    end
    isempty(res.margin_calls) ||
        println(io, "   ⚠ Margin calls (forced liquidation) at bars: ", join(res.margin_calls, ", "))
    println(io, " MODEL HEALTH")
    @printf(io, "   Health checks %6d   gate pass %5.1f %%   Johansen r≥1 %5.1f %%   quarantined %5.1f %%\n",
            h.n_checks, 100h.coint_survival, 100h.rank_survival, 100h.quarantine_rate)
    @printf(io, "   Half-life CV (median) %6.3f   Kalman innovation ACF(1) %6.3f\n", h.half_life_cv, h.innov_acf1)
    tts, tr = h.trace_trajectory
    if !isempty(tr)
        @printf(io, "   Trace stat (median across pairs): first %6.2f  median %6.2f  last %6.2f\n",
                first(tr), median(tr), last(tr))
        println(io, "   Trace trajectory ", sparkline(tr))
    end
    println(io, "   Formations: ", length(res.formations), "   pairs selected/formation: ",
            join([length(f.selected) for f in res.formations], " "))
    if any(f -> f.n_basket_candidates > 0, res.formations)
        println(io, "   Baskets selected/formation: ", join([length(f.baskets) for f in res.formations], " "))
    end
    println(io, line)
    nothing
end

const _SPARKS = ('▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')
"""Unicode sparkline of a series (downsampled to `width` points)."""
function sparkline(x::AbstractVector; width::Int = 48)
    y = filter(isfinite, x)
    isempty(y) && return ""
    idx = round.(Int, range(1, length(y); length = min(width, length(y))))
    y = y[idx]
    lo, hi = extrema(y)
    join(_SPARKS[1+round(Int, (hi > lo ? (v - lo) / (hi - lo) : 0.5) * 7)] for v in y)
end
