# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward event loop: formation → live trading with health gating.
#
# A traded *spread* is either a pair (Kalman hedge ratio) or a basket of k ≥ 3 assets
# (Johansen cointegrating vector). Per bar t, on a single monotonic clock, the loop
# runs these stages:
#
#   _stage_market!     financing, corporate actions, fills of orders decided ≤ t − latency
#   _stage_formation!  every `trading` bars: screen, test (FDR), select, calibrate
#   _stage_models!     filters + periodic Engle-Granger / Johansen health checks
#   _stage_risk!       maintenance-margin check
#   _stage_signals!    the configured signal rule (pluggable)
#   _stage_allocate!   the configured allocator (pluggable), throttle, gross cap, leg split
#   _stage_orders!     hysteresis-aware order generation (eligible from t + latency)
#   _stage_record!     mark to market, telemetry
# ─────────────────────────────────────────────────────────────────────────────

"""
    BacktestConfig(; ...)

Walk-forward, selection and signal parameters (see the field comments in the source).
Nested configs: `kalman::KalmanConfig`, `exec::ExecConfig`, `risk::RiskConfig`.
Pluggable components: `signal_rule::AbstractSignalRule` (default [`ZScoreBands`](@ref)) and
`allocator::AbstractAllocator` (default [`DefaultAllocator`](@ref)).

Baskets: with `max_baskets > 0`, formation also tests `basket_size`-asset baskets (built
by extending the closest screened pairs) with the Johansen test, and trades up to
`max_baskets` of them alongside the pairs.
"""
Base.@kwdef struct BacktestConfig
    formation::Int = 252                 # formation window F (bars)
    trading::Int = 63                    # bars between formations
    health_window::Int = 250             # rolling window W for health checks
    health_every::Int = 5                # bars between health checks
    johansen_lags::Int = 1               # p − 1 lagged differences in the VECM
    adf_lags::Int = 1
    coint_level::Float64 = 0.05          # per-test level for health checks (and selection if FDR is off)
    selection_fdr::Float64 = 0.05        # Benjamini-Hochberg FDR across formation candidates; 0 = off
    null_sims::Int = 10_000              # Monte Carlo draws behind the selection p-values
    min_tradable_frac::Float64 = 0.95    # assets need this share of tradable bars in the formation window
    max_pairs::Int = 10
    screen_top::Int = 100                # candidates kept by the distance screen
    max_baskets::Int = 0                 # baskets traded alongside pairs (0 = pairs only)
    basket_size::Int = 3                 # assets per basket (3–5)
    basket_screen_top::Int = 50          # basket candidates tested per formation
    min_basket_weight::Float64 = 0.15    # every leg needs |wᵢ| ≥ this × max|w| (else it's a pair in disguise)
    same_sector_only::Bool = true
    unique_assets::Bool = true           # an asset may appear in at most one selected spread
    hl_bounds::Tuple{Float64,Float64} = (1.0, 60.0)
    gate_on::Symbol = :johansen          # pairs: :johansen | :eg | :either (baskets: Johansen)
    recover_checks::Int = 2
    z_entry::Float64 = 2.0
    z_exit::Float64 = 0.5
    z_stop::Float64 = 4.0
    max_hold_halflives::Float64 = 4.0    # time stop in units of the (assumed) half-life
    hl_scale::Float64 = 1.0              # assumed hl = estimated hl × hl_scale (< 1 ⇒ overestimate speed)
    innov_window::Int = 60               # innovation ring for autocorrelation telemetry
    initial_capital::Float64 = 1e6
    threaded_formation::Bool = true
    kalman::KalmanConfig = KalmanConfig()
    exec::ExecConfig = ExecConfig()
    risk::RiskConfig = RiskConfig()
    signal_rule::AbstractSignalRule = ZScoreBands()
    allocator::AbstractAllocator = DefaultAllocator()
end

"""
    reconfigure(cfg; kw...)

Copy of an immutable config (`BacktestConfig`, `ExecConfig`, `RiskConfig`,
`KalmanConfig`, …) with the given fields replaced:
`reconfigure(cfg; exec = reconfigure(cfg.exec; style = :passive))`.
"""
function reconfigure(cfg::T; kw...) where {T}
    for k in keys(kw)
        k in fieldnames(T) || throw(ArgumentError("$T has no field `$k`"))
    end
    vals = map(fieldnames(T)) do f
        haskey(kw, f) ? kw[f] : getfield(cfg, f)
    end
    T(vals...)
end

"""
    with(cfg; kw...)

Deprecated alias of [`reconfigure`](@ref).
"""
function with(cfg; kw...)
    Base.depwarn("`with(cfg; ...)` is deprecated, use `reconfigure(cfg; ...)`", :with)
    reconfigure(cfg; kw...)
end

"""
    SpreadState

Live state of one traded spread: a pair (`length(assets) == 2`, hedge ratio from a Kalman
filter) or a basket (Johansen weights, [`BasketFilter`](@ref) signal).

- `assets`, `weights`: legs and log-price hedge weights (`weights[1] = 1`; the spread is
  `Σ wᵢ log pᵢ`; a pair's weights are `[1, −β_t]`, refreshed every bar)
- `model`: the filter; `zscore(sp)` is its trading signal
- `signal` / `last_signal`: desired direction this bar / last bar; `entry_t`, `blocked`
- `state`: [`HealthState`](@ref); `hl`, `rank`: latest health-check estimates
- `shares`: filled position per leg; `target`: this bar's target shares per leg
"""
mutable struct SpreadState
    id::Int
    assets::Vector{Int}
    weights::Vector{Float64}
    model::Union{KalmanHedge,OUKalmanHedge,BasketFilter}
    ring::MirroredRing          # k × W log prices
    zring::MirroredRing         # standardised innovations
    rring::MirroredRing         # spread returns per $1 gross
    spbuf::Vector{Float64}      # scratch for basket spread windows
    state::HealthState
    good::Int
    in_universe::Bool
    signal::Int
    last_signal::Int
    entry_t::Int
    blocked::Bool
    hl::Float64
    rank::Int
    created_t::Int
    shares::Vector{Float64}
    target::Vector{Float64}
end

"""Number of legs of a spread."""
nlegs(sp::SpreadState) = length(sp.assets)
"""`true` for a two-asset spread traded with a Kalman hedge ratio."""
ispair(sp::SpreadState) = length(sp.assets) == 2
"""The spread's current trading Z-score."""
zscore(sp::SpreadState) = sp.model.z
"""`true` if any leg holds shares."""
isopen(sp::SpreadState) = any(!iszero, sp.shares)
_key(assets) = sort(collect(assets))

"""
    HealthRecord

One health check of one spread: Johansen rank and statistics on the rolling window,
half-lives (Engle-Granger residual for pairs, Johansen spread for baskets), the
Engle-Granger statistic and β (pairs only, `NaN` for baskets), the filter's hedge ratio and
implied half-life, the lag-1 autocorrelation of its standardised innovations, and the
resulting gate state. `pair` is the spread id, `y`/`x` its first two assets, `k` its size.
"""
struct HealthRecord
    t::Int
    pair::Int
    y::Int
    x::Int
    k::Int
    rank::Int
    trace_r0::Float64
    trace_r1::Float64
    maxeig_r0::Float64
    half_life::Float64
    joh_half_life::Float64
    eg_adf::Float64
    eg_beta::Float64
    kf_beta::Float64
    kf_half_life::Float64
    innov_acf1::Float64
    healthy::Bool
    state::HealthState
end

struct BarRecord
    t::Int
    equity::Float64
    gross::Float64
    n_pairs::Int          # live spreads (pairs and baskets)
    n_active::Int
    n_quarantine::Int
    n_open::Int
end

"""
    FormationRecord

One formation step: bar `t`; screened pair candidates `n_candidates`; pairs that passed
selection (`n_cointegrated`: significant after FDR control and within the half-life
bounds); the smallest pair p-value; the `selected` pairs as `(y, x)`; and, for baskets,
`n_basket_candidates`, `n_basket_significant` and the `baskets` selected.
"""
struct FormationRecord
    t::Int
    n_candidates::Int
    n_cointegrated::Int
    min_pvalue::Float64
    selected::Vector{NTuple{2,Int}}
    n_basket_candidates::Int
    n_basket_significant::Int
    baskets::Vector{Vector{Int}}
end

"""
    BacktestResult

Output of [`run_backtest`](@ref): bar indices `t`, `timestamps`, `equity` and `returns`;
columnar telemetry `bars`, `health` and `fills` (`StructVector`s, so they work with any
Tables.jl consumer); `formations`; the final `book`; forced-liquidation bars
`margin_calls`; `(bar, spread id)` of every filter rebuilt after diverging
(`model_resets`); and the config used.
"""
struct BacktestResult
    config::BacktestConfig
    t::Vector{Int}
    timestamps::Vector{DateTime}
    equity::Vector{Float64}
    returns::Vector{Float64}
    bars::StructVector{BarRecord}
    health::StructVector{HealthRecord}
    fills::StructVector{FillRecord}
    formations::Vector{FormationRecord}
    book::Book
    margin_calls::Vector{Int}
    model_resets::Vector{Tuple{Int,Int}}
    bars_per_year::Float64
    symbols::Vector{Symbol}
end

Base.show(io::IO, r::BacktestResult) =
    print(io, "BacktestResult(", length(r.t), " bars, final equity ",
          round(last(r.equity); digits = 2), ", ", length(r.fills), " fills)")

"""
    SignalContext

What a signal rule or allocator may look at on bar `t`: the config, the panel (only bars
`≤ t` may be used), the execution simulator's per-asset volatility estimates `sigma2`,
current `equity`, `can_open` (false during halts or a post-margin-call cooldown) and a
reusable `overlay` workspace for allocators.
"""
struct SignalContext
    t::Int
    cfg::BacktestConfig
    panel::PricePanel
    sigma2::Vector{Float64}
    equity::Float64
    can_open::Bool
    overlay::OverlayWorkspace
end

# ─── state ───────────────────────────────────────────────────────────────────

mutable struct BacktestState
    panel::PricePanel
    cfg::BacktestConfig
    clock::Clock
    sim::ExecutionSimulator
    book::Book
    egws::EGWorkspace
    jws::Dict{Int,JohansenWorkspace}       # health-check workspaces by basket size
    gate::HealthGateConfig
    ows::OverlayWorkspace
    screen::ScreenWorkspace
    spreads::Vector{SpreadState}
    byid::Dict{Int,SpreadState}
    next_id::Int
    health::StructVector{HealthRecord}
    bars::StructVector{BarRecord}
    formations::Vector{FormationRecord}
    margin_calls::Vector{Int}
    model_resets::Vector{Tuple{Int,Int}}
    no_entry_until::Int
    liquidating::Bool
    open_sel::Vector{SpreadState}
    w::Vector{Float64}
    act::Vector{UInt8}
end

function BacktestState(panel::PricePanel, cfg::BacktestConfig)
    n, W = nassets(panel), cfg.health_window
    jws = Dict{Int,JohansenWorkspace}(2 => JohansenWorkspace(2, W; kdiff = cfg.johansen_lags))
    cfg.max_baskets > 0 && (jws[cfg.basket_size] = JohansenWorkspace(cfg.basket_size, W; kdiff = cfg.johansen_lags))
    BacktestState(panel, cfg, Clock(0), ExecutionSimulator(cfg.exec, n), Book(cfg.initial_capital, n),
                  EGWorkspace(W, cfg.adf_lags), jws,
                  HealthGateConfig(cfg.hl_bounds, cfg.gate_on, cfg.recover_checks, cfg.coint_level),
                  OverlayWorkspace(2 * (cfg.max_pairs + cfg.max_baskets)),
                  ScreenWorkspace(n, cfg.formation),
                  SpreadState[], Dict{Int,SpreadState}(), 1,
                  StructVector{HealthRecord}(undef, 0), StructVector{BarRecord}(undef, 0),
                  FormationRecord[], Int[], Tuple{Int,Int}[], 0, false, SpreadState[], Float64[], UInt8[])
end

# ─── formation ───────────────────────────────────────────────────────────────

function _new_spread!(st::BacktestState, assets::Vector{Int}, weights::Vector{Float64}, model, t::Int, hl::Float64)
    cfg, logp = st.cfg, st.panel.logmid
    F, W, L = cfg.formation, cfg.health_window, cfg.risk.cov_window
    k = length(assets)
    zr = MirroredRing(1, cfg.innov_window)
    _warmup!(model, zr, logp, assets, weights, t-F+1:t)
    if k == 2
        # the warm-up moved β: refresh the leg weights and measure the half-life with it
        weights[2] = -hedge_ratio(model)
        rng = t-F+1:t
        hl = ou_half_life(view(logp, assets[1], rng) .+ weights[2] .* view(logp, assets[2], rng))
    end
    ring = MirroredRing(k, W)
    for τ in t-W+1:t
        push_column!(ring, logp, assets, τ)
    end
    rr = MirroredRing(1, L)
    wsum = sum(abs, weights)
    for τ in t-L+1:t
        r = 0.0
        for (l, a) in enumerate(assets)
            r += weights[l] * (logp[a, τ] - logp[a, τ-1])
        end
        push!(rr, r / wsum)
    end
    sp = SpreadState(st.next_id, assets, weights, model, ring, zr, rr, zeros(k == 2 ? 0 : W), ACTIVE, 0, true,
                     0, 0, 0, false, hl, 1, t, zeros(k), zeros(k))
    st.next_id += 1
    push!(st.spreads, sp)
    st.byid[sp.id] = sp
    sp
end

# Run a freshly calibrated filter through the formation window (function barrier: the
# model arrives as one of three concrete types), recording innovations for telemetry.
function _warmup!(model, zr::MirroredRing, logp, assets, weights, rng)
    for τ in rng
        _filter_step!(model, logp, assets, weights, τ)
        push!(zr, innovation_z(model))
    end
    model.P_ref = _ref_variance(model)
    model
end

# Var(β) for pair filters, Var(s) for the basket filter: both live at P[2, 2]
_ref_variance(m) = m.P[2, 2]

# a filter is usable only while its state, covariance and outputs are finite
_model_finite(m) = isfinite(m.z) && m.S > 0 && all(isfinite, m.θ) && all(isfinite, m.P)

@inline _filter_step!(m::Union{KalmanHedge,OUKalmanHedge}, logp, assets, weights, τ) =
    kalman_update!(m, logp[assets[1], τ], logp[assets[2], τ])
@inline function _filter_step!(m::BasketFilter, logp, assets, weights, τ)
    v = 0.0
    @inbounds for l in eachindex(assets)
        v += weights[l] * logp[assets[l], τ]
    end
    kalman_update!(m, v)
end

function _stage_formation!(st::BacktestState, t::Int)
    cfg, panel = st.cfg, st.panel
    F = cfg.formation
    logp = panel.logmid
    lo, hi = cfg.hl_bounds
    # assets with enough real (tradable) bars in the window and tradable right now
    eligible = [panel.tradable[a, t] && count(view(panel.tradable, a, t-F+1:t)) >= cfg.min_tradable_frac * F
                for a in 1:nassets(panel)]
    cands = distance_screen!(st.screen, logp, t, F; top = cfg.screen_top, sector = panel.sector,
                             same_sector = cfg.same_sector_only, eligible = eligible)

    # ── pairs: Engle-Granger + Johansen, FDR across candidates ──
    tests = test_pairs(logp, cands, t, F; adf_lags = cfg.adf_lags, kdiff = cfg.johansen_lags,
                       level = cfg.coint_level, threaded = cfg.threaded_formation)
    if cfg.selection_fdr > 0
        pv = pair_pvalues(tests, F; lags = cfg.adf_lags, kdiff = cfg.johansen_lags, nsim = cfg.null_sims)
        significant = benjamini_hochberg(pv, cfg.selection_fdr)
    else
        pv = fill(NaN, length(tests))
        significant = [iscointegrated(pt.eg, cfg.coint_level) && pt.joh.ok && pt.joh.rank >= 1 for pt in tests]
    end
    keep = [significant[i] && lo <= tests[i].eg.half_life <= hi for i in eachindex(tests)]
    good = tests[keep]
    gp = pv[keep]
    order = cfg.selection_fdr > 0 ? sortperm(collect(zip(gp, [pt.eg.adf for pt in good]))) :
                                    sortperm([pt.eg.adf for pt in good])
    good = good[order]
    used = Set{Int}()
    selected = NTuple{2,Int}[]
    for pt in good
        length(selected) >= cfg.max_pairs && break
        if cfg.unique_assets
            (pt.y in used || pt.x in used) && continue
            push!(used, pt.y, pt.x)
        end
        push!(selected, (pt.y, pt.x))
    end

    # ── baskets: Johansen on k-asset candidates, FDR across candidates ──
    bsel = Vector{Int}[]
    bweights = Vector{Float64}[]
    bhl = Float64[]
    nbc = nbs = 0
    if cfg.max_baskets > 0
        k = cfg.basket_size
        bc = basket_candidates(st.screen.Z, cands, k; top = cfg.basket_screen_top, sector = panel.sector,
                               same_sector = cfg.same_sector_only, eligible = eligible)
        nbc = length(bc)
        scans = scan_johansen(logp, bc, t, F; kdiff = cfg.johansen_lags, level = cfg.coint_level,
                              threaded = cfg.threaded_formation)
        if cfg.selection_fdr > 0
            nd = johansen_null_distribution(F; k = k, kdiff = cfg.johansen_lags, nsim = cfg.null_sims)
            bp = [r.ok ? null_pvalue(nd, r.trace[1]) : 1.0 for r in scans]
            bsig = benjamini_hochberg(bp, cfg.selection_fdr)
        else
            bp = [r.ok ? -r.trace[1] : Inf for r in scans]
            bsig = [r.ok && r.rank >= 1 for r in scans]
        end
        # a basket whose cointegrating vector barely uses a leg is a pair in disguise
        legs_ok(β) = all(isfinite, β) && minimum(abs, β) >= cfg.min_basket_weight * maximum(abs, β)
        bkeep = [i for i in eachindex(scans) if bsig[i] && lo <= scans[i].half_life <= hi &&
                 legs_ok(view(scans[i].beta, 1:k))]
        nbs = length(bkeep)
        for i in bkeep[sortperm(bp[bkeep])]
            length(bsel) >= cfg.max_baskets && break
            r = scans[i]
            if cfg.unique_assets
                any(in(used), r.basket) && continue
                union!(used, r.basket)
            end
            push!(bsel, r.basket)
            push!(bweights, r.beta[1:k])
            push!(bhl, r.half_life)
        end
    end
    push!(st.formations, FormationRecord(t, length(cands), length(good), isempty(pv) ? NaN : minimum(pv),
                                         selected, nbc, nbs, bsel))

    # ── universe update: spreads are identified by their asset set ──
    selkeys = Set{Vector{Int}}(_key(p) for p in selected)
    union!(selkeys, _key(b) for b in bsel)
    existing = Set{Vector{Int}}()
    for sp in st.spreads
        key = _key(sp.assets)
        push!(existing, key)
        sp.in_universe = key in selkeys
    end
    rng = t-F+1:t
    for (yi, xi) in selected
        _key((yi, xi)) in existing && continue             # retained spread keeps its live filter
        model = calibrate_filter(view(logp, yi, rng), view(logp, xi, rng), cfg.kalman; warmup = false)
        _new_spread!(st, [yi, xi], [1.0, -hedge_ratio(model)], model, t, NaN)
    end
    for (b, w, hl) in zip(bsel, bweights, bhl)
        _key(b) in existing && continue
        v = [sum(w[l] * logp[b[l], τ] for l in eachindex(b)) for τ in rng]
        model = calibrate_basket_filter(v, cfg.kalman; warmup = false)
        _new_spread!(st, copy(b), copy(w), model, t, hl)
    end
    nothing
end

# ─── per-bar models & health ─────────────────────────────────────────────────

function _update_spread!(st::BacktestState, sp::SpreadState, t::Int)
    cfg, logp = st.cfg, st.panel.logmid
    k = nlegs(sp)
    # spread return over (t−1, t] with last bar's weights
    r = 0.0; wsum = 0.0
    @inbounds for l in 1:k
        a = sp.assets[l]
        r += sp.weights[l] * (logp[a, t] - latest(sp.ring, l))
        wsum += abs(sp.weights[l])
    end
    push_column!(sp.ring, logp, sp.assets, t)
    _filter_step!(sp.model, logp, sp.assets, sp.weights, t)
    _model_finite(sp.model) || _reset_model!(st, sp, t)
    ispair(sp) && (sp.weights[2] = -hedge_ratio(sp.model))
    push!(sp.zring, innovation_z(sp.model))
    push!(sp.rring, r / wsum)
    (t - sp.created_t) % cfg.health_every == 0 && _health_check!(st, sp, t)
    nothing
end

# A diverged filter (non-finite state or outputs) is rebuilt from the spread's own rolling
# window (data up to t only) and the spread is quarantined until it passes its health
# checks again. If even the rebuilt filter is unusable the spread becomes close-only.
function _reset_model!(st::BacktestState, sp::SpreadState, t::Int)
    cfg = st.cfg
    if ispair(sp)
        sp.model = calibrate_filter(window(sp.ring, 1), window(sp.ring, 2), cfg.kalman; warmup = true)
    else
        buf = sp.spbuf
        for s in eachindex(buf)
            v = 0.0
            for l in 1:nlegs(sp)
                v += sp.weights[l] * window(sp.ring, l)[s]
            end
            buf[s] = v
        end
        sp.model = calibrate_basket_filter(buf, cfg.kalman; warmup = true)
    end
    sp.state = QUARANTINE
    sp.good = 0
    _model_finite(sp.model) || (sp.in_universe = false)
    push!(st.model_resets, (t, sp.id))
    nothing
end

function _health_check!(st::BacktestState, sp::SpreadState, t::Int)
    cfg = st.cfg
    jws = st.jws[nlegs(sp)]
    js = johansen!(jws, window(sp.ring); level = cfg.coint_level)
    lo, hi = cfg.hl_bounds
    if ispair(sp)
        eg = engle_granger!(st.egws, window(sp.ring, 1), window(sp.ring, 2); lags = cfg.adf_lags)
        ok = healthy(st.gate, js, eg)
        hl, egadf, egβ = eg.half_life, eg.adf, eg.beta
        if cfg.kalman.recalibrate                 # re-fit the OU dynamics on the window
            xw = window(sp.ring, 2)
            recalibrate!(sp.model, view(st.egws.resid, 1:length(xw)), cfg.kalman, sum(xw) / length(xw))
        end
        kfβ = hedge_ratio(sp.model)
    else
        hl = js.half_life
        ok = js.ok && js.rank >= 1 && lo <= hl <= hi
        egadf = egβ = kfβ = NaN
        if cfg.kalman.recalibrate                 # re-fit on this spread's own window
            buf = sp.spbuf
            Wn = sp.ring.W
            for s in 1:Wn
                v = 0.0
                for l in 1:nlegs(sp)
                    v += sp.weights[l] * window(sp.ring, l)[s]
                end
                buf[s] = v
            end
            μ = sum(buf) / Wn
            buf .-= μ
            recalibrate!(sp.model, buf, cfg.kalman, 0.0)
        end
    end
    sp.state, sp.good = gate_transition(st.gate, sp.state, sp.good, ok)
    sp.hl = hl
    sp.rank = js.rank
    acf1 = autocor_lag(filled_window(sp.zring, 1), 1)
    push!(st.health, HealthRecord(t, sp.id, sp.assets[1], sp.assets[2], nlegs(sp), js.rank, js.trace_r0,
                                  js.trace_r1, js.maxeig_r0, hl, js.half_life, egadf, egβ, kfβ,
                                  model_half_life(sp.model), acf1, ok, sp.state))
    nothing
end

function _stage_models!(st::BacktestState, t::Int)
    tr = st.panel.tradable
    for sp in st.spreads
        sp.created_t == t && continue
        # a stale (missing / halted) quote carries no new information
        all(a -> tr[a, t], sp.assets) || continue
        _update_spread!(st, sp, t)
    end
end

# ─── market, risk, signals, allocation, orders ───────────────────────────────

function _stage_market!(st::BacktestState, t::Int)
    panel, sim, book = st.panel, st.sim, st.book
    if t > st.cfg.formation
        update_vol!(sim, panel, t)
        accrue_financing!(book, panel, st.cfg.exec, t)
        apply_corporate_actions!(sim, book, panel, t) do a, ratio
            for sp in st.spreads, l in eachindex(sp.assets)
                sp.assets[l] == a && (sp.shares[l] *= ratio)
            end
        end
        process_orders!(sim, book, panel, t) do id, leg, q
            st.byid[id].shares[leg] += q
        end
    else
        for τ in 2:t              # seed the impact-model volatility from history
            update_vol!(sim, panel, τ)
        end
    end
end

# maintenance margin: returns (equity, liquidate?)
function _stage_risk!(st::BacktestState, t::Int)
    E = equity(st.book, st.panel, t)
    G = gross_exposure(st.book, st.panel, t)
    liquidate = false
    if G > 0 && (E <= 0 || E < st.cfg.risk.maintenance_margin * G)
        liquidate = true
        st.liquidating || push!(st.margin_calls, t)
        st.liquidating = true
        st.no_entry_until = E <= 0 ? typemax(Int) : t + st.cfg.risk.liquidation_cooldown
    elseif G == 0
        st.liquidating = false
    end
    E, liquidate
end

function _stage_signals!(st::BacktestState, t::Int, E::Float64, liquidate::Bool, rule::AbstractSignalRule)
    tr = st.panel.tradable
    for sp in st.spreads
        sp.last_signal = sp.signal
        if liquidate || !(sp.state == ACTIVE && sp.in_universe)
            sp.signal = 0
            continue
        end
        can_open = t >= st.no_entry_until && all(a -> tr[a, t], sp.assets)
        update_signal!(rule, sp, SignalContext(t, st.cfg, st.panel, st.sim.sigma2, E, can_open, st.ows))
    end
end

function _stage_allocate!(st::BacktestState, t::Int, E::Float64, alloc::AbstractAllocator)
    cfg, panel = st.cfg, st.panel
    C = cfg.risk.pair_cap * max(E, 0.0)
    sel = st.open_sel
    empty!(sel)
    for sp in st.spreads
        sp.signal != 0 && push!(sel, sp)
        fill!(sp.target, 0.0)
    end
    w = st.w
    resize!(w, length(sel))
    isempty(sel) && return C
    target_weights!(alloc, w, sel, SignalContext(t, cfg, panel, st.sim.sigma2, E, true, st.ows))
    gross = 0.0
    for (j, sp) in enumerate(sel)
        cfg.risk.throttle_uncertainty && (w[j] *= size_multiplier(sp.model))
        gross += abs(w[j]) * C
    end
    scale = gross > cfg.risk.max_gross * E ? cfg.risk.max_gross * E / gross : 1.0
    for (j, sp) in enumerate(sel)
        G = w[j] * C * scale
        wsum = sum(abs, sp.weights)
        for (l, a) in enumerate(sp.assets)
            sp.target[l] = round(G * sp.weights[l] / wsum / panel.mid[a, t])
        end
    end
    C
end

# per-spread order actions for one bar
const ACT_NONE = 0x00      # within the rebalance band: leave it
const ACT_WAIT = 0x01      # orders in flight toward an earlier target
const ACT_SUBMIT = 0x02    # trade to target
const ACT_REPLACE = 0x03   # cancel anything in flight, then trade to target

_haspending(sim::ExecutionSimulator, id::Int) = any(o -> o.pair == id, sim.pending)

function _stage_orders!(st::BacktestState, t::Int, E::Float64, C::Float64, liquidate::Bool)
    cfg, panel, sim = st.cfg, st.panel, st.sim
    if liquidate
        for sp in st.spreads
            _haspending(sim, sp.id) && cancel_pair!(sim, sp.id)
            isopen(sp) && submit_ticket!(sim, panel, sp.id, sp.assets, .-sp.shares, t)
        end
        return
    end
    # Pass 1: the action the rebalance hysteresis would take, and the gross the book would
    # end up with (spreads that don't trade keep their holdings plus anything in flight).
    act = st.act
    resize!(act, length(st.spreads))
    projected = 0.0
    for (i, sp) in enumerate(st.spreads)
        dev = 0.0
        for (l, a) in enumerate(sp.assets)
            dev += abs(sp.target[l] - sp.shares[l]) * panel.mid[a, t]
        end
        act[i] = if sp.signal != sp.last_signal
            ACT_REPLACE
        elseif _haspending(sim, sp.id)
            ACT_WAIT
        elseif sp.signal == 0 && isopen(sp)
            ACT_SUBMIT
        elseif sp.signal != 0 && dev > cfg.risk.min_trade_frac * C
            ACT_SUBMIT
        else
            ACT_NONE
        end
        for (l, a) in enumerate(sp.assets)
            q = (act[i] == ACT_REPLACE || act[i] == ACT_SUBMIT) ? sp.target[l] :
                sp.shares[l] + pending_qty(sim, sp.id, l)
            projected += abs(q) * panel.mid[a, t]
        end
    end
    # Hysteresis must never let the book exceed max_gross: if it would, every open spread
    # is brought to its (jointly capped) target this bar.
    if projected > cfg.risk.max_gross * max(E, 0.0) * (1 + 1e-9)
        for i in eachindex(act)
            act[i] == ACT_NONE && (act[i] = ACT_SUBMIT)
            act[i] == ACT_WAIT && (act[i] = ACT_REPLACE)
        end
    end
    # Pass 2: send orders
    for (i, sp) in enumerate(st.spreads)
        act[i] == ACT_REPLACE && _haspending(sim, sp.id) && cancel_pair!(sim, sp.id)
        if act[i] == ACT_REPLACE || act[i] == ACT_SUBMIT
            d = sp.target .- sp.shares
            any(!iszero, d) && submit_ticket!(sim, panel, sp.id, sp.assets, d, t)
        end
    end
end

function _stage_record!(st::BacktestState, t::Int)
    # retire flat spreads that dropped out of the universe
    filter!(st.spreads) do sp
        keep = sp.in_universe || isopen(sp) || _haspending(st.sim, sp.id)
        keep || (sp.state = RETIRED)
        keep
    end
    e = equity(st.book, st.panel, t)
    sps = st.spreads
    push!(st.bars, BarRecord(t, e, gross_exposure(st.book, st.panel, t), length(sps),
                             count(sp -> sp.state == ACTIVE, sps), count(sp -> sp.state == QUARANTINE, sps),
                             count(isopen, sps)))
    e
end

# ─── main loop ───────────────────────────────────────────────────────────────

"""
    run_backtest(panel, cfg=BacktestConfig()) -> BacktestResult

Walk-forward statistical-arbitrage backtest over pairs (and baskets if
`cfg.max_baskets > 0`). Validates the configuration first ([`validate_config`](@ref)).
"""
function run_backtest(panel::PricePanel, cfg::BacktestConfig = BacktestConfig())
    validate_config(cfg, panel)
    _run_backtest(panel, cfg, cfg.signal_rule, cfg.allocator)
end

# function barrier: specialise on the concrete rule and allocator types
function _run_backtest(panel::PricePanel, cfg::BacktestConfig, rule::R, alloc::A) where {R<:AbstractSignalRule,A<:AbstractAllocator}
    st = BacktestState(panel, cfg)
    F, T = cfg.formation, ntimes(panel)
    ts = F:T
    eq = zeros(length(ts))
    for (k, t) in enumerate(ts)
        tick!(st.clock, t)
        _stage_market!(st, t)
        (t - F) % cfg.trading == 0 && _stage_formation!(st, t)
        _stage_models!(st, t)
        E, liquidate = _stage_risk!(st, t)
        _stage_signals!(st, t, E, liquidate, rule)
        C = _stage_allocate!(st, t, E, alloc)
        _stage_orders!(st, t, E, C, liquidate)
        eq[k] = _stage_record!(st, t)
    end
    rets = [eq[k] / eq[k-1] - 1 for k in 2:length(eq)]
    BacktestResult(cfg, collect(ts), panel.timestamps[ts], eq, rets, st.bars, st.health, st.sim.fills,
                   st.formations, st.book, st.margin_calls, st.model_resets, panel.bars_per_year, panel.symbols)
end
