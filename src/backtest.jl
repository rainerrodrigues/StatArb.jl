# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward event loop: formation → live trading with health gating.
#
# Per bar t (single monotonic clock):
#   1. fills for orders decided at ≤ t − latency are executed on bar-t prices
#   2. financing / borrow accrue over (t−1, t]
#   3. every `trading` bars: re-run formation on [t−F+1, t]
#   4. each pair: ring buffer push, Kalman update (z uses the prior θ), periodic
#      Engle-Granger + Johansen health check on the W-bar window
#   5. signals → risk allocator → orders (eligible from t + latency)
#   6. mark-to-market and telemetry
# ─────────────────────────────────────────────────────────────────────────────

"""
    BacktestConfig(; ...)

Walk-forward and signal parameters (see field comments in the source). Nested configs:
`kalman::KalmanConfig`, `exec::ExecConfig`, `risk::RiskConfig`.
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
    same_sector_only::Bool = true
    unique_assets::Bool = true           # an asset may appear in at most one selected pair
    hl_bounds::Tuple{Float64,Float64} = (1.0, 60.0)
    gate_on::Symbol = :johansen          # :johansen | :eg | :either
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
end

"""Return a copy of an immutable config with some fields replaced."""
function with(cfg::T; kw...) where {T}
    vals = map(fieldnames(T)) do f
        haskey(kw, f) ? kw[f] : getfield(cfg, f)
    end
    T(vals...)
end

mutable struct PairState
    id::Int
    y::Int
    x::Int
    kf::KalmanHedge
    ring::MirroredRing        # 2 × W log prices
    zring::MirroredRing       # standardized innovations
    rring::MirroredRing       # spread returns (per $1 gross)
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
    sh_y::Float64
    sh_x::Float64
end

struct HealthRecord
    t::Int
    pair::Int
    y::Int
    x::Int
    rank::Int
    trace_r0::Float64
    trace_r1::Float64
    maxeig_r0::Float64
    half_life::Float64        # Engle-Granger residual OU half-life
    joh_half_life::Float64
    eg_adf::Float64
    eg_beta::Float64
    kf_beta::Float64
    innov_acf1::Float64
    healthy::Bool
    state::HealthState
end

struct BarRecord
    t::Int
    equity::Float64
    gross::Float64
    n_pairs::Int
    n_active::Int
    n_quarantine::Int
    n_open::Int
end

"""
    FormationRecord

One formation step: bar `t`, number of screened `n_candidates`, number that passed
selection (`n_cointegrated`: significant after FDR control and within the half-life
bounds), the smallest selection p-value, and the `selected` pairs as `(y, x)`.
"""
struct FormationRecord
    t::Int
    n_candidates::Int
    n_cointegrated::Int
    min_pvalue::Float64
    selected::Vector{NTuple{2,Int}}
end

"""
    BacktestResult

Output of [`run_backtest`](@ref): bar indices `t`, `timestamps`, `equity` and `returns`;
columnar telemetry `bars`, `health` and `fills` (`StructVector`s, so they work with any
Tables.jl consumer); `formations`; the final `book`; and the config used.
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
    bars_per_year::Float64
    symbols::Vector{Symbol}
end

Base.show(io::IO, r::BacktestResult) =
    print(io, "BacktestResult(", length(r.t), " bars, final equity ",
          round(last(r.equity); digits = 2), ", ", length(r.fills), " fills)")

# ─── formation ───────────────────────────────────────────────────────────────

function _formation!(pairs, byid, next_id::Ref{Int}, panel::PricePanel, cfg::BacktestConfig, t::Int,
                     formations, sim)
    F, W = cfg.formation, cfg.health_window
    logp = panel.logmid
    # assets with enough real (tradable) bars in the window and tradable right now
    eligible = [panel.tradable[a, t] && count(view(panel.tradable, a, t-F+1:t)) >= cfg.min_tradable_frac * F
                for a in 1:nassets(panel)]
    cands = distance_screen(logp, t, F; top = cfg.screen_top, sector = panel.sector,
                            same_sector = cfg.same_sector_only, eligible = eligible)
    tests = test_pairs(logp, cands, t, F; adf_lags = cfg.adf_lags, kdiff = cfg.johansen_lags,
                       level = cfg.coint_level, threaded = cfg.threaded_formation)
    lo, hi = cfg.hl_bounds
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
    push!(formations, FormationRecord(t, length(cands), length(good),
                                      isempty(pv) ? NaN : minimum(pv), selected))

    # Pairs are identified by their asset set, not orientation: a pair re-selected as
    # (x, y) keeps its existing live state instead of opening a second position.
    selset = Set(minmax(p...) for p in selected)
    existing = Set{NTuple{2,Int}}()
    for ps in pairs
        key = minmax(ps.y, ps.x)
        push!(existing, key)
        ps.in_universe = key in selset
    end
    rng = t-F+1:t
    for (yi, xi) in selected
        minmax(yi, xi) in existing && continue          # retained pair keeps its live filter
        yv = view(logp, yi, rng); xv = view(logp, xi, rng)
        kf = calibrate_kalman(yv, xv, cfg.kalman; warmup = false)
        zr = MirroredRing(1, cfg.innov_window)
        for s in 1:F
            push!(zr, kalman_update!(kf, yv[s], xv[s]))
        end
        kf.P_ref = kf.P[2, 2]
        ring = MirroredRing(2, W)
        for τ in t-W+1:t
            push!(ring, (logp[yi, τ], logp[xi, τ]))
        end
        L = cfg.risk.cov_window
        rr = MirroredRing(1, L)
        β = kf.θ[2]
        for τ in t-L+1:t
            push!(rr, ((logp[yi, τ] - logp[yi, τ-1]) - β * (logp[xi, τ] - logp[xi, τ-1])) / (1 + abs(β)))
        end
        eg_hl = ou_half_life(view(logp, yi, rng) .- β .* view(logp, xi, rng))
        id = next_id[]
        next_id[] += 1
        ps = PairState(id, yi, xi, kf, ring, zr, rr, ACTIVE, 0, true, 0, 0, 0, false,
                       eg_hl, 1, t, 0.0, 0.0)
        push!(pairs, ps)
        byid[id] = ps
    end
    nothing
end

# ─── per-bar pair update & health ────────────────────────────────────────────

function _update_pair!(ps::PairState, panel::PricePanel, cfg::BacktestConfig, t::Int,
                       egws::EGWorkspace, jws::JohansenWorkspace, gate::HealthGateConfig, health)
    y = panel.logmid[ps.y, t]; x = panel.logmid[ps.x, t]
    yprev = latest(ps.ring, 1); xprev = latest(ps.ring, 2)
    βprev = ps.kf.θ[2]
    push!(ps.ring, (y, x))
    z = kalman_update!(ps.kf, y, x)
    push!(ps.zring, z)
    push!(ps.rring, ((y - yprev) - βprev * (x - xprev)) / (1 + abs(βprev)))
    if (t - ps.created_t) % cfg.health_every == 0
        eg = engle_granger!(egws, window(ps.ring, 1), window(ps.ring, 2); lags = cfg.adf_lags)
        js = johansen!(jws, window(ps.ring); level = cfg.coint_level)
        ok = healthy(gate, js, eg)
        ps.state, ps.good = gate_transition(gate, ps.state, ps.good, ok)
        ps.hl = eg.half_life
        ps.rank = js.rank
        acf1 = autocor_lag(filled_window(ps.zring, 1), 1)
        push!(health, HealthRecord(t, ps.id, ps.y, ps.x, js.rank, js.trace_r0, js.trace_r1, js.maxeig_r0,
                                   eg.half_life, js.half_life, eg.adf, eg.beta, ps.kf.θ[2], acf1, ok, ps.state))
    end
    nothing
end

function _update_signal!(ps::PairState, cfg::BacktestConfig, t::Int, can_open::Bool)
    ps.last_signal = ps.signal
    z = ps.kf.z
    if !(ps.state == ACTIVE && ps.in_universe)
        ps.signal = 0
        return
    end
    if ps.blocked
        abs(z) < cfg.z_exit && (ps.blocked = false)
        ps.signal = 0
        return
    end
    if ps.signal == 0
        can_open || return
        if z > cfg.z_entry
            ps.signal = -1; ps.entry_t = t
        elseif z < -cfg.z_entry
            ps.signal = 1; ps.entry_t = t
        end
    else
        hl_assumed = isfinite(ps.hl) ? ps.hl * cfg.hl_scale : Inf
        if abs(z) > cfg.z_stop || (t - ps.entry_t) > cfg.max_hold_halflives * hl_assumed
            ps.signal = 0
            ps.blocked = true
        elseif ps.signal * z > -cfg.z_exit
            ps.signal = 0
        end
    end
    nothing
end

# ─── main loop ───────────────────────────────────────────────────────────────

# per-pair order actions for one bar
const ACT_NONE = 0x00      # within the rebalance band: leave it
const ACT_WAIT = 0x01      # orders in flight toward an earlier target
const ACT_SUBMIT = 0x02    # trade to target
const ACT_REPLACE = 0x03   # cancel anything in flight, then trade to target

"""
    run_backtest(panel, cfg=BacktestConfig()) -> BacktestResult

Walk-forward statistical-arbitrage backtest. See the module docstring for the event order.
"""
function run_backtest(panel::PricePanel, cfg::BacktestConfig = BacktestConfig())
    validate_config(cfg, panel)
    n, T = nassets(panel), ntimes(panel)
    F, W = cfg.formation, cfg.health_window

    clock = Clock(0)
    sim = ExecutionSimulator(cfg.exec, n)
    book = Book(cfg.initial_capital, n)
    egws = EGWorkspace(W, cfg.adf_lags)
    jws = JohansenWorkspace(2, W; kdiff = cfg.johansen_lags)
    gate = HealthGateConfig(cfg.hl_bounds, cfg.gate_on, cfg.recover_checks, cfg.coint_level)
    ows = OverlayWorkspace(2 * cfg.max_pairs)

    pairs = PairState[]
    byid = Dict{Int,PairState}()
    next_id = Ref(1)
    health = StructVector{HealthRecord}(undef, 0)
    bars = StructVector{BarRecord}(undef, 0)
    formations = FormationRecord[]

    on_fill = (pid, leg, q) -> begin
        ps = byid[pid]
        leg == 1 ? (ps.sh_y += q) : (ps.sh_x += q)
    end
    on_split = (a, ratio) -> begin
        for ps in pairs
            ps.y == a && (ps.sh_y *= ratio)
            ps.x == a && (ps.sh_x *= ratio)
        end
    end
    margin_calls = Int[]
    no_entry_until = 0          # entries blocked before this bar (after a margin call)
    liquidating = false

    # scratch for allocation
    sel = Int[]; sig = Float64[]; kap = Float64[]; w = Float64[]
    tgt_y = Float64[]; tgt_x = Float64[]; act = UInt8[]
    rets = Vector{SubArray{Float64,1,Matrix{Float64},Tuple{Int,UnitRange{Int}},true}}()

    ts = F:T
    eq = zeros(length(ts))
    for (k, t) in enumerate(ts)
        tick!(clock, t)
        if t > F
            update_vol!(sim, panel, t)
            accrue_financing!(book, panel, cfg.exec, t)
            apply_corporate_actions!(on_split, sim, book, panel, t)
            process_orders!(on_fill, sim, book, panel, t)
        else
            for τ in 2:t          # seed the impact-model volatility from history
                update_vol!(sim, panel, τ)
            end
        end
        if (t - F) % cfg.trading == 0
            _formation!(pairs, byid, next_id, panel, cfg, t, formations, sim)
        end
        for ps in pairs
            ps.created_t == t && continue
            # a stale (missing / halted) quote carries no new information
            (panel.tradable[ps.y, t] && panel.tradable[ps.x, t]) || continue
            _update_pair!(ps, panel, cfg, t, egws, jws, gate, health)
        end

        # ── margin check: liquidate everything below maintenance margin ──
        E = equity(book, panel, t)
        G = gross_exposure(book, panel, t)
        liquidate = false
        if G > 0 && (E <= 0 || E < cfg.risk.maintenance_margin * G)
            liquidate = true
            liquidating || push!(margin_calls, t)
            liquidating = true
            no_entry_until = E <= 0 ? typemax(Int) : t + cfg.risk.liquidation_cooldown
        elseif G == 0
            liquidating = false
        end

        for ps in pairs
            can_open = t >= no_entry_until && panel.tradable[ps.y, t] && panel.tradable[ps.x, t]
            _update_signal!(ps, cfg, t, can_open)
            liquidate && (ps.signal = 0)
        end

        # ── risk allocation ──
        C = cfg.risk.pair_cap * max(E, 0.0)
        empty!(sel); empty!(sig); empty!(kap); empty!(rets)
        for (i, ps) in enumerate(pairs)
            if ps.signal != 0
                push!(sel, i); push!(sig, ps.signal)
                hl = ps.hl * cfg.hl_scale
                push!(kap, isfinite(hl) && hl > 0 ? log(2.0) / hl : 0.0)
                push!(rets, window(ps.rring, 1))
            end
        end
        resize!(w, length(sel))
        if cfg.risk.correlation_overlay && length(sel) >= 2
            overlay_weights!(w, ows, rets, sig, kap, cfg.risk.shrinkage)
        else
            w .= sig
        end
        gross = 0.0
        for (j, i) in enumerate(sel)
            u = cfg.risk.throttle_uncertainty ? size_multiplier(pairs[i].kf) : 1.0
            w[j] *= u
            gross += abs(w[j]) * C
        end
        scale = gross > cfg.risk.max_gross * E ? cfg.risk.max_gross * E / gross : 1.0

        # ── orders ──
        if liquidate
            for ps in pairs
                any(o -> o.pair == ps.id, sim.pending) && cancel_pair!(sim, ps.id)
                (ps.sh_y != 0 || ps.sh_x != 0) &&
                    submit_pair!(sim, panel, ps.id, (ps.y, ps.x), (-ps.sh_y, -ps.sh_x), t)
            end
        else
            # Pass 1: targets and the action the rebalance hysteresis would take. Then
            # project the gross the book would end up with: pairs that don't trade keep
            # their holdings (plus anything in flight), pairs that trade go to target.
            np = length(pairs)
            resize!(tgt_y, np); resize!(tgt_x, np); resize!(act, np)
            projected = 0.0
            j = 0
            for (i, ps) in enumerate(pairs)
                my = panel.mid[ps.y, t]; mx = panel.mid[ps.x, t]
                if ps.signal != 0
                    j += 1
                    tgt_y[i], tgt_x[i] = leg_shares(w[j] * C * scale, ps.kf.θ[2], my, mx)
                else
                    tgt_y[i], tgt_x[i] = 0.0, 0.0
                end
                haspend = any(o -> o.pair == ps.id, sim.pending)
                dy = tgt_y[i] - ps.sh_y; dx = tgt_x[i] - ps.sh_x
                act[i] = if ps.signal != ps.last_signal
                    ACT_REPLACE
                elseif haspend
                    ACT_WAIT
                elseif ps.signal == 0 && (ps.sh_y != 0 || ps.sh_x != 0)
                    ACT_SUBMIT
                elseif ps.signal != 0 && abs(dy) * my + abs(dx) * mx > cfg.risk.min_trade_frac * C
                    ACT_SUBMIT
                else
                    ACT_NONE
                end
                if act[i] == ACT_REPLACE || act[i] == ACT_SUBMIT
                    projected += abs(tgt_y[i]) * my + abs(tgt_x[i]) * mx
                else
                    projected += abs(ps.sh_y + pending_qty(sim, ps.id, 1)) * my +
                                 abs(ps.sh_x + pending_qty(sim, ps.id, 2)) * mx
                end
            end
            # Hysteresis must never let the book exceed max_gross: if it would, every open
            # pair is brought to its (jointly capped) target this bar.
            if projected > cfg.risk.max_gross * max(E, 0.0) * (1 + 1e-9)
                for i in 1:np
                    act[i] == ACT_NONE && (act[i] = ACT_SUBMIT)
                    act[i] == ACT_WAIT && (act[i] = ACT_REPLACE)
                end
            end
            # Pass 2: send orders
            for (i, ps) in enumerate(pairs)
                dy = tgt_y[i] - ps.sh_y; dx = tgt_x[i] - ps.sh_x
                if act[i] == ACT_REPLACE
                    any(o -> o.pair == ps.id, sim.pending) && cancel_pair!(sim, ps.id)
                    (dy != 0 || dx != 0) && submit_pair!(sim, panel, ps.id, (ps.y, ps.x), (dy, dx), t)
                elseif act[i] == ACT_SUBMIT && (dy != 0 || dx != 0)
                    submit_pair!(sim, panel, ps.id, (ps.y, ps.x), (dy, dx), t)
                end
            end
        end

        # ── retire flat pairs that dropped out of the universe ──
        filter!(pairs) do ps
            keep = ps.in_universe || ps.sh_y != 0 || ps.sh_x != 0 || any(o -> o.pair == ps.id, sim.pending)
            keep || (ps.state = RETIRED)
            keep
        end

        eq[k] = equity(book, panel, t)
        na = count(ps -> ps.state == ACTIVE, pairs)
        nq = count(ps -> ps.state == QUARANTINE, pairs)
        no = count(ps -> ps.sh_y != 0 || ps.sh_x != 0, pairs)
        push!(bars, BarRecord(t, eq[k], gross_exposure(book, panel, t), length(pairs), na, nq, no))
    end
    rets_out = [eq[k] / eq[k-1] - 1 for k in 2:length(eq)]
    BacktestResult(cfg, collect(ts), panel.timestamps[ts], eq, rets_out, bars, health, sim.fills,
                   formations, book, margin_calls, panel.bars_per_year, panel.symbols)
end
