# ─────────────────────────────────────────────────────────────────────────────
# Default signal rules, cost model and allocators (see strategy.jl for the interfaces).
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_signal!(rule, sp::SpreadState, ctx::SignalContext)

Set `sp.signal ∈ {-1, 0, 1}` for bar `ctx.t`. Implement this for custom
[`AbstractSignalRule`](@ref) subtypes.
"""
function update_signal! end

"""
    target_weights!(alloc, w, spreads, ctx::SignalContext) -> w

Fill `w[i]` with the signed weight of `spreads[i]`, in units of the per-spread cap.
Implement this for custom [`AbstractAllocator`](@ref) subtypes.
"""
function target_weights! end

# hysteresis bands with an entry threshold `zin`
function _band_update!(sp::SpreadState, ctx::SignalContext, zin::Float64)
    cfg = ctx.cfg
    z = zscore(sp)
    t = ctx.t
    if sp.blocked
        abs(z) < cfg.z_exit && (sp.blocked = false)
        sp.signal = 0
        return
    end
    if sp.signal == 0
        ctx.can_open || return
        if z > zin
            sp.signal = -1; sp.entry_t = t
        elseif z < -zin
            sp.signal = 1; sp.entry_t = t
        end
    else
        hl_assumed = isfinite(sp.hl) ? sp.hl * cfg.hl_scale : Inf
        if abs(z) > cfg.z_stop || (t - sp.entry_t) > cfg.max_hold_halflives * hl_assumed
            sp.signal = 0
            sp.blocked = true
        elseif sp.signal * z > -cfg.z_exit
            sp.signal = 0
        end
    end
    nothing
end

update_signal!(::ZScoreBands, sp::SpreadState, ctx::SignalContext) = _band_update!(sp, ctx, ctx.cfg.z_entry)

"""
    round_trip_cost(sp, ctx) -> Float64

Estimated cost of opening and closing `sp` at the allocator's per-spread size
(`risk.pair_cap × equity`), as a fraction of gross dollars: for every leg, its share of the
gross times (half spread / mid + commission + square-root impact `η σ √(q/V)`), doubled for
the round trip. Uses bar-`t` quotes and the simulator's current volatility estimates.
"""
function round_trip_cost(sp::SpreadState, ctx::SignalContext)
    cfg, p, t = ctx.cfg, ctx.panel, ctx.t
    ex = cfg.exec
    G = cfg.risk.pair_cap * max(ctx.equity, 0.0)
    wsum = sum(abs, sp.weights)
    c = 0.0
    @inbounds for (l, a) in enumerate(sp.assets)
        f = abs(sp.weights[l]) / wsum
        m = p.mid[a, t]; h = 0.5 * p.spread[a, t]; V = p.volume[a, t]
        q = G * f / m
        imp = V > 0 ? ex.impact_coef * sqrt(ctx.sigma2[a]) * sqrt(q / V) : 0.0
        c += f * (h / m + ex.commission_bps * 1e-4 + ex.commission_per_share / m + imp)
    end
    2c
end

"""
    cost_aware_entry(rule::CostAwareBands, sp, ctx) -> Float64

The entry threshold `CostAwareBands` applies to `sp` on this bar:
`max(z_entry, z_exit + cost_multiple · round_trip_cost · Σ|w| / spread_scale)`.
"""
function cost_aware_entry(rule::CostAwareBands, sp::SpreadState, ctx::SignalContext)
    cfg = ctx.cfg
    σ = spread_scale(sp.model)
    c = round_trip_cost(sp, ctx)
    (σ > 0 && isfinite(c)) || return Inf
    max(cfg.z_entry, cfg.z_exit + rule.cost_multiple * c * sum(abs, sp.weights) / σ)
end

function update_signal!(rule::CostAwareBands, sp::SpreadState, ctx::SignalContext)
    zin = sp.signal == 0 && !sp.blocked ? cost_aware_entry(rule, sp, ctx) : ctx.cfg.z_entry
    # an entry at or beyond the stop level would be stopped out immediately: don't open
    _band_update!(sp, ctx, zin < ctx.cfg.z_stop ? zin : Inf)
end

function target_weights!(::EqualWeightAllocator, w::Vector{Float64}, sel::Vector{SpreadState}, ::SignalContext)
    for (j, sp) in enumerate(sel)
        w[j] = sp.signal
    end
    w
end

function target_weights!(::DefaultAllocator, w::Vector{Float64}, sel::Vector{SpreadState}, ctx::SignalContext)
    cfg = ctx.cfg
    if !(cfg.risk.correlation_overlay && length(sel) >= 2)
        return target_weights!(EqualWeightAllocator(), w, sel, ctx)
    end
    sig = [Float64(sp.signal) for sp in sel]
    kap = map(sel) do sp
        hl = sp.hl * cfg.hl_scale
        isfinite(hl) && hl > 0 ? log(2.0) / hl : 0.0
    end
    rets = [window(sp.rring, 1) for sp in sel]
    overlay_weights!(w, ctx.overlay, rets, sig, kap, cfg.risk.shrinkage)
end
