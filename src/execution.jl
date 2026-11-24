# ─────────────────────────────────────────────────────────────────────────────
# Multi-leg execution, slippage, liquidity verification, legging risk, financing.
#
# Orders decided at bar t become eligible at t + latency (≥ 1). Each leg is filled
# independently against that bar's quote, volume and displayed depth:
#
#   MARKET  price = mid ± (half-spread + mid · η σ √(q / V))   (square-root impact)
#   LIMIT   rests at the decision-time touch; fills only on a strict trade-through
#           (bid_τ < L for a buy) — so passive fills come with adverse selection.
#
# Legging: for a passive pair ticket, if one leg fills and its sibling misses, the
# sibling is converted to a market order and crosses the spread `latency` bars later.
# Fill size per bar is capped by participation in bar volume and displayed depth;
# unfilled remainders roll forward (partial fills).
# ─────────────────────────────────────────────────────────────────────────────

"""
    OrderKind

`MARKET` crosses the spread and pays square-root impact; `LIMIT` rests at the touch and
fills only on a strict trade-through.
"""
@enum OrderKind MARKET LIMIT
@doc "Aggressive order: half spread + square-root impact." MARKET
@doc "Passive order resting at the decision-time touch." LIMIT

"""
    ExecConfig(; ...)

Execution and financing assumptions.

- `latency` bars between decision and earliest fill (must be ≥ 1)
- `style`: `:market` (both legs aggressive) or `:passive` (both legs post at the touch)
- `limit_wait`: bars a passive order rests before crossing
- `reprice`: re-peg resting passive orders to the latest touch every bar (otherwise they
  keep the decision-bar price for their whole life). Only matters with `limit_wait > 1`; on
  daily synthetic bars, longer resting with re-pegging produced *more* legging and a lower
  Sharpe than crossing after one bar.
- `max_participation`: max fraction of bar volume per fill; `depth_levels`: max multiple of
  displayed touch size that may be consumed per bar
- `impact_coef` η of the square-root law `Δp/p = η σ_bar √(q/V)`
- `commission_bps`, `commission_per_share`
- `borrow_rate` (annual, on short market value; `borrow_rates` per-asset override),
  `margin_rate` (annual, on debit balances), `cash_rate` (annual, on free credit; short
  proceeds are held as collateral and earn nothing)
- `vol_halflife`: EWMA half-life (bars) for the per-asset σ used in the impact model
"""
Base.@kwdef struct ExecConfig
    latency::Int = 1
    style::Symbol = :market
    limit_wait::Int = 1
    reprice::Bool = false
    max_participation::Float64 = 0.10
    depth_levels::Float64 = 5.0
    impact_coef::Float64 = 0.5
    commission_bps::Float64 = 0.5
    commission_per_share::Float64 = 0.0
    borrow_rate::Float64 = 0.005
    borrow_rates::Vector{Float64} = Float64[]
    margin_rate::Float64 = 0.06
    cash_rate::Float64 = 0.03
    vol_halflife::Float64 = 20.0
end

"""
    Order

A resting or in-flight leg order. `qty` is the remaining signed share quantity; the order
may fill from bar `active_t` (decision bar + latency); a `LIMIT` converts to `MARKET`
after `expire_t`; `legged` marks a leg that is being crossed because its sibling filled.
"""
struct Order
    ticket::Int
    pair::Int
    leg::Int            # 1 = y leg, 2 = x leg
    asset::Int
    qty::Float64        # remaining signed shares
    kind::OrderKind
    limit::Float64
    decision_mid::Float64
    submit_t::Int
    active_t::Int       # earliest fill bar
    expire_t::Int       # limit converts to market after this bar
    legged::Bool
end

"""One execution record (stored columnar in a `StructVector`)."""
struct FillRecord
    t::Int
    submit_t::Int
    ticket::Int
    pair::Int
    leg::Int
    asset::Int
    qty::Float64
    price::Float64
    mid::Float64
    decision_mid::Float64
    spread_cost::Float64      # vs mid at fill (negative = earned)
    impact_cost::Float64
    commission::Float64
    passive::Bool
    legged::Bool
end

"""Cash/position ledger with cost attribution."""
mutable struct Book
    cash::Float64
    pos::Vector{Float64}
    spread_paid::Float64
    impact_paid::Float64
    commission_paid::Float64
    borrow_paid::Float64
    interest_net::Float64     # positive = earned
    dividends_net::Float64    # received on longs minus paid on shorts
    traded_notional::Float64
    n_fills::Int
    n_legged::Int
    n_passive::Int
end
Book(cash::Real, n::Integer) = Book(Float64(cash), zeros(n), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

"""
    ExecutionSimulator(cfg::ExecConfig, n_assets)

Holds pending orders, the columnar fill log (`fills::StructVector{FillRecord}`) and the
per-asset EWMA volatility used by the impact model.
"""
mutable struct ExecutionSimulator
    cfg::ExecConfig
    pending::Vector{Order}
    fills::StructVector{FillRecord}
    sigma2::Vector{Float64}      # EWMA per-bar log-return variance per asset
    next_ticket::Int
end

function ExecutionSimulator(cfg::ExecConfig, n_assets::Integer)
    cfg.latency >= 1 || throw(ArgumentError("latency must be ≥ 1 bar to avoid lookahead"))
    cfg.style in (:market, :passive) || throw(ArgumentError("style must be :market or :passive"))
    ExecutionSimulator(cfg, sizehint!(Order[], 64), StructVector{FillRecord}(undef, 0),
                       fill(1e-4, n_assets), 1)
end

"""Update per-asset EWMA volatility with the return from t-1 to t."""
function update_vol!(sim::ExecutionSimulator, p::PricePanel, t::Int)
    t >= 2 || return
    λ = 2.0^(-1.0 / sim.cfg.vol_halflife)
    @inbounds for a in eachindex(sim.sigma2)
        r = p.logmid[a, t] - p.logmid[a, t-1]
        sim.sigma2[a] = λ * sim.sigma2[a] + (1 - λ) * r * r
    end
end

"""
    submit_pair!(sim, panel, pair, assets, qtys, t) -> ticket

Submit a (one- or two-legged) ticket decided at bar `t`. `assets`/`qtys` are tuples of
(y, x) asset indices and signed share quantities; zero quantities are skipped.
"""
submit_pair!(sim::ExecutionSimulator, p::PricePanel, pair::Int,
             assets::NTuple{2,Int}, qtys::NTuple{2,Float64}, t::Int) =
    submit_ticket!(sim, p, pair, assets, qtys, t)

"""
    submit_ticket!(sim, panel, spread_id, assets, qtys, t) -> ticket

Submit a multi-leg ticket decided at bar `t`: leg `i` trades `qtys[i]` signed shares of
`assets[i]` (zero quantities are skipped). All legs share one ticket, so a passive fill
on any leg crosses the remaining resting legs (legging risk).
"""
function submit_ticket!(sim::ExecutionSimulator, p::PricePanel, pair::Int,
                        assets, qtys, t::Int)
    length(assets) == length(qtys) || throw(DimensionMismatch("assets and qtys"))
    cfg = sim.cfg
    ticket = sim.next_ticket
    sim.next_ticket += 1
    for leg in eachindex(assets)
        q = Float64(qtys[leg])
        q == 0.0 && continue
        a = assets[leg]
        m = p.mid[a, t]
        h = 0.5 * p.spread[a, t]
        if cfg.style === :passive
            L = q > 0 ? m - h : m + h          # join the touch
            push!(sim.pending, Order(ticket, pair, leg, a, q, LIMIT, L, m, t, t + cfg.latency,
                                     t + cfg.latency + cfg.limit_wait - 1, false))
        else
            push!(sim.pending, Order(ticket, pair, leg, a, q, MARKET, NaN, m, t, t + cfg.latency,
                                     typemax(Int), false))
        end
    end
    ticket
end

@inline _with(o::Order; qty = o.qty, kind = o.kind, active_t = o.active_t, legged = o.legged,
               limit = o.limit) =
    Order(o.ticket, o.pair, o.leg, o.asset, qty, kind, limit, o.decision_mid, o.submit_t,
          active_t, o.expire_t, legged)

"""
    process_orders!(sim, book, panel, t, on_fill) -> nothing

Attempt to fill every order eligible at bar `t`. `on_fill(pair, leg, qty)` is called for
each fill so the caller can update pair-level positions.
"""
function process_orders!(on_fill::F, sim::ExecutionSimulator, book::Book, p::PricePanel, t::Int) where {F}
    cfg = sim.cfg
    pend = sim.pending
    i = 1
    while i <= length(pend)
        o = pend[i]
        if o.active_t > t
            i += 1
            continue
        end
        a = o.asset
        if !p.tradable[a, t]          # halted / missing bar: the order just waits
            i += 1
            continue
        end
        m = p.mid[a, t]; h = 0.5 * p.spread[a, t]
        V = p.volume[a, t]; Dp = p.depth[a, t]
        cap = min(cfg.max_participation * V, cfg.depth_levels * Dp)
        sgn = sign(o.qty)
        q = min(abs(o.qty), cap)
        filled = false
        if q > 0
            if o.kind == MARKET
                imp = cfg.impact_coef * sqrt(sim.sigma2[a]) * sqrt(q / V)
                price = m + sgn * (h + m * imp)
                _book_fill!(sim, book, o, t, sgn * q, price, m, q * h, q * m * imp, false)
                filled = true
            else
                through = sgn > 0 ? (m - h) < o.limit : (m + h) > o.limit
                if through
                    _book_fill!(sim, book, o, t, sgn * q, o.limit, m, q * sgn * (o.limit - m), 0.0, true)
                    filled = true
                end
            end
        end
        if filled
            on_fill(o.pair, o.leg, sgn * q)
            rem = o.qty - sgn * q
            if abs(rem) < 1e-9
                deleteat!(pend, i)
                o.kind == LIMIT && _leg_out_siblings!(sim, o, t)
                continue
            end
            pend[i] = _with(o; qty = rem)
            if o.kind == LIMIT
                _leg_out_siblings!(sim, o, t)
            end
        elseif o.kind == LIMIT && t >= o.expire_t
            pend[i] = _with(o; kind = MARKET, active_t = t + cfg.latency)
        elseif o.kind == LIMIT && cfg.reprice
            # re-peg to this bar's touch (known now) for the next bar
            pend[i] = _with(o; limit = sgn > 0 ? m - h : m + h)
        end
        i += 1
    end
    nothing
end

# A passive leg filled: any still-resting sibling LIMIT leg of the same ticket now carries
# directional exposure and is crossed as a market order (legging risk).
function _leg_out_siblings!(sim::ExecutionSimulator, o::Order, t::Int)
    pend = sim.pending
    @inbounds for j in eachindex(pend)
        s = pend[j]
        if s.ticket == o.ticket && s.leg != o.leg && s.kind == LIMIT
            pend[j] = _with(s; kind = MARKET, active_t = t + sim.cfg.latency, legged = true)
        end
    end
end

function _book_fill!(sim::ExecutionSimulator, book::Book, o::Order, t::Int, q::Float64, price::Float64,
                     m::Float64, spread_cost::Float64, impact_cost::Float64, passive::Bool)
    cfg = sim.cfg
    notional = abs(q) * price
    comm = notional * cfg.commission_bps * 1e-4 + abs(q) * cfg.commission_per_share
    book.cash -= q * price + comm
    book.pos[o.asset] += q
    book.spread_paid += spread_cost
    book.impact_paid += impact_cost
    book.commission_paid += comm
    book.traded_notional += notional
    book.n_fills += 1
    o.legged && (book.n_legged += 1)
    passive && (book.n_passive += 1)
    push!(sim.fills, FillRecord(t, o.submit_t, o.ticket, o.pair, o.leg, o.asset, q, price, m,
                                o.decision_mid, spread_cost, impact_cost, comm, passive, o.legged))
    nothing
end

_action_t(x::CorporateAction) = x.t
_action_t(x::Integer) = x

"""
    apply_corporate_actions!(on_split, sim, book, panel, t) -> Int

Apply every corporate action whose ex-date is bar `t`, before that bar's fills: the
dividend is credited on long / debited on short positions (per pre-split share), then
positions are multiplied by the split ratio and resting orders are rescaled (quantity
× split, limit and decision prices ÷ split). `on_split(asset, ratio)` lets the caller
rescale its own per-pair share counts. Returns the number of actions applied.
"""
function apply_corporate_actions!(on_split::F, sim::ExecutionSimulator, book::Book, p::PricePanel,
                                  t::Int) where {F}
    acts = p.actions
    i = searchsortedfirst(acts, t; lt = (a, b) -> _action_t(a) < _action_t(b))
    napplied = 0
    while i <= length(acts) && acts[i].t == t
        ca = acts[i]
        a = ca.asset
        if ca.dividend != 0.0 && book.pos[a] != 0.0
            d = ca.dividend * book.pos[a]
            book.cash += d
            book.dividends_net += d
        end
        if ca.split != 1.0
            book.pos[a] *= ca.split
            for j in eachindex(sim.pending)
                o = sim.pending[j]
                o.asset == a || continue
                sim.pending[j] = Order(o.ticket, o.pair, o.leg, o.asset, o.qty * ca.split, o.kind,
                                       o.limit / ca.split, o.decision_mid / ca.split, o.submit_t,
                                       o.active_t, o.expire_t, o.legged)
            end
            on_split(a, ca.split)
        end
        napplied += 1
        i += 1
    end
    napplied
end

"""Outstanding (unfilled) signed quantity for a pair leg."""
function pending_qty(sim::ExecutionSimulator, pair::Int, leg::Int)
    s = 0.0
    @inbounds for o in sim.pending
        (o.pair == pair && o.leg == leg) && (s += o.qty)
    end
    s
end

"""Cancel every resting order of a pair (e.g. before re-targeting)."""
cancel_pair!(sim::ExecutionSimulator, pair::Int) = filter!(o -> o.pair != pair, sim.pending)

"""
    accrue_financing!(book, panel, cfg, t)

Charge short-borrow fees and net interest for the interval (t−1, t], valued at t−1.
"""
function accrue_financing!(book::Book, p::PricePanel, cfg::ExecConfig, t::Int)
    t >= 2 || return
    dt = 1.0 / p.bars_per_year
    short_val = 0.0
    fee = 0.0
    hasover = !isempty(cfg.borrow_rates)
    @inbounds for a in eachindex(book.pos)
        q = book.pos[a]
        if q < 0
            v = -q * p.mid[a, t-1]
            short_val += v
            fee += v * (hasover ? cfg.borrow_rates[a] : cfg.borrow_rate) * dt
        end
    end
    bal = book.cash - short_val
    int = bal > 0 ? bal * cfg.cash_rate * dt : bal * cfg.margin_rate * dt
    book.cash += int - fee
    book.borrow_paid += fee
    book.interest_net += int
    nothing
end

"""Mark-to-market equity at bar t."""
function equity(book::Book, p::PricePanel, t::Int)
    e = book.cash
    @inbounds for a in eachindex(book.pos)
        e += book.pos[a] * p.mid[a, t]
    end
    e
end

"""Gross market value `Σ |position| · mid` at bar t."""
function gross_exposure(book::Book, p::PricePanel, t::Int)
    g = 0.0
    @inbounds for a in eachindex(book.pos)
        g += abs(book.pos[a]) * p.mid[a, t]
    end
    g
end
