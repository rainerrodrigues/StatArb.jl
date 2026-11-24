# ─────────────────────────────────────────────────────────────────────────────
# Pluggable strategy components. A backtest config holds one signal rule and one
# allocator; `run_backtest` specialises on their concrete types, so custom rules run
# at full speed. Default methods live in rules.jl.
# ─────────────────────────────────────────────────────────────────────────────

"""
    AbstractSignalRule

Decides each spread's desired direction (`-1`, `0`, `+1`). Subtype it and define

    update_signal!(rule::MyRule, sp::SpreadState, ctx::SignalContext)

which sets `sp.signal` (and may use `sp.entry_t` / `sp.blocked` for bookkeeping). The
caller has already stored the previous value in `sp.last_signal`, and forces the signal to
0 for spreads that may not trade (quarantined, out of the universe, liquidating).
Built in: [`ZScoreBands`](@ref), [`CostAwareBands`](@ref).
"""
abstract type AbstractSignalRule end

"""
    ZScoreBands()

Classic hysteresis bands on the filter's Z-score, using the config's `z_entry`, `z_exit`,
`z_stop` and the time stop `max_hold_halflives × half-life × hl_scale`. After a stop-out
the spread is blocked until `|z| < z_exit`.
"""
struct ZScoreBands <: AbstractSignalRule end

"""
    CostAwareBands(; cost_multiple=2.0)

[`ZScoreBands`](@ref) with a cost hurdle on entries: a spread opens only if the expected
reversion from the current Z-score to `z_exit`, converted to a price move with
[`spread_scale`](@ref), is at least `cost_multiple` times the estimated round-trip cost
([`round_trip_cost`](@ref): half spreads, commissions and square-root impact for the
position the allocator would take). Equivalent to raising `z_entry` per spread to
`max(z_entry, z_exit + cost_multiple · cost · Σ|w| / σ_spread)`.
"""
Base.@kwdef struct CostAwareBands <: AbstractSignalRule
    cost_multiple::Float64 = 2.0
end

"""
    AbstractAllocator

Turns the open signals into signed weights, in units of the per-spread cap
`risk.pair_cap × equity`. Subtype it and define

    target_weights!(alloc::MyAllocator, w::Vector{Float64}, spreads::Vector{SpreadState}, ctx::SignalContext)

filling `w[i]` for `spreads[i]` (every spread passed has a non-zero signal). The backtester
then applies the uncertainty throttle, scales the book to `risk.max_gross`, and splits
each spread's dollars across its legs. Built in: [`DefaultAllocator`](@ref),
[`EqualWeightAllocator`](@ref).
"""
abstract type AbstractAllocator end

"""
    DefaultAllocator()

Uses the cross-spread mean-variance overlay when `risk.correlation_overlay` is set (and
two or more spreads are open), otherwise each spread gets its signal (±1).
"""
struct DefaultAllocator <: AbstractAllocator end

"""
    EqualWeightAllocator()

Every open spread gets `±1` (its signal), regardless of correlations.
"""
struct EqualWeightAllocator <: AbstractAllocator end
