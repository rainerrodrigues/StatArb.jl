# ─────────────────────────────────────────────────────────────────────────────
# Health gate: the quarantine state machine driven by rolling cointegration checks.
# ─────────────────────────────────────────────────────────────────────────────

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
