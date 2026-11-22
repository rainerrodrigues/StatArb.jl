# ─────────────────────────────────────────────────────────────────────────────
# Position sizing and cross-spread portfolio hedging.
# ─────────────────────────────────────────────────────────────────────────────

"""
    RiskConfig(; ...)

- `pair_cap`: gross dollars per pair as a fraction of equity (equal-dollar-style legs,
  split `1 : |β|` between the y and x legs)
- `max_gross`: portfolio gross exposure cap (multiple of equity)
- `throttle_uncertainty`: scale size by the Kalman posterior throttle `min(1, P_ref/P_ββ)`
- `correlation_overlay`: mean-variance weights across open spreads, `w ∝ Σ⁻¹ μ` with
  `μᵢ = signalᵢ · κᵢ` (κ = ln2 / half-life). Correlated, slowly reverting spreads end up
  with small or even opposite weights, i.e. they act mainly as hedges for faster spreads.
- `cov_window`: bars of spread returns for Σ; `shrinkage` toward the diagonal
- `min_trade_frac`: rebalance hysteresis (fraction of the pair cap)
- `maintenance_margin`: minimum equity / gross exposure. Below it (or at non-positive
  equity) every position is closed at market and new entries are blocked for
  `liquidation_cooldown` bars (permanently if equity is gone)
"""
Base.@kwdef struct RiskConfig
    pair_cap::Float64 = 0.10
    max_gross::Float64 = 1.5
    throttle_uncertainty::Bool = true
    correlation_overlay::Bool = true
    cov_window::Int = 60
    shrinkage::Float64 = 0.5
    min_trade_frac::Float64 = 0.25
    maintenance_margin::Float64 = 0.25
    liquidation_cooldown::Int = 21
end

"""Pre-allocated buffers for the correlation overlay (grown on demand, never per bar)."""
mutable struct OverlayWorkspace
    Σ::Matrix{Float64}
    μ::Vector{Float64}
end
OverlayWorkspace(n::Integer = 16) = OverlayWorkspace(zeros(n, n), zeros(n))

function _ensure!(ws::OverlayWorkspace, n::Int)
    if size(ws.Σ, 1) < n
        ws.Σ = zeros(2n, 2n)
        ws.μ = zeros(2n)
    end
    ws
end

@inline function _cov(a::AbstractVector, b::AbstractVector)
    n = length(a)
    ma = 0.0; mb = 0.0
    @inbounds for t in 1:n
        ma += a[t]; mb += b[t]
    end
    ma /= n; mb /= n
    s = 0.0
    @inbounds for t in 1:n
        s += (a[t] - ma) * (b[t] - mb)
    end
    s / (n - 1)
end

"""
    overlay_weights!(w, ws, returns, signals, kappas, shrinkage)

Fill `w[1:n]` with mean-variance weights `Σ_shrunk⁻¹ μ`, normalised so that `max|wᵢ| = 1`.
`returns[i]` is a vector (window) of spread returns for spread `i`. Falls back to the raw
signals if Σ is not positive definite.
"""
function overlay_weights!(w::AbstractVector{Float64}, ws::OverlayWorkspace, returns, signals, kappas, shrinkage::Float64)
    n = length(signals)
    _ensure!(ws, n)
    Σ, μ = ws.Σ, ws.μ
    @inbounds for b in 1:n, a in b:n
        c = _cov(returns[a], returns[b])
        Σ[a, b] = a == b ? c : (1 - shrinkage) * c
    end
    @inbounds for i in 1:n
        μ[i] = signals[i] * kappas[i]
    end
    if chol!(Σ, n)
        cholsolve!(Σ, μ, n)
        mx = 0.0
        @inbounds for i in 1:n
            mx = max(mx, abs(μ[i]))
        end
        @inbounds for i in 1:n
            w[i] = mx > 0 ? μ[i] / mx : 0.0
        end
    else
        @inbounds for i in 1:n
            w[i] = signals[i]
        end
    end
    w
end

"""
    leg_shares(gross_dollars, beta, mid_y, mid_x) -> (shares_y, shares_x)

Split a signed gross dollar allocation to the spread `y − β x` into whole shares:
dollar legs in ratio `1 : |β|`, long y / short β·x for a positive allocation.
"""
@inline function leg_shares(G::Float64, β::Float64, my::Float64, mx::Float64)
    d = 1.0 + abs(β)
    (round(G / d / my), round(-G * β / d / mx))
end
