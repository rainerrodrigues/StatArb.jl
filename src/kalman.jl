# ─────────────────────────────────────────────────────────────────────────────
# Online Kalman filter for a time-varying intercept and hedge ratio.
#
#   observation:  y_t = F_t θ_t + ϵ_t,   F_t = [1, x_t],  ϵ_t ~ N(0, R)
#   transition:   θ_t = θ_{t-1} + ω_t,                    ω_t ~ N(0, Q)
#
# θ and P are StaticArrays held in a mutable struct, so each update is a handful of
# register-resident flops: no heap allocation, no generic matrix inversion
# (the observation is scalar, so the innovation covariance S_t is a scalar).
# ─────────────────────────────────────────────────────────────────────────────

const SV2 = SVector{2,Float64}
const SM22 = SMatrix{2,2,Float64,4}

"""
    KalmanConfig(; delta=1e-4, q_scale=1.0, r_scale=1.0)

`R = r_scale · σ²_ols` (residual variance of the formation-window OLS) and

    Q = q_scale · δ/(1−δ) · σ²_ols · diag(1, 1/x̄²)

i.e. Chan's δ-parameterisation made unit-free: per bar, the state noise moves the fitted
spread level by a variance of roughly `2δ·σ²` whatever the price scale, so `δ` is directly
"how fast the hedge relationship may drift relative to spread noise". `q_scale`/`r_scale`
exist for misspecification stress tests.
"""
Base.@kwdef struct KalmanConfig
    delta::Float64 = 1e-4
    q_scale::Float64 = 1.0
    r_scale::Float64 = 1.0
end

"""
    KalmanHedge

Filter state for one pair. After [`kalman_update!`](@ref) the fields `e` (innovation),
`S` (innovation variance `F P_{t|t-1} F' + R`) and `z = e/√S` (the uncertainty-aware
trading Z-score) describe bar `t` using only information up to `t`.
"""
mutable struct KalmanHedge
    θ::SV2
    P::SM22
    Q::SM22
    R::Float64
    e::Float64
    S::Float64
    z::Float64
    P_ref::Float64   # reference Var(β) (steady state at end of warm-up) for throttling
    n::Int
end

KalmanHedge(θ0::SV2, P0::SM22, Q::SM22, R::Real) =
    KalmanHedge(θ0, P0, Q, Float64(R), 0.0, Float64(R), 0.0, P0[2, 2], 0)

"""
    kalman_step(θ, P, Q, R, y, x) -> (θ⁺, P⁺, e, S)

Pure predict/update step (Joseph-form covariance update for numerical stability).
"""
@inline function kalman_step(θ::SV2, P::SM22, Q::SM22, R::Float64, y::Float64, x::Float64)
    F = SV2(1.0, x)
    Pp = P + Q                       # P_{t|t-1}
    e = y - dot(F, θ)                # innovation (prior θ ⇒ no lookahead)
    PF = Pp * F
    S = dot(F, PF) + R               # innovation variance
    K = PF / S
    IKF = SM22(I) - K * F'
    Pn = IKF * Pp * IKF' + (K * K') * R
    return θ + K * e, Pn, e, S
end

"""
    kalman_update!(kf, y, x) -> z

In-place update with the new observation; returns the innovation Z-score `e/√S`.
"""
@inline function kalman_update!(kf::KalmanHedge, y::Float64, x::Float64)
    θ, P, e, S = kalman_step(kf.θ, kf.P, kf.Q, kf.R, y, x)
    kf.θ = θ
    kf.P = P
    kf.e = e
    kf.S = S
    kf.z = e / sqrt(S)
    kf.n += 1
    kf.z
end

hedge_ratio(kf::KalmanHedge) = kf.θ[2]
intercept(kf::KalmanHedge) = kf.θ[1]

"""
    size_multiplier(kf; floor=0.0) -> ∈ [floor, 1]

Position throttle driven by posterior uncertainty: `min(1, P_ref / P_ββ)`. When the
hedge-ratio variance spikes above its calibrated steady state, size shrinks proportionally.
"""
@inline size_multiplier(kf::KalmanHedge; floor::Float64 = 0.0) =
    clamp(kf.P_ref / kf.P[2, 2], floor, 1.0)

"""
    calibrate_kalman(y, x, cfg; warmup=true) -> KalmanHedge

Formation-window calibration: OLS gives θ₀ and R; the prior covariance is the (diffuse,
×n) OLS covariance; with `warmup` the filter is then run through the formation window so
that `P` reaches steady state, and `P_ref` is set to the resulting `Var(β)`.
"""
function calibrate_kalman(y::AbstractVector, x::AbstractVector, cfg::KalmanConfig; warmup::Bool = true)
    n = length(y)
    α, β = ols_line(y, x)
    mx = 0.0
    @inbounds for t in 1:n
        mx += x[t]
    end
    mx /= n
    sse = 0.0; sxx = 0.0
    @inbounds for t in 1:n
        sse += (y[t] - α - β * x[t])^2
        sxx += (x[t] - mx)^2
    end
    σ2 = sse / (n - 2)
    varβ = σ2 / sxx
    varα = σ2 * (1 / n + mx^2 / sxx)
    cov = -mx * σ2 / sxx
    P0 = SM22(varα, cov, cov, varβ) * n        # diffuse prior
    q = cfg.q_scale * cfg.delta / (1 - cfg.delta) * σ2
    Q = SM22(q, 0.0, 0.0, q / max(mx^2, 1e-12))
    R = cfg.r_scale * σ2
    kf = KalmanHedge(SV2(α, β), P0, Q, R)
    if warmup
        @inbounds for t in 1:n
            kalman_update!(kf, Float64(y[t]), Float64(x[t]))
        end
    end
    kf.P_ref = kf.P[2, 2]
    kf
end

"""
    kalman_filter(y, x; cfg=KalmanConfig(), calibrate=1:length(y)) -> NamedTuple of vectors

Offline (allocating) convenience run returning `alpha, beta, z, e, S` paths. Priors are
calibrated on the observations in `calibrate` (pass a leading range to avoid lookahead),
after which the filter runs from the start of the series.
"""
function kalman_filter(y::AbstractVector, x::AbstractVector; cfg::KalmanConfig = KalmanConfig(),
                       calibrate = 1:length(y))
    n = length(y)
    kf = calibrate_kalman(view(y, calibrate), view(x, calibrate), cfg; warmup = false)
    α = zeros(n); β = zeros(n); z = zeros(n); e = zeros(n); S = zeros(n)
    for t in 1:n
        z[t] = kalman_update!(kf, Float64(y[t]), Float64(x[t]))
        α[t], β[t], e[t], S[t] = kf.θ[1], kf.θ[2], kf.e, kf.S
    end
    (alpha = α, beta = β, z = z, e = e, S = S)
end

"""Sample autocorrelation of `x` at `lag` (allocation-free)."""
function autocor_lag(x::AbstractVector, lag::Int)
    n = length(x)
    n > lag + 1 || return NaN
    μ = 0.0
    @inbounds for t in 1:n
        μ += x[t]
    end
    μ /= n
    num = 0.0; den = 0.0
    @inbounds for t in 1:n
        d = x[t] - μ
        den += d * d
        t > lag && (num += d * (x[t-lag] - μ))
    end
    den > 0 ? num / den : NaN
end

"""Ljung-Box Q statistic over lags 1…L (whiteness check for Kalman innovations)."""
function ljung_box(x::AbstractVector, L::Int)
    n = length(x)
    q = 0.0
    for k in 1:L
        ρ = autocor_lag(x, k)
        q += ρ^2 / (n - k)
    end
    n * (n + 2) * q
end
