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
    KalmanConfig(; model=:ou, delta=1e-4, q_scale=1.0, r_scale=1.0, obs_noise_frac=nothing, recalibrate=true)

Which state-space model to run and how to calibrate it from the formation window.

**`model = :rw`**: state `[α, β]` follows a random walk and the spread is treated as
observation noise ([`KalmanHedge`](@ref)). `R = r_scale · σ²_ols` and

    Q = q_scale · δ/(1−δ) · σ²_ols · diag(1, 1/x̄²)

(Chan's δ made unit-free). The trading signal is the innovation Z-score `e/√S`.

**`model = :ou`**: state `[α, β, s]` where the spread `s` is an AR(1)/Ornstein-Uhlenbeck
process ([`OUKalmanHedge`](@ref)). `φ`, the spread shock variance `σ²_η` and the
measurement noise `R` are estimated from the OLS residuals by [`ou_noise_fit`](@ref)
(AR(1)-plus-noise moments). Setting `obs_noise_frac` to a number fixes
`R = obs_noise_frac · σ²_s` instead. `R` is then multiplied by `r_scale`. The `α, β`
noise uses the same δ scheme relative to `σ²_s`. The trading signal is the filtered spread `ŝ_{t|t} / σ_s`.
With `recalibrate = true`, every health check re-fits `φ` and `σ²_η` on the rolling window.

`q_scale` and `r_scale` exist for misspecification stress tests.
"""
Base.@kwdef struct KalmanConfig
    model::Symbol = :ou
    delta::Float64 = 1e-4
    q_scale::Float64 = 1.0
    r_scale::Float64 = 1.0
    obs_noise_frac::Union{Nothing,Float64} = nothing
    recalibrate::Bool = true
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

"""Current filtered hedge ratio β_t."""
hedge_ratio(kf::KalmanHedge) = kf.θ[2]
"""Current filtered intercept α_t."""
intercept(kf::KalmanHedge) = kf.θ[1]
"""
    innovation_z(kf)

Standardised one-step prediction error `e/√S` of the last update. For a correctly
specified filter this series is white noise, so its autocorrelation is the model-health
check reported by the dashboard. (For the random-walk model it is also the trading signal.)
"""
innovation_z(kf::KalmanHedge) = kf.z

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

# ─── OU-augmented model ──────────────────────────────────────────────────────

const SV3 = SVector{3,Float64}
const SM33 = SMatrix{3,3,Float64,9}

"""
    OUKalmanHedge

Three-state filter `θ_t = [α_t, β_t, s_t]`:

    y_t = α_t + β_t x_t + s_t + ϵ_t,          ϵ_t ~ N(0, R)
    α_t = α_{t-1} + ω_α,   β_t = β_{t-1} + ω_β,   s_t = φ s_{t-1} + η_t,   η_t ~ N(0, σ²_η)

Modelling the spread as an AR(1) state instead of observation noise removes the
autocorrelation the random-walk model leaves in its innovations. The trading signal
`z = ŝ_{t|t} / σ_s` is the filtered spread in units of its stationary standard deviation
`σ_s = σ_η / √(1 − φ²)`; `innovation_z` holds the standardised prediction error. All
updates run on `StaticArrays` with zero allocations.
"""
mutable struct OUKalmanHedge
    θ::SV3
    P::SM33
    φ::Float64
    Q::SM33          # diag(q_α, q_β, σ²_η)
    R::Float64
    σs::Float64      # stationary spread std
    e::Float64
    S::Float64
    z::Float64       # trading signal ŝ/σ_s
    innov_z::Float64 # e/√S
    P_ref::Float64
    n::Int
end

"""
    ou_kalman_step(θ, P, φ, Q, R, y, x) -> (θ⁺, P⁺, e, S)

Predict/update for the OU-augmented model (Joseph form).
"""
@inline function ou_kalman_step(θ::SV3, P::SM33, φ::Float64, Q::SM33, R::Float64, y::Float64, x::Float64)
    A = SM33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, φ)
    θp = A * θ
    Pp = A * P * A' + Q
    H = SV3(1.0, x, 1.0)
    e = y - dot(H, θp)
    PH = Pp * H
    S = dot(H, PH) + R
    K = PH / S
    IKH = SM33(I) - K * H'
    Pn = IKH * Pp * IKH' + (K * K') * R
    return θp + K * e, Pn, e, S
end

@inline function kalman_update!(kf::OUKalmanHedge, y::Float64, x::Float64)
    θ, P, e, S = ou_kalman_step(kf.θ, kf.P, kf.φ, kf.Q, kf.R, y, x)
    kf.θ = θ
    kf.P = P
    kf.e = e
    kf.S = S
    kf.innov_z = e / sqrt(S)
    kf.z = θ[3] / kf.σs
    kf.n += 1
    kf.z
end

hedge_ratio(kf::OUKalmanHedge) = kf.θ[2]
intercept(kf::OUKalmanHedge) = kf.θ[1]
innovation_z(kf::OUKalmanHedge) = kf.innov_z
"""Filtered spread `ŝ_{t|t}` (OU model)."""
filtered_spread(kf::OUKalmanHedge) = kf.θ[3]
@inline size_multiplier(kf::OUKalmanHedge; floor::Float64 = 0.0) =
    clamp(kf.P_ref / kf.P[2, 2], floor, 1.0)
"""Half-life (bars) implied by the filter's AR(1) coefficient `φ`."""
model_half_life(kf::OUKalmanHedge) = 0 < kf.φ < 1 ? -log(2.0) / log(kf.φ) : Inf
model_half_life(::KalmanHedge) = Inf

"""
    ar1_fit(s) -> (φ, c, σ²_η)

Least-squares AR(1) fit `s_t = c + φ s_{t-1} + η_t` (allocation-free).
"""
function ar1_fit(s::AbstractVector)
    n = length(s)
    n >= 4 || return (NaN, NaN, NaN)
    mx = 0.0; my = 0.0
    @inbounds for t in 2:n
        mx += s[t-1]; my += s[t]
    end
    mx /= (n - 1); my /= (n - 1)
    sxx = 0.0; sxy = 0.0
    @inbounds for t in 2:n
        dx = s[t-1] - mx
        sxx += dx * dx
        sxy += dx * (s[t] - my)
    end
    sxx > 0 || return (NaN, NaN, NaN)
    φ = sxy / sxx
    c = my - φ * mx
    sse = 0.0
    @inbounds for t in 2:n
        sse += (s[t] - c - φ * s[t-1])^2
    end
    (φ, c, sse / (n - 3))
end

const _PHI_MAX = 0.999
const _R_FLOOR = 1e-3        # minimum measurement noise, as a fraction of σ²_s

@inline function _autocov(s::AbstractVector, k::Int, μ::Float64)
    n = length(s)
    a = 0.0
    @inbounds for t in k+1:n
        a += (s[t] - μ) * (s[t-k] - μ)
    end
    a / n
end

"""
    ou_noise_fit(e) -> (φ, σ²_η, R)

Fit "AR(1) spread + white measurement noise", `e_t = s_t + ϵ_t`, by matching the first
three autocovariances: `φ = γ₂/γ₁`, `σ²_s = γ₁/φ`, `R = γ₀ − σ²_s`, `σ²_η = σ²_s (1 − φ²)`.
When the moments are not informative (`γ₁ ≤ 0` or `φ ∉ (0, 1)`), falls back to a plain
AR(1) fit with no measurement noise. `R` is floored at a small fraction of `σ²_s`.
Allocation-free.
"""
function ou_noise_fit(e::AbstractVector)
    n = length(e)
    n >= 8 || return (NaN, NaN, NaN)
    μ = 0.0
    @inbounds for t in 1:n
        μ += e[t]
    end
    μ /= n
    g0 = _autocov(e, 0, μ); g1 = _autocov(e, 1, μ); g2 = _autocov(e, 2, μ)
    if g1 > 0 && 0 < g2 / g1 < 1
        φ = g2 / g1
        σs2 = min(g1 / φ, g0)
        return (φ, σs2 * (1 - φ^2), max(g0 - σs2, _R_FLOOR * σs2))
    end
    φ, _, ση2 = ar1_fit(e)
    σs2 = (isfinite(φ) && 0 < φ < 1) ? ση2 / (1 - φ^2) : g0
    (φ, ση2, _R_FLOOR * σs2)
end

# (φ, σ²_η, R) → filter parameters; φ is clamped into the stationary region
function _set_ou_dynamics!(kf::OUKalmanHedge, φ::Float64, ση2::Float64, Rhat::Float64,
                           cfg::KalmanConfig, mx::Float64)
    (isfinite(φ) && isfinite(ση2) && ση2 > 0) || return kf
    φc = clamp(φ, 0.0, _PHI_MAX)
    σs2 = ση2 / (1 - φc^2)
    q = cfg.q_scale * cfg.delta / (1 - cfg.delta) * σs2
    kf.φ = φc
    kf.σs = sqrt(σs2)
    kf.Q = SM33(q, 0.0, 0.0, 0.0, q / max(mx^2, 1e-12), 0.0, 0.0, 0.0, ση2)
    Rb = cfg.obs_noise_frac === nothing ? (isfinite(Rhat) ? Rhat : _R_FLOOR * σs2) :
                                          cfg.obs_noise_frac * σs2
    kf.R = cfg.r_scale * max(Rb, _R_FLOOR * σs2)
    kf
end

"""
    recalibrate!(kf::OUKalmanHedge, spread_window, cfg, xbar) -> kf

Re-fit `φ`, `σ²_η` and `R` with [`ou_noise_fit`](@ref) on a window of spread residuals
(past data only), keeping the filtered state. Used at every health check when
`cfg.recalibrate`.
"""
function recalibrate!(kf::OUKalmanHedge, s::AbstractVector, cfg::KalmanConfig, xbar::Real)
    φ, ση2, R = ou_noise_fit(s)
    _set_ou_dynamics!(kf, φ, ση2, R, cfg, Float64(xbar))
end
recalibrate!(kf::KalmanHedge, s, cfg, xbar) = kf

"""
    calibrate_ou_kalman(y, x, cfg; warmup=true) -> OUKalmanHedge

Formation-window calibration of the OU-augmented filter: OLS for `α, β` (diffuse prior
from the scaled OLS covariance), [`ou_noise_fit`](@ref) on the OLS residuals for `φ`,
`σ²_η` and `R`, then an optional
warm-up pass so `P` reaches steady state; `P_ref` is the resulting `Var(β)`.
"""
function calibrate_ou_kalman(y::AbstractVector, x::AbstractVector, cfg::KalmanConfig; warmup::Bool = true)
    n = length(y)
    α, β = ols_line(y, x)
    mx = sum(x) / n
    sxx = 0.0
    resid = Vector{Float64}(undef, n)
    @inbounds for t in 1:n
        resid[t] = y[t] - α - β * x[t]
        sxx += (x[t] - mx)^2
    end
    φ, ση2, Rhat = ou_noise_fit(resid)
    if !(isfinite(φ) && isfinite(ση2) && ση2 > 0)
        φ, ση2, Rhat = 0.9, max(sum(abs2, resid) / n * 0.19, 1e-12), NaN
    end
    φc = clamp(φ, 0.0, _PHI_MAX)
    σs2 = ση2 / (1 - φc^2)
    varβ = σs2 / sxx
    varα = σs2 * (1 / n + mx^2 / sxx)
    cv = -mx * σs2 / sxx
    P0 = SM33(varα * n, cv * n, 0.0, cv * n, varβ * n, 0.0, 0.0, 0.0, σs2)
    kf = OUKalmanHedge(SV3(α, β, resid[1]), P0, φc, zero(SM33), 0.0, sqrt(σs2), 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    _set_ou_dynamics!(kf, φ, ση2, Rhat, cfg, mx)
    if warmup
        @inbounds for t in 1:n
            kalman_update!(kf, Float64(y[t]), Float64(x[t]))
        end
    end
    kf.P_ref = kf.P[2, 2]
    kf
end

# ─── basket spread filter ────────────────────────────────────────────────────

"""
    BasketFilter

Two-state filter for the value `v_t = w'·log p_t` of a basket spread with fixed weights
(from the Johansen cointegrating vector):

    v_t = μ_t + s_t + ϵ_t,   μ_t = μ_{t-1} + ω_t,   s_t = φ s_{t-1} + η_t

`μ` is a slowly drifting equilibrium level and `s` the mean-reverting deviation. The
trading signal is `z = ŝ_{t|t} / σ_s`. Calibration and recalibration use the same
moment-based noise fit as the pair OU model ([`ou_noise_fit`](@ref)). Zero allocations.
"""
mutable struct BasketFilter
    θ::SV2           # (μ, s)
    P::SM22
    φ::Float64
    Q::SM22          # diag(q_μ, σ²_η)
    R::Float64
    σs::Float64
    e::Float64
    S::Float64
    z::Float64
    innov_z::Float64
    P_ref::Float64   # reference Var(s) for throttling
    n::Int
end

"""
    basket_kalman_step(θ, P, φ, Q, R, v) -> (θ⁺, P⁺, e, S)

Predict/update for [`BasketFilter`](@ref) (Joseph form).
"""
@inline function basket_kalman_step(θ::SV2, P::SM22, φ::Float64, Q::SM22, R::Float64, v::Float64)
    A = SM22(1.0, 0.0, 0.0, φ)
    θp = A * θ
    Pp = A * P * A' + Q
    H = SV2(1.0, 1.0)
    e = v - dot(H, θp)
    PH = Pp * H
    S = dot(H, PH) + R
    K = PH / S
    IKH = SM22(I) - K * H'
    Pn = IKH * Pp * IKH' + (K * K') * R
    return θp + K * e, Pn, e, S
end

"""
    kalman_update!(kf::BasketFilter, v) -> z

Update a basket filter with the new spread value `v = w'·log p_t`.
"""
@inline function kalman_update!(kf::BasketFilter, v::Float64)
    θ, P, e, S = basket_kalman_step(kf.θ, kf.P, kf.φ, kf.Q, kf.R, v)
    kf.θ = θ
    kf.P = P
    kf.e = e
    kf.S = S
    kf.innov_z = e / sqrt(S)
    kf.z = θ[2] / kf.σs
    kf.n += 1
    kf.z
end

innovation_z(kf::BasketFilter) = kf.innov_z
# a basket's weights are fixed Johansen weights: there is no single filtered hedge ratio
hedge_ratio(::BasketFilter) = NaN
filtered_spread(kf::BasketFilter) = kf.θ[2]
model_half_life(kf::BasketFilter) = 0 < kf.φ < 1 ? -log(2.0) / log(kf.φ) : Inf
@inline size_multiplier(kf::BasketFilter; floor::Float64 = 0.0) = clamp(kf.P_ref / kf.P[2, 2], floor, 1.0)

function _set_basket_dynamics!(kf::BasketFilter, φ::Float64, ση2::Float64, Rhat::Float64, cfg::KalmanConfig)
    (isfinite(φ) && isfinite(ση2) && ση2 > 0) || return kf
    φc = clamp(φ, 0.0, _PHI_MAX)
    σs2 = ση2 / (1 - φc^2)
    q = cfg.q_scale * cfg.delta / (1 - cfg.delta) * σs2
    kf.φ = φc
    kf.σs = sqrt(σs2)
    kf.Q = SM22(q, 0.0, 0.0, ση2)
    Rb = cfg.obs_noise_frac === nothing ? (isfinite(Rhat) ? Rhat : _R_FLOOR * σs2) : cfg.obs_noise_frac * σs2
    kf.R = cfg.r_scale * max(Rb, _R_FLOOR * σs2)
    kf
end

function recalibrate!(kf::BasketFilter, s::AbstractVector, cfg::KalmanConfig, ::Real = 0.0)
    φ, ση2, R = ou_noise_fit(s)
    _set_basket_dynamics!(kf, φ, ση2, R, cfg)
end

"""
    calibrate_basket_filter(v, cfg; warmup=true) -> BasketFilter

Calibrate a [`BasketFilter`](@ref) on a formation-window spread series `v`.
"""
function calibrate_basket_filter(v::AbstractVector, cfg::KalmanConfig; warmup::Bool = true)
    n = length(v)
    μ = sum(v) / n
    d = [v[t] - μ for t in 1:n]
    φ, ση2, Rhat = ou_noise_fit(d)
    if !(isfinite(φ) && isfinite(ση2) && ση2 > 0)
        φ, ση2, Rhat = 0.9, max(sum(abs2, d) / n * 0.19, 1e-12), NaN
    end
    φc = clamp(φ, 0.0, _PHI_MAX)
    σs2 = ση2 / (1 - φc^2)
    kf = BasketFilter(SV2(μ, d[1]), SM22(σs2, 0.0, 0.0, σs2), φc, zero(SM22), 0.0, sqrt(σs2),
                      0.0, 0.0, 0.0, 0.0, 0.0, 0)
    _set_basket_dynamics!(kf, φ, ση2, Rhat, cfg)
    if warmup
        @inbounds for t in 1:n
            kalman_update!(kf, Float64(v[t]))
        end
    end
    kf.P_ref = kf.P[2, 2]
    kf
end

"""
    spread_scale(model) -> Float64

Standard deviation of the traded spread, in log-price units, implied by the filter: `σ_s`
for the OU and basket models, `√S` (the innovation standard deviation) for the
random-walk model. Converts a Z-score into an expected price move (cost-aware entries).
"""
spread_scale(kf::OUKalmanHedge) = kf.σs
spread_scale(kf::BasketFilter) = kf.σs
spread_scale(kf::KalmanHedge) = sqrt(kf.S)

"""
    calibrate_filter(y, x, cfg; warmup=true)

Calibrate the filter selected by `cfg.model` (`:rw` → [`calibrate_kalman`](@ref),
`:ou` → [`calibrate_ou_kalman`](@ref)).
"""
function calibrate_filter(y::AbstractVector, x::AbstractVector, cfg::KalmanConfig; warmup::Bool = true)
    cfg.model === :ou && return calibrate_ou_kalman(y, x, cfg; warmup = warmup)
    cfg.model === :rw && return calibrate_kalman(y, x, cfg; warmup = warmup)
    throw(ArgumentError("KalmanConfig.model must be :rw or :ou, got :$(cfg.model)"))
end

"""
    kalman_filter(y, x; cfg=KalmanConfig(), calibrate=1:length(y)) -> NamedTuple of vectors

Offline (allocating) convenience run of the filter selected by `cfg.model`, returning
`alpha, beta, z` (trading signal), `innov_z`, `e`, `S` and `spread` (the filtered OU
spread; `NaN` for the random-walk model) paths. Priors are calibrated on the observations
in `calibrate` (pass a leading range to avoid lookahead), after which the filter runs from
the start of the series.
"""
function kalman_filter(y::AbstractVector, x::AbstractVector; cfg::KalmanConfig = KalmanConfig(),
                       calibrate = 1:length(y))
    kf = calibrate_filter(view(y, calibrate), view(x, calibrate), cfg; warmup = false)
    _run_filter(kf, y, x)
end

function _run_filter(kf, y::AbstractVector, x::AbstractVector)
    n = length(y)
    α = zeros(n); β = zeros(n); z = zeros(n); iz = zeros(n); e = zeros(n); S = zeros(n); sp = fill(NaN, n)
    for t in 1:n
        z[t] = kalman_update!(kf, Float64(y[t]), Float64(x[t]))
        α[t], β[t], iz[t], e[t], S[t] = intercept(kf), hedge_ratio(kf), innovation_z(kf), kf.e, kf.S
        kf isa OUKalmanHedge && (sp[t] = filtered_spread(kf))
    end
    (alpha = α, beta = β, z = z, innov_z = iz, e = e, S = S, spread = sp)
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
