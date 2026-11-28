# ─────────────────────────────────────────────────────────────────────────────
# Structural-break monitoring on standardised Kalman innovations.
#
# A spread's filter innovations z_t = e_t / √S_t are white N(0, 1) while the model holds
# (the OU-augmented filter makes them so). A broken relationship shows up as
#   - a mean shift in z (hedge ratio or equilibrium level moved): two-sided Page CUSUM
#   - a variance increase in z (the spread stopped mean-reverting): CUSUM on z²
# Both run online in O(1) per bar with no allocation. Thresholds are calibrated by Monte
# Carlo to a target in-control average run length (bars between false alarms).
# ─────────────────────────────────────────────────────────────────────────────

"""
    BreakMonitorConfig(; enabled=false, arl0=5000, k=0.5, k_variance=0.5, variance=true,
                       cooldown=60, warmup=20, max_abs_z=8.0)

Online structural-break monitoring of every spread's standardised Kalman innovations
(opt-in: `BacktestConfig(breaks = BreakMonitorConfig(enabled = true))`). An alarm
quarantines the spread immediately, instead of waiting for its health checks to fail.

- `arl0`: target average number of bars between false alarms, both detectors together,
  *if the innovations were exactly N(0, 1)*. Real filter innovations have fatter tails and
  small persistent biases, so actual false alarms come about 4–5× more often: on synthetic
  universes the default 5000 gives one per ~1000 spread-bars, and 1000 gives one per ~470.
- `k`: reference value of the two-sided mean CUSUM, in standard deviations (detects mean
  shifts of about `2k` fastest); `k_variance`: reference value of the variance CUSUM on
  `(z² − 1)/√2`
- `variance`: also monitor variance increases
- `cooldown`: bars after an alarm during which the spread stays quarantined even if its
  health checks pass
- `warmup`: bars after a spread is created (or after an alarm) before monitoring starts
- `max_abs_z`: innovations are clipped to `±max_abs_z` so a single bad print can't
  trigger an alarm on its own
"""
Base.@kwdef struct BreakMonitorConfig
    enabled::Bool = false
    arl0::Float64 = 5000.0
    k::Float64 = 0.5
    k_variance::Float64 = 0.5
    variance::Bool = true
    cooldown::Int = 60
    warmup::Int = 20
    max_abs_z::Float64 = 8.0
end

# Monte Carlo average run length of the CUSUM statistic(s) under N(0,1) input
function _arl(h::Float64, k::Float64, kind::Symbol, nsim::Int, cap::Int, seed::UInt64)
    rng = Xoshiro(seed)
    total = 0
    for _ in 1:nsim
        cp = 0.0; cn = 0.0; n = 0
        while n < cap
            n += 1
            z = randn(rng)
            if kind === :mean
                cp = max(0.0, cp + z - k)
                cn = max(0.0, cn - z - k)
                (cp > h || cn > h) && break
            else
                cp = max(0.0, cp + (z * z - 1) / sqrt(2.0) - k)
                cp > h && break
            end
        end
        total += n
    end
    total / nsim
end

const _CUSUM_CACHE = Dict{Tuple{Symbol,Float64,Float64,Int},Float64}()
const _CUSUM_LOCK = ReentrantLock()

"""
    cusum_threshold(arl; k=0.5, kind=:mean, nsim=2000) -> h

Decision threshold `h` giving an in-control average run length of `arl` bars for N(0, 1)
input, found by bisection on Monte Carlo run lengths (fixed seed, cached).
`kind = :mean` is the two-sided Page CUSUM `C⁺ = max(0, C⁺ + z − k)`,
`C⁻ = max(0, C⁻ − z − k)`, alarm when either exceeds `h`; `kind = :variance` is the
one-sided CUSUM on `(z² − 1)/√2`.
"""
function cusum_threshold(arl::Real; k::Real = 0.5, kind::Symbol = :mean, nsim::Integer = 2000)
    kind in (:mean, :variance) || throw(ArgumentError("kind must be :mean or :variance"))
    arl > 1 || throw(ArgumentError("arl must be > 1"))
    k >= 0 || throw(ArgumentError("k must be ≥ 0"))
    key = (kind, Float64(arl), Float64(k), Int(nsim))
    lock(_CUSUM_LOCK) do
        get!(_CUSUM_CACHE, key) do
            seed = UInt64(0x5eed_0b0c) + hash(key) % UInt64(1 << 20)
            cap = ceil(Int, 50arl)
            lo, hi = 0.0, 1.0
            while _arl(hi, Float64(k), kind, nsim, cap, seed) < arl
                hi *= 2
                hi > 1e3 && throw(ArgumentError("no threshold reaches ARL $arl"))
            end
            for _ in 1:30
                mid = (lo + hi) / 2
                _arl(mid, Float64(k), kind, nsim, cap, seed) < arl ? (lo = mid) : (hi = mid)
                hi - lo < 1e-3 && break
            end
            (lo + hi) / 2
        end
    end
end

"""
    BreakMonitor(cfg::BreakMonitorConfig)

Online state of one spread's break detectors: the two mean CUSUMs and the variance CUSUM,
their thresholds, and the number of observations since the last reset. Update with
[`update_monitor!`](@ref) once per bar.
"""
mutable struct BreakMonitor
    cpos::Float64
    cneg::Float64
    cvar::Float64
    h_mean::Float64
    h_var::Float64
    k::Float64
    k_var::Float64
    variance::Bool
    clip::Float64
    warmup::Int
    n::Int
end

function BreakMonitor(cfg::BreakMonitorConfig)
    # each detector gets half the false-alarm budget, so together they average ≈ arl0
    split = cfg.variance ? 2 * cfg.arl0 : cfg.arl0
    hm = cusum_threshold(split; k = cfg.k, kind = :mean)
    hv = cfg.variance ? cusum_threshold(split; k = cfg.k_variance, kind = :variance) : Inf
    BreakMonitor(0.0, 0.0, 0.0, hm, hv, cfg.k, cfg.k_variance, cfg.variance, cfg.max_abs_z, cfg.warmup, 0)
end

"""Clear the CUSUM statistics and restart the warm-up."""
reset_monitor!(m::BreakMonitor) = (m.cpos = 0.0; m.cneg = 0.0; m.cvar = 0.0; m.n = 0; m)

"""
    update_monitor!(m, z) -> Symbol

Feed one standardised innovation. Returns `:none`, `:mean_up`, `:mean_down` or
`:variance`; after an alarm the statistics reset and the warm-up restarts. Non-finite
inputs are ignored. Zero allocations.
"""
@inline function update_monitor!(m::BreakMonitor, z::Float64)
    isfinite(z) || return :none
    m.n += 1
    m.n <= m.warmup && return :none
    z = clamp(z, -m.clip, m.clip)
    m.cpos = max(0.0, m.cpos + z - m.k)
    m.cneg = max(0.0, m.cneg - z - m.k)
    if m.variance
        m.cvar = max(0.0, m.cvar + (z * z - 1) / sqrt(2.0) - m.k_var)
    end
    alarm = m.cpos > m.h_mean ? :mean_up : m.cneg > m.h_mean ? :mean_down :
            m.cvar > m.h_var ? :variance : :none
    alarm === :none || reset_monitor!(m)
    alarm
end

"""
    page_cusum(z; k=0.5, arl=1000, variance=true) -> Vector{NamedTuple}

Offline replay of the online detectors over a series of standardised innovations `z`:
returns every alarm as `(t, kind)`, restarting after each one (no warm-up).
"""
function page_cusum(z::AbstractVector{<:Real}; k::Real = 0.5, arl::Real = 1000, variance::Bool = true)
    m = BreakMonitor(BreakMonitorConfig(; arl0 = arl, k = k, variance = variance, warmup = 0))
    out = NamedTuple{(:t, :kind),Tuple{Int,Symbol}}[]
    for (t, x) in enumerate(z)
        a = update_monitor!(m, Float64(x))
        a === :none || push!(out, (t = t, kind = a))
    end
    out
end

"""
    cusum_test(w; level=0.05) -> NamedTuple

Brown-Durbin-Evans CUSUM test for parameter stability on a series of recursive (one-step
prediction) residuals `w`, such as Kalman innovations. The path `W_t = Σ wᵢ / σ̂` is
compared with the 5% (or 1%, 10%) boundary `±a (√n + 2t/√n)`.

Returns `(stable, statistic, first_crossing, path)`: `statistic` is
`max |W_t| / (√n + 2t/√n)` relative to `a` (above 1 means the boundary was crossed), and
`first_crossing` the first bar outside the boundary (0 if none).
"""
function cusum_test(w::AbstractVector{<:Real}; level::Real = 0.05)
    a = level == 0.05 ? 0.948 : level == 0.01 ? 1.143 : level == 0.10 ? 0.850 :
        throw(ArgumentError("level must be 0.01, 0.05 or 0.10"))
    n = length(w)
    n >= 10 || throw(ArgumentError("need at least 10 residuals"))
    σ = std(w)
    σ > 0 || return (stable = true, statistic = 0.0, first_crossing = 0, path = zeros(n))
    path = cumsum(w ./ σ)
    sn = sqrt(n)
    ratio = [abs(path[t]) / (a * (sn + 2t / sn)) for t in 1:n]
    fc = findfirst(>(1), ratio)
    (stable = fc === nothing, statistic = maximum(ratio), first_crossing = fc === nothing ? 0 : fc, path = path)
end
