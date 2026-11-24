# ─────────────────────────────────────────────────────────────────────────────
# Augmented Dickey-Fuller, Engle-Granger two-step test, OU half-life.
# ─────────────────────────────────────────────────────────────────────────────

# MacKinnon (2010) response-surface coefficients: crit(T) = b0 + b1/T + b2/T² + b3/T³.
# Index: [N variables][level: 1%, 5%, 10%]. "c" = constant in the (cointegrating) regression.
const MACKINNON_C = (
    ((-3.43035, -6.5393, -16.786, -79.433), (-2.86154, -2.8903, -4.234, -40.040), (-2.56677, -1.5384, -2.809, 0.0)),
    ((-3.89644, -10.9519, -22.527, 0.0), (-3.33613, -6.1101, -6.823, 0.0), (-3.04445, -4.2412, -2.720, 0.0)),
    ((-4.29374, -14.4354, -33.195, 47.433), (-3.74066, -8.5631, -10.852, 27.982), (-3.45218, -6.2143, -3.718, 4.866)),
)
# No deterministic terms, N = 1 (plain unit-root test on a zero-mean series).
const MACKINNON_N = ((-2.56574, -2.2358, -3.627, 0.0), (-1.94100, -0.2686, -3.365, 31.223), (-1.61682, 0.2656, -2.714, 25.364))

"""
    mackinnon_crit(N, T; regression=:c) -> (cv1%, cv5%, cv10%)

Finite-sample critical values for ADF (`N = 1`) and Engle-Granger residual tests
(`N` = number of variables in the cointegrating regression, 1–3).
"""
function mackinnon_crit(N::Integer, T::Integer; regression::Symbol = :c)
    tab = if regression === :c
        1 <= N <= 3 || throw(ArgumentError("MacKinnon table available for N = 1…3"))
        MACKINNON_C[N]
    elseif regression === :n
        N == 1 || throw(ArgumentError("regression=:n only for N = 1"))
        MACKINNON_N
    else
        throw(ArgumentError("regression must be :c or :n"))
    end
    f(b) = b[1] + b[2] / T + b[3] / T^2 + b[4] / T^3
    (f(tab[1]), f(tab[2]), f(tab[3]))
end

@inline _level_index(level) = level == 0.01 ? 1 : level == 0.05 ? 2 : level == 0.10 ? 3 :
    throw(ArgumentError("level must be 0.01, 0.05 or 0.10"))

"""
    ADFResult

Augmented Dickey-Fuller regression output: `stat` (t-statistic on γ), `gamma`, `lags`,
`nobs` and the residual sum of squares `sse`.
"""
struct ADFResult
    stat::Float64     # t-statistic on γ
    gamma::Float64
    lags::Int
    nobs::Int
    sse::Float64
end

# Δx_t = [c] + γ x_{t-1} + Σ_{i=1}^p φ_i Δx_{t-i} + u_t,  t = start … n
function _adf_reg!(ws::OLSWorkspace, x::AbstractVector, p::Int, constant::Bool, start::Int)
    n = length(x)
    nobs = n - start + 1
    m = 1 + p + constant
    X, y = ws.X, ws.y
    row = 0
    @inbounds for t in start:n
        row += 1
        y[row] = x[t] - x[t-1]
        X[row, 1] = x[t-1]
        for i in 1:p
            X[row, 1+i] = x[t-i] - x[t-i-1]
        end
        constant && (X[row, m] = 1.0)
    end
    sse, ok = ols!(ws, nobs, m)
    ok || return ADFResult(NaN, NaN, p, nobs, NaN)
    σ2 = sse / (nobs - m)
    se = sqrt(σ2 * inv_diag(ws, 1, m))
    γ = @inbounds ws.coef[1]
    ADFResult(γ / se, γ, p, nobs, sse)
end

"""
    adf!(ws, x, p; constant=true) -> ADFResult

ADF regression with a fixed number of lagged differences `p`. `ws` must hold at least
`length(x)` rows and `p + 2` columns. Allocation-free.
"""
adf!(ws::OLSWorkspace, x::AbstractVector, p::Int; constant::Bool = true) =
    _adf_reg!(ws, x, p, constant, p + 2)

"""
    adf_auto!(ws, x, maxlag; constant=true) -> ADFResult

ADF with the lag order chosen by AIC over a common estimation sample.
"""
function adf_auto!(ws::OLSWorkspace, x::AbstractVector, maxlag::Int; constant::Bool = true)
    start = maxlag + 2
    best_p, best_aic = 0, Inf
    for p in 0:maxlag
        r = _adf_reg!(ws, x, p, constant, start)
        m = 1 + p + constant
        aic = r.nobs * log(r.sse / r.nobs) + 2m
        if aic < best_aic
            best_aic, best_p = aic, p
        end
    end
    _adf_reg!(ws, x, best_p, constant, best_p + 2)
end

"""
    ou_half_life(s) -> Float64

Ornstein-Uhlenbeck half-life (in bars) from the AR(1) regression `Δs_t = a + b s_{t-1}`:
`hl = -ln 2 / ln(1 + b)`. Returns `Inf` when the series shows no mean reversion.
"""
function ou_half_life(s::AbstractVector)
    n = length(s)
    n < 3 && return Inf
    mx = 0.0; my = 0.0
    @inbounds for t in 2:n
        mx += s[t-1]; my += s[t] - s[t-1]
    end
    mx /= (n - 1); my /= (n - 1)
    sxx = 0.0; sxy = 0.0
    @inbounds for t in 2:n
        dx = s[t-1] - mx
        sxx += dx * dx
        sxy += dx * ((s[t] - s[t-1]) - my)
    end
    sxx > 0 || return Inf
    φ = 1.0 + sxy / sxx
    (0.0 < φ < 1.0) ? -log(2.0) / log(φ) : Inf
end

"""
    EGResult

Engle-Granger result: cointegrating regression `alpha`, `beta`; residual `adf` statistic;
MacKinnon critical values `crit` (1%, 5%, 10%); residual OU `half_life`; `resid_std`;
`nobs`; and the ADF lag order `lags`. For an exact p-value see [`eg_pvalue`](@ref).
"""
struct EGResult
    alpha::Float64
    beta::Float64
    adf::Float64
    crit::NTuple{3,Float64}   # 1%, 5%, 10%
    half_life::Float64
    resid_std::Float64
    nobs::Int
    lags::Int
end

"""`true` if the Engle-Granger statistic rejects “no cointegration” at `level`."""
iscointegrated(r::EGResult, level = 0.05) = r.adf < r.crit[_level_index(level)]

"""
    EGWorkspace(nmax, maxlag)

Buffers for repeated Engle-Granger tests on windows of up to `nmax` observations.
"""
struct EGWorkspace
    ols::OLSWorkspace
    resid::Vector{Float64}
end
EGWorkspace(nmax::Integer, maxlag::Integer = 4) = EGWorkspace(OLSWorkspace(nmax, maxlag + 2), zeros(nmax))

"""
    engle_granger!(ws, y, x; lags=1) -> EGResult

Two-step test: OLS `y = α + β x + ε`, then ADF (no deterministic terms) on the residuals,
compared against MacKinnon (2010) N=2 critical values. Allocation-free.
"""
function engle_granger!(ws::EGWorkspace, y::AbstractVector, x::AbstractVector; lags::Int = 1)
    n = length(y)
    length(x) == n || throw(DimensionMismatch("y and x must have equal length"))
    α, β = ols_line(y, x)
    r = ws.resid
    ss = 0.0
    @inbounds for t in 1:n
        e = y[t] - α - β * x[t]
        r[t] = e
        ss += e * e
    end
    e = view(r, 1:n)
    a = _adf_reg!(ws.ols, e, lags, false, lags + 2)
    EGResult(α, β, a.stat, mackinnon_crit(2, n), ou_half_life(e), sqrt(ss / (n - 2)), n, lags)
end
