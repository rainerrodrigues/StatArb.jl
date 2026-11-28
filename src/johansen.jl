# ─────────────────────────────────────────────────────────────────────────────
# Rolling Johansen VECM cointegration test on pre-allocated LAPACK workspaces.
#
#   ΔX_t = Π X_{t-1} + Σ_{i=1}^{p-1} Γ_i ΔX_{t-i} + μ + ε_t,      Π = α β'
#
# Unrestricted constant (data demeaned). Concentrating out the short-run dynamics gives
# residual moment matrices S00, S01, S11 and the generalized symmetric eigenproblem
#
#   (S10 S00⁻¹ S01) v = λ S11 v
#
# solved with LAPACK dsygvd (the routine behind `LAPACK.sygvd!`) called directly on
# workspace-owned WORK/IWORK buffers so that the rolling step performs zero allocations.
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

# Critical values, rows indexed by k − r, columns (90%, 95%, 99%).
# Restricted constant (constant inside the cointegrating relation, no drift in levels):
# Osterwald-Lenum (1992) Table 1*. This is the natural specification for price spreads.
const JOH_TRACE_CV_RC = (
    (7.52, 9.24, 12.97), (17.85, 19.96, 24.60), (32.00, 34.91, 41.07),
    (49.65, 53.12, 60.16), (71.86, 76.07, 84.45))
const JOH_MAXEIG_CV_RC = (
    (7.52, 9.24, 12.97), (13.75, 15.67, 20.20), (19.77, 22.00, 26.81),
    (25.56, 28.14, 33.24), (31.66, 34.40, 39.79))
# Unrestricted constant (linear trends in levels): MacKinnon-Haug-Michelis (1999).
const JOH_TRACE_CV = (
    (2.7055, 3.8415, 6.6349), (13.4294, 15.4943, 19.9349), (27.0669, 29.7961, 35.4628),
    (44.4929, 47.8545, 54.6815), (65.8202, 69.8189, 77.8202), (91.1090, 95.7542, 104.9637))
const JOH_MAXEIG_CV = (
    (2.7055, 3.8415, 6.6349), (12.2971, 14.2639, 18.5200), (18.8928, 21.1314, 25.8650),
    (25.1236, 27.5858, 32.7172), (31.2379, 33.8777, 39.3693), (37.2786, 40.0763, 45.8662))
const JOHANSEN_MAX_K = length(JOH_TRACE_CV_RC)

@inline _joh_col(level) = level == 0.10 ? 1 : level == 0.05 ? 2 : level == 0.01 ? 3 :
    throw(ArgumentError("level must be 0.10, 0.05 or 0.01"))

@inline _trace_tab(det::Symbol) = det === :restricted ? JOH_TRACE_CV_RC : JOH_TRACE_CV
@inline _maxeig_tab(det::Symbol) = det === :restricted ? JOH_MAXEIG_CV_RC : JOH_MAXEIG_CV

"""Critical value of the trace statistic for `k - r` common trends (`det` = `:restricted` or `:unrestricted`)."""
johansen_trace_cv(kr::Integer, level = 0.05; det::Symbol = :restricted) = _trace_tab(det)[kr][_joh_col(level)]
"""Critical value of the maximum-eigenvalue statistic for `k - r` common trends."""
johansen_maxeig_cv(kr::Integer, level = 0.05; det::Symbol = :restricted) = _maxeig_tab(det)[kr][_joh_col(level)]

"""
    JohansenWorkspace(k, W; kdiff=1, det=:restricted)

All buffers needed to run the Johansen test on a `k × W` window of log prices with
`kdiff = p − 1` lagged differences. One workspace per thread.

`det = :restricted` puts the constant inside the cointegrating relation (spread with a
non-zero mean, no drift in prices); `det = :unrestricted` demeans the data (allows linear
trends in levels).
"""
struct JohansenWorkspace
    k::Int
    k1::Int                  # k + 1 with a restricted constant, else k
    W::Int
    kdiff::Int
    T::Int                   # effective observations
    m::Int                   # k · kdiff short-run regressors
    det::Symbol
    D::Matrix{Float64}       # (k+k1+m) × T stacked [ΔX_t; X_{t-1}(;1); ΔX_{t-1}; …]
    M::Matrix{Float64}       # second moments (upper triangle)
    SZZ::Matrix{Float64}
    G::Matrix{Float64}       # L_Z⁻¹ M_{Z,·}
    S00::Matrix{Float64}     # k × k
    S01::Matrix{Float64}     # k × k1
    S11::Matrix{Float64}     # k1 × k1
    C::Matrix{Float64}       # k × k1
    A::Matrix{Float64}       # k1 × k1 LHS, overwritten by eigenvectors
    B::Matrix{Float64}       # k1 × k1 RHS (S11), overwritten by its Cholesky factor
    w::Vector{Float64}       # eigenvalues, ascending (LAPACK order)
    lambda::Vector{Float64}  # k largest eigenvalues, descending
    trace::Vector{Float64}   # trace[r+1] = λ_trace(r)
    maxeig::Vector{Float64}  # maxeig[r+1] = λ_max(r, r+1)
    beta::Vector{Float64}    # leading cointegrating vector (k1 entries; last = constant if restricted), beta[1] = 1
    spread::Vector{Float64}  # β'X (+ constant) over the window
    work::Vector{Float64}
    iwork::Vector{BlasInt}
    info::Vector{BlasInt}
end

function JohansenWorkspace(k::Integer, W::Integer; kdiff::Integer = 1, det::Symbol = :restricted)
    1 <= k <= JOHANSEN_MAX_K || throw(ArgumentError("k must be in 1…$JOHANSEN_MAX_K"))
    kdiff >= 0 || throw(ArgumentError("kdiff must be ≥ 0"))
    det in (:restricted, :unrestricted) || throw(ArgumentError("det must be :restricted or :unrestricted"))
    k1 = det === :restricted ? k + 1 : k
    T = W - kdiff - 1
    m = k * kdiff
    T > k + k1 + m + 5 || throw(ArgumentError("window W=$W too short for k=$k, kdiff=$kdiff"))
    d = k + k1 + m
    lwork = k1 == 1 ? 1 : 1 + 6k1 + 2k1^2
    liwork = k1 == 1 ? 1 : 3 + 5k1
    JohansenWorkspace(k, k1, W, kdiff, T, m, det, zeros(d, T), zeros(d, d),
                      zeros(max(m, 1), max(m, 1)), zeros(max(m, 1), k + k1),
                      zeros(k, k), zeros(k, k1), zeros(k1, k1), zeros(k, k1), zeros(k1, k1), zeros(k1, k1),
                      zeros(k1), zeros(k), zeros(k), zeros(k), zeros(k1), zeros(W),
                      zeros(lwork), zeros(BlasInt, liwork), zeros(BlasInt, 1))
end

"""Compact, isbits summary of one Johansen run (safe to return from hot loops)."""
struct JohansenSummary
    ok::Bool
    rank::Int
    trace_r0::Float64
    trace_r1::Float64
    maxeig_r0::Float64
    lambda1::Float64
    half_life::Float64
    nobs::Int
end
const JOHANSEN_FAIL = JohansenSummary(false, 0, NaN, NaN, NaN, NaN, Inf, 0)
# not run because it could not change a decision (see `test_pairs(...; eg_gate)`)
const JOHANSEN_SKIPPED = JohansenSummary(false, -1, NaN, NaN, NaN, NaN, Inf, 0)
_skipped(s::JohansenSummary) = s.rank == -1

# Direct dsygvd call: A v = λ B v, ITYPE=1, JOBZ='V', UPLO='U'. Mirrors LAPACK.sygvd! but
# uses workspace-owned WORK/IWORK so nothing is allocated.
function _dsygvd!(ws::JohansenWorkspace)
    n = ws.k1
    ccall((@blasfunc(dsygvd_), libblastrampoline), Cvoid,
          (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
           Ptr{BlasInt}, Clong, Clong),
          1, UInt8('V'), UInt8('U'), n, ws.A, n, ws.B, n, ws.w,
          ws.work, length(ws.work), ws.iwork, length(ws.iwork), ws.info, 1, 1)
    @inbounds ws.info[1]
end

@inline _msym(M, a, b) = @inbounds a <= b ? M[a, b] : M[b, a]

"""
    johansen!(ws, X; level=0.05) -> JohansenSummary

Johansen trace / max-eigenvalue test on the `k × W` window `X` (columns are time,
typically `window(ring)`). Results (eigenvalues, statistics, cointegrating vector, spread)
are left in `ws`; the estimated rank uses the sequential trace test at `level`.
Zero heap allocations.
"""
function johansen!(ws::JohansenWorkspace, X::AbstractMatrix; level::Real = 0.05)
    k, k1, W, kd, T, m = ws.k, ws.k1, ws.W, ws.kdiff, ws.T, ws.m
    size(X, 1) == k && size(X, 2) == W || throw(DimensionMismatch("window must be $k × $W"))
    restricted = ws.det === :restricted
    D, M = ws.D, ws.M
    zo = k + k1                 # offset of the short-run block
    d = zo + m
    # 1. stacked data matrix
    @inbounds for s in 1:T
        t = s + kd + 1
        for i in 1:k
            D[i, s] = X[i, t] - X[i, t-1]
            D[k+i, s] = X[i, t-1]
            for l in 1:kd
                D[zo+(l-1)*k+i, s] = X[i, t-l] - X[i, t-l-1]
            end
        end
        restricted && (D[2k+1, s] = 1.0)
    end
    # 2. unrestricted constant ⇒ demean every row
    if !restricted
        @inbounds for r in 1:d
            μ = 0.0
            @simd for s in 1:T
                μ += D[r, s]
            end
            μ /= T
            @simd for s in 1:T
                D[r, s] -= μ
            end
        end
    end
    # 3. all second moments with a single BLAS-3 call: M = D D' (upper triangle)
    BLAS.syrk!('U', 'N', 1.0, D, 0.0, M)
    # 4. concentrate out short-run dynamics
    S00, S01, S11 = ws.S00, ws.S01, ws.S11
    G = ws.G
    if m > 0
        SZZ = ws.SZZ
        @inbounds for b in 1:m, a in b:m
            SZZ[a, b] = _msym(M, zo + a, zo + b)
        end
        chol!(SZZ, m) || return JOHANSEN_FAIL
        @inbounds for j in 1:zo
            for a in 1:m
                G[a, j] = _msym(M, j, zo + a)
            end
            forward_solve!(SZZ, view(G, :, j), m)
        end
    end
    invT = 1.0 / T
    @inbounds for j in 1:k, i in 1:k
        s = _msym(M, i, j)
        for a in 1:m
            s -= G[a, i] * G[a, j]
        end
        S00[i, j] = s * invT
    end
    @inbounds for j in 1:k1, i in 1:k
        s = _msym(M, i, k + j)
        for a in 1:m
            s -= G[a, i] * G[a, k+j]
        end
        S01[i, j] = s * invT
    end
    @inbounds for j in 1:k1, i in 1:k1
        s = _msym(M, k + i, k + j)
        for a in 1:m
            s -= G[a, k+i] * G[a, k+j]
        end
        S11[i, j] = s * invT
    end
    # 5. A = S10 S00⁻¹ S01 = C'C with C = L00⁻¹ S01 ;  B = S11
    chol!(S00, k) || return JOHANSEN_FAIL
    C, A, B = ws.C, ws.A, ws.B
    @inbounds for j in 1:k1
        for i in 1:k
            C[i, j] = S01[i, j]
        end
        forward_solve!(S00, view(C, :, j), k)
    end
    @inbounds for j in 1:k1, i in 1:k1
        s = 0.0
        for a in 1:k
            s += C[a, i] * C[a, j]
        end
        A[i, j] = s
        B[i, j] = S11[i, j]
    end
    # 6. generalized symmetric-definite eigenproblem (LAPACK dsygvd)
    _dsygvd!(ws) == 0 || return JOHANSEN_FAIL
    λ, w = ws.lambda, ws.w
    @inbounds for i in 1:k
        λ[i] = clamp(w[k1+1-i], 0.0, 1.0 - 1e-12)
    end
    # 7. statistics and rank (sequential trace test)
    trace, maxeig = ws.trace, ws.maxeig
    @inbounds for r in 0:k-1
        s = 0.0
        for i in r+1:k
            s -= log1p(-λ[i])
        end
        trace[r+1] = T * s
        maxeig[r+1] = -T * log1p(-λ[r+1])
    end
    tab = _trace_tab(ws.det)
    col = _joh_col(level)
    rank = 0
    @inbounds while rank < k && trace[rank+1] > tab[k-rank][col]
        rank += 1
    end
    # 8. leading cointegrating vector (eigenvector of the largest λ), β₁ = 1
    β = ws.beta
    v1 = @inbounds A[1, k1]
    @inbounds for i in 1:k1
        β[i] = abs(v1) > 1e-14 ? A[i, k1] / v1 : A[i, k1]
    end
    sp = ws.spread
    c0 = restricted ? @inbounds(β[k1]) : 0.0
    @inbounds for t in 1:W
        s = c0
        for i in 1:k
            s += β[i] * X[i, t]
        end
        sp[t] = s
    end
    hl = ou_half_life(view(sp, 1:W))
    JohansenSummary(true, rank, trace[1], k >= 2 ? trace[2] : NaN, maxeig[1], λ[1], hl, T)
end

"""Full (allocating) Johansen result, used for scans and reporting."""
struct JohansenResult
    basket::Vector{Int}
    t::Int
    ok::Bool
    rank::Int
    eigenvalues::Vector{Float64}
    trace::Vector{Float64}
    maxeig::Vector{Float64}
    beta::Vector{Float64}
    half_life::Float64
    nobs::Int
end

JohansenResult(ws::JohansenWorkspace, s::JohansenSummary, basket, t) =
    JohansenResult(collect(Int, basket), t, s.ok, s.rank, copy(ws.lambda), copy(ws.trace),
                   copy(ws.maxeig), copy(ws.beta), s.half_life, s.nobs)

"""
    johansen(X; kdiff=1, level=0.05, det=:restricted) -> JohansenResult

Convenience (allocating) wrapper for a one-off test on a `k × W` matrix of log prices.
"""
function johansen(X::AbstractMatrix; kdiff::Integer = 1, level::Real = 0.05, det::Symbol = :restricted)
    ws = JohansenWorkspace(size(X, 1), size(X, 2); kdiff = kdiff, det = det)
    s = johansen!(ws, X; level = level)
    JohansenResult(ws, s, 1:size(X, 1), size(X, 2))
end
