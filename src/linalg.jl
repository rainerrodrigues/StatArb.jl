# ─────────────────────────────────────────────────────────────────────────────
# Small allocation-free dense kernels used inside rolling loops.
# Dimensions here are tiny (≤ ~20), so hand-written loops beat LAPACK call overhead
# and, unlike the high-level wrappers, never touch the heap.
# ─────────────────────────────────────────────────────────────────────────────

"""
    chol!(A, m) -> Bool

In-place lower Cholesky factorisation of the leading `m × m` block of `A` (only the lower
triangle is read/written). Returns `false` if the block is not positive definite.
"""
@inline function chol!(A::AbstractMatrix{Float64}, m::Int)
    @inbounds for j in 1:m
        s = A[j, j]
        for k in 1:j-1
            s -= A[j, k]^2
        end
        (s > 0 && isfinite(s)) || return false
        d = sqrt(s)
        A[j, j] = d
        for i in j+1:m
            v = A[i, j]
            for k in 1:j-1
                v -= A[i, k] * A[j, k]
            end
            A[i, j] = v / d
        end
    end
    return true
end

"""Solve `L z = b` in place (`b` ← `z`) with `L` the lower factor in the leading `m×m` block."""
@inline function forward_solve!(L::AbstractMatrix{Float64}, b::AbstractVector{Float64}, m::Int)
    @inbounds for i in 1:m
        v = b[i]
        for k in 1:i-1
            v -= L[i, k] * b[k]
        end
        b[i] = v / L[i, i]
    end
    b
end

"""Solve `L' x = z` in place."""
@inline function backward_solve!(L::AbstractMatrix{Float64}, b::AbstractVector{Float64}, m::Int)
    @inbounds for i in m:-1:1
        v = b[i]
        for k in i+1:m
            v -= L[k, i] * b[k]
        end
        b[i] = v / L[i, i]
    end
    b
end

"""Solve `(L L') x = b` in place."""
@inline cholsolve!(L, b, m) = backward_solve!(L, forward_solve!(L, b, m), m)

"""
    OLSWorkspace(nmax, mmax)

Pre-allocated design matrix / normal-equation buffers for repeated small regressions.
Fill `ws.X[1:n, 1:m]` and `ws.y[1:n]`, then call [`ols!`](@ref).
"""
struct OLSWorkspace
    X::Matrix{Float64}
    y::Vector{Float64}
    XtX::Matrix{Float64}
    Xty::Vector{Float64}
    coef::Vector{Float64}
    tmp::Vector{Float64}
end
OLSWorkspace(nmax::Integer, mmax::Integer) =
    OLSWorkspace(zeros(nmax, mmax), zeros(nmax), zeros(mmax, mmax), zeros(mmax), zeros(mmax), zeros(mmax))

"""
    ols!(ws, n, m) -> (sse, ok)

Least squares on the first `n` rows / `m` columns of the workspace. Coefficients land in
`ws.coef[1:m]`; the Cholesky factor of `X'X` stays in `ws.XtX` for [`inv_diag`](@ref).
"""
function ols!(ws::OLSWorkspace, n::Int, m::Int)
    X, y, A, b, coef = ws.X, ws.y, ws.XtX, ws.Xty, ws.coef
    @inbounds for a in 1:m
        for c in a:m
            s = 0.0
            @simd for i in 1:n
                s += X[i, a] * X[i, c]
            end
            A[c, a] = s
        end
        s = 0.0
        @simd for i in 1:n
            s += X[i, a] * y[i]
        end
        b[a] = s
    end
    chol!(A, m) || return (NaN, false)
    @inbounds for a in 1:m
        coef[a] = b[a]
    end
    cholsolve!(A, coef, m)
    sse = 0.0
    @inbounds for i in 1:n
        r = y[i]
        for a in 1:m
            r -= X[i, a] * coef[a]
        end
        sse += r * r
    end
    return (sse, true)
end

"""`[(X'X)⁻¹]_{cc}` from the Cholesky factor left in `ws.XtX` by the last `ols!`."""
function inv_diag(ws::OLSWorkspace, c::Int, m::Int)
    z = ws.tmp
    @inbounds for a in 1:m
        z[a] = a == c ? 1.0 : 0.0
    end
    forward_solve!(ws.XtX, z, m)
    s = 0.0
    @inbounds for a in 1:m
        s += z[a]^2
    end
    s
end

"""Closed-form simple regression `y = α + β x` (two passes, no allocation)."""
@inline function ols_line(y::AbstractVector, x::AbstractVector)
    n = length(y)
    mx = 0.0; my = 0.0
    @inbounds for t in 1:n
        mx += x[t]; my += y[t]
    end
    mx /= n; my /= n
    sxx = 0.0; sxy = 0.0
    @inbounds for t in 1:n
        dx = x[t] - mx
        sxx += dx * dx
        sxy += dx * (y[t] - my)
    end
    β = sxy / sxx
    return (my - β * mx, β)
end
