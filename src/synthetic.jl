# ─────────────────────────────────────────────────────────────────────────────
# Synthetic ground truth: OU spreads and VECM systems with known parameters.
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_ou_pair(rng, T; alpha=0.2, beta=1.2, half_life=10, sigma_x=0.01, sigma_s=0.005, x0=log(50),
                     break_at=nothing, break_kind=:beta, break_size=0.3)

Log prices `(y, x)` with `x` a random walk and `y = α + β x + s`, where `s` is an AR(1)/OU
spread with the given half-life (bars). Returns `(y, x, s)`.

With `break_at = t_b` the relationship breaks at bar `t_b` (see [`apply_break`](@ref)):
`break_kind = :beta` (hedge ratio jumps by `break_size`), `:level` (equilibrium level
shifts by `break_size` log points) or `:unit_root` (the spread stops mean-reverting).
The random draws are identical with or without a break.
"""
function simulate_ou_pair(rng::AbstractRNG, T::Integer; alpha = 0.2, beta = 1.2, half_life = 10.0,
                          sigma_x = 0.01, sigma_s = 0.005, x0 = log(50.0), break_at = nothing,
                          break_kind::Symbol = :beta, break_size = 0.3)
    φ = 2.0^(-1.0 / half_life)
    x = zeros(T); s = zeros(T); y = zeros(T)
    x[1] = x0
    s[1] = sigma_s / sqrt(1 - φ^2) * randn(rng)
    for t in 2:T
        x[t] = x[t-1] + sigma_x * randn(rng)
        s[t] = φ * s[t-1] + sigma_s * randn(rng)
    end
    @. y = alpha + beta * x + s
    break_at === nothing || (y = apply_break(y, x, s, φ, beta, Int(break_at), break_kind, break_size))
    (y = y, x = x, s = s)
end

"""
    apply_break(y, x, s, φ, β, t_b, kind, size) -> y′

Return a copy of the dependent log price `y = α + β x + s` (spread `s` an AR(1) with
coefficient `φ`) whose relationship breaks from bar `t_b` on:

- `:beta`: the hedge ratio becomes `β + size`, with the level adjusted so prices stay
  continuous at `t_b`; the old spread then drifts with `x`
- `:level`: the equilibrium shifts by `size` log points
- `:unit_root`: the spread's own shocks `ηₜ = sₜ − φ sₜ₋₁` accumulate as a random walk,
  so cointegration is lost
"""
function apply_break(y::AbstractVector, x::AbstractVector, s::AbstractVector, φ::Real, β::Real,
                     tb::Integer, kind::Symbol, size::Real)
    T = length(y)
    2 <= tb <= T || throw(ArgumentError("break bar must be in 2:$T"))
    y2 = copy(y)
    if kind === :beta
        for t in tb:T
            y2[t] += size * (x[t] - x[tb])
        end
    elseif kind === :level
        for t in tb:T
            y2[t] += size
        end
    elseif kind === :unit_root
        sw = s[tb-1]
        for t in tb:T
            sw += s[t] - φ * s[t-1]
            y2[t] += sw - s[t]
        end
    else
        throw(ArgumentError("break kind must be :beta, :level or :unit_root"))
    end
    y2
end

"""
    simulate_vecm(rng, T, α, β; Γ=nothing, Σ=I·1e-4, x0=fill(log(100), k))

Simulate `ΔX_t = α (β'X_{t-1} − μ) + Γ ΔX_{t-1} + ε_t`, `ε ~ N(0, Σ)`, with `μ = β'x0` so the
system starts in equilibrium. `α`, `β` are `k × r`. Returns the `k × T` matrix of log prices.
"""
function simulate_vecm(rng::AbstractRNG, T::Integer, α::AbstractMatrix, β::AbstractMatrix;
                       Γ = nothing, Σ = nothing, x0 = nothing)
    k, r = size(β)
    size(α) == (k, r) || throw(DimensionMismatch("α and β must both be k × r"))
    X0 = x0 === nothing ? fill(log(100.0), k) : collect(Float64, x0)
    Sig = Σ === nothing ? Matrix(1e-4I, k, k) : Matrix{Float64}(Σ)
    L = cholesky(Symmetric(Sig)).L
    μ = β' * X0
    X = zeros(k, T)
    X[:, 1] = X0
    dprev = zeros(k)
    for t in 2:T
        d = α * (β' * X[:, t-1] - μ) + L * randn(rng, k)
        Γ === nothing || (d .+= Γ * dprev)
        X[:, t] = X[:, t-1] + d
        dprev = d
    end
    X
end

"""
    synthetic_universe(rng; n_pairs=8, n_noise=8, T=1500, half_lives, betas, ...) -> (panel, truth)

Panel of `2·n_pairs` cointegrated assets (pair `i` uses assets `2i-1` (y) and `2i` (x)) plus
`n_noise` independent random walks. Every asset shares a weak common market factor.
Pairs are assigned their own sector; noise assets are spread across all sectors.
`obs_noise` adds i.i.d. Gaussian noise of that standard deviation to every log price, which
mimics bid-ask bounce and stale quotes (the spread dynamics are unchanged).

`n_baskets` plants three-asset baskets `y = α + b₁x₁ + b₂x₂ + s` (assets placed after the
pairs, each basket in its own sector) with half-lives `basket_half_lives`; no pair inside
a basket is cointegrated on its own.

`n_breaks` makes the first `n_breaks` planted pairs break at a uniformly random bar in
`break_window` (fractions of `T`), with [`apply_break`](@ref) of `break_kind` and
`break_size`. Break dates are drawn after everything else, so a universe with breaks is
identical to its unbroken twin up to each break.

`truth` lists, for every planted spread, `(y, x, beta, half_life, assets, weights, break_t)`
(`break_t = 0` for spreads that never break):
`weights` are the log-price weights normalised to `weights[1] = 1` (`[1, −β]` for a pair,
`[1, −b₁, −b₂]` for a basket, which has `beta = NaN`).
"""
function synthetic_universe(rng::AbstractRNG; n_pairs::Int = 8, n_noise::Int = 8, T::Int = 1500,
                            half_lives = range(2.0, 12.0; length = n_pairs),
                            betas = range(0.7, 1.5; length = n_pairs),
                            sigma_x = 0.012, sigma_s = 0.006, market_vol = 0.006,
                            spread_bps = 4.0, volume = 2e6, depth_frac = 0.02, obs_noise = 0.0,
                            n_baskets::Int = 0, basket_half_lives = range(3.0, 8.0; length = n_baskets),
                            n_breaks::Int = 0, break_kind::Symbol = :unit_root, break_size = 0.3,
                            break_window = (0.4, 0.8))
    n = 2n_pairs + 3n_baskets + n_noise
    L = zeros(n, T)
    mkt = cumsum(market_vol .* randn(rng, T))
    0 <= n_breaks <= n_pairs || throw(ArgumentError("n_breaks must be in 0:n_pairs"))
    truth = NamedTuple{(:y, :x, :beta, :half_life, :assets, :weights, :break_t),
                       Tuple{Int,Int,Float64,Float64,Vector{Int},Vector{Float64},Int}}[]
    sims = Any[]
    sector = zeros(Int, n)
    for i in 1:n_pairs
        sim = simulate_ou_pair(rng, T; alpha = 0.1randn(rng), beta = betas[i], half_life = half_lives[i],
                               sigma_x = sigma_x, sigma_s = sigma_s, x0 = log(20 + 80rand(rng)))
        L[2i-1, :] = sim.y .+ betas[i] .* mkt
        L[2i, :] = sim.x .+ mkt
        sector[2i-1] = sector[2i] = i
        push!(sims, sim)
        push!(truth, (y = 2i - 1, x = 2i, beta = Float64(betas[i]), half_life = Float64(half_lives[i]),
                      assets = [2i - 1, 2i], weights = [1.0, -Float64(betas[i])], break_t = 0))
    end
    for b in 1:n_baskets
        a0 = 2n_pairs + 3(b - 1)
        b1, b2 = 0.4 + 0.6rand(rng), 0.3 + 0.5rand(rng)
        φ = 2.0^(-1.0 / basket_half_lives[b])
        x1 = log(20 + 80rand(rng)) .+ cumsum(sigma_x .* randn(rng, T))
        x2 = log(20 + 80rand(rng)) .+ cumsum(sigma_x .* randn(rng, T))
        s = zeros(T)
        s[1] = sigma_s / sqrt(1 - φ^2) * randn(rng)
        for t in 2:T
            s[t] = φ * s[t-1] + sigma_s * randn(rng)
        end
        L[a0+1, :] = 0.1randn(rng) .+ b1 .* x1 .+ b2 .* x2 .+ s .+ (b1 + b2) .* mkt
        L[a0+2, :] = x1 .+ mkt
        L[a0+3, :] = x2 .+ mkt
        sector[a0+1:a0+3] .= n_pairs + b
        push!(truth, (y = a0 + 1, x = a0 + 2, beta = NaN, half_life = Float64(basket_half_lives[b]),
                      assets = [a0 + 1, a0 + 2, a0 + 3], weights = [1.0, -b1, -b2], break_t = 0))
    end
    for j in 1:n_noise
        a = 2n_pairs + 3n_baskets + j
        L[a, :] = log(20 + 80rand(rng)) .+ cumsum(sigma_x .* randn(rng, T)) .+ mkt
        sector[a] = 1 + (j - 1) % max(n_pairs, 1)
    end
    # breaks are drawn last, so everything before each break matches the unbroken universe
    for i in 1:n_breaks
        lo, hi = break_window
        tb = clamp(round(Int, T * (lo + (hi - lo) * rand(rng))), 2, T)
        sim = sims[i]
        φ = 2.0^(-1.0 / half_lives[i])
        L[2i-1, :] = apply_break(sim.y, sim.x, sim.s, φ, betas[i], tb, break_kind, break_size) .+ betas[i] .* mkt
        truth[i] = merge(truth[i], (break_t = tb,))
    end
    obs_noise > 0 && (L .+= obs_noise .* randn(rng, n, T))
    mid = exp.(L)
    vol = volume .* exp.(0.3 .* randn(rng, n, T))
    panel = PricePanel(mid; spread_bps = spread_bps, volume = vol, depth = vol .* depth_frac,
                       sector = sector, symbols = [Symbol("S", a) for a in 1:n])
    panel, truth
end

"""
    recovery_study(rng; nsim=200, T=500, beta=1.2, half_life=10, kdiff=1) -> NamedTuple

Monte-Carlo check of unbiased parameter recovery on OU pairs with a known hedge ratio and
half-life: OLS β (Engle-Granger), Johansen β, final Kalman β, and the OU half-life; plus
the rejection rates (power) of Engle-Granger and Johansen at 5%.
"""
function recovery_study(rng::AbstractRNG; nsim::Int = 200, T::Int = 500, beta = 1.2, half_life = 10.0,
                        kdiff::Int = 1, sigma_x = 0.01, sigma_s = 0.005)
    eg = EGWorkspace(T, 4)
    jw = JohansenWorkspace(2, T; kdiff = kdiff)
    X = zeros(2, T)
    b_eg = zeros(nsim); b_j = zeros(nsim); b_kf = zeros(nsim); hl = zeros(nsim)
    rej_eg = 0; rej_j = 0
    for s in 1:nsim
        sim = simulate_ou_pair(rng, T; beta = beta, half_life = half_life, sigma_x = sigma_x, sigma_s = sigma_s)
        r = engle_granger!(eg, sim.y, sim.x)
        X[1, :] .= sim.y; X[2, :] .= sim.x
        js = johansen!(jw, X)
        kf = calibrate_kalman(sim.y, sim.x, KalmanConfig(delta = 1e-6))
        b_eg[s] = r.beta; b_j[s] = -jw.beta[2]; b_kf[s] = kf.θ[2]; hl[s] = r.half_life
        rej_eg += iscointegrated(r); rej_j += js.rank >= 1
    end
    (beta_eg = (mean = mean(b_eg), sd = std(b_eg)), beta_johansen = (mean = median(b_j), sd = std(b_j)),
     beta_kalman = (mean = mean(b_kf), sd = std(b_kf)),
     half_life = (median = median(filter(isfinite, hl)), mean = mean(filter(isfinite, hl))),
     power_eg = rej_eg / nsim, power_johansen = rej_j / nsim)
end
