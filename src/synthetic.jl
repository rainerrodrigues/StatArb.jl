# ─────────────────────────────────────────────────────────────────────────────
# Synthetic ground truth: OU spreads and VECM systems with known parameters.
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_ou_pair(rng, T; alpha=0.2, beta=1.2, half_life=10, sigma_x=0.01, sigma_s=0.005, x0=log(50))

Log prices `(y, x)` with `x` a random walk and `y = α + β x + s`, where `s` is an AR(1)/OU
spread with the given half-life (bars). Returns `(y, x, s)`.
"""
function simulate_ou_pair(rng::AbstractRNG, T::Integer; alpha = 0.2, beta = 1.2, half_life = 10.0,
                          sigma_x = 0.01, sigma_s = 0.005, x0 = log(50.0))
    φ = 2.0^(-1.0 / half_life)
    x = zeros(T); s = zeros(T); y = zeros(T)
    x[1] = x0
    s[1] = sigma_s / sqrt(1 - φ^2) * randn(rng)
    for t in 2:T
        x[t] = x[t-1] + sigma_x * randn(rng)
        s[t] = φ * s[t-1] + sigma_s * randn(rng)
    end
    @. y = alpha + beta * x + s
    (y = y, x = x, s = s)
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
`truth` lists `(y, x, beta, half_life)` for each planted pair.
"""
function synthetic_universe(rng::AbstractRNG; n_pairs::Int = 8, n_noise::Int = 8, T::Int = 1500,
                            half_lives = range(2.0, 12.0; length = n_pairs),
                            betas = range(0.7, 1.5; length = n_pairs),
                            sigma_x = 0.012, sigma_s = 0.006, market_vol = 0.006,
                            spread_bps = 4.0, volume = 2e6, depth_frac = 0.02, obs_noise = 0.0)
    n = 2n_pairs + n_noise
    L = zeros(n, T)
    mkt = cumsum(market_vol .* randn(rng, T))
    truth = NamedTuple{(:y, :x, :beta, :half_life),Tuple{Int,Int,Float64,Float64}}[]
    sector = zeros(Int, n)
    for i in 1:n_pairs
        sim = simulate_ou_pair(rng, T; alpha = 0.1randn(rng), beta = betas[i], half_life = half_lives[i],
                               sigma_x = sigma_x, sigma_s = sigma_s, x0 = log(20 + 80rand(rng)))
        L[2i-1, :] = sim.y .+ betas[i] .* mkt
        L[2i, :] = sim.x .+ mkt
        sector[2i-1] = sector[2i] = i
        push!(truth, (y = 2i - 1, x = 2i, beta = Float64(betas[i]), half_life = Float64(half_lives[i])))
    end
    for j in 1:n_noise
        a = 2n_pairs + j
        L[a, :] = log(20 + 80rand(rng)) .+ cumsum(sigma_x .* randn(rng, T)) .+ mkt
        sector[a] = 1 + (j - 1) % max(n_pairs, 1)
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
