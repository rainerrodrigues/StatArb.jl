# Statistical validation of the econometric engine: Monte Carlo studies that are too
# slow for the unit tests. Prints a Markdown table and exits non-zero if any check falls
# outside its tolerance.
#   julia --project -t auto benchmark/validation.jl
using StatArb, Random, Statistics, Printf

const RESULTS = Tuple{String,String,String,Bool}[]
check(name, value, target, ok) = push!(RESULTS, (name, value, target, ok))
fmt(x) = @sprintf("%.4f", x)

function size_study(seed, nsim, T)
    rng = Xoshiro(seed)
    eg = EGWorkspace(T, 4)
    j2 = JohansenWorkspace(2, T); j3 = JohansenWorkspace(3, T)
    re = rj2 = rj3 = 0
    for _ in 1:nsim
        y, x = cumsum(randn(rng, T)), cumsum(randn(rng, T))
        re += iscointegrated(engle_granger!(eg, y, x))
        rj2 += johansen!(j2, cumsum(randn(rng, 2, T); dims = 2)).rank > 0
        rj3 += johansen!(j3, cumsum(randn(rng, 3, T); dims = 2)).rank > 0
    end
    re / nsim, rj2 / nsim, rj3 / nsim
end

function vecm_rank_recovery(seed, nsim)
    rng = Xoshiro(seed)
    β = [1.0 0.0; 0.0 1.0; -0.6 -0.8]
    α = [-0.15 0.0; 0.05 -0.2; 0.05 0.1]
    count(_ -> johansen(simulate_vecm(rng, 600, α, β)).rank == 2, 1:nsim) / nsim
end

function lookahead_check(seed)
    rng = Xoshiro(seed)
    panel, _ = synthetic_universe(rng; n_pairs = 6, n_noise = 6, T = 900)
    cfg = BacktestConfig(threaded_formation = false)
    res = run_backtest(panel, cfg)
    tcut = 650
    mid = copy(panel.mid)
    mid[:, tcut+1:end] .*= exp.(0.05 .* randn(rng, size(mid, 1), ntimes(panel) - tcut))
    p2 = PricePanel(mid; symbols = panel.symbols, timestamps = panel.timestamps, spread = panel.spread,
                    volume = panel.volume, depth = panel.depth, sector = panel.sector)
    res2 = run_backtest(p2, cfg)
    k = findfirst(==(tcut), res.t)
    res.equity[1:k] == res2.equity[1:k]
end

# False selections on a universe with no cointegration at all, with and without FDR control
function noise_selection(seeds, fdr)
    tot = 0; nform = 0; anysel = 0
    for s in seeds
        panel, _ = synthetic_universe(Xoshiro(s); n_pairs = 0, n_noise = 30, T = 1200)
        res = run_backtest(panel, BacktestConfig(selection_fdr = fdr, threaded_formation = false))
        for f in res.formations
            tot += length(f.selected); nform += 1; anysel += !isempty(f.selected)
        end
    end
    tot / nform, anysel / nform
end

# Calibration of the selection p-values on unscreened null pairs
function pvalue_calibration(seed, n)
    rng = Xoshiro(seed)
    ps = Float64[]
    for _ in 1:n
        L = cumsum(randn(rng, 2, 252); dims = 2)
        append!(ps, pair_pvalues(test_pairs(L, [(1, 2)], 252, 252; threaded = false), 252))
    end
    mean(ps .< 0.05)
end

# Kalman model specification on OU pairs with quote noise: innovation whiteness, spread
# tracking and the OU filter's implied half-life
function kalman_spec(seed, nsim)
    rng = Xoshiro(seed)
    acf_ou = Float64[]; acf_rw = Float64[]; trk = Float64[]; hl = Float64[]
    for _ in 1:nsim
        sim = simulate_ou_pair(rng, 1500; beta = 1.2, half_life = 8.0, sigma_s = 0.006)
        y = sim.y .+ 0.003 .* randn(rng, 1500)
        ou = kalman_filter(y, sim.x; cfg = KalmanConfig(model = :ou), calibrate = 1:250)
        rw = kalman_filter(y, sim.x; cfg = KalmanConfig(model = :rw), calibrate = 1:250)
        r = 300:1500
        push!(acf_ou, autocor_lag(ou.innov_z[r], 1)); push!(acf_rw, autocor_lag(rw.innov_z[r], 1))
        push!(trk, cor(ou.spread[r], sim.s[r]))
        push!(hl, model_half_life(calibrate_ou_kalman(view(y, 1:500), view(sim.x, 1:500), KalmanConfig())))
    end
    mean(acf_ou), mean(acf_rw), mean(trk), median(hl)
end

# Basket selection on universes with planted three-asset baskets
function basket_precision(seeds)
    hits = 0; tot = 0
    for s in seeds
        panel, truth = synthetic_universe(Xoshiro(s); n_pairs = 4, n_baskets = 4, n_noise = 8, T = 1500)
        planted = Set(sort(tr.assets) for tr in truth if length(tr.assets) == 3)
        res = run_backtest(panel, BacktestConfig(max_baskets = 4, threaded_formation = false))
        for f in res.formations, b in f.baskets
            tot += 1; hits += b in planted
        end
    end
    hits / max(tot, 1), tot
end

# ADF Monte Carlo p-values under the null
function adf_calibration(seed, n)
    rng = Xoshiro(seed)
    ws = OLSWorkspace(250, 3)
    mean(adf_pvalue(adf!(ws, cumsum(randn(rng, 250)), 1)) < 0.05 for _ in 1:n)
end

t0 = time()
eg, j2, j3 = size_study(11, 4000, 250)
check("Engle-Granger size (5% test, T=250)", fmt(eg), "0.035 – 0.065", 0.035 <= eg <= 0.065)
check("Johansen k=2 size (5% test, T=250)", fmt(j2), "0.03 – 0.09", 0.03 <= j2 <= 0.09)
check("Johansen k=3 size (5% test, T=250)", fmt(j3), "≤ 0.10", j3 <= 0.10)

st = recovery_study(Xoshiro(12); nsim = 1000, T = 500, beta = 1.2, half_life = 8.0)
check("β bias, Engle-Granger (true 1.2)", fmt(st.beta_eg.mean - 1.2), "abs. bias < 0.01", abs(st.beta_eg.mean - 1.2) < 0.01)
check("β bias, Johansen (true 1.2)", fmt(st.beta_johansen.mean - 1.2), "abs. bias < 0.02", abs(st.beta_johansen.mean - 1.2) < 0.02)
check("β bias, Kalman (true 1.2)", fmt(st.beta_kalman.mean - 1.2), "abs. bias < 0.01", abs(st.beta_kalman.mean - 1.2) < 0.01)
check("half-life median (true 8)", fmt(st.half_life.median), "6.4 – 9.6", 6.4 <= st.half_life.median <= 9.6)
check("power, Engle-Granger (hl=8, T=500)", fmt(st.power_eg), "≥ 0.90", st.power_eg >= 0.90)
check("power, Johansen (hl=8, T=500)", fmt(st.power_johansen), "≥ 0.85", st.power_johansen >= 0.85)

loose_n, loose_any = noise_selection(31:42, 0.0)
fdr_n, fdr_any = noise_selection(31:42, 0.05)
check("pure noise, per-test 5% selection: spurious pairs / formation", fmt(loose_n), "(reference)", true)
check("pure noise, FDR 5% selection: spurious pairs / formation", fmt(fdr_n), "≤ 0.15", fdr_n <= 0.15)
check("pure noise, FDR 5% selection: formations selecting anything", fmt(fdr_any), "≤ 0.10", fdr_any <= 0.10)
pc = pvalue_calibration(14, 1000)
check("selection p-values on null pairs: P(p < 0.05)", fmt(pc), "≤ 0.06 (valid)", pc <= 0.06)

ka, kr, kt, kh = kalman_spec(15, 200)
check("Kalman RW model: innovation ACF(1) on OU pairs + quote noise", fmt(kr), "(reference)", true)
check("Kalman OU model: innovation ACF(1) on OU pairs + quote noise", fmt(ka), "abs < 0.10", abs(ka) < 0.10)
check("Kalman OU model: corr(filtered spread, true spread)", fmt(kt), "≥ 0.90", kt >= 0.90)
check("Kalman OU model: implied half-life median (true 8)", fmt(kh), "6.0 – 10.0", 6.0 <= kh <= 10.0)

bp, bn = basket_precision(51:56)
check("baskets: share of selections that are planted baskets ($bn selections)", fmt(bp), "≥ 0.85", bp >= 0.85)
ac = adf_calibration(16, 2000)
check("ADF Monte Carlo p-values on random walks: P(p < 0.05)", fmt(ac), "0.035 – 0.065", 0.035 <= ac <= 0.065)

rr = vecm_rank_recovery(13, 300)
check("VECM rank recovery (true r=2)", fmt(rr), "≥ 0.90", rr >= 0.90)

la = all(lookahead_check(s) for s in 21:23)
check("lookahead: future perturbation leaves past equity identical (3 seeds)", string(la), "true", la)

println("| Check | Value | Target | |")
println("|---|---|---|---|")
for (n, v, t, ok) in RESULTS
    println("| ", n, " | ", v, " | ", t, " | ", ok ? "✅" : "❌", " |")
end
@printf("\nCompleted in %.1f s with %d thread(s).\n", time() - t0, Threads.nthreads())
all(r -> r[4], RESULTS) || exit(1)
