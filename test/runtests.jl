using Aqua
using StatArb
using StatArb: chol!, cholsolve!, _book_fill!
using Test, Random, LinearAlgebra, Statistics, Dates, StaticArrays

# Independent (allocating, textbook) Johansen with restricted constant, used as a reference.
function naive_johansen(X::AbstractMatrix, kdiff::Int)
    k, W = size(X)
    T = W - kdiff - 1
    dX = diff(X; dims = 2)                       # k × (W-1); column s ↔ ΔX_{s+1}
    R0 = zeros(k, T); L1 = zeros(k + 1, T); Z = zeros(k * kdiff, T)
    for s in 1:T
        t = s + kdiff + 1
        R0[:, s] = X[:, t] - X[:, t-1]
        L1[1:k, s] = X[:, t-1]; L1[k+1, s] = 1.0
        for l in 1:kdiff
            Z[(l-1)*k+1:l*k, s] = X[:, t-l] - X[:, t-l-1]
        end
    end
    if kdiff > 0
        R0 = R0 - (R0 / Z) * Z
        L1 = L1 - (L1 / Z) * Z
    end
    S00 = R0 * R0' / T; S01 = R0 * L1' / T; S11 = L1 * L1' / T
    λ = sort(real(eigvals(S11 \ (S01' * (S00 \ S01)))); rev = true)[1:k]
    trace = [-T * sum(log.(1 .- λ[r+1:k])) for r in 0:k-1]
    λ, trace
end

@testset "StatArb" begin

@testset "Aqua code quality" begin
    Aqua.test_all(StatArb)
end

@testset "data layer & temporal guardrails" begin
    @test_throws ArgumentError PricePanel([1.0 2.0; 3.0 -1.0])
    @test_throws ArgumentError PricePanel(ones(2, 3); timestamps = [DateTime(2020, 1, 3), DateTime(2020, 1, 2), DateTime(2020, 1, 4)])
    p = PricePanel(fill(10.0, 3, 50))
    @test nassets(p) == 3 && ntimes(p) == 50
    @test asset_bars(p, 2).mid[5] == 10.0

    c = Clock(0)
    tick!(c, 10)
    @test_throws ArgumentError tick!(c, 9)
    @test_throws LookaheadError assert_past(c, 11)
    @test size(history(p, c, 5)) == (3, 5)
    @test_throws ArgumentError history(p, c, 11)
    @test history(p, c, 2, 5; field = :mid) == fill(10.0, 5)
    @test_throws ArgumentError history(p, c, 5; field = :bars_per_year)
    @test_throws ArgumentError latest(MirroredRing(1, 3))

    # mirrored ring == naive last-W window, always contiguous & chronological
    r = MirroredRing(2, 7)
    data = randn(Xoshiro(1), 2, 40)
    for t in 1:40
        push!(r, view(data, :, t))
        if t >= 7
            @test window(r) == data[:, t-6:t]
            @test latest(r, 2) == data[2, t]
        end
    end
    r1 = MirroredRing(1, 5)
    for v in 1.0:3.0
        push!(r1, v)
    end
    @test filled_window(r1, 1) == [1.0, 2.0, 3.0]
    f(r, x) = @allocated push!(r, x)
    f(r, (1.0, 2.0))
    @test f(r, (1.0, 2.0)) == 0
end

@testset "small linear algebra" begin
    A = randn(Xoshiro(2), 6, 6); S = A * A' + I
    L = copy(S); @test chol!(L, 6)
    b = randn(6); x = copy(b); cholsolve!(L, x, 6)
    @test S * x ≈ b
    @test !chol!([1.0 2.0; 2.0 1.0], 2)
end

@testset "ADF / Engle-Granger / half-life" begin
    @test mackinnon_crit(2, 10^9)[2] ≈ -3.33613 atol = 1e-4
    @test mackinnon_crit(1, 100)[2] ≈ -2.86154 - 2.8903 / 100 - 4.234 / 100^2 - 40.040 / 100^3
    rng = Xoshiro(3)
    ws = EGWorkspace(300, 4)
    # size under the null (independent random walks) ≈ 5 %
    rej = sum(iscointegrated(engle_granger!(ws, cumsum(randn(rng, 300)), cumsum(randn(rng, 300)))) for _ in 1:1000)
    @test 0.03 < rej / 1000 < 0.075
    # power and recovery under cointegration
    sim = simulate_ou_pair(rng, 300; beta = 1.3, half_life = 5.0)
    r = engle_granger!(ws, sim.y, sim.x)
    @test iscointegrated(r)
    @test r.beta ≈ 1.3 atol = 0.1
    @test 2.5 < r.half_life < 10
    # AR(1) half-life
    φ = 2^(-1 / 8); s = zeros(20_000)
    for t in 2:length(s)
        s[t] = φ * s[t-1] + randn(rng)
    end
    @test ou_half_life(s) ≈ 8 rtol = 0.1
    @test ou_half_life(1.01 .^ (1:200)) == Inf                    # explosive: φ > 1
    @test ou_half_life(fill(3.0, 50)) == Inf                       # constant: no variation
    # ADF on white noise strongly rejects; auto-lag works
    ows = OLSWorkspace(500, 8)
    @test adf!(ows, randn(rng, 500), 1).stat < -10
    @test adf_auto!(ows, cumsum(randn(rng, 500)), 6).stat > -3.5
    g(ws, y, x) = @allocated engle_granger!(ws, y, x)
    g(ws, sim.y, sim.x)
    @test g(ws, sim.y, sim.x) == 0
end

@testset "Johansen VECM" begin
    rng = Xoshiro(4)
    for kdiff in (0, 1, 2)
        X = cumsum(0.01 .* randn(rng, 3, 400); dims = 2)
        X[3, :] .= 0.5 .* X[1, :] .- 0.3 .* X[2, :] .+ 0.01 .* randn(rng, 400)
        ws = JohansenWorkspace(3, 400; kdiff = kdiff)
        s = johansen!(ws, X)
        λref, trref = naive_johansen(X, kdiff)
        @test ws.lambda ≈ λref rtol = 1e-6
        @test ws.trace ≈ trref rtol = 1e-6
        @test s.trace_r0 ≈ trref[1] rtol = 1e-6
    end
    # rank recovery on a simulated VECM with known α, β
    β = [1.0 0.0; 0.0 1.0; -0.6 -0.8]
    α = [-0.15 0.0; 0.05 -0.2; 0.05 0.1]
    ranks = [johansen(simulate_vecm(rng, 600, α, β)).rank for _ in 1:50]
    @test count(==(2), ranks) >= 40
    b1 = [1.0 -1.2]'; a1 = [-0.1 0.05]'
    res = [johansen(simulate_vecm(rng, 800, a1, b1)) for _ in 1:50]
    @test count(r -> r.rank == 1, res) >= 40
    @test median([-r.beta[2] for r in res]) ≈ 1.2 atol = 0.03
    # size of the trace test (restricted constant) for independent random walks
    ws2 = JohansenWorkspace(2, 250)
    rej = sum(johansen!(ws2, cumsum(randn(rng, 2, 250); dims = 2)).rank > 0 for _ in 1:1000)
    @test rej / 1000 < 0.10
    # critical values & zero allocations
    @test johansen_trace_cv(2) == 19.96
    @test johansen_trace_cv(1; det = :unrestricted) == 3.8415
    X = simulate_vecm(rng, 250, a1, b1)
    h(ws, X) = @allocated johansen!(ws, X)
    h(ws2, X)
    @test h(ws2, X) == 0
    # ring-buffer window works as Johansen input without copies
    ring = MirroredRing(2, 250)
    for t in 1:250
        push!(ring, view(X, :, t))
    end
    @test johansen!(ws2, window(ring)).trace_r0 ≈ johansen!(ws2, X).trace_r0
end

@testset "Kalman hedge ratio" begin
    rng = Xoshiro(5)
    # matches a straightforward dense implementation
    θ = SVector(0.1, 1.0); P = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0); Q = SMatrix{2,2}(1e-5, 0, 0, 1e-5); R = 1e-3
    θn, Pn, e, S = kalman_step(θ, P, Q, R, 4.2, 4.0)
    Fd = [1.0 4.0]; Pp = Matrix(P + Q)
    Sd = (Fd * Pp * Fd')[1] + R; Kd = Pp * Fd' / Sd
    @test S ≈ Sd
    @test θn ≈ Vector(θ) + vec(Kd) * (4.2 - (Fd*Vector(θ))[1])
    @test Pn ≈ (I - Kd * Fd) * Pp rtol = 1e-8

    # tracks a drifting hedge ratio; z-scores are ~N(0,1) when the model is correct
    T = 3000
    x = 4 .+ cumsum(0.01 .* randn(rng, T))
    βt = 1.0 .+ cumsum(0.002 .* randn(rng, T))
    y = 0.2 .+ βt .* x .+ 0.01 .* randn(rng, T)
    # correctly specified filter (true Q, R): innovations must be white with unit variance
    kft = KalmanHedge(SVector(0.2, 1.0), SMatrix{2,2}(1.0, 0, 0, 1.0), SMatrix{2,2}(1e-12, 0, 0, 0.002^2), 0.01^2)
    zs = [kalman_update!(kft, y[t], x[t]) for t in 1:T]
    bs_err = abs(kft.θ[2] - βt[end])
    @test bs_err < 0.05
    @test 0.9 < std(zs[200:end]) < 1.1
    @test abs(autocor_lag(zs[200:end], 1)) < 0.06
    @test ljung_box(zs[200:end], 10) < 25          # χ²₁₀ 99% ≈ 23.2
    # convenience path with formation-window calibration tracks a stable hedge ratio
    y2 = 0.2 .+ 1.4 .* x .+ 0.01 .* randn(rng, T)
    out = kalman_filter(y2, x; calibrate = 1:250)
    @test mean(abs.(out.beta[500:end] .- 1.4)) < 0.05

    kf = calibrate_kalman(view(y, 1:250), view(x, 1:250), KalmanConfig())
    @test size_multiplier(kf) ≈ 1.0
    kf.P = kf.P * 4
    @test size_multiplier(kf) ≈ 0.25
    u(kf, a, b) = @allocated kalman_update!(kf, a, b)
    u(kf, 1.0, 2.0)
    @test u(kf, 1.0, 2.0) == 0
end

@testset "execution simulator" begin
    T = 10
    mid = fill(100.0, 2, T)
    mid[1, 3] = 99.0                                          # price dips at bar 3
    p = PricePanel(mid; spread = fill(0.10, 2, T), volume = fill(1e5, 2, T), depth = fill(1e3, 2, T))
    log_fills = Tuple{Int,Int,Float64}[]
    cb = (pair, leg, q) -> push!(log_fills, (pair, leg, q))

    # market orders: latency, half spread + impact, participation/depth cap → partial fill
    cfg = ExecConfig(latency = 1, max_participation = 0.01, depth_levels = 5.0, impact_coef = 0.5, commission_bps = 0.0)
    sim = ExecutionSimulator(cfg, 2); book = Book(1e6, 2)
    sim.sigma2 .= 0.01^2
    submit_pair!(sim, p, 1, (1, 2), (1500.0, -1500.0), 1)
    process_orders!(cb, sim, book, p, 1)
    @test isempty(log_fills)                                   # nothing at the decision bar
    process_orders!(cb, sim, book, p, 2)
    @test book.pos ≈ [1000.0, -1000.0]                         # capped at 1% of volume
    f1 = sim.fills[1]
    @test f1.price ≈ 100 + 0.05 + 100 * 0.5 * 0.01 * sqrt(1000 / 1e5)
    process_orders!(cb, sim, book, p, 3)
    @test book.pos ≈ [1500.0, -1500.0]
    @test isempty(sim.pending)

    # passive legging: leg 1 (buy at 99.95) fills on the dip, leg 2 misses and is crossed later
    empty!(log_fills)
    cfgp = ExecConfig(latency = 1, style = :passive, limit_wait = 3, commission_bps = 0.0)
    sim = ExecutionSimulator(cfgp, 2); book = Book(1e6, 2)
    submit_pair!(sim, p, 7, (1, 2), (100.0, -100.0), 2)
    process_orders!(cb, sim, book, p, 3)
    @test book.pos[1] == 100.0 && book.pos[2] == 0.0
    @test sim.fills[1].passive && sim.fills[1].price == 99.95
    @test sim.pending[1].kind == MARKET && sim.pending[1].legged
    process_orders!(cb, sim, book, p, 4)
    @test book.pos[2] == -100.0
    @test sim.fills[2].legged && sim.fills[2].price < 100.0    # crossed the spread to sell
    @test book.n_legged == 1

    # financing: short borrow fee + no interest on short proceeds
    book = Book(0.0, 2); book.pos .= [0.0, -100.0]; book.cash = 100 * 100.0
    accrue_financing!(book, p, ExecConfig(borrow_rate = 0.0252, cash_rate = 0.0), 2)
    @test book.borrow_paid ≈ 100 * 100 * 0.0252 / 252
    @test book.interest_net == 0.0
    @test_throws ArgumentError ExecutionSimulator(ExecConfig(latency = 0), 2)
end

@testset "screening & threaded scans" begin
    rng = Xoshiro(6)
    panel, truth = synthetic_universe(rng; n_pairs = 4, n_noise = 4, T = 600)
    c = distance_screen(panel.logmid, 400, 250; top = 10, sector = panel.sector)
    @test length(c) <= 10
    tests = test_pairs(panel.logmid, c, 400, 250; threaded = false)
    found = Set(Set((pt.y, pt.x)) for pt in tests if iscointegrated(pt.eg) && pt.joh.rank >= 1)
    @test count(tr -> Set((tr.y, tr.x)) in found, truth) >= 2   # hl = 2 and 5.3 must be found
    @test Set((truth[1].y, truth[1].x)) in found
    baskets = [[tr.y, tr.x] for tr in truth]
    push!(baskets, [1, 2, 9])
    a = scan_johansen(panel, baskets, 500, 250; threaded = true)
    b = scan_johansen(panel, baskets, 500, 250; threaded = false)
    @test [x.trace for x in a] ≈ [x.trace for x in b]
    @test length(a[end].trace) == 3
    rj = rolling_johansen(panel.logmid, [1, 2], 200; step = 50)
    ws = JohansenWorkspace(2, 200)
    @test rj.trace_r0[2] ≈ johansen!(ws, panel.logmid[[1, 2], rj.t[2]-199:rj.t[2]]).trace_r0
    @test with_single_threaded_blas(() -> BLAS.get_num_threads()) == 1
end

@testset "health gate" begin
    g = HealthGateConfig(hl_bounds = (1.0, 30.0), recover_checks = 2)
    @test gate_transition(g, ACTIVE, 3, false) == (QUARANTINE, 0)
    @test gate_transition(g, QUARANTINE, 0, true) == (QUARANTINE, 1)
    @test gate_transition(g, QUARANTINE, 1, true) == (ACTIVE, 2)
    @test gate_transition(g, RETIRED, 0, true) == (RETIRED, 0)
end

@testset "portfolio overlay" begin
    rng = Xoshiro(7)
    common = randn(rng, 200)
    r1 = common .+ 0.3 .* randn(rng, 200)              # fast spread
    r2 = common .+ 0.3 .* randn(rng, 200)              # slow, highly correlated spread
    w = zeros(2)
    overlay_weights!(w, OverlayWorkspace(2), [r1, r2], [1.0, 1.0], [log(2) / 3, log(2) / 30], 0.0)
    @test w[1] ≈ 1.0
    @test w[2] < 0                                      # slow spread becomes a hedge
    sy, sx = leg_shares(10_000.0, 2.0, 50.0, 20.0)
    @test sy * 50 ≈ 10_000 / 3 atol = 50
    @test sx * 20 ≈ -2 * 10_000 / 3 atol = 20
end

@testset "end-to-end backtest" begin
    rng = Xoshiro(42)
    panel, truth = synthetic_universe(rng; T = 1200)
    cfg = BacktestConfig()
    res = run_backtest(panel, cfg)
    @test length(res.equity) == ntimes(panel) - cfg.formation + 1
    @test res.book.n_fills > 50
    @test all(isfinite, res.equity)
    planted = Set(Set((tr.y, tr.x)) for tr in truth)
    sel = [Set(p) for f in res.formations for p in f.selected]
    @test count(in(planted), sel) / length(sel) > 0.8      # selection mostly finds planted pairs
    perf = performance(res)
    @test perf.sharpe > 0.5
    h = health_telemetry(res)
    @test 0 <= h.coint_survival <= 1 && h.n_checks > 0
    buf = IOBuffer()
    F, nm = panel_factors(panel, res.t[2:end])
    report(buf, res; factors = F, factor_names = nm)
    s = String(take!(buf))
    @test occursin("MODEL HEALTH", s) && occursin("FACTOR ATTRIBUTION", s)
    # fills never happen before decision + latency
    @test all(res.fills.t .>= res.fills.submit_t .+ cfg.exec.latency)
    # a pair is tracked by its asset set: never two live states on the same two assets
    for f in res.formations
        @test length(unique(minmax(p...) for p in f.selected)) == length(f.selected)
    end
    @test allunique(minmax(h.y, h.x) => h.t for h in res.health)
    # passive style runs and records passive / legged fills
    resp = run_backtest(panel, with(cfg; exec = ExecConfig(style = :passive)))
    @test resp.book.n_passive > 0

    # ── lookahead test: scrambling the future must not change the past ──
    tcut = 800
    mid2 = copy(panel.mid)
    mid2[:, tcut+1:end] .*= exp.(0.05 .* randn(rng, size(mid2, 1), ntimes(panel) - tcut))
    panel2 = PricePanel(mid2; symbols = panel.symbols, timestamps = panel.timestamps, spread = panel.spread,
                        volume = panel.volume, depth = panel.depth, sector = panel.sector)
    res2 = run_backtest(panel2, cfg)
    k = findfirst(==(tcut), res.t)
    @test res.equity[1:k] == res2.equity[1:k]
    @test res.equity[end] != res2.equity[end]
end

@testset "synthetic ground truth recovery" begin
    st = recovery_study(Xoshiro(8); nsim = 200, T = 500, beta = 1.2, half_life = 8.0)
    @test st.beta_eg.mean ≈ 1.2 atol = 0.02
    @test st.beta_johansen.mean ≈ 1.2 atol = 0.02
    @test st.beta_kalman.mean ≈ 1.2 atol = 0.03
    @test st.half_life.median ≈ 8.0 rtol = 0.25
    @test st.power_eg > 0.8 && st.power_johansen > 0.8
end

@testset "stress sweep" begin
    panel, _ = synthetic_universe(Xoshiro(9); n_pairs = 6, n_noise = 4, T = 900)
    sw = sensitivity_sweep(panel; q_scales = (1.0, 100.0), r_scales = (1.0,), windows = (120,),
                           hl_scales = (0.25, 1.0, 4.0))
    @test length(sw) == 7
    @test all(s -> isfinite(s.total_return), sw)
    asym = misspecification_asymmetry(sw)
    @test length(asym.rows) == 1
    buf = IOBuffer(); print_sweep(buf, sw); @test occursin("hl_scale", String(take!(buf)))
end

@testset "Arrow IO extension" begin
    using Arrow
    panel, _ = synthetic_universe(Xoshiro(10); n_pairs = 2, n_noise = 1, T = 50)
    path = tempname() * ".arrow"
    save_panel_arrow(path, panel)
    p2 = load_panel_arrow(path)
    @test p2.mid ≈ panel.mid
    @test p2.volume ≈ panel.volume
    @test p2.timestamps == panel.timestamps
    @test p2.sector == panel.sector
end

end
