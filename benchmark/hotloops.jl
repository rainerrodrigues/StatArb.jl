# Micro-benchmarks for the allocation-free hot loops.
#   julia --project=. benchmark/hotloops.jl
using StatArb, Random, StaticArrays, Printf

@noinline function bench(f::F, n) where {F}
    f()                                         # compile
    a = @allocated f()
    t = @elapsed for _ in 1:n
        f()
    end
    t / n * 1e9, a
end

function main()
rng = Xoshiro(1)
W = 250
sim = simulate_ou_pair(rng, W)
X = permutedims(hcat(sim.y, sim.x))
X3 = simulate_vecm(rng, W, [-0.1 0.05 0.02]', [1.0 -0.5 -0.5]')
eg = EGWorkspace(W, 4)
jw2 = JohansenWorkspace(2, W); jw3 = JohansenWorkspace(3, W)
kf = calibrate_kalman(sim.y, sim.x, KalmanConfig(model = :rw))
kou = calibrate_ou_kalman(sim.y, sim.x, KalmanConfig())
ring = MirroredRing(2, W)
for t in 1:W; push!(ring, view(X, :, t)); end
y0, x0 = 4.1, 3.9

rows = [
    ("Kalman update (2-state RW)", () -> kalman_update!(kf, y0, x0), 10^7),
    ("Kalman update (3-state OU)", () -> kalman_update!(kou, y0, x0), 10^7),
    ("Ring push + window view", () -> (push!(ring, (4.0, 3.9)); window(ring)), 10^7),
    ("Engle-Granger (W=250)", () -> engle_granger!(eg, sim.y, sim.x), 10^5),
    ("Johansen k=2 (W=250)", () -> johansen!(jw2, X), 10^5),
    ("Johansen k=3 (W=250)", () -> johansen!(jw3, X3), 10^5),
]
@printf("%-28s %12s %10s\n", "operation", "time/op", "allocs")
for (name, f, n) in rows
    ns, a = bench(f, n)
    @printf("%-28s %9.1f ns %9d B\n", name, ns, a)
end

panel, _ = synthetic_universe(Xoshiro(2); n_pairs = 10, n_noise = 10, T = 2000)
run_backtest(panel)
t = @elapsed res = run_backtest(panel)
@printf("\nFull walk-forward backtest: %d assets × %d bars in %.1f ms (%d health checks, %d fills)\n",
        nassets(panel), ntimes(panel), 1e3t, length(res.health), length(res.fills))
end
main()
