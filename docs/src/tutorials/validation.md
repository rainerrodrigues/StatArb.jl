# Validating the engine

Check the econometrics against data with a known answer before you trust any backtest
built on them.

## Parameter recovery

[`recovery_study`](@ref) simulates Ornstein-Uhlenbeck pairs with a known hedge ratio and
half-life. It reports:
- how well Engle-Granger (OLS), Johansen and the Kalman filter recover the hedge ratio;
- the OU half-life estimate;
- the power of both cointegration tests.

```@example val
using StatArb, Random
st = recovery_study(Xoshiro(1); nsim = 200, T = 500, beta = 1.2, half_life = 8.0)
```

## Test size

Under the null hypothesis (independent random walks), a 5% test should reject about 5% of
the time:

```@example val
rng = Xoshiro(2)
ws = EGWorkspace(250, 4)
jw = JohansenWorkspace(2, 250)
n = 1000
eg = count(_ -> iscointegrated(engle_granger!(ws, cumsum(randn(rng, 250)), cumsum(randn(rng, 250)))), 1:n) / n
jo = count(_ -> johansen!(jw, cumsum(randn(rng, 2, 250); dims = 2)).rank > 0, 1:n) / n
(engle_granger = eg, johansen = jo)
```

Johansen rejects slightly more often than 5% in small samples, which is well documented.
The default deterministic specification is the *restricted constant* (`det = :restricted`).
The unrestricted variant assumes the prices drift, and on drift-free pairs it rejects far
too often.

## A known VECM

[`simulate_vecm`](@ref) generates a system with a chosen loading matrix α and cointegrating
matrix β. Here the cointegrating rank is 2:

```@example val
β = [1.0 0.0; 0.0 1.0; -0.6 -0.8]
α = [-0.15 0.0; 0.05 -0.2; 0.05 0.1]
r = johansen(simulate_vecm(Xoshiro(3), 600, α, β))
(rank = r.rank, trace = round.(r.trace; digits = 2))
```

## Misspecification stress tests

[`sensitivity_sweep`](@ref) perturbs one thing at a time: the Kalman noises Q and R, the
health-check window, and the assumed reversion speed. Each run is an independent backtest,
so the sweep is threaded.

```@example val
panel, _ = synthetic_universe(Xoshiro(4); n_pairs = 6, n_noise = 6, T = 1200)
sw = sensitivity_sweep(panel; q_scales = (1.0, 100.0), r_scales = (1.0, 4.0),
                       windows = (120, 250), hl_scales = (0.25, 1.0, 4.0))
print_sweep(sw)
```

[`misspecification_asymmetry`](@ref) compares overestimating the reversion speed with
underestimating it by the same factor:

```@example val
misspecification_asymmetry(sw)
```
