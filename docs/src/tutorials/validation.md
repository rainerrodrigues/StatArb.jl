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

## Is the best configuration real?

A sensitivity sweep or a parameter grid always produces a "best" configuration, and some
of its edge is luck: try enough settings and one of them will look good.
[`overfitting_report`](@ref) runs a grid (sharing formations through a
[`FormationCache`](@ref)) and asks three questions about the winner:

- **[`deflated_sharpe`](@ref):** does it beat the Sharpe ratio the best of that many
  *skill-less* configurations would reach by luck ([`expected_max_sharpe`](@ref))? This
  accounts for sample length, skewness and fat tails. Look for a value above 0.95.
- **[`pbo`](@ref):** in combinatorially symmetric cross-validation, how often does the
  in-sample winner rank in the bottom half out of sample? Look for a value below 0.5,
  ideally below 0.2. The out-of-sample over in-sample Sharpe ratio of the winner should be
  close to 1.
- **[`bootstrap_ci`](@ref):** a stationary-bootstrap confidence interval for the winner's
  Sharpe ratio, which preserves the serial dependence of mean-reversion P&L.

On a universe with real (planted) pairs:

```@example val
panel, _ = synthetic_universe(Xoshiro(3); T = 1500)
grid = parameter_grid(BacktestConfig(); z_entry = [1.5, 2.0, 2.5], z_exit = [0.25, 0.75],
                      kalman = [KalmanConfig(q_scale = q) for q in (1.0, 10.0)])
overfitting_report(panel, grid; samples = 1000)
```

And on pure noise, with false-discovery control switched off so that spurious pairs
actually get traded:

```@example val
noise, _ = synthetic_universe(Xoshiro(3); n_pairs = 0, n_noise = 30, T = 1500)
noise_grid = parameter_grid(BacktestConfig(selection_fdr = 0.0); z_entry = [1.5, 2.0, 2.5],
                            z_exit = [0.25, 0.75], kalman = [KalmanConfig(q_scale = q) for q in (1.0, 10.0)])
overfitting_report(noise, noise_grid; samples = 1000)
```

The noise winner's probabilistic Sharpe ratio still looks encouraging, but it fails the
deflated Sharpe test, and almost none of its in-sample performance carries over out of
sample. Judge the metrics together. PBO alone can look acceptable when the configurations
in a grid are highly correlated, because the in-sample winner then tends to be close to
the median anyway.

!!! note "Calibration"
    The CI validation job checks the statistics on skill-less strategies. Among the best
    of 20, PSR exceeds 0.95 about 62% of the time (theory: 64%), while the deflated Sharpe
    ratio exceeds it about 0.1% of the time. PBO averages 0.49, and the bootstrap interval
    covers the true Sharpe about 93% of the time. The DSR is deliberately conservative:
    the maximum of `N` Sharpe ratios varies much less than a single one does.
