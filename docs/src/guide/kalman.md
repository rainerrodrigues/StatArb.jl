# Kalman hedge ratios

The state ``\theta_t = [\alpha_t, \beta_t]'`` follows a random walk and is observed through
the log-price regression:

```math
y_t = F_t \theta_t + \epsilon_t,\quad F_t = [1, x_t],\quad \epsilon_t \sim \mathcal N(0, R);
\qquad \theta_t = \theta_{t-1} + \omega_t,\quad \omega_t \sim \mathcal N(0, Q)
```

## Allocation-free updates

[`KalmanHedge`](@ref) stores `θ` as an `SVector{2}` and `P` and `Q` as `SMatrix{2,2}`
fields of a mutable struct. The observation is scalar, so the innovation covariance `S_t`
is a scalar and the gain needs a division, not a matrix inversion. The covariance update
uses the Joseph form for numerical stability. [`kalman_update!`](@ref) takes about 30 ns
and makes 0 allocations.

## Uncertainty-aware signals

The trading Z-score is the standardised innovation

```math
z_t = \frac{y_t - F_t \theta_{t-1}}{\sqrt{S_t}}, \qquad S_t = F_t P_{t|t-1} F_t' + R
```

It is not a residual divided by a rolling standard deviation: when the filter becomes less
sure of the hedge ratio, `S_t` grows and the same price move produces a smaller signal.
On top of that, [`size_multiplier`](@ref) scales position size by
`min(1, P_ref / P_ββ)`, where `P_ref` is the steady-state variance at the end of
calibration.

## Calibration

[`calibrate_kalman`](@ref) works in three steps on the formation window:
1. OLS gives the initial `θ` and the residual variance `σ²`, which sets `R = r_scale · σ²`.
2. The OLS covariance, scaled up by the sample size `n`, gives a diffuse prior.
3. The filter is run through the formation window, so that `P` reaches steady state before
   any live trading.

[`KalmanConfig`](@ref) sets `Q` in a unit-free form of Chan's δ:

```math
Q = q_{\text{scale}} \cdot \frac{\delta}{1-\delta} \cdot \sigma^2 \cdot \mathrm{diag}(1, 1/\bar x^2)
```

The textbook `δ/(1−δ)·I` depends on price units. On log prices it lets the intercept absorb
the spread itself, and the filter then produces almost no trading signals.

## Reading the telemetry

The dashboard reports the lag-1 autocorrelation of the innovations. A correctly specified
filter produces white innovations; the test suite checks this with known `Q` and `R`. A
high value means the spread has dynamics the random-walk-coefficient model doesn't capture,
for example Ornstein-Uhlenbeck mean reversion.
