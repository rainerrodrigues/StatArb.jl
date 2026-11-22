# Kalman hedge ratios

StatArb has two state-space models, selected with `KalmanConfig(model = …)`:

| | `:ou` (default) | `:rw` |
|---|---|---|
| State | ``[\alpha_t, \beta_t, s_t]`` | ``[\alpha_t, \beta_t]`` |
| Spread | AR(1) / OU state | treated as observation noise |
| Trading signal | filtered spread ``\hat s_{t\mid t} / \sigma_s`` | innovation ``e_t / \sqrt{S_t}`` |
| Innovations | white when the spread is OU | inherit the spread's autocorrelation |
| Update time | ~71 ns | ~29 ns |

Both run on `StaticArrays` with zero allocations and use the same uncertainty throttle.

## The random-walk model

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

## The OU-augmented model

A mean-reverting spread isn't observation noise: it has memory. [`OUKalmanHedge`](@ref)
makes it a state:

```math
y_t = \alpha_t + \beta_t x_t + s_t + \epsilon_t,\quad \epsilon_t \sim \mathcal N(0, R);\qquad
s_t = \phi\, s_{t-1} + \eta_t,\quad \eta_t \sim \mathcal N(0, \sigma^2_\eta)
```

with ``\alpha`` and ``\beta`` still random walks, and the update done by
[`ou_kalman_step`](@ref).

**Calibration** ([`calibrate_ou_kalman`](@ref)) starts from the OLS residuals of the
formation window. [`ou_noise_fit`](@ref) matches their first three autocovariances to an
AR(1) spread plus white measurement noise:

```math
\phi = \gamma_2/\gamma_1,\qquad \sigma^2_s = \gamma_1/\phi,\qquad R = \gamma_0 - \sigma^2_s,
\qquad \sigma^2_\eta = \sigma^2_s (1-\phi^2)
```

The moments recover the measurement noise even on 250-bar windows. Without that step,
bid-ask bounce would be read as fast mean reversion. If the moments aren't informative,
the fit falls back to a plain AR(1). Set `obs_noise_frac` to a number to fix
`R = obs_noise_frac · σ²_s` instead.

**Recalibration.** With `recalibrate = true` (the default), every health check re-fits
``\phi``, ``\sigma^2_\eta`` and ``R`` on the rolling window ([`recalibrate!`](@ref)),
keeping the filtered state. [`model_half_life`](@ref) reports the implied half-life, which
is logged in `res.health.kf_half_life`.

**Signal.** Once the spread is modelled, the innovations are white, so the innovation
Z-score no longer carries a mean-reversion signal. The trading signal is the filtered
spread in units of its stationary standard deviation,
``z_t = \hat s_{t\mid t} / \sigma_s`` with ``\sigma_s = \sigma_\eta/\sqrt{1-\phi^2}``. It
uses bar ``t``, which is known at ``t``. Orders still fill no earlier than ``t + 1``.

### What it changes

On simulated OU pairs with 0.3% quote noise (the CI validation job):

| | `:rw` | `:ou` |
|---|---|---|
| innovation ACF(1) | 0.86 | 0.02 |
| corr(filtered spread, true spread) | — | 0.96 |
| implied half-life (true 8) | — | 7.0 |

In walk-forward backtests on synthetic universes (8 seeds), the OU model makes the
innovations white and trades 30–58% less. Its Sharpe ratio is about the same as the
random-walk model's (2.71 vs 2.74 on clean data, 0.90 vs 0.90 with 0.6% noise), or a little
lower (1.90 vs 2.10 with 0.3% noise). Part of the random-walk model's return in noisy
simulations comes from trading the quote noise itself, which reverts in the very next mid.
Real bid-ask bounce is far harder to capture, so that profit is optimistic.

## Reading the telemetry

The dashboard reports the lag-1 autocorrelation of the standardised innovations
([`innovation_z`](@ref)). A correctly specified filter produces white innovations; the test
suite checks this with known `Q` and `R`. A high value means the spread has dynamics the
model doesn't capture. With `model = :rw` on OU spreads, it is typically 0.6–0.9.
