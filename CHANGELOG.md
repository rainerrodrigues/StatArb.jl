# Changelog

## Unreleased

- CI on Julia 1.10, 1.11 and latest across Linux, macOS and Windows, with
  coverage, a benchmark job, Monte Carlo statistical validation, and JET
  static analysis. Documenter site, TagBot and CompatHelper.
- Aqua quality checks in the test suite.
- Fix: a pair re-selected in the opposite orientation (`(x, y)` instead of
  `(y, x)`) no longer opens a second live position on the same two assets.
- Fix: `history(...; field)` only accepts the panel's bar matrices and is
  type-stable (`field = :bars_per_year` used to crash inside `view`).
- `latest` on an empty ring buffer throws instead of reading stale memory.
- Docstrings for every exported name.

## v0.1.0 (2026-11-22)

First version.

- Data layer: contiguous `n_assets × T` price panels, mirrored ring buffers
  that always expose the last `W` bars as one contiguous view, a monotonic
  clock with `LookaheadError`, Arrow IO through a package extension.
- Econometrics on pre-allocated workspaces with zero allocations: ADF (fixed
  and AIC lag order), Engle-Granger with MacKinnon (2010) critical values, OU
  half-life, and a rolling Johansen VECM (restricted or unrestricted constant)
  solved with LAPACK `dsygvd`.
- Kalman filter for a time-varying intercept and hedge ratio, built on
  StaticArrays, with Z-scores from the innovation variance and a size throttle
  driven by posterior uncertainty.
- Execution simulator: latency, half spread plus square-root impact,
  participation and depth caps, partial fills, passive orders with legging
  risk, borrow fees and margin financing.
- Walk-forward backtest with a distance pre-screen, threaded pair tests, a
  health gate that quarantines pairs (close-only), per-pair caps and a
  cross-spread mean-variance overlay.
- Dashboard covering performance (on excess returns), factor attribution,
  execution costs and model-health telemetry; synthetic OU / VECM generators,
  a parameter-recovery study and misspecification sweeps.
