# Changelog

## Unreleased

### Strategy, baskets and tooling

- **Basket trading**: `max_baskets`, `basket_size` (3–5), `basket_screen_top` and
  `min_basket_weight`. Formation grows screened pairs into baskets
  (`basket_candidates`), tests them with the `k`-asset Johansen test (exact p-values,
  Benjamini-Hochberg), and rejects pairs in disguise. Baskets trade with fixed Johansen
  weights, a two-state `BasketFilter` (~47 ns per update, zero allocations) and
  `k`-asset health checks. `synthetic_universe(...; n_baskets)` plants baskets for
  validation.
- **Pluggable strategy components**: `run_backtest` now runs as stages over a
  `BacktestState`, dispatching on `AbstractSignalRule` (`ZScoreBands`, `CostAwareBands`)
  and `AbstractAllocator` (`DefaultAllocator`, `EqualWeightAllocator`). Implement
  `update_signal!` or `target_weights!` for your own. `PairState` is replaced by
  `SpreadState` (k legs, log-price weights). The refactor reproduces the previous
  release's equity curves bit for bit.
- **Cost-aware entries**: `CostAwareBands(cost_multiple)` with `round_trip_cost` and
  `cost_aware_entry`.
- **Exact p-values**: `adf_pvalue`, `eg_pvalue`, `johansen_pvalue` and their cached null
  distributions (`adf_null_distribution`, `eg_null_distribution_single`,
  `johansen_null_distribution(T; k)`). `EGResult` now stores `lags`.
- **Multi-leg execution**: `submit_ticket!` for any number of legs (`submit_pair!` is a
  wrapper). Optional re-pegging of resting passive orders (`ExecConfig(reprice = true)`,
  off by default: it increased legging on daily bars).
- **Export and plotting**: `export_results` (CSV, or Arrow with `using Arrow`),
  `result_tables` and `write_csv`; a Makie extension with `plot_backtest`, `plot_health`
  and `plot_sweep`.
- `with(cfg; …)` is renamed to `reconfigure(cfg; …)`, which also rejects unknown field
  names. `with` is deprecated.
- CI: a plotting job, and validation checks for basket precision and ADF p-value
  calibration. Docs: tutorials on baskets and custom rules, and on plotting and export
  (rendered with CairoMakie).

### Kalman filter

- **OU-augmented Kalman filter (now the default, `KalmanConfig(model = :ou)`)**: the state
  is `[α, β, s]`, with the spread as an AR(1)/OU state. `φ`, `σ²_η` and the measurement
  noise `R` are estimated by autocovariance moments (`ou_noise_fit`), so quote noise
  isn't mistaken for reversion. They're re-fitted on the rolling window at every health
  check (`recalibrate = true`). The trading signal is the filtered spread `ŝ/σ_s`. About
  71 ns per update, zero allocations. Innovation ACF(1) on noisy OU pairs drops from 0.86
  to 0.02, and turnover falls 30–58% at similar Sharpe. `model = :rw` restores the
  previous filter.
- New: `OUKalmanHedge`, `ou_kalman_step`, `calibrate_ou_kalman`, `calibrate_filter`,
  `innovation_z`, `filtered_spread`, `model_half_life`, `ar1_fit`, `ou_noise_fit`,
  `recalibrate!`. `kalman_filter` returns `innov_z` and `spread` too.
- The telemetry's innovation autocorrelation now uses the standardised innovation for
  both models, and `res.health` records the OU filter's implied half-life
  (`kf_half_life`).
- `synthetic_universe(...; obs_noise)` adds i.i.d. quote noise to log prices.
- The validation job checks innovation whiteness, spread tracking and half-life recovery.

### Robustness

- **False-discovery control in pair selection** (`selection_fdr = 0.05` by default):
  exact Monte Carlo p-values for the two-orientation Engle-Granger statistic and the
  Johansen trace (cached per window length), combined so both tests must reject, then
  Benjamini-Hochberg. On pure noise this cuts spurious selections from 2.6 to 0.04 pairs
  per formation. New: `NullDistribution`, `null_pvalue`, `eg_null_distribution`,
  `johansen_null_distribution`, `benjamini_hochberg`, `pair_pvalues`.
- **Config validation**: `validate_config` reports every inconsistency at once, and
  `run_backtest` calls it first.
- **Missing data and halts**: `PricePanel(...; missing_policy = :ffill)` forward-fills
  gaps as non-tradable bars; `panel.tradable` (gaps, zero volume, optional halt mask)
  blocks fills, filter updates and new entries; formation needs `min_tradable_frac` real
  bars. The Arrow loader accepts `missing_policy` as well.
- **Corporate actions**: `CorporateAction(asset, t; split, dividend)`; raw prices for
  execution, causally adjusted `logmid` for signals; positions, cash and resting orders
  are adjusted on the ex-date.
- **Margin**: `risk.maintenance_margin` and `risk.liquidation_cooldown`; forced
  liquidation is reported in `res.margin_calls` and on the dashboard.
- **Fix**: rebalance hysteresis could let the book exceed `max_gross` when a new pair
  opened while the existing pairs' required downsizing fell inside the rebalance band.
  The projected gross is now checked, and downsizing is forced whenever needed.
- Validation job: false-selection rate on pure noise and p-value calibration.

### Infrastructure

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
