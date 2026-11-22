# Architecture and temporal guardrails

## Layers

| Layer | Files | Job |
|---|---|---|
| Data | `data.jl` | contiguous `n_assets × T` panels, mirrored ring buffers, the simulation clock |
| Econometrics | `linalg.jl`, `adf.jl`, `johansen.jl`, `kalman.jl` | allocation-free statistics on rolling windows |
| Selection & health | `screening.jl` | distance pre-screen, threaded scans, quarantine state machine |
| Execution | `execution.jl` | order lifecycle, fills, costs, financing |
| Portfolio | `portfolio.jl` | pair caps, leg split, cross-spread overlay |
| Orchestration | `backtest.jl` | walk-forward event loop |
| Reporting | `metrics.jl`, `stress.jl` | performance, attribution, telemetry, sweeps |

The econometric layers know nothing about orders, and the execution layer knows nothing
about cointegration. They only meet inside the event loop.

## The event loop

[`run_backtest`](@ref) moves a single monotonic [`Clock`](@ref) forward one bar at a time.
At bar `t` it:

1. accrues borrow fees and interest for the interval `(t−1, t]`, applies that bar's
   corporate actions, then fills orders decided at or before `t − latency` using bar-`t`
   quotes, volume and depth (halted bars don't fill);
2. checks maintenance margin and liquidates if it's breached;
3. every `trading` bars, re-runs formation on `[t−F+1, t]`: distance screen, then
   Engle-Granger and Johansen, then pair selection, then Kalman calibration;
4. for each pair, pushes bar `t` into its ring buffer, updates the Kalman filter and, every
   `health_every` bars, runs the health checks;
5. turns signals into target positions and submits orders, which become eligible from
   `t + latency`;
6. marks to market and records telemetry.

## Guardrails against lookahead

- **Latency of at least one bar.** [`ExecutionSimulator`](@ref) refuses `latency < 1`, so an
  order can never fill on the bar that triggered it.
- **Prior-state Z-scores.** The Kalman innovation is `y_t − F_t θ_{t−1}`, so the Z-score
  traded at `t` hasn't seen bar `t` through the hedge ratio.
- **Checked data access.** [`history`](@ref) and [`assert_past`](@ref) throw a
  [`LookaheadError`](@ref) for any bar after `clock.t`.
- **Perturbation test.** The test suite scrambles every price after bar 800 and checks that
  the equity curve up to bar 800 is *bit-identical*. That check catches leaks the other
  guardrails can't, such as a statistic accidentally computed over the full sample.

## Data hygiene

[`PricePanel`](@ref) validates its input:
- timestamps must strictly increase;
- prices must be finite and positive, and microstructure fields finite and non-negative.

Gaps either raise an error (`missing_policy = :error`, the default) or are forward-filled
as non-tradable bars (`:ffill`). [`validate_config`](@ref) checks a configuration before
any work starts and reports every problem at once, for example `z_exit ≥ z_entry`,
`latency < 1`, a half-life bound longer than the window can estimate, or a leverage and
margin combination that would start in a margin call.

## Mirrored ring buffers

A [`MirroredRing`](@ref) with lookback `W` stores every sample twice, at columns `p` and
`p + W` of a `k × 2W` matrix. That means the latest `W` samples always form one contiguous
block in chronological order. [`window`](@ref) returns a plain `view` of that block, and
the Johansen and Engle-Granger routines read it directly, with no copying and no
wrap-around arithmetic.

## Threading

[`test_pairs`](@ref), [`scan_johansen`](@ref), [`rolling_johansen`](@ref) and
[`sensitivity_sweep`](@ref) split their work into contiguous chunks. Each task allocates
its own workspaces, so the threads share no mutable state. BLAS is pinned to one thread for
the duration of a scan ([`with_single_threaded_blas`](@ref)), so that LAPACK and Julia's
thread pool don't compete for the same cores.
