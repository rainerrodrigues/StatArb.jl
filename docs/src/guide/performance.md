# Performance

## Benchmarks

`benchmark/hotloops.jl`, single thread:

| Operation | Time | Allocations |
|---|---|---|
| Kalman update, OU model (3 states) | ~71 ns | 0 |
| Kalman update, random-walk model (2 states) | ~29 ns | 0 |
| Basket spread filter update | ~47 ns | 0 |
| Ring push + window view | ~3 ns | 0 |
| Engle-Granger, W = 250 | ~2.2 µs | 0 |
| Johansen, k = 2, W = 250 | ~10 µs | 0 |
| Johansen, k = 3, W = 250 | ~14 µs | 0 |
| Full walk-forward, 30 assets × 2000 bars | ~27 ms | — |

At a larger scale (270 assets × 2,500 bars, 20 pairs and 5 baskets, formation threading
off):

| | Time | Allocated |
|---|---|---|
| Build the panel (all fields supplied) | 19 ms | 26 MiB (= the panel itself) |
| Backtest, same-sector screen | 155 ms | 12.7 MiB |
| Backtest, screen across all pairs | 236 ms | 12.6 MiB |
| Default 15-run sensitivity sweep, single thread | 1.40 s | — |
| Distance screen, 1,500 assets (1.1M pairs) | 18 ms | 8.7 MiB |

## Memory model

- **Panel.** A [`PricePanel`](@ref) holds five `n_assets × T` `Float64` matrices (mid,
  log mid, spread, volume, depth) plus a `BitMatrix` of tradable bars, so about
  `41·n·T` bytes. The constructor converts each input in one pass and only builds a
  default matrix for a field you leave out.
- **Live state.** Each spread holds `k × 2W` doubles for its mirrored ring buffer, two
  short rings and a 2×2 or 3×3 filter state. All per-check statistics run in shared
  workspaces.
- **Formation.** [`distance_screen!`](@ref) runs on a reusable [`ScreenWorkspace`](@ref):
  the normalised window (`W × n`), one Gram block (`n × 256`) and a heap of `top`
  candidates. It never builds the `n²/2` list of pairs. Basket candidates reuse the same
  normalised window.
- **Results.** Telemetry is stored as `StructVector`s of small immutable records, so each
  column is one contiguous vector.

## Formation shortcuts

- **Johansen skip gate.** A pair's selection p-value is `max(p_EG, p_Johansen)`, and
  Benjamini-Hochberg at level `q` never accepts `p > q`. Formation therefore skips the
  Johansen test whenever `p_EG > q`, which is about two thirds of screened candidates and
  removes ~70% of each skipped pair's cost. Selections are provably unchanged, and the
  logged minimum p-value is kept exact by testing skipped pairs lazily only when they
  could hold it.
- **[`FormationCache`](@ref).** A formation's selection depends only on the data window and
  the selection settings, so backtests that differ in filter, signal, risk or execution
  settings can share it. [`sensitivity_sweep`](@ref) does this by default, which makes the
  default 15-run sweep 1.9× faster.

```julia
cache = FormationCache(panel)
for q in (0.5, 1.0, 2.0)
    run_backtest(panel, BacktestConfig(kalman = KalmanConfig(q_scale = q)); formation_cache = cache)
end
cache          # FormationCache(16 selections, 32 hits, 16 misses)
```

## Where the time goes

On the 270-asset run, formation takes about 55% of the time (pair tests, basket scans and
the screen) and the rolling health checks about 40%. Formation runs threaded by default.
The health checks are exact recomputations of Engle-Granger and Johansen on the rolling
window, about 2 µs and 10 µs each. They are deliberately not replaced by incremental
moment updates, which would trade exactness for speed.

CI runs the benchmark on every push and posts the results in the job summary.

## Robustness guards on the hot path

- A filter whose state, covariance or outputs become non-finite is rebuilt from the
  spread's own rolling window, and the spread is quarantined. Every rebuild is logged in
  `res.model_resets`.
- Square-root impact is capped at `exec.max_impact` (50% by default), so no fill can have
  a non-positive price.
- BLAS's global thread setting is saved and restored once across nested or concurrent
  threaded scans.

## How the kernels avoid allocating

- **Workspaces:** [`OLSWorkspace`](@ref), [`EGWorkspace`](@ref) and
  [`JohansenWorkspace`](@ref) own every buffer they need. Create one per thread and reuse
  it.
- **Small dense algebra:** Cholesky factorisation and triangular solves on matrices of
  dimension ≤ ~20 use hand-written loops, which avoids LAPACK's call overhead and the heap
  allocations of the high-level wrappers.
- **LAPACK directly:** the Johansen eigenproblem calls `dsygvd` through `ccall`, with
  `WORK` and `IWORK` buffers owned by the workspace.
- **Static arrays:** the Kalman state lives in `SVector`/`SMatrix` fields, so updates stay
  in registers.
- **Contiguous windows:** mirrored ring buffers hand out contiguous views, so the rolling
  statistics read memory in order.

The test suite asserts `@allocated == 0` for every kernel, measured inside a function.
At the top level of a test block, Julia also counts the boxing of captured variables. CI
runs JET over the whole package, so a type instability or a possible `MethodError` fails
the build.

## Threading tips

- Start Julia with `-t auto`.
- Formation scans and sweeps pin BLAS to one thread while they run.
- [`sensitivity_sweep`](@ref) runs one backtest per task. It turns off threading inside
  formation, so the two levels of parallelism don't oversubscribe the cores.
