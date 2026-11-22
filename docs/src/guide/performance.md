# Performance

## Benchmarks

`benchmark/hotloops.jl`, single thread:

| Operation | Time | Allocations |
|---|---|---|
| Kalman update | ~29 ns | 0 |
| Ring push + window view | ~3 ns | 0 |
| Engle-Granger, W = 250 | ~2.2 µs | 0 |
| Johansen, k = 2, W = 250 | ~10 µs | 0 |
| Johansen, k = 3, W = 250 | ~14 µs | 0 |
| Full walk-forward, 30 assets × 2000 bars | ~40 ms | — |

CI runs the benchmark on every push and posts the results in the job summary.

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
