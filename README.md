# StatArb.jl

A statistical-arbitrage backtester that keeps **econometric health monitoring** (rolling
Engle-Granger / Johansen cointegration, OU half-life, Kalman state-space hedge ratios)
separate from **multi-leg execution simulation** (latency, spread, square-root impact,
liquidity caps, legging risk, borrow and financing costs). Every inner-loop kernel runs on
pre-allocated workspaces with **zero heap allocations**, and JET reports no runtime dispatch in any of them.

```julia
using StatArb, Random
panel, truth = synthetic_universe(Xoshiro(1); n_pairs = 10, n_noise = 10, T = 2000)
res = run_backtest(panel, BacktestConfig())
report(res)                                   # performance + execution + model health
sw  = sensitivity_sweep(panel)                # Q / R / window / half-life misspecification
```

Run with threads for parallel formation scans and sweeps: `julia --project=. -t auto`.

## Layout

| File | Responsibility |
|---|---|
| `src/data.jl` | `PricePanel` (contiguous `n_assets × T` matrices), `MirroredRing` buffers, `Clock` + `LookaheadError` |
| `src/linalg.jl` | allocation-free Cholesky / triangular solves / OLS workspace |
| `src/adf.jl` | ADF (fixed & AIC lag), Engle-Granger two-step, MacKinnon (2010) critical values, OU half-life |
| `src/johansen.jl` | rolling Johansen VECM: trace & max-eigenvalue statistics, rank, cointegrating vector |
| `src/kalman.jl` | `StaticArrays` Kalman filter for `θ_t = [α_t, β_t]`, innovation Z-scores, uncertainty throttle |
| `src/execution.jl` | order lifecycle, market/limit fills, impact, participation/depth caps, legging, financing |
| `src/screening.jl` | distance pre-screen, threaded pair/basket scans, health-gate state machine |
| `src/portfolio.jl` | pair caps, hedge-ratio leg split, cross-spread mean-variance overlay |
| `src/backtest.jl` | walk-forward event loop |
| `src/metrics.jl` | Sharpe/Sortino/MDD/Calmar, factor attribution, health telemetry, text dashboard |
| `src/synthetic.jl` | OU pairs, VECM simulator, synthetic universes, parameter-recovery study |
| `src/stress.jl` | parameter-misspecification sweeps |
| `ext/StatArbArrowExt.jl` | memory-mapped Arrow IO (`using Arrow` activates it) |

## Design notes

**Temporal guardrails.** The event loop has one monotonic clock. At bar `t` it does the following, in order:
1. execute orders decided at `≤ t − latency` on bar-`t` quotes;
2. accrue financing;
3. (re)run formation on `[t−F+1, t]`;
4. update the filters and health checks with bar `t`;
5. decide orders, which become eligible from `t + latency`. `latency ≥ 1` is enforced.

The Kalman Z-score uses the *prior* state, so it contains no information from bar `t` itself. The test
suite checks this with a perturbation test: it scrambles every price after bar 800 and asserts the equity
curve up to bar 800 doesn't change at all.

**Ring buffers.** `MirroredRing` writes each sample at positions `p` and `p+W` of a
`k × 2W` matrix. That makes the latest `W` samples always one contiguous, chronologically ordered
`view`, with no wrap-around logic and no copies. Johansen and Engle-Granger consume `window(ring)` directly.

**Johansen.** The test stacks `[ΔX_t; X_{t−1}(;1); ΔX_{t−1…}]`, forms every second moment with a single
`BLAS.syrk!`, and concentrates out the short-run dynamics with in-place Cholesky solves. It then solves
`(S10 S00⁻¹ S01) v = λ S11 v` with LAPACK `dsygvd`. This is the routine behind
`LAPACK.sygvd!`, called through `ccall` with workspace-owned `WORK`/`IWORK` buffers, because the
high-level wrapper allocates them on every call. The test suite cross-checks the eigenvalues and trace
statistics against an independent textbook implementation for `kdiff ∈ {0,1,2}`.

The default deterministic spec is the **restricted constant**: a constant inside the cointegrating
relation and no drift in prices, which is the natural model for a spread. The unrestricted-constant
variant (`det = :unrestricted`) is available, but on drift-free random walks it over-rejected badly
(12–31% empirical size against a nominal 5%). The restricted version measured 4–7%.

**Kalman filter.** `θ`, `P` and `Q` are `SVector`/`SMatrix` fields of a mutable struct. The observation
is scalar, so the "inversion" is a scalar division, and an update takes about 30 ns with 0 allocations. Q
uses a unit-free version of Chan's δ: `Q = q·δ/(1−δ)·σ²·diag(1, 1/x̄²)`. The textbook `δ/(1−δ)·I`
depends on price units, and on log prices it let the intercept absorb the spread. Size is throttled by
`min(1, P_ref / P_ββ)`.

**Execution.**
- **Market orders** pay the half spread plus square-root impact `η σ √(q/V)`.
- **Fill caps:** each bar's fill is limited by participation in bar volume and by displayed depth;
  anything unfilled rolls forward as a partial fill.
- **Limit orders** rest at the decision-time touch and fill only on a strict trade-through, so they carry
  adverse selection.
- **Legging:** in a passive ticket, if one leg fills and the other doesn't, the missed leg is crossed
  `latency` bars later.
- **Financing:** shorts pay borrow fees; short proceeds earn no interest; debit balances pay the margin
  rate.

**Health gate.** A pair is moved to *quarantine* (close-only) when the rolling Johansen rank drops to 0
(`gate_on = :eg` or `:either` are also available) or when its half-life leaves `hl_bounds`. It
recovers after `recover_checks` consecutive passing checks.

**Metrics.** Sharpe, Sortino and alpha are computed on returns **in excess of the cash rate**.
Otherwise the interest earned on a market-neutral book's idle cash shows up as alpha (on the demo, the
raw Sharpe of 10.9 drops to 2.9).

## Benchmarks (`benchmark/hotloops.jl`, single thread)

| Operation | Time | Allocations |
|---|---|---|
| Kalman update | ~29 ns | 0 |
| Ring push + window view | ~3 ns | 0 |
| Engle-Granger, W=250 | ~2.2 µs | 0 |
| Johansen k=2, W=250 | ~10 µs | 0 |
| Johansen k=3, W=250 | ~14 µs | 0 |
| Full walk-forward, 30 assets × 2000 bars | ~38 ms | — |

## What the demo shows (`examples/walkforward_demo.jl`)

- **Parameter recovery** (300 simulations, β = 1.2, hl = 8): the EG, Johansen and Kalman β estimates
  average 1.1996, 1.1968 and 1.1996. The median half-life comes out at 7.1, a mild downward bias.
  Power is 99.7% for EG and 94% for Johansen.
- **Execution style:** market orders give a Sharpe of 2.71 after costs. Passive limits give −1.81:
  adverse selection and 147 legged fills cost about 46 bps against roughly 4 bps.
- **Misspecification:** overestimating reversion speed 4× costs 0.73 Sharpe; underestimating it 4×
  costs 0.04. Short health windows (W = 60/120) don't have the power to keep pairs out of quarantine,
  so Sharpe drops to 0.85 and 1.51.
- **Telemetry:** the Kalman innovation ACF(1) is about 0.74 because the planted spreads are OU
  processes, which the random-walk-coefficient model treats as white noise. This is exactly the kind
  of misspecification the dashboard is there to surface.

## Limitations / not included

- No Fama-French data is bundled. `factor_attribution` accepts any user-supplied factor matrix, and
  `panel_factors` computes market and short-term-reversal factors from the panel itself.
- ADF and Engle-Granger give MacKinnon critical values, not p-values. The Johansen tables cover up to
  5 assets with a restricted constant and 6 with an unrestricted one.
- Bars only. There's no tick-level order-book replay; liquidity checks use bar volume and top-of-book
  depth.
- `LoopVectorization.jl` isn't used: the kernels are small hand-written `@simd` loops.
