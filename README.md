# StatArb.jl

[![CI](https://github.com/rainerrodrigues/StatArb.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/rainerrodrigues/StatArb.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://rainerrodrigues.github.io/StatArb.jl/dev/)
[![codecov](https://codecov.io/gh/rainerrodrigues/StatArb.jl/graph/badge.svg)](https://codecov.io/gh/rainerrodrigues/StatArb.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A statistical-arbitrage backtester that keeps **econometric health monitoring** (rolling
Engle-Granger / Johansen cointegration, OU half-life, Kalman state-space hedge ratios)
separate from **multi-leg execution simulation** (latency, spread, square-root impact,
liquidity caps, legging risk, borrow and financing costs). Every inner-loop kernel runs on
pre-allocated workspaces with **zero heap allocations**, and JET reports no runtime dispatch in any of them.

```julia
using StatArb, Random
panel, truth = synthetic_universe(Xoshiro(1); n_pairs = 10, n_noise = 10, T = 2000)
res = run_backtest(panel, BacktestConfig(max_baskets = 3))   # pairs + 3-asset baskets
report(res)                                   # performance + execution + model health
sw  = sensitivity_sweep(panel)                # Q / R / window / half-life misspecification
export_results("out", res)                    # CSV logs (or format = :arrow after `using Arrow`)
# using CairoMakie; plot_backtest(res)        # dashboard via the Makie extension
```

Install from the General registry with `using Pkg; Pkg.add("StatArb")`. Run with threads for
parallel formation scans and sweeps: `julia -t auto`.

📖 **[Documentation](https://rainerrodrigues.github.io/StatArb.jl/dev/)**: tutorials, guides for
each layer, and the full API.

## Continuous integration

| Workflow | What it checks |
|---|---|
| `ci.yml` → `test` | the full test suite (Aqua included) on Julia 1.10, 1.11 and latest on Linux, plus latest on macOS and Windows; coverage goes to Codecov |
| `ci.yml` → `validation` | Monte Carlo checks of test size and power, parameter-recovery bias, VECM rank recovery, false selections on pure noise, p-value calibration, Kalman innovation whiteness and spread tracking, and multi-seed lookahead perturbation (`benchmark/validation.jl`); fails the build if any check leaves its tolerance |
| `ci.yml` → `benchmark` | hot-loop latency, allocation counts and backtest throughput, posted to the job summary |
| `ci.yml` → `plotting` | renders every plot of the Makie extension with CairoMakie |
| `ci.yml` → `jet` | JET static analysis of the whole package (pinned to Julia 1.12) |
| `docs.yml` | builds the Documenter site, running every example, and deploys it to `gh-pages` |
| `ci.yml` → `lint` | the generic pre-commit checks (whitespace, line endings, YAML/TOML syntax, merge markers, large files) |
| `TagBot.yml`, `CompatHelper.yml` | release tagging after registration, and `[compat]` bumps for Julia dependencies |
| `dependabot.yml`, `dependabot-automerge.yml` | weekly grouped GitHub Actions updates; minor and patch bumps merge automatically once CI and Documentation pass |
| `codeql.yml` | CodeQL security analysis of the workflows themselves |

## Layout

| File | Responsibility |
|---|---|
| `src/data.jl` | `PricePanel` (contiguous `n_assets × T` matrices, tradable mask, corporate actions), `MirroredRing` buffers, `Clock` + `LookaheadError` |
| `src/linalg.jl` | allocation-free Cholesky / triangular solves / OLS workspace |
| `src/adf.jl` | ADF (fixed & AIC lag), Engle-Granger two-step, MacKinnon (2010) critical values, OU half-life |
| `src/johansen.jl` | rolling Johansen VECM: trace & max-eigenvalue statistics, rank, cointegrating vector |
| `src/kalman.jl` | `StaticArrays` Kalman filters: random-walk `[α, β]` and OU-augmented `[α, β, s]` with moment-based noise estimation; uncertainty throttle |
| `src/execution.jl` | order lifecycle, market/limit fills, impact, participation/depth caps, legging, financing |
| `src/screening.jl` | distance pre-screen, threaded pair/basket scans, health-gate state machine |
| `src/selection.jl` | exact null distributions, Monte Carlo p-values, Benjamini-Hochberg FDR control |
| `src/validate.jl` | configuration validation |
| `src/portfolio.jl` | caps, cross-spread mean-variance overlay |
| `src/strategy.jl`, `src/rules.jl` | pluggable signal rules (`ZScoreBands`, `CostAwareBands`) and allocators |
| `src/backtest.jl` | walk-forward event loop over pairs and baskets, in stages |
| `src/metrics.jl` | Sharpe/Sortino/MDD/Calmar, factor attribution, health telemetry, text dashboard |
| `src/synthetic.jl` | OU pairs, VECM simulator, synthetic universes, parameter-recovery study |
| `src/stress.jl` | parameter-misspecification sweeps |
| `src/export.jl` | CSV / Arrow export of results; plotting API |
| `ext/StatArbArrowExt.jl` | memory-mapped Arrow IO (`using Arrow` activates it) |
| `ext/StatArbMakieExt.jl` | `plot_backtest`, `plot_health`, `plot_sweep` (any Makie backend activates it) |

## Design notes

**Temporal guardrails.** The event loop has one monotonic clock. At bar `t` it does the following, in order:
1. execute orders decided at `≤ t − latency` on bar-`t` quotes;
2. accrue financing;
3. (re)run formation on `[t−F+1, t]`;
4. update the filters and health checks with bar `t`;
5. decide orders, which become eligible from `t + latency`. `latency ≥ 1` is enforced.

Signals at bar `t` use data up to and including `t`, and their orders fill no earlier than
`t + latency`. The test suite checks this with a perturbation test: it scrambles every price after bar
800 and asserts the equity curve up to bar 800 doesn't change at all.

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

**Kalman filter.** Two models, both on `SVector`/`SMatrix` state with scalar observations, so
there's no matrix inversion and no allocation.
- **`:ou` (default):** the state is `[α, β, s]`, and the spread `s` is an AR(1)/OU state:
  `s_t = φ s_{t−1} + η_t`.
  - `φ`, `σ²_η` and the quote noise `R` are estimated from the residuals' first three
    autocovariances (AR(1) plus noise), so bid-ask bounce isn't mistaken for spread dynamics.
  - Every health check re-fits them on the rolling window.
  - The signal is the filtered spread `ŝ_{t|t}/σ_s`, and the innovations are kept as a whiteness
    check.
  - Update time is about 71 ns.
- **`:rw`:** the classic state `[α, β]` follows a random walk, and the signal is the innovation
  Z-score `e/√S`. Update time is about 29 ns.

Both use a unit-free version of Chan's δ for the α/β noise: `Q ∝ δ/(1−δ)·σ²·diag(1, 1/x̄²)`. The
textbook `δ/(1−δ)·I` depends on price units, and on log prices it let the intercept absorb the
spread. Size is throttled by `min(1, P_ref / P_ββ)`.

On noisy synthetic quotes, the OU model trades 30–58% less, at the same or slightly lower
Sharpe. The random-walk model partly profits from trading the quote noise itself, which this
simulator makes easier to capture than real markets do.

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

**Pair selection.** Formation controls the false discovery rate across candidates
(Benjamini-Hochberg at `selection_fdr = 0.05`). The p-values are exact Monte Carlo p-values for the
statistics formation actually computes (two-orientation Engle-Granger and the Johansen trace),
cached per window length, and both tests must reject.

**Data hygiene and risk.**
- **Gaps:** they raise an error, or with `missing_policy = :ffill` they're forward-filled as
  non-tradable bars.
- **Halts and zero-volume bars:** no fills and no new entries.
- **Corporate actions:** splits and dividends adjust positions, cash and resting orders on the
  ex-date, while signals use a causally adjusted `logmid`.
- **Margin:** a breach of `maintenance_margin` triggers a forced liquidation and an entry cooldown.
- **Config checks:** `validate_config` reports every inconsistent setting at once.

**Baskets.** With `max_baskets > 0`, formation grows screened pairs into 3–5 asset baskets,
tests them with Johansen (exact p-values, FDR control), and rejects "pairs in disguise"
(a leg with negligible weight). Baskets trade with fixed Johansen weights and a two-state
spread filter. On synthetic universes with planted baskets, about 90% of basket
selections are planted ones, and adding baskets raised Sharpe in all 6 test seeds (mean
1.78 → 2.30).

**Pluggable strategy.** `run_backtest` is a sequence of stages, and the signal rule and
allocator are config fields (`signal_rule`, `allocator`). Subtype `AbstractSignalRule` or
`AbstractAllocator` and implement one method to plug in your own; the backtester
specialises on the concrete types. `CostAwareBands` only enters when the expected
reversion pays for the estimated round trip. It never binds when costs are small; with
30 bps spreads on low-volatility spreads, it turned a −1.6% excess return into +0.1%
while trading 93% less.

**Exact p-values.** `adf_pvalue`, `eg_pvalue` and `johansen_pvalue` give finite-sample
Monte Carlo p-values for the statistics as computed, cached per configuration.

**Health gate.** A pair is moved to *quarantine* (close-only) when the rolling Johansen rank drops to 0
(`gate_on = :eg` or `:either` are also available) or when its half-life leaves `hl_bounds`. It
recovers after `recover_checks` consecutive passing checks.

**Metrics.** Sharpe, Sortino and alpha are computed on returns **in excess of the cash rate**.
Otherwise the interest earned on a market-neutral book's idle cash shows up as alpha (on the demo, the
raw Sharpe of 10.9 drops to 2.9).

## Benchmarks (`benchmark/hotloops.jl`, single thread)

| Operation | Time | Allocations |
|---|---|---|
| Kalman update, OU model (3 states) | ~71 ns | 0 |
| Kalman update, random-walk model (2 states) | ~29 ns | 0 |
| Basket spread filter update | ~47 ns | 0 |
| Ring push + window view | ~3 ns | 0 |
| Engle-Granger, W=250 | ~2.2 µs | 0 |
| Johansen k=2, W=250 | ~10 µs | 0 |
| Johansen k=3, W=250 | ~14 µs | 0 |
| Full walk-forward, 30 assets × 2000 bars | ~27 ms | — |
| Full walk-forward, 270 assets × 2500 bars, pairs + baskets | ~190 ms | 12.7 MiB total |
| Distance screen, 1,500 assets (1.1M pairs) | ~18 ms | 8.7 MiB |

## What the demo shows (`examples/walkforward_demo.jl`)

- **Parameter recovery** (300 simulations, β = 1.2, hl = 8): the EG, Johansen and Kalman β estimates
  average 1.1996, 1.1968 and 1.1996. The median half-life comes out at 7.1, a mild downward bias.
  Power is 99.7% for EG and 94% for Johansen.
- **Execution style:** market orders give a Sharpe of 2.29 after costs. Passive limits give −1.37:
  adverse selection and 102 legged fills cost about 48 bps against roughly 4 bps.
- **Misspecification:** overestimating reversion speed 4× costs 0.75 Sharpe; underestimating it 4×
  costs nothing (−0.01). Short health windows (W = 60/120) don't have the power to keep pairs out of
  quarantine, so Sharpe drops to 0.43 and 1.83.
- **API change:** `with(cfg; …)` is now `reconfigure(cfg; …)` (`with` still works, with a
  deprecation warning).
- **Pair selection:** with false-discovery control, a universe of pure random walks yields 0.02
  spurious pairs per formation, against 2.6 with plain per-test selection (see the CI validation job).
- **Kalman model:** with the default OU-augmented filter, the innovation ACF(1) is 0.04, so the
  innovations are white and the model fits. With `KalmanConfig(model = :rw)` it is about 0.7: the
  random-walk model treats the OU spread as white noise. That's exactly the misspecification the
  dashboard is designed to surface.

## Limitations / not included

- No Fama-French data is bundled. `factor_attribution` accepts any user-supplied factor matrix, and
  `panel_factors` computes market and short-term-reversal factors from the panel itself.
- Johansen critical-value tables cover up to 5 assets with a restricted constant and 6 with an
  unrestricted one. Exact p-values work for any supported `k`, but cost about 0.1 s to simulate once
  per window length.
- Basket weights are fixed between formations; only pairs have a time-varying hedge ratio.
- Bars only. There's no tick-level order-book replay; liquidity checks use bar volume and top-of-book
  depth.
- `LoopVectorization.jl` isn't used: the kernels are small hand-written `@simd` loops.
