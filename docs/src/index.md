# StatArb.jl

A statistical-arbitrage backtester that keeps **econometric health monitoring** apart from
**multi-leg execution simulation**.

- **Monitoring:** rolling Engle-Granger and Johansen cointegration, Ornstein-Uhlenbeck
  half-lives, and Kalman state-space hedge ratios.
- **Execution:** latency, spreads, square-root impact, liquidity caps, legging risk, and
  borrow and financing costs.

Every inner-loop kernel (ring buffers, ADF, Johansen, Kalman) runs on pre-allocated
workspaces and makes zero heap allocations.

```julia
using StatArb, Random
panel, truth = synthetic_universe(Xoshiro(1); n_pairs = 10, n_noise = 10, T = 2000)
res = run_backtest(panel, BacktestConfig())
report(res)                       # performance + execution costs + model health
```

## Where to start

- [Getting started](@ref) walks through a full backtest on synthetic data.
- [Validating the engine](@ref) checks parameter recovery and stress-tests the strategy
  before you use real data.
- [Baskets and custom rules](@ref) covers trading 3–5 asset baskets, cost-aware entries,
  and writing your own signal rule or allocator.
- [Plotting and exporting results](@ref) shows the Makie dashboard and CSV/Arrow export.
- The guides explain each layer: [Architecture and temporal guardrails](@ref),
  [Cointegration diagnostics](@ref), [Kalman hedge ratios](@ref),
  [Execution model](@ref) and [Performance](@ref).

## Installation

```julia
using Pkg
Pkg.add("StatArb")
```

Start Julia with threads (`julia -t auto`) so formation scans and sensitivity sweeps run in
parallel. To read and write Arrow files, run `using Arrow` to load the extension.
