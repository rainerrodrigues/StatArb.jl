# Getting started

This tutorial runs a complete walk-forward backtest on a synthetic universe with planted
cointegrated pairs, so you can see every part of the engine without real market data.

## A synthetic universe

[`synthetic_universe`](@ref) plants `n_pairs` cointegrated pairs and adds `n_noise`
unrelated random walks. All assets share a weak market factor. `truth` records the hedge
ratio and half-life of every planted pair.

```@example gs
using StatArb, Random
panel, truth = synthetic_universe(Xoshiro(2026); n_pairs = 8, n_noise = 8, T = 1500)
panel
```

```@example gs
truth[1:3]
```

## Running the backtest

[`BacktestConfig`](@ref) holds the walk-forward settings, with the execution, risk and
Kalman settings nested inside it. The defaults are:
- a 252-bar formation window, re-run every 63 bars;
- health checks every 5 bars on a rolling window of 250 bars;
- entry at |z| > 2 and exit at |z| < 0.5;
- market-order execution with one bar of latency.

```@example gs
cfg = BacktestConfig()
res = run_backtest(panel, cfg)
```

## The dashboard

[`report`](@ref) prints three sections:
- trading performance (Sharpe and Sortino are computed on returns *in excess of the cash
  rate*);
- execution costs broken down by source;
- model-health telemetry.

Pass a factor matrix to add a factor attribution. [`panel_factors`](@ref) computes a market
factor and a short-term-reversal factor from the panel itself.

```@example gs
F, names = panel_factors(panel, res.t[2:end])
report(res; factors = F, factor_names = names)
```

## Did it find the right pairs?

```@example gs
planted = Set(Set((tr.y, tr.x)) for tr in truth)
selected = [Set(p) for f in res.formations for p in f.selected]
count(in(planted), selected) / length(selected)
```

## Passive execution

Switching to passive limit orders brings in adverse selection and legging risk. On bar data,
a limit order fills only when the price trades through it, which is exactly when the price
has moved against you.

```@example gs
resp = run_backtest(panel, with(cfg; exec = ExecConfig(style = :passive)))
cost_breakdown(resp)
```

## Telemetry tables

`res.bars`, `res.health` and `res.fills` are `StructVector`s, so each column is a plain
vector and any Tables.jl consumer (DataFrames, CSV, Arrow) can read them directly.

```@example gs
h = res.health
(n_checks = length(h), mean_trace = sum(h.trace_r0) / length(h), quarantined = count(==(QUARANTINE), h.state))
```
