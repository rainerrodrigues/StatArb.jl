# Baskets and custom rules

## Trading baskets

Pairs are the special case `k = 2` of a cointegrated *spread*. With `max_baskets > 0`,
each formation also looks for `basket_size`-asset baskets:

1. **Candidates.** [`basket_candidates`](@ref) grows each screened pair into a basket by
   repeatedly adding the eligible asset (from the same sector) that is closest to the
   current members.
2. **Testing.** Every candidate gets a Johansen test on the formation window, an exact
   Monte Carlo p-value for its trace statistic
   ([`johansen_null_distribution`](@ref) with `k = basket_size`), and Benjamini-Hochberg
   control of the false discovery rate, the same as pairs.
3. **Pairs in disguise.** A basket containing a cointegrated pair plus an unrelated asset
   *is* cointegrated, but the Johansen vector puts almost no weight on the extra leg, which
   would only add trading cost. Baskets where any leg's weight is below
   `min_basket_weight × max|w|` are rejected.
4. **Trading.** The leg weights `w` (the cointegrating vector, `w₁ = 1`) stay fixed until
   the next formation. A [`BasketFilter`](@ref) tracks the spread `w'·log p` as a drifting
   level plus an AR(1) deviation, and supplies the Z-score. Gross dollars are split across
   legs in proportion to `wᵢ / Σ|w|`. Health checks run a `k`-asset Johansen test on the
   rolling window and quarantine the basket when the rank drops to 0 or the half-life
   leaves its bounds.

[`synthetic_universe`](@ref) can plant baskets (`y = α + b₁x₁ + b₂x₂ + s`) in which no
pair is cointegrated on its own:

```@example br
using StatArb, Random, Statistics
panel, truth = synthetic_universe(Xoshiro(5); n_pairs = 4, n_baskets = 4, n_noise = 8, T = 1500)
planted = Set(sort(tr.assets) for tr in truth if length(tr.assets) == 3)

res = run_backtest(panel, BacktestConfig(max_baskets = 4))
sel = [b for f in res.formations for b in f.baskets]
(basket_selections = length(sel), planted_share = mean(in(planted), sel))
```

```@example br
(pairs_only = performance(run_backtest(panel)).sharpe, with_baskets = performance(res).sharpe)
```

## Cost-aware entries

[`CostAwareBands`](@ref) adds a cost hurdle to the usual Z-score bands. A spread opens only
if the expected move back to `z_exit` is worth at least `cost_multiple` round trips
([`round_trip_cost`](@ref): half spreads, commissions and square-root impact at the size
the allocator would trade).

When trading is cheap relative to the spread's volatility, the hurdle never binds and the
results are identical to [`ZScoreBands`](@ref). When it's expensive, the hurdle filters out
trades that can't pay for themselves:

```@example br
hp, _ = synthetic_universe(Xoshiro(84); n_pairs = 10, n_noise = 10, T = 1500,
                           sigma_s = 0.0015, spread_bps = 30.0)
bands = run_backtest(hp)
aware = run_backtest(hp, BacktestConfig(signal_rule = CostAwareBands(cost_multiple = 1.0)))
(fills = (bands.book.n_fills, aware.book.n_fills),
 final_equity = round.((last(bands.equity), last(aware.equity))))
```

## Writing your own rule or allocator

Signal rules and allocators are plain Julia types. Subtype [`AbstractSignalRule`](@ref)
and implement [`update_signal!`](@ref), or subtype [`AbstractAllocator`](@ref) and
implement [`target_weights!`](@ref). `run_backtest` specialises on the concrete types, so
custom components run at full speed.

A rule that only trades spreads whose filter puts the half-life in a target range, and
only on the reversion side of a fresh extreme:

```@example br
struct HalfLifeBands <: AbstractSignalRule
    max_hl::Float64
end

function StatArb.update_signal!(r::HalfLifeBands, sp::SpreadState, ctx::SignalContext)
    cfg = ctx.cfg
    z = zscore(sp)
    if sp.signal == 0
        ctx.can_open && sp.hl <= r.max_hl && abs(z) > cfg.z_entry && (sp.signal = z > 0 ? -1 : 1)
    elseif sp.signal * z > -cfg.z_exit || abs(z) > cfg.z_stop
        sp.signal = 0
    end
end

custom = run_backtest(panel, BacktestConfig(max_baskets = 4, signal_rule = HalfLifeBands(5.0)))
performance(custom).sharpe
```

An allocator that sizes every open spread by the inverse of its half-life:

```@example br
struct SpeedWeighted <: AbstractAllocator end

function StatArb.target_weights!(::SpeedWeighted, w, spreads, ctx)
    speed = [1 / max(sp.hl, 1.0) for sp in spreads]
    w .= [sp.signal for sp in spreads] .* speed ./ maximum(speed)
end

performance(run_backtest(panel, BacktestConfig(max_baskets = 4, allocator = SpeedWeighted()))).sharpe
```

The backtester still applies the uncertainty throttle, the `max_gross` cap, the rebalance
band, the health gate and the margin rules on top of any custom component.
