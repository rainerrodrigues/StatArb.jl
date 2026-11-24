# Execution model

Fills assuming zero spread and passive limit orders make mean reversion look much more
profitable than it is. [`ExecutionSimulator`](@ref) simulates each leg independently,
against the quote, volume and displayed depth of the bar it fills on.

## Market orders

```math
p_{\text{fill}} = m \pm \left(\tfrac{1}{2}\text{spread} + m\,\eta\,\sigma_{\text{bar}}\sqrt{q / V}\right)
```

`σ_bar` is an EWMA of each asset's bar returns, `V` is the bar volume and `η` is
`impact_coef`.

## Liquidity checks

Each bar's fill is capped at `min(max_participation · V, depth_levels · depth)`. Anything
left over carries to the next bar as a partial fill. If a bar has no volume, the order
waits.

## Passive orders and legging risk

With `style = :passive`, both legs rest as limit orders at the touch quoted on the decision
bar. A limit order fills only on a strict trade-through: for a buy, the next bid has to be
below the limit price. That builds in adverse selection, because passive fills happen when
the price has moved against you.

When one leg fills and its sibling hasn't, the sibling becomes a market order and crosses
the spread `latency` bars later. Its fill is flagged as `legged`. A limit order that hasn't
filled within `limit_wait` bars crosses as well.

In the demo, passive execution pays about 42 bps of effective spread against about 2 bps
for market orders, and its Sharpe ratio turns negative.

### Multi-leg tickets

[`submit_ticket!`](@ref) sends any number of legs as one ticket (baskets use 3–5).
Legging generalises directly: the first passive fill on any leg turns *every* other
resting leg of the ticket into a market order.

### Re-pegging

With `limit_wait > 1`, a resting order normally keeps its decision-bar price. Setting
`reprice = true` re-pegs it to each new touch. Both stay off by default
(`limit_wait = 1`, `reprice = false`). On daily synthetic bars, resting longer, with or
without re-pegging, produced *more* legged fills (74 → 160 per run) and a lower Sharpe
ratio (−0.98 → −1.19) than crossing after one bar.

## Halts and missing data

`panel.tradable` is `false` on bars that were missing in the source data, had zero volume,
or were marked halted. On those bars:

- no order fills; resting and in-flight orders simply wait;
- pairs don't update their filters or ring buffers, because a stale quote carries no new
  information;
- no new positions open, although exits keep their signal and fill once trading resumes.

## Corporate actions

A [`CorporateAction`](@ref) records a split ratio and/or a cash dividend on its ex-date.
Panel prices are raw, so they jump on the ex-date. Before that bar's fills,
[`apply_corporate_actions!`](@ref):
1. credits the dividend to long positions and debits it from short positions;
2. multiplies share positions by the split ratio;
3. rescales resting orders: quantity × split, and limit and reference prices ÷ split.

Signals read `logmid`, which is adjusted forward in time only. That keeps it continuous
through the event without ever depending on actions that haven't happened yet. A test
injects a 2-for-1 split into the raw data and checks that the equity curve doesn't change
(beyond share rounding).

## Margin and forced liquidation

After each bar's fills, the backtester compares equity with `risk.maintenance_margin ×
gross exposure`. If equity falls below that, or to zero or less, it records a margin call,
cancels every resting order and closes every position at market. New entries are then
blocked for `risk.liquidation_cooldown` bars, or permanently if equity is gone. The
dashboard lists every margin call.

Allocations are capped at `risk.max_gross` when they're decided. Rebalance hysteresis
(`min_trade_frac`) skips small adjustments, unless skipping them would leave the book above
the cap. In that case, every open pair is brought to its jointly capped target.

## Financing

[`accrue_financing!`](@ref) charges, for every bar:
- a borrow fee on the market value of each short position (`borrow_rate`, with optional
  per-asset `borrow_rates`);
- interest on cash, *excluding* short-sale proceeds, which are held as collateral. A
  positive balance earns `cash_rate` and a negative one pays `margin_rate`.

## Cost attribution

Every fill is logged in `res.fills` with its spread cost relative to the mid, its impact
cost, its commission, and whether it was passive or legged. [`cost_breakdown`](@ref) adds
these up in currency units and in basis points of traded notional.
