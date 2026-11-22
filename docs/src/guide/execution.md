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

In the demo, passive execution pays about 46 bps of effective spread against about 2 bps
for market orders, and its Sharpe ratio turns negative.

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
