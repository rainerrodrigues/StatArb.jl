# Cointegration diagnostics

## Engle-Granger

[`engle_granger!`](@ref) works in two steps:
1. it regresses `y = α + β x + ε` by OLS;
2. it runs an ADF test on the residuals, with no deterministic terms, and compares the
   statistic with the MacKinnon (2010) response-surface critical values for two variables
   ([`mackinnon_crit`](@ref)).

During formation, [`test_pairs`](@ref) runs it in both orientations and keeps the more
negative statistic.

The ADF regression is
`Δx_t = [c] + γ x_{t−1} + Σ φ_i Δx_{t−i} + u_t`. [`adf!`](@ref) uses a fixed lag order
and [`adf_auto!`](@ref) picks the order by AIC over a common sample.

## Johansen

For a `k × W` window, [`johansen!`](@ref) estimates the VECM

```math
\Delta X_t = \alpha(\beta' X_{t-1} + \rho_0) + \sum_{i=1}^{p-1} \Gamma_i \Delta X_{t-i} + \varepsilon_t
```

It then solves the generalized symmetric eigenproblem
``S_{10} S_{00}^{-1} S_{01} v = \lambda S_{11} v`` with LAPACK `dsygvd`, the routine
behind `LAPACK.sygvd!`. The call goes through `ccall` with buffers owned by the workspace,
because the high-level wrapper allocates them on every call. From the eigenvalues it
computes

```math
\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{k} \ln(1-\lambda_i), \qquad
\lambda_{\max}(r, r+1) = -T \ln(1-\lambda_{r+1})
```

The rank comes from the sequential trace test. The workspace also holds the leading
cointegrating vector (normalised so its first entry is 1) and the half-life of the implied
spread.

All second moments come from a single `BLAS.syrk!` on the stacked data matrix. The
short-run dynamics are concentrated out with in-place Cholesky solves, so no residual
matrices are ever formed.

### Deterministic terms

| `det` | Model | Critical values |
|---|---|---|
| `:restricted` (default) | constant inside the cointegrating relation, no drift in prices | Osterwald-Lenum (1992), Table 1* |
| `:unrestricted` | data demeaned, so prices may trend | MacKinnon-Haug-Michelis (1999), case 3 |

A price spread has a mean but no trend, so `:restricted` is the right null for pairs. On
drift-free random walks the unrestricted tables reject 12–31% of the time at a nominal 5%.

## Exact p-values

Critical-value tables only give a reject/accept answer at fixed levels. For p-values,
StatArb simulates the null distribution of exactly the statistic you computed (same series
length, lag order and deterministic terms), once per configuration with a fixed seed, and
caches it:

| Function | Statistic |
|---|---|
| [`adf_pvalue`](@ref) | ADF t-statistic ([`adf!`](@ref)), with or without a constant |
| [`eg_pvalue`](@ref) | one-orientation Engle-Granger statistic |
| [`eg_null_distribution`](@ref) | the two-orientation minimum that formation uses |
| [`johansen_pvalue`](@ref) | rank-0 trace statistic, `k` assets, restricted constant |

The p-values are `(#{draws at least as extreme} + 1) / (n + 1)`, which is valid for any
number of draws. The simulated 5% point of the Engle-Granger statistic matches MacKinnon's
response surface, and the CI validation job checks that ADF p-values on random walks fall
below 0.05 about 5% of the time.

## Half-life

[`ou_half_life`](@ref) fits `Δs_t = a + b s_{t−1}` and returns
`−ln 2 / ln(1 + b)`. If the spread doesn't mean-revert (`1 + b ∉ (0, 1)`), it returns `Inf`.

## The health gate

Every `health_every` bars, each live pair is re-tested on its rolling window. If the
Johansen rank drops to 0, or the half-life leaves `hl_bounds`, the pair moves to
[`QUARANTINE`](@ref): it can close positions but not open new ones. It returns to
[`ACTIVE`](@ref) after `recover_checks` passing checks in a row. Setting `gate_on = :eg`
gates on Engle-Granger instead, and `:either` keeps a pair active while at least one of the
two tests passes. See [`HealthGateConfig`](@ref).

!!! note "Window length and power"
    Short windows lack the power to detect slowly reverting spreads. With a half-life of 10
    bars, a 120-bar window often fails to reject even when the pair really is cointegrated,
    so healthy pairs keep getting quarantined. The default `health_window = 250`, and the
    sensitivity sweep shows what shorter windows cost.

## Selecting pairs with false-discovery control

Formation tests up to `screen_top` candidates at once. If each one is tested at 5%, about
5% of the candidates that *aren't* cointegrated get through anyway. On a universe of pure
random walks, that selects 2.6 spurious pairs per formation, and 95% of formations pick at
least one.

By default (`selection_fdr = 0.05`), selection controls the false discovery rate instead:

1. Each candidate gets an exact finite-sample p-value from a simulated null distribution
   of the *same* statistic formation computes: the more negative of the two Engle-Granger
   orientations ([`eg_null_distribution`](@ref)), and the k = 2 Johansen trace with a
   restricted constant ([`johansen_null_distribution`](@ref)). Each distribution is
   simulated once per window length with a fixed seed (about 0.1 s) and cached.
2. A pair's p-value is `max(p_EG, p_Johansen)`, so it only counts as significant when
   *both* tests reject ([`pair_pvalues`](@ref)). Requiring both is conservative: on
   unscreened null pairs, P(p < 0.05) is about 1.8%.
3. [`benjamini_hochberg`](@ref) keeps the expected share of false discoveries among the
   selected pairs at or below `q`.

On the same noise universe, this selects 0.04 pairs per formation. Set
`selection_fdr = 0` to go back to per-test selection at `coint_level`.

!!! note "Screening bias"
    The distance screen keeps the pairs whose paths look most alike, so the candidates'
    statistics are more extreme than an unselected null. The p-values ignore this, so the
    *nominal* guarantee doesn't strictly hold. The CI validation job measures what actually
    happens on pure noise and fails the build if the false-selection rate rises above
    0.15 pairs per formation.

### Baskets

Basket candidates come from [`basket_candidates`](@ref) and are tested with a `k`-asset
Johansen test. They get their own Benjamini-Hochberg family, using
[`johansen_null_distribution`](@ref) with `k = basket_size`. Two extra rules apply:
- the half-life of the Johansen spread must be within bounds;
- no leg's weight may fall below `min_basket_weight × max|w|`, which rules out a pair plus
  an irrelevant asset.

On universes with planted baskets, about 90% of basket selections are planted baskets (CI
validation job).

## Pre-screening

Testing every pair takes ``N(N-1)/2`` tests. [`distance_screen`](@ref) first z-normalises
log prices over the formation window and keeps the `screen_top` pairs with the smallest
sum of squared differences, optionally only within a sector. Assets are only eligible if
at least `min_tradable_frac` of the formation window's bars are real, tradable bars and
the asset is tradable on the formation bar.
