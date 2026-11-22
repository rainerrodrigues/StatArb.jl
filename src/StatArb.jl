"""
    StatArb

Statistical-arbitrage backtester that separates *econometric health monitoring* (rolling
Engle-Granger / Johansen cointegration, OU half-life, Kalman state-space hedge ratios)
from *multi-leg execution simulation* (latency, spread, square-root impact, liquidity
caps, legging risk, borrow and financing costs). Hot loops (ring buffers, ADF, Johansen,
Kalman) run on pre-allocated workspaces with zero heap allocations.
"""
module StatArb

using Dates
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Statistics
using StructArrays

include("data.jl")
include("linalg.jl")
include("adf.jl")
include("johansen.jl")
include("kalman.jl")
include("execution.jl")
include("screening.jl")
include("selection.jl")
include("portfolio.jl")
include("backtest.jl")
include("validate.jl")
include("metrics.jl")
include("synthetic.jl")
include("stress.jl")

"""
    load_panel_arrow(path; bars_per_year=252, spread_bps=5, depth_frac=0.01, missing_policy=:error) -> PricePanel

Memory-mapped Arrow reader for long-format bar data (`timestamp, symbol, mid` and
optionally `spread, volume, depth, sector`). Available after `using Arrow` (package extension).
"""
function load_panel_arrow end

"""
    save_panel_arrow(path, panel)

Write a panel in the long format read by [`load_panel_arrow`](@ref). Requires `using Arrow`.
"""
function save_panel_arrow end

# data
export PricePanel, CorporateAction, nassets, ntimes, asset_bars, Clock, tick!, assert_past, history, LookaheadError,
       MirroredRing, push_column!, window, filled_window, isfull, latest, reset!
# econometrics
export OLSWorkspace, ols!, ols_line, ADFResult, adf!, adf_auto!, mackinnon_crit, ou_half_life,
       EGWorkspace, EGResult, engle_granger!, iscointegrated,
       JohansenWorkspace, JohansenSummary, JohansenResult, johansen!, johansen,
       johansen_trace_cv, johansen_maxeig_cv
# kalman
export KalmanConfig, KalmanHedge, kalman_step, kalman_update!, calibrate_kalman, kalman_filter,
       hedge_ratio, intercept, size_multiplier, autocor_lag, ljung_box,
       OUKalmanHedge, ou_kalman_step, calibrate_ou_kalman, calibrate_filter, innovation_z,
       filtered_spread, model_half_life, ar1_fit, ou_noise_fit, recalibrate!
# execution
export ExecConfig, ExecutionSimulator, Book, Order, FillRecord, OrderKind, MARKET, LIMIT,
       submit_pair!, process_orders!, accrue_financing!, apply_corporate_actions!, update_vol!, pending_qty, cancel_pair!,
       equity, gross_exposure
# screening / health / parallel
export distance_screen, test_pairs, PairTest, scan_johansen, rolling_johansen, with_single_threaded_blas,
       HealthState, ACTIVE, QUARANTINE, RETIRED, HealthGateConfig, healthy, gate_transition,
       NullDistribution, null_pvalue, eg_null_distribution, johansen_null_distribution,
       benjamini_hochberg, pair_pvalues
# portfolio / backtest
export RiskConfig, OverlayWorkspace, overlay_weights!, leg_shares,
       BacktestConfig, BacktestResult, FormationRecord, run_backtest, with, validate_config
# metrics
export performance, factor_attribution, panel_factors, health_telemetry, cost_breakdown, report, sparkline
# synthetic / stress
export simulate_ou_pair, simulate_vecm, synthetic_universe, recovery_study,
       sensitivity_sweep, misspecification_asymmetry, print_sweep
# io
export load_panel_arrow, save_panel_arrow

end # module
