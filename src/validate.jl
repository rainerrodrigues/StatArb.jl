# ─────────────────────────────────────────────────────────────────────────────
# Configuration validation: fail fast, with every problem listed at once.
# ─────────────────────────────────────────────────────────────────────────────

_need!(errs, cond::Bool, msg::AbstractString) = (cond || push!(errs, msg); errs)

function _check!(errs, k::KalmanConfig)
    _need!(errs, k.model in (:rw, :ou), "kalman.model must be :rw or :ou, got :$(k.model)")
    _need!(errs, 0 < k.delta < 1, "kalman.delta must be in (0, 1), got $(k.delta)")
    _need!(errs, k.obs_noise_frac === nothing || 0 <= k.obs_noise_frac < 1,
           "kalman.obs_noise_frac must be nothing (estimate) or in [0, 1), got $(k.obs_noise_frac)")
    _need!(errs, k.q_scale > 0, "kalman.q_scale must be positive, got $(k.q_scale)")
    _need!(errs, k.r_scale > 0, "kalman.r_scale must be positive, got $(k.r_scale)")
end

function _check!(errs, e::ExecConfig)
    _need!(errs, e.latency >= 1, "exec.latency must be ≥ 1 bar (a fill on the decision bar is lookahead), got $(e.latency)")
    _need!(errs, e.style in (:market, :passive), "exec.style must be :market or :passive, got :$(e.style)")
    _need!(errs, e.limit_wait >= 1, "exec.limit_wait must be ≥ 1, got $(e.limit_wait)")
    _need!(errs, 0 < e.max_participation <= 1, "exec.max_participation must be in (0, 1], got $(e.max_participation)")
    _need!(errs, e.depth_levels > 0, "exec.depth_levels must be positive, got $(e.depth_levels)")
    _need!(errs, e.impact_coef >= 0, "exec.impact_coef must be ≥ 0, got $(e.impact_coef)")
    _need!(errs, 0 <= e.max_impact < 1, "exec.max_impact must be in [0, 1), got $(e.max_impact)")
    _need!(errs, e.commission_bps >= 0 && e.commission_per_share >= 0, "exec commissions must be ≥ 0")
    _need!(errs, e.borrow_rate >= 0 && all(>=(0), e.borrow_rates), "exec borrow rates must be ≥ 0")
    _need!(errs, e.margin_rate >= 0, "exec.margin_rate must be ≥ 0, got $(e.margin_rate)")
    _need!(errs, isfinite(e.cash_rate), "exec.cash_rate must be finite")
    _need!(errs, e.vol_halflife > 0, "exec.vol_halflife must be positive, got $(e.vol_halflife)")
end

function _check!(errs, r::RiskConfig)
    _need!(errs, r.pair_cap > 0, "risk.pair_cap must be positive, got $(r.pair_cap)")
    _need!(errs, r.max_gross > 0, "risk.max_gross must be positive, got $(r.max_gross)")
    _need!(errs, r.pair_cap <= r.max_gross, "risk.pair_cap ($(r.pair_cap)) cannot exceed risk.max_gross ($(r.max_gross))")
    _need!(errs, r.cov_window >= 10, "risk.cov_window must be ≥ 10 bars, got $(r.cov_window)")
    _need!(errs, 0 <= r.shrinkage <= 1, "risk.shrinkage must be in [0, 1], got $(r.shrinkage)")
    _need!(errs, r.min_trade_frac >= 0, "risk.min_trade_frac must be ≥ 0, got $(r.min_trade_frac)")
    _need!(errs, 0 <= r.maintenance_margin < 1, "risk.maintenance_margin must be in [0, 1), got $(r.maintenance_margin)")
    _need!(errs, r.maintenance_margin * r.max_gross < 1,
           "risk.max_gross × maintenance_margin must be < 1, otherwise a fully invested book starts in a margin call")
    _need!(errs, r.liquidation_cooldown >= 0, "risk.liquidation_cooldown must be ≥ 0")
end

function _check!(errs, c::BacktestConfig)
    _need!(errs, c.formation >= 30, "formation must be ≥ 30 bars, got $(c.formation)")
    _need!(errs, c.trading >= 1, "trading must be ≥ 1 bar, got $(c.trading)")
    _need!(errs, c.health_window <= c.formation, "health_window ($(c.health_window)) must be ≤ formation ($(c.formation))")
    _need!(errs, c.health_window > 11 + 3 * c.johansen_lags,
           "health_window ($(c.health_window)) is too short for johansen_lags = $(c.johansen_lags)")
    _need!(errs, c.health_every >= 1, "health_every must be ≥ 1, got $(c.health_every)")
    _need!(errs, c.johansen_lags >= 0 && c.adf_lags >= 0, "lag orders must be ≥ 0")
    _need!(errs, c.coint_level in (0.01, 0.05, 0.10), "coint_level must be 0.01, 0.05 or 0.10, got $(c.coint_level)")
    _need!(errs, 0 <= c.selection_fdr <= 1, "selection_fdr must be in [0, 1] (0 disables FDR control), got $(c.selection_fdr)")
    _need!(errs, c.null_sims >= 999, "null_sims must be ≥ 999 for usable p-values, got $(c.null_sims)")
    _need!(errs, c.selection_fdr == 0 || 1 / (c.null_sims + 1) < c.selection_fdr / max(c.screen_top, 1),
           "null_sims ($(c.null_sims)) is too small to resolve the BH threshold selection_fdr / screen_top")
    _need!(errs, 0 < c.min_tradable_frac <= 1, "min_tradable_frac must be in (0, 1], got $(c.min_tradable_frac)")
    _need!(errs, c.max_pairs >= 0 && c.screen_top >= 1, "max_pairs must be ≥ 0 and screen_top ≥ 1")
    _need!(errs, c.max_pairs + c.max_baskets >= 1, "max_pairs + max_baskets must be ≥ 1")
    _need!(errs, c.max_baskets >= 0, "max_baskets must be ≥ 0, got $(c.max_baskets)")
    _need!(errs, 3 <= c.basket_size <= JOHANSEN_MAX_K, "basket_size must be in 3:$JOHANSEN_MAX_K, got $(c.basket_size)")
    _need!(errs, c.basket_screen_top >= 1, "basket_screen_top must be ≥ 1")
    _need!(errs, 0 <= c.min_basket_weight < 1, "min_basket_weight must be in [0, 1), got $(c.min_basket_weight)")
    _need!(errs, c.health_window > 11 + (c.basket_size + 1) * c.johansen_lags + c.basket_size || c.max_baskets == 0,
           "health_window ($(c.health_window)) is too short for $(c.basket_size)-asset baskets")
    lo, hi = c.hl_bounds
    _need!(errs, 0 < lo < hi, "hl_bounds must satisfy 0 < lo < hi, got $(c.hl_bounds)")
    _need!(errs, hi < c.health_window / 2,
           "hl_bounds upper limit ($hi) must be below health_window / 2 ($(c.health_window / 2)): longer half-lives can't be estimated reliably")
    _need!(errs, c.gate_on in (:johansen, :eg, :either), "gate_on must be :johansen, :eg or :either, got :$(c.gate_on)")
    _need!(errs, c.recover_checks >= 1, "recover_checks must be ≥ 1")
    _need!(errs, 0 <= c.z_exit < c.z_entry < c.z_stop,
           "need 0 ≤ z_exit < z_entry < z_stop, got $(c.z_exit), $(c.z_entry), $(c.z_stop)")
    _need!(errs, c.max_hold_halflives > 0, "max_hold_halflives must be positive")
    _need!(errs, c.hl_scale > 0, "hl_scale must be positive")
    _need!(errs, c.innov_window >= 3, "innov_window must be ≥ 3")
    _need!(errs, c.initial_capital > 0, "initial_capital must be positive")
    _need!(errs, c.risk.cov_window < c.formation, "risk.cov_window ($(c.risk.cov_window)) must be < formation ($(c.formation))")
    c.signal_rule isa CostAwareBands &&
        _need!(errs, c.signal_rule.cost_multiple >= 0, "CostAwareBands.cost_multiple must be ≥ 0")
    _check!(errs, c.kalman); _check!(errs, c.exec); _check!(errs, c.risk)
end

"""
    validate_config(cfg)
    validate_config(cfg, panel)

Check a [`BacktestConfig`](@ref) (including its nested Kalman, execution and risk
configs), and optionally its fit with a panel. Throws one `ArgumentError` listing every
problem; returns `nothing` when the configuration is consistent.
[`run_backtest`](@ref) calls this before doing any work.
"""
function validate_config(cfg::BacktestConfig)
    errs = String[]
    _check!(errs, cfg)
    isempty(errs) || throw(ArgumentError("invalid BacktestConfig:\n  • " * join(errs, "\n  • ")))
    nothing
end

function validate_config(cfg::BacktestConfig, p::PricePanel)
    errs = String[]
    _check!(errs, cfg)
    _need!(errs, cfg.formation + 2 <= ntimes(p),
           "panel has $(ntimes(p)) bars, fewer than one formation window + 2 ($(cfg.formation + 2))")
    br = cfg.exec.borrow_rates
    _need!(errs, isempty(br) || length(br) == nassets(p),
           "exec.borrow_rates has $(length(br)) entries for $(nassets(p)) assets")
    isempty(errs) || throw(ArgumentError("invalid BacktestConfig for this panel:\n  • " * join(errs, "\n  • ")))
    nothing
end
