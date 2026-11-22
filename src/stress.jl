# ─────────────────────────────────────────────────────────────────────────────
# Parameter-misspecification stress tests.
# ─────────────────────────────────────────────────────────────────────────────

"""
    sensitivity_sweep(panel, base=BacktestConfig(); q_scales, r_scales, windows, hl_scales,
                      threaded=true) -> Vector{NamedTuple}

One-at-a-time perturbation sweep around `base`:

- `q_scales`  multiply the Kalman state-noise `Q`
- `r_scales`  multiply the observation noise `R`
- `windows`   rolling health-check lookbacks `W`
- `hl_scales` multiply every estimated half-life used for time stops and the overlay's
  reversion speeds (< 1 ⇒ *over*-estimating reversion speed)

Each run is an independent backtest, so runs are executed in parallel threads (formation
threading is disabled inside to avoid nested oversubscription).
"""
function sensitivity_sweep(panel::PricePanel, base::BacktestConfig = BacktestConfig();
                           q_scales = (0.1, 1.0, 10.0, 100.0), r_scales = (0.25, 1.0, 4.0),
                           windows = (60, 120, 250), hl_scales = (0.25, 0.5, 1.0, 2.0, 4.0),
                           threaded::Bool = true)
    base = with(base; threaded_formation = false)
    specs = Tuple{Symbol,Float64,BacktestConfig}[]
    for q in q_scales
        push!(specs, (:q_scale, q, with(base; kalman = with(base.kalman; q_scale = q))))
    end
    for r in r_scales
        push!(specs, (:r_scale, r, with(base; kalman = with(base.kalman; r_scale = r))))
    end
    for w in windows
        w <= base.formation && push!(specs, (:health_window, w, with(base; health_window = w)))
    end
    for h in hl_scales
        push!(specs, (:hl_scale, h, with(base; hl_scale = h)))
    end
    out = Vector{Any}(undef, length(specs))
    foreach_chunk(length(specs); threaded = threaded) do chunk
        for i in chunk
            nm, v, cfg = specs[i]
            res = run_backtest(panel, cfg)
            p = performance(res)
            out[i] = (param = nm, value = v, sharpe = p.sharpe, total_return = p.total_return,
                      max_drawdown = p.max_drawdown, n_fills = res.book.n_fills)
        end
    end
    [o for o in out]
end

"""
    misspecification_asymmetry(sweep) -> NamedTuple

Compare the Sharpe loss from over- vs under-estimating reversion speed by the same factor
(`hl_scale = 1/c` vs `c`). Positive `asymmetry` means overestimation hurts more.
"""
function misspecification_asymmetry(sweep)
    hs = filter(s -> s.param === :hl_scale, sweep)
    base = only(filter(s -> s.value == 1.0, hs)).sharpe
    rows = NamedTuple[]
    for s in hs
        s.value < 1 || continue
        c = 1 / s.value
        m = filter(u -> isapprox(u.value, c), hs)
        isempty(m) && continue
        over = base - s.sharpe
        under = base - only(m).sharpe
        push!(rows, (factor = c, loss_overestimate = over, loss_underestimate = under, asymmetry = over - under))
    end
    (baseline_sharpe = base, rows = rows)
end

"""Pretty-print a sensitivity sweep."""
function print_sweep(io::IO, sweep)
    @printf(io, "%-14s %8s %8s %10s %9s %7s\n", "parameter", "value", "Sharpe", "return %", "maxDD %", "fills")
    for s in sweep
        @printf(io, "%-14s %8.3g %8.2f %10.2f %9.2f %7d\n", s.param, s.value, s.sharpe,
                100s.total_return, 100s.max_drawdown, s.n_fills)
    end
end
print_sweep(sweep) = print_sweep(stdout, sweep)
