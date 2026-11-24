# Walk-forward demo on a synthetic universe with planted cointegrated pairs.
#   julia --project=. -t auto examples/walkforward_demo.jl
using StatArb, Random

rng = Xoshiro(2026)
panel, truth = synthetic_universe(rng; n_pairs = 10, n_noise = 10, T = 2000)
println(panel)

# 1. Ground-truth validation before trusting the engine
st = recovery_study(Xoshiro(1); nsim = 300, T = 500, beta = 1.2, half_life = 8.0)
println("\nParameter recovery (true β = 1.2, half-life = 8):")
println("  OLS/EG β      ", round(st.beta_eg.mean; digits = 4), " ± ", round(st.beta_eg.sd; digits = 4))
println("  Johansen β    ", round(st.beta_johansen.mean; digits = 4))
println("  Kalman β      ", round(st.beta_kalman.mean; digits = 4))
println("  half-life     ", round(st.half_life.median; digits = 2), " (median)")
println("  power EG / Johansen  ", st.power_eg, " / ", st.power_johansen)

# 2. Backtest: aggressive vs passive execution
cfg = BacktestConfig()
res = run_backtest(panel, cfg)
F, names = panel_factors(panel, res.t[2:end])
println("\n=== Aggressive (market-order) execution ===")
report(res; factors = F, factor_names = names)

resp = run_backtest(panel, reconfigure(cfg; exec = ExecConfig(style = :passive)))
println("\n=== Passive limit execution (with legging risk) ===")
report(resp)

# 3. Misspecification stress test
println("\n=== Sensitivity sweep ===")
sw = sensitivity_sweep(panel, cfg)
print_sweep(sw)
asym = misspecification_asymmetry(sw)
println("\nReversion-speed misspecification (baseline Sharpe ", round(asym.baseline_sharpe; digits = 2), "):")
for r in asym.rows
    println("  ×", r.factor, ": loss if speed overestimated ", round(r.loss_overestimate; digits = 2),
            ", underestimated ", round(r.loss_underestimate; digits = 2))
end
