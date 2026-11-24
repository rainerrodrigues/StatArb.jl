# Plotting extension smoke test (needs a Makie backend; run separately from the main
# suite so that the core tests stay light):
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add("CairoMakie"); include("test/plotting.jl")'
using StatArb, CairoMakie, Random, Test

@testset "Makie plotting extension" begin
    panel, _ = synthetic_universe(Xoshiro(3); n_pairs = 4, n_baskets = 2, n_noise = 6, T = 900)
    res = run_backtest(panel, BacktestConfig(max_baskets = 2))
    dir = mktempdir()
    f1 = plot_backtest(res)
    @test f1 isa Makie.Figure
    save(joinpath(dir, "dashboard.png"), f1)
    @test filesize(joinpath(dir, "dashboard.png")) > 20_000
    for id in unique(res.health.pair)[1:min(3, end)]
        f = plot_health(res, id)
        save(joinpath(dir, "health_$id.png"), f)
        @test filesize(joinpath(dir, "health_$id.png")) > 10_000
    end
    @test_throws ArgumentError plot_health(res, -1)
    sw = sensitivity_sweep(panel; q_scales = (1.0, 10.0), r_scales = (1.0,), windows = (250,), hl_scales = (0.5, 1.0))
    f3 = plot_sweep(sw)
    save(joinpath(dir, "sweep.png"), f3)
    @test filesize(joinpath(dir, "sweep.png")) > 5_000
    println("plots written to ", dir)
end
