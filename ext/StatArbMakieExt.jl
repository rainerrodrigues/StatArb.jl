module StatArbMakieExt

using StatArb
using StatArb: BacktestResult, QUARANTINE, cost_breakdown, johansen_trace_cv
using Makie

const BLUE = Makie.RGBf(0.16, 0.42, 0.70)
const RED = Makie.RGBf(0.80, 0.25, 0.22)
const GREY = Makie.RGBf(0.55, 0.55, 0.55)

function StatArb.plot_backtest(res::BacktestResult; size = (1400, 1000))
    fig = Figure(; size = size)
    t = res.t
    eq = res.equity ./ first(res.equity)
    peak = accumulate(max, eq)
    dd = eq ./ peak .- 1

    ax1 = Axis(fig[1, 1]; title = "Equity (× initial)", ylabel = "equity")
    lines!(ax1, t, eq; color = BLUE)
    ax2 = Axis(fig[2, 1]; title = "Drawdown", ylabel = "%")
    band!(ax2, t, 100 .* dd, zeros(length(t)); color = (RED, 0.35))
    lines!(ax2, t, 100 .* dd; color = RED)
    linkxaxes!(ax1, ax2)

    b = res.bars
    ax3 = Axis(fig[3, 1]; title = "Gross exposure / equity and open spreads", ylabel = "leverage", xlabel = "bar")
    lines!(ax3, b.t, b.gross ./ b.equity; color = BLUE, label = "gross / equity")
    ax3r = Axis(fig[3, 1]; yaxisposition = :right, ylabel = "open spreads")
    hidespines!(ax3r); hidexdecorations!(ax3r)
    stairs!(ax3r, b.t, b.n_open; color = GREY, step = :post)
    linkxaxes!(ax1, ax3, ax3r)
    isempty(res.margin_calls) || vlines!(ax3, res.margin_calls; color = RED, linestyle = :dash)

    h = res.health
    # each spread's trace relative to its own 95% critical value: comparable across pairs
    # and baskets, > 1 means the rolling window rejects "no cointegration"
    ax4 = Axis(fig[1, 2]; title = "Johansen trace / 95% critical value, per spread", ylabel = "ratio")
    for (j, id) in enumerate(unique(h.pair))
        idx = findall(==(id), h.pair)
        cv = johansen_trace_cv(h.k[first(idx)])
        lines!(ax4, h.t[idx], h.trace_r0[idx] ./ cv; color = j, colormap = :tab10, colorrange = (1, 10))
    end
    hlines!(ax4, [1.0]; color = RED, linestyle = :dash)
    ax5 = Axis(fig[2, 2]; title = "Half-life per spread (bars)", ylabel = "bars")
    if !isempty(h)
        ok = isfinite.(h.half_life)
        scatter!(ax5, h.t[ok], h.half_life[ok]; color = h.pair[ok], colormap = :tab10, markersize = 4)
    end
    ax6 = Axis(fig[3, 2]; title = "Kalman innovation autocorrelation, lag 1", ylabel = "ACF(1)", xlabel = "bar")
    if !isempty(h)
        ok = isfinite.(h.innov_acf1)
        scatter!(ax6, h.t[ok], h.innov_acf1[ok]; color = (BLUE, 0.5), markersize = 4)
        hlines!(ax6, [0.0]; color = GREY)
    end

    c = cost_breakdown(res)
    carry = "net interest $(round(Int, c.interest_net)), net dividends $(round(Int, c.dividends_net))"
    ax7 = Axis(fig[4, 1:2]; title = "Trading costs (carry: $carry)", ylabel = "currency",
               xticks = (1:4, ["spread", "impact", "commission", "borrow"]))
    vals = [c.spread, c.impact, c.commission, c.borrow]
    barplot!(ax7, 1:4, vals; color = [v >= 0 ? RED : BLUE for v in vals])
    rowsize!(fig.layout, 4, Relative(0.18))
    Label(fig[0, :], "StatArb backtest — $(first(res.timestamps)) → $(last(res.timestamps))"; fontsize = 18)
    fig
end

function StatArb.plot_health(res::BacktestResult, id::Integer; size = (1000, 800))
    h = res.health
    idx = findall(==(id), h.pair)
    isempty(idx) && throw(ArgumentError("no health checks recorded for spread $id"))
    t = h.t[idx]
    k = h.k[first(idx)]
    fig = Figure(; size = size)
    shade!(ax) = begin
        q = h.state[idx] .== QUARANTINE
        for i in eachindex(t)
            q[i] || continue
            hi = i < length(t) ? t[i+1] : t[i] + 1
            vspan!(ax, t[i], hi; color = (RED, 0.15))
        end
    end
    ax1 = Axis(fig[1, 1]; title = "Spread $id ($(k) assets: $(join(_assets(res, id), ", "))) — Johansen trace",
               ylabel = "trace(r = 0)")
    shade!(ax1)
    lines!(ax1, t, h.trace_r0[idx]; color = BLUE)
    hlines!(ax1, [johansen_trace_cv(k)]; color = RED, linestyle = :dash)
    ax2 = Axis(fig[2, 1]; title = "Half-life (bars)", ylabel = "bars")
    shade!(ax2)
    lines!(ax2, t, h.half_life[idx]; color = BLUE, label = k == 2 ? "Engle-Granger" : "Johansen spread")
    kh = h.kf_half_life[idx]
    any(isfinite, kh) && lines!(ax2, t, kh; color = GREY, label = "filter (φ)")
    axislegend(ax2; position = :lt)
    ax3 = Axis(fig[3, 1]; title = "Innovation ACF(1)", ylabel = "ACF(1)", xlabel = "bar")
    shade!(ax3)
    lines!(ax3, t, h.innov_acf1[idx]; color = BLUE)
    hlines!(ax3, [0.0]; color = GREY)
    linkxaxes!(ax1, ax2, ax3)
    fig
end

_assets(res::BacktestResult, id) = begin
    i = findfirst(==(id), res.health.pair)
    k = res.health.k[i]
    legs = sort(unique(res.fills.asset[res.fills.pair .== id]))
    length(legs) == k ? res.symbols[legs] : res.symbols[[res.health.y[i], res.health.x[i]]]
end

function StatArb.plot_sweep(sweep; size = (1200, 350))
    params = unique(s.param for s in sweep)
    fig = Figure(; size = size)
    for (j, p) in enumerate(params)
        rows = filter(s -> s.param === p, sweep)
        ax = Axis(fig[1, j]; title = string(p), ylabel = j == 1 ? "Sharpe" : "",
                  xticks = (1:length(rows), [string(r.value) for r in rows]))
        barplot!(ax, 1:length(rows), [r.sharpe for r in rows];
                 color = [r.sharpe >= 0 ? BLUE : RED for r in rows])
        hlines!(ax, [0.0]; color = GREY)
    end
    fig
end

end # module
