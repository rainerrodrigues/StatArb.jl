using Documenter
using StatArb

makedocs(
    sitename = "StatArb.jl",
    modules = [StatArb],
    authors = "Rainer Rodrigues",
    # DOCS_DRAFT=true skips running the @example blocks (quick text-only builds)
    draft = get(ENV, "DOCS_DRAFT", "false") == "true",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             size_threshold = 400 * 1024, size_threshold_warn = 300 * 1024),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/getting_started.md",
            "tutorials/validation.md",
            "tutorials/baskets_and_rules.md",
            "tutorials/plotting.md",
        ],
        "Guides" => [
            "guide/architecture.md",
            "guide/econometrics.md",
            "guide/kalman.md",
            "guide/execution.md",
            "guide/performance.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(repo = "github.com/rainerrodrigues/StatArb.jl.git", devbranch = "main")
