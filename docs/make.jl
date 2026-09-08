using Documenter
using DocumenterPages
using DocumenterTools: Themes
using Obliqua

ASSETS_DIR = joinpath(@__DIR__, "src", "assets")

# metadata
footer::String = "Copyright © 2025-Present by the Forming Worlds Lab collaborators. Source code is available under the [MIT](https://opensource.org/license/mit) license. Documentation and assets are available under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license."

# following https://github.com/JuliaMusic/JuliaMusic_documentation.jl/blob/master/docs/make.jl
# combine style and defs files into single scss files for compilation
for w in ("light", "dark")
    style = read(joinpath(ASSETS_DIR, "style.scss"), String)
    theme = read(joinpath(ASSETS_DIR, "$(w)defs.scss"), String)
    write(joinpath(ASSETS_DIR, "$(w).scss"), style*"\n"*theme)
end

# compile styles into scss files
Themes.compile(joinpath(ASSETS_DIR, "light.scss"),
                joinpath(ASSETS_DIR, "themes/documenter-light.css"))
Themes.compile(joinpath(ASSETS_DIR, "dark.scss"),
                joinpath(ASSETS_DIR, "themes/documenter-dark.css"))

format = Documenter.HTML(
    edit_link = "main",
    collapselevel = 1,
    size_threshold = 300 * 1024,
    size_threshold_warn = 200 * 1024,
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = [
        "assets/style.css",
        "assets/glyph_tidal.ico",
        "assets/theme-toggle.js",
        "assets/os-tabs.css",
        "assets/os-tabs.js",
    ]
)

# Build the docs
makedocs(
    sitename = "Obliqua",
    checkdocs = :none,
    format = format,
    modules = [Obliqua],
    pages = [
        "Home" => "index.md",
        "Get started" => "get_started.md",
        "How-to guides" => [
            "1 - Installation" => "how-to-guides/install.md",
            "2 - Configuration" => "how-to-guides/config_file.md",
            "3 - Usage" => [
                "Overview" => "how-to-guides/usage.md",
                "Running and output" => "how-to-guides/running_output.md",
            ],
            "4 - Stabilise simulations" => "how-to-guides/stabilise_sim.md",
            "5 - Troubleshooting" => "how-to-guides/troubleshooting.md",
            "6 - Testing and profiling" => "how-to-guides/testing.md",
        ],
        "Tutorials" => [
            "1 - 0D Test" => "tutorials/0d_test.md",
            "2 - 1D Test" => "tutorials/1d_test.md",
        ],
        "Reference" => [
            "Overview" => "reference/index.md",
            "0 - Configuring Model" => "reference/configuration-file.md",
            "1 - Forcing Frequency" => "reference/forcing-frequency.md",
            "2 - Rheology" => "reference/rheology.md",
            "3 - Solid-phase" => [
                "Overview" => "reference/solid-phase.md",
                "Solid0d" => "reference/solid/solid0d.md",
                "Solid1d" => "reference/solid/solid1d.md",
                "Solid1d-mush" => "reference/solid/solid1d_mush.md",
                "Solid1d-relax" => "reference/solid/solid1d_relax.md",
                "Solid1d-mush-relax" => "reference/solid/solid1d_mush_relax.md",
                "Solid1d-equil-relax" => "reference/solid/solid1d_equil_relax.md",
            ],
            "4 - Mush layer" => "reference/mush-layer.md",
            "5 - Liquid-phase" => "reference/liquid-phase.md",
            "6 - Surface Loading" => "reference/surface-loading.md",
            "7 - Tidal potential" => "reference/tidal-potentials.md",
        ],
        "Explanation" => [
            "Overview" => "explanation/index.md",
            "1 - Technical overview" => "explanation/technical_overview.md",
            "2 - Main loop" => "explanation/main_loop.md",
            "3 - Iterative process" => "explanation/iter_proc.md",
            "4 - Post-processing" => "explanation/post_proc.md",
            "5 - Stabilization techniques" => "explanation/stabilize.md",
        ],
        "Development" => "development.md",
        "Related codes" => "ecosystem.md",
    ]
)

deploydocs(
    repo = "https://github.com/FormingWorlds/Obliqua.git",
    push_preview=true
)

