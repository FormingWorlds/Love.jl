```@raw html
    <div align="center">
    <h1>Obliqua</h1>
    <p><i>A Julia package to calculate the tidal deformation (i.e., tidal Love numbers) of solid, partially-solid, and liquid planetary mantles and the associated volumetric tidal dissipation and global heating rates.</i></p>
    <p>
        <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
        <a href="https://codecov.io/gh/FormingWorlds/Obliqua"><img src="https://codecov.io/gh/FormingWorlds/Obliqua/graph/badge.svg?token=wNmXt5ld7H" alt="Codecov"></a>
        <a href="https://github.com/FormingWorlds/Obliqua/actions/workflows/runtests.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FMarijnJ0%2Febf04f7a12a5b14edb8198927121d149%2Fraw%2Ftests-unit.json" alt="Unit Tests"></a>
        <a href="https://github.com/FormingWorlds/Obliqua/actions/workflows/runtests.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FMarijnJ0%2Febf04f7a12a5b14edb8198927121d149%2Fraw%2Ftests-integration.json" alt="Integration Tests"></a>
        <a href="https://github.com/FormingWorlds/Obliqua/actions/workflows/runtests.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FMarijnJ0%2Febf04f7a12a5b14edb8198927121d149%2Fraw%2Ftests-total.json" alt="Total Tests"></a>
    </p>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:d31027,100:ea384d&height=90&section=header" width="100%"/>
    <!-- Moving the video above the header divider wave -->
    <p align="center" style="margin-top: -85px; margin-bottom: 10px;">
        <!-- Resize width here (e.g., 60%, 80%, or fixed pixel width like 500px) -->
        <video width="60%" autoplay loop muted playsinline style="max-width: 600px; height: auto;">
            <source src="assets/banner.webm" type="video/webm">
            Your browser does not support the video tag.
        </video>
    </p>
    </div>
```

### Documentation

The documentation for Obliqua is hosted on the PROTEUS website. The documentation includes a user guide, installation instructions, and examples of how to use the package. Additionally, the documentation provides information on the underlying theory and algorithms used in Obliqua. The Obliqua package is designed to be used in conjunction with the PROTEUS simulation framework, and the documentation includes information on how to integrate Obliqua with PROTEUS.

```@raw html
    <p align="center">
    <a href="https://proteus-framework.org/Obliqua/"><b>📖 Obliqua Documentation</b></a>
    &nbsp;&nbsp;•&nbsp;&nbsp;
    <a href="https://proteus-framework.org/PROTEUS/"><b>🌌 PROTEUS Framework</b></a>
    </p>
```

---

### Get started

```@raw html
<table width="100%" style="border-collapse: separate; border-spacing: 10px;">
<tr>
    <td width="33%" valign="top" style="border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: left;">
    <h5 style="margin-top: 0; text-align: left;">📥 Install Obliqua</h5>
    <p style="text-align: left;">Set up the package and its dependencies within the Julia environment.</p>
    <div style="margin-top: 16px; text-align: left;">

```

**[Installation guide →](@ref "Quick installation")**

```@raw html
    </div>
    </td>
    <td width="33%" valign="top" style="border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: left;">
    <h5 style="margin-top: 0; text-align: left;">🎓 New to Obliqua?</h5>
    <p style="text-align: left;">Start with the quick-start guide to calculate basic tidal Love numbers.</p>
    <div style="margin-top: 16px; text-align: left;">

```

**[Quick start tutorial →](@ref "Zero-dimensional test case: The Moon")**

```@raw html
    </div>
    </td>
    <td width="33%" valign="top" style="border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: left;">
    <h5 style="margin-top: 0; text-align: left;">🌍 Ready for science?</h5>
    <p style="text-align: left;">Run full planet-mantle dissipation models integrated with PROTEUS.</p>
    <div style="margin-top: 16px; text-align: left;">

```

**[Science examples →](@ref "One-dimensional test case: The Earth")**

```@raw html
    </div>
    </td>
</tr>
</table>

```


---

### Repository structure

The repository structure is laid out below. The main components of the repository are the source code, documentation, tests, and examples. The source code is located in the `src/` directory, while the documentation is located in the `docs/` directory. The tests are located in the `test/` directory, and the examples are located in the `examples/` directory.

```@raw html
<details>
<summary><b>📁 Click to view repository structure</b></summary>

```

```bash
Obliqua/
├── 📂 src/
│   ├── 📄 common.jl
│   ├── 📄 constants.jl
│   ├── 📄 fluid0d.jl
│   ├── 📄 Hansen.jl
│   ├── 📄 interior.jl
│   ├── 📄 load.jl
│   ├── 📄 Obliqua.jl
│   ├── 📄 plotting.jl
│   ├── 📄 solid0d.jl
│   ├── 📄 solid1d_equil_relax.jl
│   ├── 📄 solid1d_mush_relax.jl
│   ├── 📄 solid1d_mush.jl
│   ├── 📄 solid1d_relax.jl
│   ├── 📄 solid1d.jl
│   └── 📄 SphericalHarmonics.jl
├── 📂 examples/
│   ├── 📄 hansen_k_table.ipynb
│   └── 📄 visualize_output.ipynb
├── 📂 docs/
│   ├── 📂 src/
│   ├── 📄 make.jl
│   └── 📝 Project.toml
├── 📂 out/
├── 📂 res/
│   ├── 📂 config
│   └── 📂 interior_data
├── 📂 test/
│   ├── 📄 runcoverage.jl
│   ├── 📄 runprofiler.jl
│   ├── 📄 runtests.jl
│   ├── 📄 test_common.jl
│   ├── 📄 test_constants.jl
│   ├── 📄 test_fluid0d.jl
│   ├── 📄 test_obliqua.jl
│   ├── 📄 test_setup.jl
│   ├── 📄 test_solid0d.jl
│   ├── 📄 test_solid1d_equil_relax.jl
│   ├── 📄 test_solid1d_mush_relax.jl
│   ├── 📄 test_solid1d_mush.jl
│   ├── 📄 test_solid1d_relax.jl
│   ├── 📄 test_solid1d.jl
│   └── 📝 test.toml
├── 🚫 .gitignore
├── 📦 package.json
├── 📝 README.md
├── 📝 LICENSE.txt
├── 📝 CODE_OF_CONDUCT.md
└── 📝 Project.toml

```

```@raw html
</details>

```

---

### Source code

The source code for Obliqua is organized into several modules, each responsible for a specific aspect of the functionality. The main modules include:

- `common.jl`: Common functions and utilities used throughout the package.
- `constants.jl`: Physical constants and parameters used in the calculations.
- `fluid0d.jl`: 0D fluid model.
- `Hansen.jl`: Hansen coefficients for tidal dissipation.
- `interior.jl`: Complementary models for the interior properties of planets.
- `load.jl`: Load interior data functions.
- `Obliqua.jl`: Main module for the Obliqua package.
- `plotting.jl`: Functions for plotting and visualization.
- `solid0d.jl`: 0D solid model.
- `solid1d_equil_relax.jl`: 1D solid equilibrium tide model using relaxation based solver.
- `solid1d_mush_relax.jl`: 1D solid porovisco-elastic model using relaxation based solver.
- `solid1d_mush.jl`: 1D solid porovisco-elastic model using shooting method.
- `solid1d_relax.jl`: 1D solid visco-elastic model using relaxation based solver.
- `solid1d.jl`: 1D solid visco-elastic model using shooting method.
- `SphericalHarmonics.jl`: Spherical harmonics functions.

All source code is written in the Julia programming language and is availible on GitHub. 

```@raw html
    <p align="center">
    <a href="https://github.com/FormingWorlds/Obliqua/"><b>💻 Obliqua Github</b></a>
    </p>
```

This software is available under the MIT license. Different components within the PROTEUS framework carry different licenses. Please find information about the use of licenses within the PROTEUS framework on the PROTEUS [licence page](https://proteus-framework.org/license).

---

### Integrations

This project is being developed in close collaboration with the [PROTEUS](https://github.com/FormingWorlds/PROTEUS) similation framework. Obliqua can be used seamlessly within PROTEUS to model tidal dissipation and Love number evolution in planetary interiors.

For detailed instructions on installing and using Obliqua in combination with PROTEUS, refer to the [installation guide](@ref "Quick installation") in the Obliqua documentation.