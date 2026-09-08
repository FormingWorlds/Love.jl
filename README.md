

<p align="center">
  <a href="https://opensource.org/license/mit">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://codecov.io/gh/FormingWorlds/Obliqua"> 
    <img src="https://codecov.io/gh/FormingWorlds/Obliqua/graph/badge.svg?token=wNmXt5ld7H" alt="Codecov">
  </a>
  <a href="https://github.com/FormingWorlds/Obliqua/actions/workflows/runtests.yml">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FMarijnJ0%2Febf04f7a12a5b14edb8198927121d149%2Fraw%2Ftests-unit.json" alt="Unit Tests">
  </a>
  <a href="https://github.com/FormingWorlds/Obliqua/actions/workflows/runtests.yml">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FMarijnJ0%2Febf04f7a12a5b14edb8198927121d149%2Fraw%2Ftests-integration.json" alt="Integration Tests">
  </a>
  <a href="https://github.com/FormingWorlds/Obliqua/actions/workflows/runtests.yml">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FMarijnJ0%2Febf04f7a12a5b14edb8198927121d149%2Fraw%2Ftests-total.json" alt="Total Tests">
  </a>
</p>

## Obliqua (Tidal heating model)

A Julia package to calculate the tidal deformation (i.e., tidal Love numbers) of solid, partially-solid, and liquid planetary mantles and the associated volumetric tidal dissipation and global heating rates.

### Documentation

Obliqua: [https://proteus-framework.org/Obliqua/](https://proteus-framework.org/Obliqua/) \
PROTEUS: [https://proteus-framework.org/PROTEUS/](https://proteus-framework.org/PROTEUS/)

### Repository structure
* `README.md`           - This file
* `CODE_OF_CONDUCT.md`  - Code of conduct for the community
* `LICENSE.txt`         - License for modification, distribution, etc.
* `Project.toml`        - Project file for Julia package manager
* `src/`                - Source files
* `docs/`               - Documentation files
* `out/`                - Output files
* `res/`                - Resources
* `test/`               - Tests for the code
* `examples/`           - Tools and notebooks showing how to use the code

### Source code

GitHub: [https://github.com/FormingWorlds/Obliqua](https://github.com/FormingWorlds/Obliqua)

This software is available under the MIT license. Different components within the PROTEUS framework carry different licenses. Please find information about the use of licenses within the PROTEUS framework on the PROTEUS [licence page](https://proteus-framework.org/license).

### Installation

1. Install Julia's package manager (Mac and Linux): 
   ```sh
   curl -fsSL https://install.julialang.org | sh
   ```
2. Clone the repo
   ```sh
   git clone https://github.com/FormingWorlds/Obliqua.git
   ```
3. Move to the repository directory
   ```sh
   cd Obliqua
   ```
4. Open Julia package manager
   ```sh
   julia
   ```
5. Install `Obliqua`
   ```sh
   julia> ]
   pkg> add .
   ```

### Integrations

This project is being developed in close collaboration with the [PROTEUS](https://github.com/FormingWorlds/PROTEUS) similation framework. Obliqua can be used seamlessly within PROTEUS to model tidal dissipation and Love number evolution in planetary interiors.

For detailed instructions on installing and using Obliqua in combination with PROTEUS, please refer to the [installation guide](https://proteus-framework.org/Obliqua/dev/install/) in the Obliqua documentation.
