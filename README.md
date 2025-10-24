

<a href="https://opensource.org/license/mit">
   <img src="https://img.shields.io/badge/License-MIT-blue.svg">
</a>
<a href="https://arxiv.org/abs/2507.11266">
  <img src="https://img.shields.io/badge/arXiv-2507.11266-b31b1b">
</a>

## Love.jl (Tidal heating model)

A Julia package to calculate the tidal deformation (i.e., tidal Love numbers) of solid and partially-solid planetary bodies.

Forked from the [original repository](https://github.com/hamishHay/Love.jl) of Hamish Hay. Distributed under the MIT License.

### Documentation
https://fwl-proteus.readthedocs.io

## Contributors

| Name  | Email address |
| -     | -             |
Hamish Hay              | hamish[at]tides.rocks |
Tim Lichtenberg         | tim.lichtenberg[at]rug.nl |
Harrison Nicholls       | harrison.nicholls[at]physics.ox.ac.uk |


### Repository structure
* `README.md`           - This file
* `LICENSE.txt`         - License for modification, distribution, etc.
* `src/`                - Source files
* `examples/`           - Tools and notebooks showing how to use the code

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/FormingWorlds/Love.jl.git
   ```
2. Move to the repository directory and start Julia
   ```sh
   cd Love.jl
   julia
   ```
3. Install `Love.jl`
   ```sh
   julia> ]
   pkg> add https://github.com/FormingWorlds/Love.jl.git#julia

