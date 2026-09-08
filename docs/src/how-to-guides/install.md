
# Quick installation

This page outlines requirements and installation steps for the code. Currently,
GNU/Linux and MacOS (including ARM) are supported.

---

### Software requirements

!!! warning
    Do not install Julia using your system package manager. Install only from julialang.org as below.

---

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

---

### Testing
If you want to run the tests manually, simply use the script located in the `test/` folder. Run the follwoing command from the Obliqua root directory:
```bash
julia --project test/runtests.jl
```
This will print information on whether tests passed or failed.

---

### Python environment (for output visualization)

Obliqua itself only needs Julia, but the `visualize_output.ipynb` notebook used to plot its results needs Python **3.12**, installed via [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [miniforge](https://github.com/conda-forge/miniforge).

!!! tip
    If you already have a `proteus` conda environment set up (e.g. from installing [PROTEUS](https://proteus-framework.org/PROTEUS/How-to/installation.html)), you can reuse it here instead of creating a new one — just `conda activate proteus` and skip to the last step below.

If you do not have miniconda or miniforge installed yet:

```@raw html
<div class="os-tabs" data-tab-group="os">
  <div role="tablist">
    <button role="tab" data-tab="macos" aria-selected="true">macOS (Homebrew)</button>
    <button role="tab" data-tab="debian" aria-selected="false">Debian / Ubuntu</button>
    <button role="tab" data-tab="fedora" aria-selected="false">Fedora / RHEL</button>
  </div>
  <div role="tabpanel" data-tab="macos" data-active="true"><pre><code>curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh</code></pre></div>
  <div role="tabpanel" data-tab="debian" data-active="false"><pre><code>mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh
rm ~/miniconda3/miniconda.sh</code></pre></div>
  <div role="tabpanel" data-tab="fedora" data-active="false"><pre><code>mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh
rm ~/miniconda3/miniconda.sh</code></pre></div>
</div>
```

!!! tip "Install miniconda/miniforge in your personal directory"
    This gives you full control over your environment and is recommended even on shared/cluster systems.

Then, from the Obliqua repository directory, create and activate a Python 3.12 environment:

```sh
conda create -n obliqua python=3.12
conda activate obliqua
```

Install the notebook's plotting dependencies:

```sh
pip install numpy matplotlib netCDF4 jupyter cmcrameri proteus_mpl
```

Then launch the notebook:

```sh
jupyter lab visualize_output.ipynb
```
