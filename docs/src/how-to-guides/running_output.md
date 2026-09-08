
# Running and output

This page covers how to work with the main `Obliqua.run_tides` function within Obliqua, and where the output files are stored.

---

### Loading dummy interior data

The `Obliqua` package comes with with several tidal models. Each model can be accessed through one comprehensive function. Depending on which module you want to use `Obliqua` requires different input parameters. In order to get started, several data files are included, these can be used in combination with the different functions. First we shall show how to load these data files.

The data files are stored in the `/path/to/Obliqua/res` folder, and are of type `JSON`. They have following structure

```json
{
  "omega": "Float",
  "axial": "Float",
  "ecc": "Float",
  "sma": "Float",
  "S_mass": "Float",
  "density": "[Array]",
  "radius": "[Array]",
  "visc": "[Array]",
  "shear": "[Array]",
  "bulk": "[Array]",
  "bulkd": "[Array]",
  "phi": "[Array]",
  "perm": "[Array]",
  "cfg": "Dict"
}
```

Load the interior data using the `Obliqua.load` module. For example, to load the `test_mantle_mush_full_test.json` data file, use the following command:

```julia
omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, phi, ncalc =
        load.load_interior_mush_full("res/interior_data/test_mantle_mush_full_test.json", false)
```

!!! tip 
    First let's test if the provided data is compatible.

    ```julia
    using Obliqua

    # test data validity using included data file
    ok = load.load_interior_mush_full("res/interior_data/test_mantle_mush_full_test.json", true)

    if ok
        print("Pass")
    else
        print("Fail")
    end

    ```

---

### Input parameters

Obliqua is run through the `Obliqua.run_tides` function, which takes in the relevant parameters and configuration settings. Depending on which module is being used the following parameters need to be provided.

```@raw html
<p class="class-header"><span class="class-label">input</span> <span class="class-name">[Table of inputs]</span></p>
```

```@raw html
<div class="attributes-table">
```

| Input   | solid1d\_equil | solid1d          | solid1d\_mush    | fluid1d  | Description                      | Symbol |
|:--------|:--------------:|:----------------:|:----------------:|:--------:|:---------------------------------|-------------:|
| omega   | ✔️              | ✔️              | ✔️               | ✔️       | Orbital Frequency                | ``\omega``   |
| axial   | ✔️              | ✔️              | ✔️               | ✔️       | Axial Frequency                  | ``\Omega``   |
| ecc     | ✔️              | ✔️              | ✔️               | ✔️       | Eccentricity                     | ``\epsilon`` |
| sma     | ✔️              | ✔️              | ✔️               | ✔️       | Semi major axis                  | ``a``        |
| S_mass  | ✔️              | ✔️              | ✔️               | ✔️       | Stellar mass                     | ``M_\star `` |
| density | ✔️              | ✔️              | ✔️               | ✔️       | Density profile                  | ``\rho``     |
| radius  | ✔️              | ✔️              | ✔️               | ✔️       | Radii                            | ``r``        |
| visc    | ❌              | ✔️              | ✔️               | ✔️       | Viscosity profile                | ``\eta``     |
| shear   | ❌              | ✔️              | ✔️               | ❌       | Shear profile                    | ``\mu``      |
| bulk    | ❌              | ✔️              | ✔️               | ❌       | Solid Bulk Modulus               | ``\kappa_s`` |
| bulkd   | ❌              | ❌              | ✔️               | ❌       | Drained Bulk Modulus             | ``\kappa_d`` |
| phi     | ❌              | ❌              | ✔️               | ❌       | Melt Fraction profile (porosity) | ``\phi``     |
| perm    | ❌              | ❌              | ✔️               | ❌       | Permeability profile             | ``k``        |
| cfg     | ✔️              | ✔️              | ✔️               | ✔️       | Configuration                    |  |

```@raw html
</div>
```

\

!!! tip
    It is important to note that the `radius` array contains the radial values at the boundaries of the spherical shells that make up the planetary mantle, whilst the `density`, `visc`, `shear`, `bulk`, `bulk_d`, `phi`, and `perm` arrays contain the mean of these in the spherical shells. As such it follows that there are ``N+1`` values in the `radius` array, and ``N`` values in the `density`, `visc`, `shear`, `bulk`, `bulk_d`, `phi`, and `perm` arrays, where ``N`` is the number of spherical shells. 

!!! info
    Load the configuration file (TOML) using the `Obliqua.open_config` function:

    ```julia
    cfg = Obliqua.open_config("res/config/all_options.toml")
    ```

    The default configuration file is located in the `res/config` folder, and is called `all_options.toml`. 

---

### Running Obliqua 

Given these parameters, the `Obliqua.run_tides` function can be called as follows:

```julia
# call the desired model
power_prf, power_blk, nmk, σ_range, LNk = Obliqua.run_tides(
    omega, axial, ecc, sma, S_mass, density, radius, visc, shear, bulk, bulkd, phi, perm, cfg
)
```

This writes its results to a timestamped `out/<timestamp>_obliqua.nc` [NETCDF](https://www.unidata.ucar.edu/software/netcdf/) file inside your Obliqua directory, because the config sets `params.out.path = "out"`. Set path to a fixed string in the config to choose the folder name instead. The timestamp is provided in the config file as `params.out.time`, and is used to distinguish between different runs of the model when setup with PROTEUS. 

---

### Output and results

In addition to the NETCDF datafile, the immidiate output of the function includes the power profile (W/m³), the total power (W), the tidal modes (n,m,k), the forcing frequencies per mode (s⁻¹), and the complex tidal k-Love number per mode. This is all the information needed within the PROTEUS framework, and as such omits additional file reading overhead. The tree below outlines the purpose of the main files and subfolders:

```bash
Obliqua/
├── 📂 out/
|   └── 📄 <timestamp>_obliqua.nc
│   └── 🖼️ other plots
├── 📂 res/
│   └── 📂 config/
|       └── 📄 all_options.toml
├── 📂 examples/
│   └── 📄 visualize_output.ipynb
```

!!! tip
    Use the included `examples/visualize_output.ipynb` Jupyter notebook to visualize the output of the model. The notebook reads the NETCDF file and generates plots of the tidal response, power distribution, and other relevant quantities! All output is saved to the `out` folder.
