
# Configuration file

Obliqua uses [TOML](https://toml.io/en/) to structure its configuration files. This page lists all parameters with their types and descriptions. For topic-specific parameter guides, see the configuration reference pages:



For worked examples, see the [tutorials](@ref "Zero-dimensional test case: The Moon").

---

### Global Parameters

These define metadata and output behavior.

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `title` | str | Identifier for the simulation setup. |
| `version` | str | Configuration file version for reproducibility. |

```@raw html
</div>
```

---

### Execution Parameters

This block controls output and logging.

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[params]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `path` | str | Directory where output files are stored. |
| `time` | float | Current time of the simulation run in years (used for output file naming). |
| `logging` | str | Logging level (`INFO`, `WARNING`, `DEBUG`). |
| `plot_fmt` | str | Output format for generated plots. |

```@raw html
</div>
```

---

### Stellar Parameters

Defines the host star.

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[star]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `mass` | float | Stellar mass in solar masses ($M_\odot$). |

```@raw html
</div>
```

---

### Tidal Model Parameters

Controls the tidal response model.

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[orbit.obliqua]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `store_3D` | bool | Boolean flag to store 3D tidal response data. |
| `enforce_ec` | bool | Boolean flag to enforce energy conservation in tidal response calculations. |
| `optimize_scales` | bool | Boolean flag to optimize scaling factors for numerical stability. |
| `solid_shell` | bool | Boolean flag to add an infinitesimal solid shell around the core to couple y2 and y4 in fluid mantles. |
| `min_frac` | float | Minimum segment fraction of total mantle before it is considered. |
| `max_frac` | float | Maximum segment fraction of total mantle before it is considered. |
| `visc_l` | float | Liquid viscosity. |
| `visc_lus` | float | Liquid-Mush handoff viscosity. |
| `visc_s` | float | Solid viscosity. |
| `visc_sus` | float | Solid-Mush handoff viscosity. |
| `n` | array | Radial dependence exponent in $(r/a)^n$. |
| `m` | array | Tidal harmonic (e.g., $m=2$ for semidiurnal tides). |
| `spectrum` | str | Frequency sampling strategy (`"full"` or `"adaptive"`). |
| `N_sigma` | int | Number of sampled forcing frequencies. |
| `p_min` | float | Minimum period ($\log_{10}$ kyr). |
| `p_max` | float | Maximum period ($\log_{10}$ kyr). |
| `s_min` | int | Minimum Fourier mode. |
| `s_max` | int | Maximum Fourier mode. |
| `material_mu` | str | Rheological model for shear modulus (`"andrade"`, `"maxwell"`, or `"elastic"`). |
| `material_k` | str | Rheological model for bulk modulus (`"andrade"`, `"maxwell"`, or `"elastic"`). |
| `alpha` | float | Andrade power-law exponent. |
| `module_solid` | str | Solid interior model (`"solid0d"`, `"solid1d"`, `"solid1d-relax"`, `"solid1d-mush"`, `"solid1d-mush-relax"`, or `"solid1d-equil-relax"`). |
| `module_mushy` | str | Mushy layer model (`"none"` or `"interp"`). |
| `module_fluid` | str | Fluid layer model (`"none"`, `"fluid0d"`, or `"fluid1d"`). |

```@raw html
</div>
```

---

### Solid Interior Parameters

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[orbit.obliqua.solid]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `ncalc` | int | Number of radial layers (shooting method). |
| `dr_min` | int | Minimum grid spacing for relaxation solver [m]. |
| `dr_max` | int | Maximum grid spacing for relaxation solver [m]. |
| `core` | str | Core boundary condition (`"liquid"`, `"solid"`, `"inertial"`). |
| `core_props` | str | Core properties (shear modulus, bulk modulus) to use for CMB boundary condition (`"core"`, `"mantle"`). |
| `inertial_terms` | bool | Boolean flag to include inertial terms in the motion matrix. |
| `bulk_l` | float | Liquid bulk modulus [Pa]. |
| `dbulk_power` | float | Drained bulk modulus power-law scaling exponent. |
| `porosity_thresh` | float | Threshold below which no mush is formed. |

```@raw html
</div>
```

---

### Fluid Parameters

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[orbit.obliqua.fluid]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `sigma_R` | float | Rayleigh drag at the interface. |
| `sigma_R_inf` | float | Drag in the bulk fluid. |
| `sigma_R_prf` | str | Vertical drag profile (`"uniform"`, `"exp"`, `"linear"`, `"quadratic"`, `"dynamic"`, `"dynamic_interp"`). |
| `H_R` | float | Scale height [m]. |
| `efficiency` | float | Drag efficiency factor. |

```@raw html
</div>
```

---

### Mushy Layer Parameters

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[orbit.obliqua.mushy]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `b_width` | float | Width of the bottom dissipation peak as a fraction of layer thickness. |
| `t_width` | float | Width of the top dissipation peak as a fraction of layer thickness. |

```@raw html
</div>
```

---

### Planetary Structure

Defines bulk planetary properties.

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[struct]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `core_density` | float | Core density [kg m$^{-3}$]. |
| `core_shear` | float | Core shear [Pa s]. |
| `core_bulk` | float | Core bulk modulus [Pa]. |

```@raw html
</div>
```

---

### Interior Energetics

Defines interior energy transfer dynamics related properties.

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[interior_energetics]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `grain_size` | float | Grain size [m]. |

```@raw html
</div>
```



