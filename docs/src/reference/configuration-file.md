```@meta
CollapsedDocStrings = true
```

### Reference (0)

# Configuration

The configuration files follow the conventions used within PROTEUS. The default `all_options.toml` file (`res/config/all_options.toml`) contains all available parameters together with their defaults; the full parameter table is also available on the [Configuration file](@ref) how-to-guide page. This page instead walks through the `[orbit.obliqua]` block topic by topic and links each group of parameters to the reference page that derives the underlying model.

### Globals

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

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[params.out]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `path` | str | Directory where output files are stored. |
| `time` | float | Current time of the simulation run in years (used for output file naming). |
| `logging` | str | Logging level (e.g. `"INFO"`, `"DEBUG"`). |
| `plot_fmt` | str | Output format for generated plots (`"png"` or `"pdf"` recommended). |

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
| `store_3D` | bool | Store 3D tidal response (displacement, stress, strain) for each layer. Generates large output files; if `false`, radial profiles are stored instead. |
| `enforce_ec` | bool | Enforce energy conservation in tidal response calculations. Improves stability in fluid-mush cases at low forcing frequencies (< 1e-7 Hz); does not affect the Love numbers. |
| `optimize_scales` | bool | Optimize non-dimensionalization scales for the relaxation method, for numerical stability at low forcing frequencies. Do not combine with BigFloat precision. |
| `solid_shell` | bool | Insert an infinitesimal solid shell around the core to patch a $y_2$/$y_4$ decoupling instability in fluid layers. Only relevant for `solid1d-relax` or `solid1d-mush-relax`. |

#### Rheology and Viscosity

See [Rheology](@ref) for the underlying model.

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `min_frac` | float | Minimal segment radius fraction before smoothing [dimensionless]. |
| `visc_l` | float | Pure liquid viscosity [Pa s]. |
| `visc_lus` | float | Liquidus viscosity to use [Pa s]. |
| `visc_s` | float | Pure solid viscosity [Pa s]. |
| `visc_sus` | float | Solidus viscosity to use [Pa s]. |
| `material_mu` | str | Rheological model for the complex shear modulus (`"andrade"`, `"maxwell"`, or `"elastic"`). |
| `material_k` | str | Rheological model for the complex bulk modulus (`"andrade"`, `"maxwell"`, or `"elastic"`). |
| `alpha` | float | Andrade power-law exponent (free parameter), only used for Andrade rheology. |

#### Spectral and Forcing Parameters

See [Forcing Frequency](@ref) for the underlying model.

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `n` | array | Radial dependence exponent(s) in $(r/a)^n$; since $r \ll a$, only $n=2$ contributes significantly. |
| `m` | array | Tidal harmonic(s) of the true anomaly (e.g. $m=2$ semidiurnal, $m=1$ diurnal). |
| `spectrum` | str | Frequency sampling strategy: `"full"` samples the whole k2 spectrum, `"adaptive"` samples only the region of interest. |
| `N_sigma` | int | Number of probe frequencies to evaluate k2 at (used when `spectrum = "full"`). |
| `p_min` | float | Minimum period for orbital and axial frequencies [$\log_{10}$ kyr]. |
| `p_max` | float | Maximum period for orbital and axial frequencies [$\log_{10}$ kyr]. |
| `s_min` | int or `"none"` | Minimum tidal mode (Fourier index in mean anomaly). `"none"` derives it from the eccentricity relation. |
| `s_max` | int or `"none"` | Maximum tidal mode (Fourier index in mean anomaly). `"none"` derives it from the eccentricity relation. |

#### Phase specific Tidal Models

See [Tidal Models](@ref) for all included models.

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `module_solid` | str | Solid-phase tidal model, see below. |
| `module_mushy` | str | Mushy-phase tidal model, see below. |
| `module_fluid` | str | Fluid-phase tidal model, see below. |

```@raw html
</div>
```

---

### Tidal Models

```@raw html
<p class="class-header"><span class="class-label">models</span> <span class="class-name">[orbit.obliqua.module_solid]</span></p>
```

See [Solid-Phase](@ref) for the underlying theory.

* **`"none"`**: No solid-phase tidal model.
* **`"solid0d"`**: Homogeneous solid approximation. The solid region is treated as a single effective layer with averaged mechanical properties; fast, but cannot resolve radial structure.
* **`"solid1d"`**: Radially resolved structure, solved with the shooting method. Allows realistic rigidity, density, and rheology profiles.
* **`"solid1d-relax"`**: Same as `solid1d`, but uses the relaxation method instead of shooting. Generally more stable, but slightly slower.
* **`"solid1d-mush"`**: Same as `solid1d`, but accounts for a partially molten/porous ("mushy") (interface) layer.
* **`"solid1d-mush-relax"`**: Same as `solid1d-relax`, but accounts for partially molten/porous ("mushy") regions. Uses the relaxation method instead of shooting.
* **`"solid1d-equil-relax"`**: Same as `solid1d-relax`, but limited to the fully fluid interior case (equilibrium tide). Automatically selected by the other `solid1d*` models when the forcing frequency is below the inverse Hubble time.

```@raw html
<p class="class-header"><span class="class-label">models</span> <span class="class-name">[orbit.obliqua.module_mushy]</span></p>
```

See [Mush layer - interp](@ref) for the underlying model.

* **`"none"`**: No explicit mushy layer treatment.
* **`"interp"`**: Tidal dissipation transitions smoothly between solid and fluid regimes using interpolation across the melt fraction range. Accounts for imaginary part of the Lovenumber, but not the real part.

```@raw html
<p class="class-header"><span class="class-label">models</span> <span class="class-name">[orbit.obliqua.module_fluid]</span></p>
```

See [Liquid-Phase](@ref) for the underlying model.

* **`"none"`**: No fluid-phase tidal model.
* **`"fluid0d"`**: Bulk fluid approximation. The fluid region is treated as a single effective layer with averaged mechanical properties; fast, but cannot resolve radial structure.
* **`"fluid1d"`**: Same as `fluid0d`, but allows a user-specified radial heating distribution (see `[orbit.obliqua.fluid].sigma_R_prf`).

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
| `ncalc` | int | Number of sublayers for `solid1d` solvers using the shooting method. |
| `dr_min` | float | Minimum spacing between grid points, relaxation method [m]. |
| `dr_max` | float | Maximum spacing between grid points, relaxation method [m]. |
| `core` | str | Core solution used as the CMB boundary condition: `"liquid"`, `"solid"`, `"inertial-liquid"`, or `"inertial"`. See [Solid-Phase](@ref) ("The four `core` options"). |
| `core_props` | str | Core properties (shear modulus, bulk modulus) to use for the CMB boundary condition: `"core"` or `"mantle"`. |
| `inertial_terms` | bool | Include inertial terms in the solid tidal response equations. |
| `bulk_l` | float | Liquid bulk modulus [Pa]. |
| `dbulk_power` | float | Drained bulk modulus power-law scaling exponent. |
| `porosity_thresh` | float | Percolation threshold, at which there is a first-order transition from fully connected to fully isolated pore space [dimensionless]. |

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
| `sigma_R` | float | Rayleigh drag coefficient at the interface [dimensionless]. |
| `sigma_R_inf` | float | Rayleigh drag coefficient in the bulk (pure) fluid [dimensionless]. |
| `sigma_R_prf` | str | Vertical drag profile: `"uniform"`, `"exp"`, `"linear"`, `"quadratic"`, `"dynamic"`, or `"dynamic_interp"` (default). See [Liquid-Phase](@ref) for the physical meaning of each. |
| `H_R` | float | Rayleigh drag scale height [m]. |
| `efficiency` | float | Rayleigh drag efficiency at the core interface [dimensionless]. |

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
| `b_width` | float | Width of the bottom dissipation peak, as a fraction of layer thickness. |
| `t_width` | float | Width of the top dissipation peak, as a fraction of layer thickness. |

```@raw html
</div>
```

---

### Planetary Structure

```@raw html
<p class="class-header"><span class="class-label">config</span> <span class="class-name">[struct]</span></p>
```

```@raw html
<div class="attributes-table">
```

| NAME | TYPE | DESCRIPTION |
| :--- | :--- | :--- |
| `core_density` | float | Core density [kg m$^{-3}$]. |
| `core_shear` | float | Core shear modulus [Pa]. |
| `core_bulk` | float | Core bulk modulus [Pa]. |

```@raw html
</div>
```

---

### Interior Energetics

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
