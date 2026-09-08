
# Stabilise simulations

Given the multi-phase nature of the Obliqua model, and the wide range of orbital regimes it can explore, it is posible for simulations to fail to converge. In this guide we provide some tips to help stabilise your simulations.

!!! tip
    The first place to look is always the terminal output. If the simulation fails to converge, the terminal output will usually provide a hint as to what went wrong. 
---

### Solid tide models

The solid models included within the Obliqua module are based based on different solving schemes. In general, these models can also be used to describe the tidal response of a mushy interior. However, this poses nummerical challenges. Specifically, traditional methods employing the **shooting method** to solve the tidal response are prone to fail in this regime. For this, we recommend using the **relaxation method** to solve the tidal response. This method is more robust and can handle the mushy regime better.

The additional complexity of the poroviscoelastic model can also lead to convergence issues. In this case, we again recommend using the **relaxation method** to solve the tidal response. In addition, at low forcing frequencies, the additional nummerical precision provided by the `BigFloat` type may be required to preserve convergence. This can be set in the `src/constants.jl` file by setting `const prec  = BigFloat` and `const precc = Complex{BigFloat}`.

### The equilibrium tide

As the forcing frequency approaches zero, the tidal response transitions to the equilibrium tide. In this regime, the tidal response is dominated by the hydrostatic response of the interior. generally, this leads to the `solid1d*` models becoming singular, regardless of the solving scheme used. In this case, we recommend using the `solid1d_equil_relax` model, which is specifically designed to handle the equilibrium tide. Currently, Obliqua automatically switches to this model when the forcing frequency is below a threshold of `1e-18`, the inverse of the Hubble time. 

### Radial resolution

All `solid1d` models rely on a radial discretisation of the interior properties. The number of radial shells can be set in the configuration file using the `orbit.obliqua.solid.ncalc` parameter for shooting method, and the `orbit.obliqua.solid.dr_min` and `orbit.obliqua.solid.dr_max` parameters for the relaxation method. Generally, good convergence is achieved with `ncalc = 1000`, `dr_min = 10`, and `dr_max = 3000`. However, in some cases, such as around the natural frequency of the body, increasing the radial resolution may be required to achieve convergence. In this case, we recommend increasing the number of radial shells for the shooting method, and decreasing the minimum spacing between grid points for the relaxation method.

### Fluid regime

When modeling the fluid response through the solid formalisms, the motion matrix drops in rank, leading to singular matrices. In this case, we recommend setting `orbit.obliqua.solid.solid_shell = true` in the configuration file. This adds an infinitesimal solid shell around the core, which couples the y2 and y4 Love numbers in fluid layers, and stabilises the solution. 

### Extremely low forcing frequencies

Additionally, at low forcing frequencies, where the overall tidal dissipation rates become negligible, the tidal response can become numerically unstable. In this case, we recommend setting `orbit.obliqua.solid.enforce_ec = true` in the configuration file. This enforces energy conservation in the tidal response calculations, which improves stability in fluid-mush cases at low forcing frequencies (< 1e-7 s⁻¹). Note that this does not affect the Love numbers. This merely acts to clean up the already insiginficant part of the parameter space. As such, one may also consider disabling tides completely in this regime.