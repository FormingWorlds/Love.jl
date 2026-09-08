
### Reference (5)

# Liquid-Phase

When the melt fraction of a magma layer exceeds a critical value ``F_m \gtrsim F_{m,c}``, its response to tidal forcing becomes fluid-like rather than solid-like. In this regime, tidal deformation is governed by the linearized Laplace tidal equations (LTEs), which describe the dynamics of a thin, global fluid shell of uniform thickness ``H`` on a planet of radius ``R_p``.

Neglecting nonlinear advection and mean flows, momentum and mass conservation for the magma ocean read
```math
\partial_t \vec{u} + \sigma_R \vec{u} + \vec{f} \times \vec{u} + g \nabla \zeta = g \nabla \zeta_{\mathrm{eq}}, \qquad
\partial_t \zeta + \nabla \cdot (H \vec{u}) = 0,
```
where ``\vec{u}`` is the horizontal velocity, ``\zeta`` the tidally induced surface displacement, ``\zeta_{\mathrm{eq}} = U_T/g`` the equilibrium tide associated with the tidal potential ``U_T``, ``g`` the surface gravity, and ``\sigma_R`` a Rayleigh drag frequency parameterizing dissipation in the magma ocean.

We focus on the strongly damped, highly viscous limit (``\sigma_R \gg 2\Omega``), appropriate for magma oceans, in which dissipative processes dominate over rotational effects. In this high-Ekman-number regime, the Coriolis term is negligible and the tidal response becomes overdamped. Dissipation is encapsulated by ``\sigma_R``, which should be interpreted as an effective damping rate accounting for viscous resistance, boundary-layer friction, form drag, and porous-flow dissipation near the rheological transition.

Transforming the governing equations to the frequency domain and expanding tidal quantities in spherical harmonics yields a closed-form expression for the complex, frequency-dependent tidal Love number ``k_\ell(\sigma)``. 

```math
k_{\ell}(\sigma)
= -\frac{\varrho_\ell \bar{\sigma}_\ell^2}{\sigma\tilde{\sigma} - \bar{\sigma}_\ell^2},
```

where ``\varrho_\ell = 3/(2n+1)(\rho_f/\rho_b)`` is the degree-``\ell`` density ratio between the fluid and the bulk part of the planet. The characteristic frequency is given by

```math
\bar{\sigma}_\ell = \sqrt{\frac{\mu_\ell g H_\mathrm{magma}}{R^2}},
```

where ``H_\mathrm{magma}`` the ocean depth, ``R`` the radius of the planet, and with 

```math
\mu_\ell = \ell(\ell+1),
```

the eigenvalue of the nth degree spherical harmonic, and 

```math
\tilde{\sigma} = \sigma - i \sigma_R.
```

### Radially resolved dissipation

Given that there is no consensus on where in a fluid layer most tidal energy is dissipated, we provide several parameterized radial dissipation profiles. These profiles represent different physical assumptions about how turbulent mixing, wave breaking, or viscous damping distribute tidal energy with depth. The available profiles should be interpreted as idealized parameterizations of where energy is dissipated within the fluid layer.

### Uniform dissipation

The uniform profile assumes dissipation is constant throughout the layer.

This corresponds to the simplest assumption that turbulence or small-scale mixing distributes energy evenly with depth. It is useful as a baseline case when no information about the dissipation structure is available.

### Exponential dissipation

The exponential profile concentrates dissipation near the lower boundary and decreases with height:

```math
D(z) \propto e^{-z/H_R}
```

This can approximate scenarios where dissipation is strongest near the interface with the underlying solid layer, for example when tidal flows interact with boundary roughness or generate shear-driven turbulence.

### Linear dissipation

The linear profile decreases linearly with height and reaches zero at a characteristic scale (H_R).

This provides a simple way to model dissipation that is concentrated toward the bottom of the fluid layer but not as strongly localized as an exponential profile.

### Quadratic dissipation

The quadratic profile falls off more steeply with height:

```math
D(z) \propto (1 - z/H_R)^2
```

This concentrates dissipation even more strongly toward the base of the layer. It can approximate cases where turbulence is primarily generated near the boundary and decays rapidly away from it.

### Dynamic dissipation

The dynamic profile introduces a depth-dependent mixing length

```math
\ell_{\text{mix}} = \min(z, H_R)
```

so that the dissipation scale adjusts with distance from the boundary.

This mimics mixing-length arguments commonly used in geophysical and astrophysical fluid dynamics, where turbulence intensity depends on the available eddy size.

### Choosing a dissipation profile

Since the physical location of tidal energy dissipation is uncertain in many systems, these profiles should be viewed as parameterized hypotheses. Comparing results across multiple profiles is often more informative than adopting a single choice.

In practice:

| Profile     | Physical assumption                           |
| ----------- | --------------------------------------------- |
| Uniform     | Energy dissipated evenly throughout the layer |
| Exponential | Strong boundary-layer dissipation             |
| Linear      | Mild bottom-enhanced dissipation              |
| Quadratic   | Strongly localized bottom dissipation         |
| Dynamic     | Mixing-length controlled turbulence           |


---