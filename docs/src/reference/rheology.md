
### Reference (2)

# Rheology

The frequency response of the mantle is computed assuming a homogeneous viscoelastic mantle. `Obliqua` implements two rheological models: Maxwell and Andrade. Unlike the Maxwell model, which captures only elastic–viscous behavior through a single Maxwell timescale, the Andrade rheology includes a transient anelastic component. This yields an elastic response at high forcing frequencies, a viscous response at low frequencies governed by the Maxwell timescale ``\tau_M = \eta / \mu``, and a smooth anelastic transition in between characterized by the Andrade timescale ``\tau_A`` and creep exponent ``\alpha``. As a result, the Andrade rheology reproduces the observed attenuation behavior of planetary mantles more accurately, particularly at high tidal forcing frequencies. The rheologies are incorporated via their definitions of the complex shear modulus. For the Andrade rheology, the complex shear modulus is

```math
\tilde{\mu}(\omega) =
\frac{\mu}{
1 + (\mathrm{i}\omega\eta/\mu)^{-\alpha}\Gamma(1+\alpha)
  + (\mathrm{i}\omega\eta/\mu)^{-1}}, \qquad \alpha = 0.3,
```

where ``\mu`` is the elastic shear modulus and ``\Gamma`` the Gamma function.

For the Maxwell limit ``\alpha = 0`` and 

```math
\tilde{\mu}(\omega) =
\frac{\mu}{1 + (\mathrm{i}\omega\eta/\mu)^{-1}}.
```

The same equations hold for the (drained) bulk modulus, just replace ``\mu`` with ``\kappa`` and ``\eta`` with ``\zeta``, with compaction viscosity

```math
    \zeta (\phi) \approx \frac{\eta (\phi)}{\phi}.
```

---