### Reference

This is the theory behind `Obliqua`: where the forcing comes from, how a
material responds to it, and how that response is actually solved for
across a planet's interior. Read it start to finish for the full story,
or jump straight to the model you're configuring — each page stands on
its own and links back to the ones it builds on.

#### 1. Setting the stage: what's forcing the tides?

A planet in an eccentric orbit feels tidal forcing at more than just one
frequency. Before any material response can be computed, we first need
to know *which* frequencies matter and *how strongly* each one drives
the response.

- [Forcing Frequency](@ref) — which harmonics and Fourier modes actually
  contribute, and why we can usually stop at $n=2$.
- [Tidal potentials](@ref) — normalizing the forcing itself, and turning
  a Love-number spectrum into a heating rate.

#### 2. How a material responds

- [Rheology](@ref) — Maxwell, Andrade, and the purely elastic limit: the
  three ways `Obliqua` lets shear and bulk moduli become
  frequency-dependent (and dissipative).

#### 3. The solid interior

The bulk of `Obliqua`'s machinery lives here: propagating the tidal
response equations through a radially-structured solid mantle, from a
homogeneous zeroth-order estimate up to a fully poro-viscoelastic,
core-inertia-aware solver.

- [Solid-Phase](@ref) — the shared theory: $y$-functions, the motion
  matrix $\pmb{A}_n(r)$, the core boundary matrix $\pmb{I}_C$ and its
  four `core` options, and the shooting vs. relaxation split.
- [Solid-Phase - solid0d](@ref) — the zero-dimensional, single-layer
  approximation. Fast, and a good sanity check for everything else.
- [Solid-Phase - solid1d](@ref) — radially resolved, shooting method.
- [Solid-Phase - solid1d-relax](@ref) — the same physics, but solved
  with a Henyey-style relaxation scheme instead — generally far more
  stable.
- [Solid-Phase - solid1d-mush](@ref) and
  [Solid-Phase - solid1d-mush-relax](@ref) — their poro-viscoelastic
  extensions, for a mantle with a partially molten layer.
- [Solid-Phase (Equilibrium) - solid1d-equil-relax](@ref) — the reduced,
  two-component equilibrium-tide limit used automatically at very low
  forcing frequencies.

#### 4. The mushy transition

- [Mush layer - interp](@ref) — a lightweight, purely interpolated
  treatment of dissipation across a thin mushy or transitional region,
  for when the full poro-viscoelastic machinery above is overkill.

#### 5. The fluid response

- [Liquid-Phase](@ref) — the Laplace tidal equations for a magma ocean
  or other fully fluid layer, and the family of radial dissipation
  profiles (including the energy-conserving `dynamic_interp` default)
  used to distribute that heating with depth.

#### 6. Reading out the answer

- [Surface Loading](@ref) — turning surface boundary conditions into
  tidal and load Love numbers, and how the solid and fluid contributions
  are currently combined into one global $k_n$.
