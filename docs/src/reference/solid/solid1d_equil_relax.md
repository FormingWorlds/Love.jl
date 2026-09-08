
### Reference (3)

# Solid-Phase (Equilibrium) - solid1d-equil-relax

Like `solid1d-relax`, the `solid1d-equil-relax` model avoids shooting-method complications by solving a global relaxation system in one pass. The key structural difference is that it propagates a reduced two-component state vector rather than the full six-component elastic solution. The governing matrix `get_A!` (following Saito 1974, Eq. 18) depends only on density, gravity, and degree $n$ — shear modulus $\mu$ and bulk modulus $K$ never enter it — so this model captures the gravitational-potential response of a body deforming in hydrostatic (equilibrium-tide) balance, rather than solving the coupled elastic stress-displacement system. 

### Numerical Formulation

The problem statement retains the same form as the elastic case, but with a $2\times2$ system matrix in place of the $6\times6$ one:

$$\frac{d\pmb{y}_{n,m}(r)}{dr} = \pmb{A}_n(r)\, \pmb{y}_{n,m}(r)$$

with

$$\pmb{A}(r) = \begin{pmatrix} \dfrac{4\pi G \rho}{g} - \dfrac{n+1}{r} & 1 \\[6pt] \dfrac{2(n-1)}{r}\dfrac{4\pi G \rho}{g} & \dfrac{n-1}{r} - \dfrac{4\pi G \rho}{g} \end{pmatrix}$$

Discretizing with the same second-order (trapezoidal) finite-difference scheme as before gives the identical relaxation form:

$$\pmb{C}_n \pmb{y}_n + \pmb{D}_{n+1}\pmb{y}_{n+1} = \pmb{0}, \qquad \pmb{C}_n = \pmb{I}_2 + \frac{\Delta r}{2}\pmb{A}_n, \quad \pmb{D}_{n+1} = -\pmb{I}_2 + \frac{\Delta r}{2}\pmb{A}_{n+1}$$

Because the elastic degrees of freedom drop out, $\pmb{y}$ carries only two entries instead of six.

---

### Boundary Conditions as Constraints

#### Lower Boundary (Core)
At the core boundary, `get_core_bc!` builds a $1\times2$ constraint row $B_1$ so that

$$B_1\, \pmb{y}(r_C^+) = 0$$

Rather than hand-deriving this row, it is obtained numerically as the left null space of the matrix $I_c$ of admissible interior solutions returned by `get_Ic`. Currently only the **liquid-core** case is implemented, with the single regular solution

$$I_c = \begin{pmatrix} r^n \\ 2(n-1)r^{n-1} \end{pmatrix}$$

`"solid"` and `"inertial"` core types are declared in the interface but raise an error if selected, so the constraint construction is presently liquid-core-only. An optional `patch` keyword can insert an infinitesimally thin solid shell just above the core (via placeholder $\mu$, $K$ values) to work around a known $y_2$–$y_4$ decoupling instability in fluid layers — though since $\mu, K$ do not appear in $\pmb{A}$, this patch's effect on the reduced system is limited to whatever downstream consumers of $\mu, K$ exist outside this reduced ODE.

#### Upper Boundary (Surface)
At the surface, `get_surface_bc!` is called twice — once for the tidal-forcing case and once for the load-forcing case — each returning its own $1\times2$ matrix $B_N$ and right-hand side $b$, together with directly-evaluated surface values $y_2$ and $y_6$ (obtained from a separate analytic relation rather than from the propagated 2-vector). The reduced system is then solved for the potential-related pair, and the classical surface relation

$$y_1 = \frac{y_2}{g\rho} + \frac{y_5}{g}$$

is used to recover $y_1$. The full six-component output vector is finally assembled as

$$\pmb{y} = (y_1,\ y_2,\ 0,\ 0,\ y_5,\ y_6)$$

with $y_3 = y_4 = 0$ imposed by construction, since no shear stresses are solved for in this reduced formulation.

---

### The Global System and Henyey Relaxation

The assembled system again has the block-tridiagonal Henyey structure, but with $2\times2$ blocks instead of $6\times6$:

* **Core step** (`core_boundary`): combines $B_1$ with the upper half of $C_1$ to form $S_1$, and initializes $R_1 = -S_1^{-1}Q_1$.
* **Propagation step** (`propagate_solid`): for each interior layer, carries forward the "stored" lower half-rows of $C_n$ and $D_{n+1}$ from the previous step to build $P_n$, $S_n$, $Q_n$, then updates
$$X_n = P_n R_{n-1} + S_n, \qquad R_n = -X_n^{-1}Q_n$$
* **Surface step** (`surface_boundary`): applies $B_N$ (separately for the tidal and load cases) in place of the interior $C_N$ upper half, solves $X_N y = b$ for the surface potential pair, and back-fills $y_1$ and the shear-free components.

The recursion logic is otherwise identical to the elastic $6\times6$ solver — only the block dimension changes, since the physics being relaxed is restricted to the potential equation rather than the full stress-displacement-potential system. Both a tidal ($y_t$) and a load ($y_l$) surface solution are produced from a single forward sweep, since the two cases share every $C_n$, $D_{n+1}$ block and differ only in the surface right-hand side $b$.

---