
### Reference (6)

# Surface Loading

### Total ``k_{2}``

To calculate the response of a planet to various external forcings, we define general boundary conditions at the surface ($r = R$). These conditions account for external gravitational potentials, mass loading, tangential tractions, and surface pressure.

### General Boundary Conditions
For a given degree $n$, the boundary conditions for the state vector components $y_i$ are summarized in the following table:

| State Function | External Potential ($U$) | Mass Load ($\zeta_n$) | Traction ($\tau$) | Pressure ($P$) |
| :--- | :--- | :--- | :--- | :--- |
| **$y_3(R)$** (Normal Stress) | $0$ | $-g_e \zeta_n$ | $0$ | $-P_n$ |
| **$y_4(R)$** (Tangential Stress) | $0$ | $0$ | $\tau_n$ | $0$ |
| **$y_6(R)$** (Potential Stress) | $\frac{2n+1}{R} U_n$ | $4\pi G \zeta_n$ | $0$ | $0$ |

No $y_5$ term appears in the third row: in Obliqua's convention $y_6$ is already Takeuchi & Saito's (1972) combined "potential stress" variable, whose own radial equation has no coupling back to $y_5$, so the surface condition is a direct statement about $y_6(R)$ alone (see [Solid-Phase](@ref) for the underlying motion matrix).

By expressing a surface mass load $\zeta_n$ as an equivalent external potential $U'$, where $\zeta_n = \frac{2n + 1}{4 \pi G R} U'_n$, the system simplifies to:

$$\begin{aligned}
y_{3}(R) &= - \frac{(2n + 1)g_e}{4 \pi G R} U'_n - P_n \\
y_{4}(R) &= \tau_n \\
y_6(R) &= \frac{2n+1}{R} (U_n + U'_n)
\end{aligned}$$

Note that Obliqua's actual `get_surface_bc!` (`src/common.jl`) does not literally apply this $\zeta_n \leftrightarrow U'_n$ conversion; it sets $(U,U',\tau,P)$ directly as dimensionless $0$/$1$ selector flags, which for the load case numerically works out to $y_3(R) = -(2n+1)g(R)/(4\pi R^2)$ and $y_6(R) = (2n+1)G/R^2$ — see [Solid-Phase - solid1d](@ref) for the concrete tidal/load values the code actually produces.

### Calculation of Love Numbers
In `Obliqua`, Love numbers are non-dimensionalized by setting the forcing terms to either $1$ (present) or $0$ (absent).
*   **Tidal Love Number (TLN):** Calculated by setting $(U, U', \tau, P) = (1, 0, 0, 0)$.
*   **Load Love Number (LLN):** Calculated by setting $(U, U', \tau, P) = (0, 1, 0, 0)$.

Setting $U=1$ allows us to determine the intrinsic response of the planet; the final physical solution is obtained by scaling these non-dimensional Love numbers by the actual tidal potential $U_{n,m,k}$ of the system.

---

### Global Potential and Layer Coupling
The total distortion potential of the planet, $U_n^D$, can be constructed by combining the responses to different forcings:

$$U_n^D = k_n^T + (1+k_n^L)U_n^L + K_n^P U_n^P$$

Where:
*   $U_n^L$: Potential due to surface mass-loading (e.g., a magma ocean or ice sheet).
*   $U_n^P$: Potential due to surface pressure (e.g., atmospheric loading).

#### Current Implementation
Currently, `Obliqua` couples the solid and fluid responses to calculate a **Global Tidal Love Number** $k_n$ using the following recursive form:

$$k_n = k^T_n + (1 + k^L_n) k_n^{T, \text{(fluid)}}$$

In this formulation:
1.  $k^T_n$ and $k^L_n$ are the tidal and load Love numbers of the **solid mantle** only.
2.  $k_n^{T, \text{(fluid)}}$ represents the response of the fluid layer.
3.  The term $(1 + k^L_n)$ accounts for how the solid interior deforms under the weight of the fluid's own tidal response.

While additional corrections for atmospheric pressure or specialized crustal rheologies (Andrade/Maxwell) are not yet active, the framework is designed to incorporate these by calculating the respective pressure and loading Love numbers which are already implemented in the solver.

---

### Function Documentation

```@docs
Obliqua.solid1d.common.get_surface_bc!
```