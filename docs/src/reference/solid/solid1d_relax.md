
### Reference (3)

# Solid-Phase - solid1d-relax

A significant advantage of the `solid1d-relax` model is that it eliminates the need for the cumbersome imposition of internal boundary conditions required by shooting methods. Instead, the relaxation method solves for the modal solution across the entire domain simultaneously.

### Numerical Formulation
We begin with the standard problem statement:

$$\frac{d\pmb{y}_{n,m}(r)}{dr} = \pmb{A}_n (r) \pmb{y}_{n,m}(r) - \pmb{f}_{n,m}(r)$$

Assuming no internal sources ($\pmb{f}_{n,m}(r) = \pmb{0}$), we approximate the system using second-order finite differences:

$$\pmb{y}_{n+1} − \pmb{y}_{n} = \frac{\Delta r}{2} (\pmb{A}_{n+1} \pmb{y}_{n+1} + \pmb{A}_n \pmb{y}_n)$$

Rearranging terms yields the fundamental relaxation equation:

$$\pmb{C}_n \pmb{y}_n + \pmb{D}_{n+1} \pmb{y}_{n+1} = \pmb{0}$$

where:
*   $\pmb{C}_n = \pmb{I} + \frac{\Delta r}{2} \pmb{A}_n$
*   $\pmb{D}_{n+1} = −\pmb{I} + \frac{\Delta r}{2} \pmb{A}_{n+1}$

---

### Boundary Conditions as Constraints
Unlike the shooting method, where we track elementary solutions, the relaxation scheme treats the Core-Mantle Boundary (CMB) and the surface as direct constraints on a single modal solution vector.

#### Lower Boundary (CMB)
At the CMB ($r = r_C^+$), we eliminate unknown coefficients to write the boundary condition as:

$$B_1 \mathbf{y}(r_C^+) = 0$$

where $B_1$ is a $3 \times 6$ matrix that enforces continuity. By setting the RHS to zero, we ensure the solution remains physically consistent at the interface without needing to manually restart the integration.

#### Upper Boundary (Surface)
Similarly, at the surface ($r = a^-$), we define the $3 \times 6$ matrix $B_N$:

$$B_N = \begin{pmatrix} 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{pmatrix}$$

This satisfies $B_N \mathbf{y} = b$, where $b$ contains the surface forcing terms (see [Solid-Phase - solid1d](@ref) for the explicit tidal/load values of $b$). No $y_5$ coefficient appears in the third row: $y_6$ is already the combined potential-stress variable, so matching to the exterior vacuum solution is a direct condition on $y_6(a^-)$ alone.

---

### The Global Matrix and Henyey Relaxation
Combining the difference equations and boundary conditions results in a large, banded global system:

$$\mathcal{H} \pmb{y} = \pmb{b}$$

Because $\mathcal{H}$ is typically $6N \times 6N$ (with $N \approx 2000$), direct inversion is too expensive. We use a **Henyey-type relaxation** to solve the system recursively. We define square $6 \times 6$ submatrices $P_n, S_n,$ and $Q_n$:

*   **$P_n$**: Lower half of $C_{n-1}$
*   **$S_n$**: Combined lower half of $D_n$ and upper half of $C_n$
*   **$Q_n$**: Upper half of $D_{n+1}$

The system is then solved in two passes:

1.  **Forward Sweep (Recursive Relations):**
    Calculate the response matrices $R_n$ from the CMB to the surface:

    $$R_n = -X_n^{-1} Q_n, \quad X_n = P_n R_{n-1} + S_n$$

    starting with $R_1 = -S_1^{-1} Q_1$.

2.  **Backward Sweep (Solution):**
    Determine the surface solution:

    $$y_N = X_N^{-1} b$$

    Then propagate the solution back down to the CMB:

    $$y_n = R_n y_{n+1}$$

> **Conceptual Analogy:** Imagine a string in a river tied to two endpoints. The river's flow (the physics/differential equations) dictates the string's curve, while the endpoints (boundaries) fix its absolute position. If a rock (internal boundary) is in the river, the string naturally bends around it. 

---

### The Grid
To resolve high gradients near the surface, we use an uneven exponential grid. The total number of nodes $N$ and their positions $r_i$ are determined by the ratio between the maximum interval $\Delta r_{\max}$ (at the CMB) and the minimum interval $\Delta r_{\min}$ (at the surface):

$$\alpha = \ln\left(\frac{\Delta r_{\max}}{\Delta r_{\min}}\right)$$

$$r_i = R_E + (R_c - R_E) \left( \frac{\exp\left( \alpha \frac{N - i}{N - 1} \right) - 1}{\exp(\alpha) - 1} \right), \quad i = 1, \ldots, N$$

---

### Function Documentation

```@docs
Obliqua.run_solid1d_relax
```
