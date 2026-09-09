
### Reference (3)

# Solid-Phase - solid1d-mush

The `solid1d-mush` model also uses the shooting method approach and is in many regards identical to the original **Love.jl** model by Hamish Hay (see [original repository](https://github.com/hamishHay/Love.jl)). In addition to the `solid1d` model, it includes the effects of porosity making it a **poro-viscoelastic model**. 

The governing equations are an extension of the `solid1d` model. The additional complexity is handled through two extra variables: the 7th and 8th $y$-functions, representing **pore pressure** and **Darcy flux** (relative tangential displacement), respectively.

### Porous Transitions and Elementary Solutions
The solver initially treats the system as a $6 \times 6$ problem. When it encounters a porous layer, it updates the elementary solutions by embedding the $3 \times 1$ coefficient set into a $4 \times 1$ matrix and adding a fourth elementary solution vector, $\pmb{y}^{(4)}_n(r_i)$.

At the initial solid-to-porous transition ($r_i$), the new elementary vector is defined as:

$$\pmb{y}^{(4)}_{n,m}(r_i) = (0, 0, 0, 0, 0, 0, 1, 0)^T$$

### Continuity and Source Terms
By turning back to the governing differential equation:

$$\frac{d\pmb{y}_{n,m}(r)}{dr} = \pmb{A}_n (r) \pmb{y}_{n,m}(r) - \pmb{f}_{n,m}(r)$$

We impose continuity across an infinitesimally thin layer ($\Delta r = 0$) where $\pmb\Pi_n(r_{i}) = \pmb{1}$. We allow $\pmb{y}_{n,m}(r_i^+)$ to be an $8 \times 1$ vector and $\pmb{y}_{n,m}(r_i^-)$ to be $6 \times 1$. To preserve continuity while introducing pore pressure, we include a source term $\pmb{f}_{n,m}(r_i^+)$:

$$\begin{pmatrix} \pmb{1}_{(8\times8)} \end{pmatrix} \pmb{y}_{n,m}(r_i^+) - \pmb{f}_{n,m}(r_i^+) = \begin{pmatrix} \pmb{1}_{(8\times6)} \\ \pmb{0}_{(2\times6)} \end{pmatrix} \pmb{y}_{n,m}(r_i^-)$$

Since the elementary solutions are linearly independent, the modal solution at the interface is:

$$\pmb{y}_n(r_i^+) = a_1 \mathbf{y}^{(1)}(r_i^+) + a_2 \mathbf{y}^{(2)}(r_i^+) + a_3 \mathbf{y}^{(3)}(r_i^+) + a_4 \mathbf{y}^{(4)}(r_i^+)$$

$$\pmb{y}_{n,m}(r_i^+) = \{ \text{solution of } 6\times 6 \text{ system} \} + (0, 0, 0, 0, 0, 0, a_4, 0)^T$$

To preserve continuity, the source term must exactly cancel the introduced pressure:

$$\pmb{f}_{n,m}(r_i^+) = (0, 0, 0, 0, 0, 0, a_4, 0)^T$$

Understanding these source/sink steps is key for including effects like porosity or seismic activity in the relaxation-based solver.

---

### Removing Porous Functions
If the medium becomes solid again before the surface, we reverse the process. We use a sink term $\pmb{f}_{n,m}(r_i^-)$ at the upper interface to remove excess pore pressure:

$$\begin{pmatrix} \pmb{1}_{(8\times6)} \\ \pmb{0}_{(2\times6)} \end{pmatrix}^T \pmb{y}_{n,m}(r_i^+) = \begin{pmatrix} \pmb{1}_{(8\times8)} \end{pmatrix} \pmb{y}_{n,m}(r_i^-) - \pmb{f}_{n,m}(r_i^-)$$

We must constrain the **Darcy flux to be zero** at the interface (upper boundary condition). While a sink term can remove excess pore pressure, it cannot "force" a boundary condition like zero Darcy flux if the propagation has already diverged. Instead, we find:

$$\pmb{f}_{n,m}(r_i^-) = (0, 0, 0, 0, 0, 0, y_7(r_i^-), 0)^T$$

A simpler numerical approach is to manually set $y_7 = 0$ for all $r > r_i^-$, as it no longer interacts with the $6 \times 6$ system.

---

### Final Solution
The remaining steps are identical to the `solid1d` model:
1. Propagate the elementary solution vectors ($8 \times 1$ or $6 \times 1$) to the surface.
2. Determine the coefficients $\pmb{C}$ (now including $a_4$ for the porous component).
3. Apply $\pmb{C}$ to obtain the final modal solution.

---

### Function Documentation

```@docs
Obliqua.run_solid1d_mush
```