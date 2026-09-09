
### Reference (7)

# Tidal potentials

### Tidal Dissipation

To determine the total tidal heating, `Obliqua` loops over all $(n,m,k)$ pairs and calculates heating rates based on general eccentric expressions. While the model currently assumes **co-planarity** (precluding obliquity tides), it fully accounts for eccentricity through Hansen coefficients.

### Tidal Potential and Normalization
For every triplet $(n, m, k)$, we calculate the Hansen coefficient $X_k^{-(n+1),m}(e)$ and the normalization factor $A_{n,m,k}$:

$$A_{n,m,k} = 2\sqrt{\frac{4\pi}{2n+1}\frac{(n-m)!}{(n+m)!}} P_n^m(0) X_k^{-(n+1),m}(e)$$

(When `Obliqua`'s adaptive mode enumeration restricts the loop to $m \geq 0$ to exploit the $\pm m$ symmetry of the problem, it applies additional $m=0$ correction factors to the implementation of this sum so that the $m=0$ terms are not double-counted. Those factors are a bookkeeping detail of that loop-restriction optimization, not part of the definition of $A_{n,m,k}$ itself, and are omitted here.)

The associated tidal potential $U_{n,m,k}$ is defined as:

$$U_{n,m,k} = \frac{GM}{a} \left(\frac{R}{a}\right)^n A_{n,m,k}(e)$$

---

### Heating Rates
The "black box" tidal models return an unnormalized heating profile $\tilde{H}(r, \sigma)$. To obtain the physical heating profile $H(r, \sigma)$ at a specific forcing frequency $\sigma$, we scale the output by the square of the tidal potential:

$$H(r, \sigma) = \tilde{H}(r, \sigma) \times |U_{n,m,k}|^2$$

The **global heating rate** $H(\sigma)$ for a given frequency is determined using the imaginary part of the tidal Love number $k_n(\sigma)$:

$$H(\sigma) = \text{prefactor} \times (-\text{Im}[k_n(\sigma)]) \times |U_{n,m,k}|^2$$

where the energy dissipation prefactor is:

$$\text{prefactor} = \frac{(2n + 1)R}{8\pi G} \sigma$$

---

### Spectral Summation
Since the tidal forcing consists of a discrete set of frequencies, the total heating is determined by summing across the forcing frequency domain:

*   **Radial Heating Profile:** $H(r) = \sum_{\sigma} H(r, \sigma)$
*   **Total Global Heating:** $H = \sum_{\sigma} H(\sigma)$

> **Consistency Check:** The volume integration of the radial profile $H(r)$ should match the global heating rate $H$, provided the surface load Love number $k'_n(\sigma)$ remains small compared to unity ($k'_n \ll 1$).

### Outputs for Orbital Dynamics
In addition to the heating rates, `Obliqua` returns the complex tidal Love numbers $k_n(\sigma)$ and their corresponding forcing frequencies $\sigma$. these are required for calculating tidal torques and the long-term orbital evolution of the system.

---

### Function Documentation

```@docs
Obliqua.run_tides
```