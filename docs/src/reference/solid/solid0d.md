
### Reference (3)

# Solid-Phase - solid0d

The mechanical response of a planetary mantle with radially varying properties is approximated by a single effective complex shear modulus, $\bar{\mu}^*$. To account for the competition between uniform strain (Voigt limit) and uniform stress (Reuss limit) within a spherical segment, we employ the **Hill Average**. 

### The Hill Average
Given a radial profile of the complex shear modulus $\mu^*(r)$ and radial shells defined by $r_i$, the volume of each shell $V_i$ and its corresponding volume fraction $f_i$ are given by:

$$V_i = \frac{4}{3}\pi \left( r_{i+1}^3 - r_i^3 \right), \quad f_i = \frac{V_i}{\sum V_i}$$

The effective modulus $\bar{\mu}^*$ is the arithmetic mean of the Voigt average ($\mu_V$) and the Reuss average ($\mu_R$):

$$\mu_V = \sum_{i} f_i \mu_i^*, \quad \mu_R = \left( \sum_{i} \frac{f_i}{\mu_i^*} \right)^{-1}$$

$$\bar{\mu}^* = \frac{1}{2} \left( \mu_V + \mu_R \right)$$

---

### Dimensionless Rigidity
For a homogeneous incompressible sphere of radius $R$ and total mass $M$, the tidal ($k_n^T$) and load ($k_n^L$) Love numbers of degree $n$ are determined by the dimensionless rigidity $\bar{\mu}_n$. The scaling parameter $A_n$, which relates the elastic restoring force to the self-gravitational stability of the body, is defined as:

$$A_n = \frac{4(2n^2 + 4n + 3)}{3n G M^2} \pi R^4$$

We define the effective dimensionless complex rigidity for the segment as:

$$\mu^*_n = A_n \bar{\mu}^*$$

---

### Complex Love Numbers
The complex Love numbers are calculated using the transfer function factor $\mathcal{F} = (1 + \mu^*_n)^{-1}$. 

#### Tidal Love Number ($k_n^T$)
The tidal Love number represents the potential modification due to an external body. For $n > 1$:

$$k_n^T = \frac{1}{1 + \mu^*_n} \left( \frac{3}{2(n - 1)} \right)$$

#### Load Love Number ($k_n^L$)
The load Love number represents the response to a surface mass load:

$$k_n^L = -\frac{1}{1 + \mu^*_n}$$

---

### Function Documentation

```@docs
Obliqua.solid0d.mean_cmu
Obliqua.solid0d.compute_solid_lovenumbers
Obliqua.run_solid0d
```