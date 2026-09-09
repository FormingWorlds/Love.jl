
### Reference (3)

# Solid-Phase - solid1d

The `solid1d` model uses the shooting method approach and is in many regards identical to the original **Lovepy** model (now called **Love.jl**) by Hamish Hay (see [original repository](https://github.com/hamishHay/Love.jl)). 

The structure of the model is as follows: 
1. We impose the Core-Mantle Boundary (CMB) condition (three elementary solutions). 
2. We solve the system at each radius:

   $$\frac{d\pmb{y}_{n,m}(r)}{dr} = \pmb{A}_n (r) \pmb{y}_{n,m}(r) - \pmb{f}_{n,m}(r)$$

   stepping (shooting) from the CMB to the surface. 

For this model, we assume $\pmb{f}_{n,m}(r) = \pmb{0}$, as porosity effects are not included. 

### The Propagator Matrix
A convenient way to treat the state vectors $\pmb{y}_{n,m}(r)$ is to define the propagator matrix $\pmb\Pi_n(r)$ (a $6 \times 6$ matrix) that takes the state vector at $r_i$ to $r_{i+1}$:

$$\pmb{y}_{n,m}(r_{i+1}) = \pmb\Pi_n(r_{i}) \pmb{y}_{n,m}(r_{i})$$

Using the **RK4 (Runge-Kutta 4th Order)** method, we have:

$$\pmb\Pi_n(r_{i}) = \pmb{1} + \frac{1}{6} \left(\pmb{K}_1 + 2\pmb{K}_2 + 2\pmb{K}_3 + \pmb{K}_4 \right)$$

where:

$$\begin{aligned}
    \pmb{K}_1 &= \Delta r_i \pmb{A}_n (r_i) \\
    \pmb{K}_2 &= \Delta r_i \pmb{A}_n (r_i + \Delta r_i / 2) \left[ \pmb{1} + \frac{1}{2} \pmb{K}_1 \right] \\
    \pmb{K}_3 &= \Delta r_i \pmb{A}_n (r_i + \Delta r_i / 2) \left[ \pmb{1} + \frac{1}{2} \pmb{K}_2 \right] \\
    \pmb{K}_4 &= \Delta r_i \pmb{A}_n (r_i + \Delta r_i) \left[ \pmb{1} +  \pmb{K}_3 \right] 
\end{aligned}$$

---

### Shooting vs. Relaxation
It is worth highlighting that this method purely propagates the solution vectors to the surface. Effectively, we attempt to solve the two-point **Boundary Value Problem (BVP)** by rewriting it as an initial value problem and imposing a constraint at the end. 

*   **Shooting Method:** The propagator matrix $\pmb\Pi_n(r)$ has no knowledge of the surface boundary until it arrives there, which means it may diverge before the boundary conditions can be applied. 
*   **Relaxation Method:** Solves the state vectors globally based on both boundary conditions simultaneously, making it significantly more stable.

---

### Numerical Integration
Imposing continuity of the propagator (implying the solution itself is continuous):

$$\pmb \Pi_n (r_i^+,r') = \pmb \Pi_n (r_i^-,r')$$

And imposing CMB conditions in the general solution:

$$\pmb y_{n,m}(r_C^+) = \pmb y_0 = \pmb I_C \pmb C$$

We find:

$$\pmb y_{n,m}(r) = \pmb \Pi_n (r,r_C^+) \pmb I_C \pmb C$$

We iterate through the mantle, repeatedly multiplying the elementary solutions with the propagator matrix until we reach the surface ($R = a^-$).

### Surface Boundary Conditions
At the surface, we impose the boundary conditions on the upper components of the $y$-functions using a projector $\pmb P_1$:

$$\pmb P_1 \pmb y (a^-) = 
\begin{pmatrix}
y_3(a^-) \\
y_4(a^-) \\
y_6(a^-)
\end{pmatrix} = 
\pmb P_1 \left( \pmb\Pi_n (a^-, r_C^+) \pmb I_C \pmb C \right) = \pmb b$$

where $\pmb b$ is the 3-vector composed of the RHS of the surface boundary conditions. No explicit $y_5$ term is needed in the third row because $y_6$ is already the combined "potential stress" variable of Takeuchi & Saito (1972) — its own radial equation, $dy_6/dr = (n-1)y_6/r + \ldots$, carries no coupling to $y_5$, unlike the raw potential-gradient definition. For unit-amplitude forcing, $\pmb b$ takes the classical free-surface values

$$\pmb b_{\text{tidal}} = \begin{pmatrix} 0 \\ 0 \\ \dfrac{2n+1}{a} \end{pmatrix}, \qquad \pmb b_{\text{load}} = \begin{pmatrix} -\dfrac{(2n+1)g(a)}{4\pi a^2} \\ 0 \\ \dfrac{(2n+1)G}{a^2} \end{pmatrix}$$

for the tidal and load Love-number problems respectively. This allows for the determination of the coefficient vector $\pmb C$, which is used to combine the three elementary solutions into the true modal solution.

---

### Function Documentation

```@docs
Obliqua.run_solid1d
```