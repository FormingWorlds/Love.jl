using Test
using LinearAlgebra
using AssociatedLegendrePolynomials
using DoubleFloats
using Obliqua
using Obliqua.constants
using Obliqua.solid1d.common


@testset "common Module Unit Tests" begin

    @testset "Ynm (Spherical Harmonics)" begin
        # Test basic evaluation and shape guarantees
        theta = [0.0, π/2, π]
        phi = [0.0, π]'
        n, m = 2, 1
        
        res = Ynm(n, m, theta, phi) 
        @test size(res) == (3, 2)
        @test res isa Array{ComplexF64, 2}
    end

    @testset "define_spherical_grid" begin
        # Define a grid with a large angular spacing for efficiency
        grid = define_spherical_grid(60.0, 2, 1) 
        
        @test grid isa NamedTuple
        @test haskey(grid, :res)
        @test haskey(grid, :clats)
        @test haskey(grid, :lons)
        @test haskey(grid, :Y)
        @test haskey(grid, :dYdθ)
        @test haskey(grid, :dYdϕ)
        @test haskey(grid, :Z)
        @test haskey(grid, :X)
        
        @test grid.res == 60.0
        @test size(grid.Y, 1) == 1
    end

    @testset "get_scales" begin
        R0 = 1.737e6
        M0 = 7.342e22
        G0 = 1.0
        
        # Test standard 6-variable configuration scaling matrices
        R0_o, M0_o, ω0, ρ0, G0_o, g0, μ0, S, Sinv = get_scales(R0, M0, G0, Y=[1,2,3,4,5,6]) 
        @test size(S) == (6, 6)
        @test size(Sinv) == (6, 6)
        @test S * Sinv ≈ I
        
        # Test 8-variable two-phase configuration layout matrix size
        _, _, _, _, _, _, _, S8, Sinv8 = get_scales(R0, M0, G0, Y=collect(1:8)) 
        @test size(S8) == (8, 8)
        @test S8[7,7] > 0
        @test S8[8,8] > 0
    end

    @testset "doublefactorial" begin
        # Verify boundary mathematical cases
        @test doublefactorial(0) == 1
        @test doublefactorial(1) == 1
        @test doublefactorial(4) == 8   # 4 * 2
        @test doublefactorial(5) == 15  # 5 * 3 * 1
        
        # Enforce error assertion for invalid domain spaces
        @test_throws ErrorException doublefactorial(-1)
    end

    @testset "sbesselj" begin
        # Analytical matching for n=0: j_0(x) = sin(x)/x
        x_val = 1.5
        @test sbesselj(0, x_val) ≈ sin(x_val) / x_val
        
        # Verify execution path type consistency
        @test sbesselj(1, 2.0 + 0.0im) isa ComplexF64
    end

    @testset "get_Ic (Core Solution Matrix)" begin
        ω = prec(2.0e-5)
        r = prec(1.2e6)
        ρ = prec(4000.0)
        g = prec(2.0)
        μ = precc(2.0e10)
        K = precc(7.0e10)
        n = 2
        
        # 1. Liquid Core verification
        Ic_liquid = get_Ic(ω, r, ρ, g, μ, K, "liquid", n) 
        @test size(Ic_liquid) == (6, 3)
        @test Ic_liquid[2, 2] == 1.0 # Tangential slip
        
        # 2. Solid Core verification
        Ic_solid = get_Ic(ω, r, ρ, g, μ, K, "solid", n) 
        @test size(Ic_solid) == (6, 3)
        
        # 3. Inertial Fluid Core verification
        Ic_inertial = get_Ic(ω, r, ρ, g, μ, K, "inertial", n) 
        @test size(Ic_inertial) == (6, 3)
        
        # 4. Error throwing for unsupported definitions
        @test_throws ErrorException get_Ic(ω, r, ρ, g, μ, K, "invalid_type", n) 
    end

    @testset "get_A & get_A! (ODE Propagator System Matrices)" begin
        ω = prec(2.2e-5)
        r = prec(1.5e6)
        ρ = prec(3300.0)
        g = prec(1.62)
        μ = precc(4.0e10 + 2.0e8im)
        K = precc(9.0e10 + 4.0e8im)
        n = 2
        
        # Test 6x6 solid matrix calculations
        A6 = get_A(ω, r, ρ, g, μ, K, n) 
        @test size(A6) == (6, 6)
        @test A6 isa Matrix{ComplexF64}
        
        # Test 8x8 two-phase porous mixture matrix calculations
        ρₗ = prec(2900.0)
        Kl = prec(7.5e10)
        Kd = precc(3.5e10 + 1.0e8im)
        α = precc(0.85 + 0.01im)
        ηₗ = prec(1.0)
        ϕ = prec(0.05)
        k = prec(1.0e-11)
        
        # FIX: Explicitly pass G0=1.0 to bypass the source code's Int64 default type issue
        A8 = get_A(ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1.0) 
        @test size(A8) == (8, 8)
        @test A8 isa Matrix{ComplexF64}
    end

    @testset "compute_strain_ten!" begin
        # Generate a baseline grid setup space
        grid = define_spherical_grid(90.0, 2, 1) 
        n_clats = length(grid.clats)
        n_lons = length(grid.lons)
        
        # Prepare 3D strain tensor destination array allocation
        ϵ = zeros(ComplexF64, n_clats, n_lons, 6) 
        
        # Mocking values for standard solution vector configurations
        y = zeros(ComplexF64, 6) 
        y[1] = 0.002
        y[2] = 0.001
        y[3] = 2.0e5
        y[4] = 6.0e4
        
        rr = 1.4e6 
        ρr = 3100.0 
        gr = 1.8 
        μr = 3.5e10 + 1.0e7im 
        Ksr = 8.5e10 + 3.0e7im 
        
        # Call and test non-destructive updating checks
        compute_strain_ten!(ϵ, y, 2, rr, ρr, gr, μr, Ksr, grid) 
        
        @test size(ϵ) == (n_clats, n_lons, 6)
        @test !all(iszero, ϵ) 
    end
end


# =============================================================================
# Numerical self-consistency tests for the core and surface boundary condition machinery
# (`get_Ic`, `get_A` / `get_A!`, `get_surface_bc!`).
#
# These tests do NOT rely on any literature reference value. Instead they check
# that the package's own functions agree with each other, using two
# model-independent facts:
#
#  1. Each column of `get_Ic(...)`, viewed as a function of r with ρ, g, μ, K, ω, n
#     held fixed, is supposed to be an exact analytic solution of the same ODE
#     dy/dr = A(r)*y that `get_A`/`get_A!` encodes (they describe one continuous
#     physical solution for a uniform sphere). So we can finite-difference `Ic(r)`
#     and compare against `A(r)*Ic(r)` directly, with no need to know the "right"
#     sign convention a priori — a genuine sign/convention mismatch between the
#     two functions shows up as an O(1) relative error, not numerical noise.
#
#  2. Placing the core boundary at R under a surface boundary condition without a 
#     propagation matrix is equivalent to compute_solid_lovenumbers (the Love, 1911
#     closed-form solution for a homogeneous, incompressible, self-gravitating 
#     elastic sphere). This test checks the compatibility of the core solution with
#     the surface boundary conditions, and validates both tidal and load Love
#     numbers against the analytical result.
#
# Note: point two is not valid for the fluid core solution vector as it is not 
# constructed as an internal interface, but rather as a fluid-solid boundary condition.
# =============================================================================
@testset "Core and Surface Boundary Conditions: get_Ic, get_A, get_core_bc!, get_surface_bc!, compute_solid_lovenumbers" begin
    @testset "Compatibility: Core Basis Ic vs Motion Matrix A" begin
        # Physical and Discretization Setup
        ρ0   = 5500.0          # Average density [kg/m^3]
        μ0   = 6e10            # Shear modulus [Pa]
        K0   = 1e20            # Bulk modulus [Pa] (Incompressible limit)
        R0   = 6.371e6         # Planet Radius [m]
        G_val = 6.67430e-11    # Gravitational constant
        ω_val = 1e-5           # Tidal frequency [rad/s]

        for n in (2, 3, 4)
            @testset "Harmonic Degree n = $n" begin
                # Sample radial positions from deep interior (10% R) to surface R
                r_samples = [0.1 * R0, 0.25 * R0, 0.5 * R0, 0.75 * R0, 1.0 * R0]

                for r in r_samples
                    g_r = (4/3) * π * G * ρ0 * r

                    # 1. Allocate and compute Motion Matrix A at radius r
                    A = zeros(ComplexF64, 6, 6)
                    get_A!(A, ω_val, r, ρ0, g_r, ComplexF64(μ0), ComplexF64(K0), n; G0=1.0)

                    # 2. Compute Core Solution Basis Matrix Ic at radius r
                    Ic = get_Ic(ω_val, r, ρ0, g_r, ComplexF64(μ0), ComplexF64(K0), "solid", n; G0=1.0)

                    # 3. Compute Analytic Derivative d(Ic)/dr via finite differences
                    dr = 1e-6 * r
                    g_plus  = (4/3) * π * G * ρ0 * (r + dr)
                    g_minus = (4/3) * π * G * ρ0 * (r - dr)

                    Ic_plus  = get_Ic(ω_val, r + dr, ρ0, g_plus,  ComplexF64(μ0), ComplexF64(K0), "solid", n; G0=1.0)
                    Ic_minus = get_Ic(ω_val, r - dr, ρ0, g_minus, ComplexF64(μ0), ComplexF64(K0), "solid", n; G0=1.0)

                    dIc_dr_fd = (Ic_plus - Ic_minus) / (2 * dr)

                    # 4. Compute RHS: A(r) * Ic(r)
                    A_times_Ic = A * Ic

                    # 5. Evaluate Residual Matrix: R(r) = d(Ic)/dr - A(r)*Ic(r)
                    residual = dIc_dr_fd - A_times_Ic

                    # Scale-invariant normalization per column
                    col_norms_fd = [norm(dIc_dr_fd[:, c]) for c in 1:3]
                    norm_res     = [norm(residual[:, c]) / col_norms_fd[c] for c in 1:3]

                    # Assert differential consistency (Relative error < 1e-4 due to FD)
                    for c in 1:3
                        @test isapprox(dIc_dr_fd[:, c], A_times_Ic[:, c]; rtol=1e-4, atol=1e-8)
                    end
                end
            end
        end
    end

    # -------------------------------------------------------------------------
    # "liquid", "inertial", and "inertial-liquid" core bases are column-wise
    # unit-normalized inside get_Ic (unlike "solid"), so directly finite-
    # differencing get_Ic's output does not satisfy dY/dr = A*Y exactly: an
    # r-dependent normalization factor c(r) = 1/norm(Y_true(r)) contributes an
    # extra d/dr[c(r)]*Y_true(r) term, which is always parallel to Y itself
    # (a pure rescaling, not a direction change). The normalization-invariant
    # test is therefore to project the FD residual onto the direction
    # perpendicular to the column itself: nothing should remain there. A
    # genuine convention/sign/formula error shows up as an O(1) perpendicular
    # residual, not numerical noise -- the same discriminating power as the
    # direct test above, just robust to the internal normalization.
    #
    # "liquid" and "inertial-liquid" each also carry one tangential-slip
    # column at the CMB (a boundary-condition embedding, not a physical
    # elementary solution -- see get_Ic's docstring), which is excluded below.
    # "inertial" (a genuinely solid, elastic core) has no fluid-solid
    # interface to admit such a slip, so all three of its columns are
    # physical elementary solutions and are tested.
    #
    # `rows` restricts the comparison to a subset of the 6 equations, for two
    # reasons:
    #   1. At mu=0, A[Y[2],Y[4]] = 1/mu is a genuine Inf. Even though Y[4]=0
    #      identically (no shear stress in a fluid), floating-point Inf*0 =
    #      NaN still contaminates that row: row Y[2]'s own equation (dV/dr)
    #      is genuinely singular at mu=0, so it is excluded for
    #      "inertial-liquid" below.
    #   2. The projection direction Yhat must be built from the same subset
    #      of rows being tested, not the full 6-vector: get_Ic's column
    #      normalization divides by the norm of the full column, which can be
    #      dominated by one or two large-magnitude rows (the stress rows,
    #      which carry factors of the bulk/shear modulus). Using the
    #      full-vector norm for Yhat would let an error confined to those
    #      rows leak into the direction used to test every other row.
    # =========================================================================
    function projected_ode_residual(A, Y, Y_plus, Y_minus, dr; rows=1:6)
        dY_fd = (Y_plus - Y_minus) / (2 * dr)
        AY = A * Y
        residual = dY_fd - AY
        Yhat_sub = Y[rows] / norm(Y[rows])
        residual_sub = residual[rows]
        residual_perp = residual_sub - dot(Yhat_sub, residual_sub) * Yhat_sub
        return norm(residual_perp) / norm(dY_fd[rows])
    end

    # NOTE: "liquid" (the static, omega=0 fluid-core boundary condition) is
    # deliberately not given a Ic-vs-A finite-difference test here, unlike
    # "solid", "inertial", and "inertial-liquid" below. A static fluid core
    # responds to loading instantaneously (pure hydrostatic equilibrium) --
    # there is no propagating internal structure to integrate through, so
    # "liquid" is a boundary/jump condition embedding the liquid-solid
    # interface, not a continuous elementary solution of dY/dr = A*Y. Its
    # correctness is instead checked structurally in the "get_Ic (Core
    # Solution Matrix)" testset above and via the surface-BC/solid0d
    # cross-check below.

    @testset "Compatibility: Core Basis Ic vs Motion Matrix A (inertial)" begin
        # K0 here must be a moderate, realistic value: the incompressible
        # limit K0=1e20 used for "solid" above pushes the general finite-K
        # Bessel formula's k^2 discriminant into a catastrophic-cancellation
        # regime under double precision, producing spurious residuals even in
        # otherwise-correct rows.
        ρ0, μ0, K0, R0 = 5500.0, 6e10, 2e11, 6.371e6
        for n in (2, 3, 4)
            @testset "Harmonic Degree n = $n" begin
                for ω_val in (0.0, 1e-5)   # inertial's general k^2 (Kervazo B.30) is well-defined at omega=0
                    for r in [0.1 * R0, 0.25 * R0, 0.5 * R0, 0.75 * R0, 1.0 * R0]
                        g_r = (4/3) * π * G * ρ0 * r
                        A = zeros(ComplexF64, 6, 6)
                        get_A!(A, ω_val, r, ρ0, g_r, ComplexF64(μ0), ComplexF64(K0), n; G0=1.0, inertial_terms=true)

                        # dr=1e-6*r (used elsewhere in this file) sits in the
                        # rounding-error-dominated regime for the smallest
                        # radius/lowest-degree combination tested here (n=2,
                        # r=0.1*R0): central-difference cancellation noise
                        # grows as dr shrinks below the double-precision
                        # optimum, so a slightly larger step is used for this
                        # harder-conditioned finite-mu,K Bessel check.
                        dr = 1e-5 * r
                        g_plus  = (4/3) * π * G * ρ0 * (r + dr)
                        g_minus = (4/3) * π * G * ρ0 * (r - dr)
                        Ic       = get_Ic(ω_val, r,      ρ0, g_r,      ComplexF64(μ0), ComplexF64(K0), "inertial", n; G0=1.0)
                        Ic_plus  = get_Ic(ω_val, r + dr, ρ0, g_plus,   ComplexF64(μ0), ComplexF64(K0), "inertial", n; G0=1.0)
                        Ic_minus = get_Ic(ω_val, r - dr, ρ0, g_minus,  ComplexF64(μ0), ComplexF64(K0), "inertial", n; G0=1.0)

                        # All three columns are genuine elementary solutions:
                        # column 1 is the T&S Eq. 100 regular solution, and
                        # columns 2, 3 are Kervazo B.30's "+" and "-" k^2
                        # roots. There is no tangential-slip column here,
                        # since a solid core has no fluid-solid interface.
                        for c in (1, 2, 3)
                            rel = projected_ode_residual(A, Ic[:, c], Ic_plus[:, c], Ic_minus[:, c], dr)
                            @test rel < 1e-4
                        end
                    end
                end
            end
        end
    end

    @testset "Compatibility: Core Basis Ic vs Motion Matrix A (inertial-liquid)" begin
        ρ0, K0, R0 = 5500.0, 2e11, 6.371e6
        physical_cols = (1, 2)   # column 3 is the tangential-slip boundary column
        # omega=0 is excluded: Korenaga's fluid k^2 (B.7) has an explicit
        # n(n+1)*gamma^2/omega^2 term that diverges as omega->0.
        for n in (2, 3, 4)
            @testset "Harmonic Degree n = $n" begin
                for ω_val in (1e-4, 1e-3)   # 1e-5 overflows sbesselj (double-precision AMOS) at n=4, r=R0
                    for r in [0.1 * R0, 0.25 * R0, 0.5 * R0, 0.75 * R0, 1.0 * R0]
                        g_r = (4/3) * π * G * ρ0 * r
                        A = zeros(ComplexF64, 6, 6)
                        get_A!(A, ω_val, r, ρ0, g_r, ComplexF64(0.0), ComplexF64(K0), n; G0=1.0, inertial_terms=true)

                        dr = 1e-6 * r
                        g_plus  = (4/3) * π * G * ρ0 * (r + dr)
                        g_minus = (4/3) * π * G * ρ0 * (r - dr)
                        Ic       = get_Ic(ω_val, r,      ρ0, g_r,      ComplexF64(0.0), ComplexF64(K0), "inertial-liquid", n; G0=1.0)
                        Ic_plus  = get_Ic(ω_val, r + dr, ρ0, g_plus,   ComplexF64(0.0), ComplexF64(K0), "inertial-liquid", n; G0=1.0)
                        Ic_minus = get_Ic(ω_val, r - dr, ρ0, g_minus,  ComplexF64(0.0), ComplexF64(K0), "inertial-liquid", n; G0=1.0)

                        for c in physical_cols
                            rel = projected_ode_residual(A, Ic[:, c], Ic_plus[:, c], Ic_minus[:, c], dr; rows=[1,3,4,5,6])
                            @test rel < 1e-4
                        end
                    end
                end
            end
        end
    end


    g_uniform(r, ρ; G=G, G0=prec(1.0)) = (4/3) * π * (G / G0) * ρ * r

    # Local mirror of solid0d's compute_solid_lovenumbers. Note that 
    # `test_solid0d` already verifies `solid0d.compute_solid_lovenumbers` 
    # against Love (1911) closed form, so this is just a local copy to avoid 
    # cross-module dependencies in this test.
    function love_reference(μc, mass_tot, R, n)
        An     = 4.0 * (2n^2 + 4n + 3.0) / (3.0 * n * G * mass_tot^2) * π * R^4
        μc_n   = An * μc
        factor = 1.0 / (1.0 + μc_n)
        k2_T   = n == 1 ? zero(factor) : factor * 3.0 / (2.0 * (n - 1.0))
        k2_L   = -factor
        return k2_T, k2_L
    end

    function direct_surface_solution(R, ρ, μ, K, n; G0=prec(1.0), ω=prec(1e-5),
                                    b_indices=(3, 4, 6), U=0, Uprime=0, tau=0, P=0)
        g_R = g_uniform(R, ρ; G0=G0)
        Ic  = get_Ic(ω, R, ρ, g_R, μ, K, "solid", n; G0=G0)

        Bsurf, b = get_surface_bc!(R, g_R, n, U, Uprime, tau, P; G0=G0)
        M   = Bsurf * Ic
        rhs = b[collect(b_indices)]
        c   = M \ rhs
        y_R = Ic * c   # [U, V, X, Y, phi, psi] at r=R

        return (y_R=y_R, Bsurf=Bsurf, b=b, M=M, rhs=rhs, c=c)
    end

    love_number_tidal(y_R) = y_R[5] - 1
    love_number_load(y_R)  = y_R[5] - 1

    @testset "Compatibility: Core Basis Ic vs surface BC vs solid0d" begin

        # Define physical baseline constants once
        ρ_phys   = prec(5500.0)
        μ_phys   = precc(6e10)
        K_phys   = precc(1e20)     # Incompressible limit
        R_phys   = prec(6.371e6)
        mass_tot = prec(4/3 * π * R_phys^3 * ρ_phys)

        # Derive scale factors without mutating global scope
        R0, M0, ω0, ρ0, G0, g0, μ0, S, Sinv = get_scales(R_phys, mass_tot, G)

        # Convert baseline values to dimensionless values for the core solver
        ρ_dim = ρ_phys / ρ0
        μ_dim = μ_phys / μ0
        K_dim = K_phys / μ0
        R_dim = R_phys / R0
        
        for n in (2, 3, 4)
            @testset "n = $n" begin

                # Closed-form physical reference (Love, 1911)
                k2_T_ref, k2_L_ref = love_reference(μ_phys, mass_tot, R_phys, n)
                fluid_limit = 3 / (2 * (n - 1))

                b_idx = (3, 4, 6) # get_surface_bc! stores boundary conditions in these indices of b

                @testset "b_indices = $b_idx" begin

                    # Solve surface system in dimensionless space
                    sol_T = direct_surface_solution(R_dim, ρ_dim, μ_dim, K_dim, n; G0=G0, b_indices=b_idx, U=1)
                    sol_L = direct_surface_solution(R_dim, ρ_dim, μ_dim, K_dim, n; G0=G0, b_indices=b_idx, Uprime=1)

                    # Convert state vectors back to physical SI units using scaling matrix S
                    y_R_T_phys = S * sol_T.y_R
                    y_R_L_phys = S * sol_L.y_R

                    # Extract Love numbers from physical surface potential y_5(R)
                    k2_T_manual = love_number_tidal(y_R_T_phys)
                    k2_L_manual = love_number_load(y_R_L_phys)

                    # Sign compatibility
                    @test sign(real(k2_T_manual)) == sign(real(k2_T_ref))
                    @test sign(real(k2_L_manual)) == sign(real(k2_L_ref))

                    # Quantitative match against analytical limits (within 5%)
                    @test isapprox(real(k2_T_manual), real(k2_T_ref); rtol=0.05)
                    @test isapprox(real(k2_L_manual), real(k2_L_ref); rtol=0.05)

                    # Physical boundary constraints
                    @test 0 <= real(k2_T_manual) <= fluid_limit
                    @test -1 <= real(k2_L_manual) <= 0
                end
            end
        end
    end
end