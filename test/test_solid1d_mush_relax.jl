using Test
using LinearAlgebra
using Obliqua
using Obliqua.solid1d_mush_relax
using Obliqua.solid1d_mush_relax.common
using Obliqua.constants


@testset "solid1d_mush_relax Module Tests" begin

    # Define a realistic physical configuration for a planetary body (SI units)
    # This ensures N > 1 during grid resampling, preventing empty vectors.
    radius = [1000000.0, 1500000.0, 2000000.0] # Layer boundaries in meters
    rho    = [3000.0, 3500.0]                 # Density (kg/m^3)
    visc   = [1e21, 1e22]                     # Viscosity (Pa*s)
    shear  = complex.([1e10, 2e10])           # Complex shear modulus (Pa)
    bulk_s = complex.([5e10, 6e10])           # Solid bulk modulus (Pa)
    bulk_l = [4e10, 4.5e10]                   # Liquid bulk modulus (Pa)
    bulk_d = complex.([4.8e10, 5.5e10])       # Drained bulk modulus (Pa)
    alpha  = complex.([0.8, 0.9])             # Biot coefficient
    visc_l = [1.0, 2.0]                       # Liquid viscosity (Pa*s)
    phi    = [0.1, 0.05]                      # Porosity
    k      = [1e-12, 0.0]                     # Permeability (m^2) - Layer 1 is mushy, Layer 2 is solid
    m_core = 1e23                             # Core mass (kg)
    dr_min = 10000                            # Minimum spacing (10 km)
    dr_max = 50000                            # Maximum spacing (50 km)
    
    # Global physics parameters
    ω = 1e-4                                  # Angular frequency
    n = 2                                     # Tidal degree
    ρ_core = 7000.0                           # Core density
    μ_core = complex(0.0)                     # Liquid core shear modulus
    κ_core = complex(1e11)                    # Core bulk modulus
    M_tot  = 6e24                             # Approximate total mass target

    @testset "1. Grid Generation & Gravity Calculations" begin
        # Test profile resampling
        r_new_b, ρ_new, η_new, μ_new, κs_new, κl_new, κd_new, α_new, ηl_new, φ_new, k_new, g_new, M_tot_calc = 
            solid1d_mush_relax.resample_profiles(
                radius, rho, visc, shear, bulk_s, bulk_l, bulk_d, alpha, visc_l, phi, k, m_core, dr_min, dr_max
            )

        # Basic size and dimension validations
        N = length(r_new_b)
        @test N > 1
        @test length(ρ_new)  == N - 1
        @test length(μ_new)  == N - 1
        @test length(g_new)  == N - 1
        @test length(k_new)  == N - 1

        # Check boundary integrity
        @test r_new_b[1]   ≈ radius[1]
        @test r_new_b[end] ≈ radius[end]

        # Test the underlying gravity compiler routine directly
        g_test, M_enc_test = solid1d_mush_relax.get_g(r_new_b, ρ_new, m_core)
        @test length(g_test) == N - 1
        @test M_enc_test > m_core
    end

    @testset "2. Core Boundary Condition Drivers" begin
        # Generate non-dimensional profiles similar to solve_radial_system execution
        Nr = 5
        rs = collect(range(0.5, 1.0, length=Nr))
        ρs = fill(1.0, Nr)
        gs = fill(1.0, Nr)
        μs = fill(complex(1.0), Nr)
        Ks = fill(complex(2.0), Nr)
        
        # Additional mushy specific arrays
        ρₗs = fill(0.9, Nr)
        Kls = fill(1.5, Nr)
        Kds = fill(complex(1.2), Nr)
        αs  = fill(complex(0.8), Nr)
        ηₗs = fill(1.0, Nr)
        ϕs  = fill(0.1, Nr)
        ks  = fill(1e-4, Nr)

        # Allocate state vector integration arrays
        R_solid = [zeros(precc, 8, 8) for _ in 1:Nr]
        R_mush  = [zeros(precc, 8, 8) for _ in 1:Nr]
        
        idx_6 = [1, 2, 3, 5, 6, 7]
        R6_view = [view(R_solid[i], idx_6, idx_6) for i in 1:Nr]
        R8_view = [view(R_mush[i], 1:8, 1:8) for i in 1:Nr]

        # Test Solid Core Boundary initialization (6x6 context)
        C1l, D2l = solid1d_mush_relax.core_boundary(
            R6_view, (1, 2), rs, ρs, gs, μs, Ks, ω, ρ_core, μ_core, κ_core, "liquid", n; G0=1.0, Y=[1,2,4,5,3,6]
        )
        @test size(C1l) == (3, 6)
        @test size(D2l) == (3, 6)
        @test any(R_solid[1] .!= 0.0)

        # Test Mushy Core Boundary initialization (8x8 context)
        C1l_m, D2l_m = solid1d_mush_relax.core_boundary_mush(
            R8_view, (1, 2), rs, ρs, gs, μs, Ks, ω, ρₗs, Kls, Kds, αs, ηₗs, ϕs, ks, ρ_core, μ_core, κ_core, "liquid", n; G0=1.0, Y=[1,2,5,6,3,7,4,8]
        )
        @test size(C1l_m) == (4, 8)
        @test size(D2l_m) == (4, 8)
        @test any(R_mush[1] .!= 0.0)
    end
    
    @testset "Boundary Conditions: get_core_bc!" begin
        ω, r, ρ, g, μ, K, n = 1.0, 1.0, 1.0, 1.0, complex(1.0), complex(1.0), 2
        
        Ic = solid1d_mush_relax.common.get_Ic(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0)
        B  = solid1d_mush_relax.get_core_bc!(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0)
        
        # Check dimensions
        @test size(B) == (3, 6)
        
        # Check linear independence of constraint rows
        @test rank(B) == 3

        # Check left-nullspace orthogonality constraint: B * Ic ≈ 0
        @test B * Ic ≈ zeros(ComplexF64, 3, size(Ic, 2)) atol=1e-12
    end

    @testset "Boundary Conditions: get_core_bc! (mush)" begin
        ω, r, ρ, g, μ, K, n = 1.0, 1.0, 1.0, 1.0, complex(1.0), complex(1.0), 2
        
        Ic = solid1d_mush_relax.common.get_Ic(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0, Y=[1,2,3,4,5,6,7,8])
        B  = solid1d_mush_relax.get_core_bc!(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0, Y=[1,2,3,4,5,6,7,8])
        
        # Check dimensions
        @test size(B) == (4, 8)
        
        # Check linear independence of constraint rows
        @test rank(B) == 4

        # Check left-nullspace orthogonality constraint: B * Ic ≈ 0
        @test B * Ic ≈ zeros(ComplexF64, 4, size(Ic, 2)) atol=1e-12
    end

    @testset "Boundary Conditions: get_core_bc! (inertial)" begin
        # Mirrors the liquid/solid checks above: dimensions, linear independence of
        # the constraint rows, and left-nullspace orthogonality B * Ic ~ 0. The
        # inertial core has two structurally different elementary-solution formulas
        # (a low-frequency algebraic approximation and a full Bessel-function-based
        # solution, selected by get_Ic's internal is_low_freq switch), so both code
        # paths are exercised independently below rather than relying on a single
        # omega value to happen to hit one of them.
        # NOTE: this file's @testset blocks do not fully isolate variables from the
        # enclosing scope (assignment to a name already bound in an enclosing scope
        # rebinds it there), so every local here uses an "_iner" suffix to avoid
        # colliding with the outer testset's r/rho/n/omega/etc. fixture variables.
        r_iner, ρ_iner, g_iner, μ_iner, K_iner, n_iner = 1.0, 1.0, 1.0, complex(1.0), complex(4.0), 2

        @testset "high-frequency / full Bessel branch" begin
            ω_iner = 1.0  # >> the is_low_freq switch for these parameters; exercises the "else" branch

            Ic = solid1d_mush_relax.common.get_Ic(ω_iner, r_iner, ρ_iner, g_iner, μ_iner, K_iner, "inertial", n_iner; G0=1.0)
            B  = solid1d_mush_relax.get_core_bc!(ω_iner, r_iner, ρ_iner, g_iner, μ_iner, K_iner, "inertial", n_iner; G0=1.0)

            @test size(B) == (3, 6)
            @test rank(B) == 3
            @test B * Ic ≈ zeros(ComplexF64, 3, size(Ic, 2)) atol=1e-10

            # mush (8x8) variant
            Ic8 = solid1d_mush_relax.common.get_Ic(ω_iner, r_iner, ρ_iner, g_iner, μ_iner, K_iner, "inertial", n_iner; G0=1.0, Y=[1,2,3,4,5,6,7,8])
            B8  = solid1d_mush_relax.get_core_bc!(ω_iner, r_iner, ρ_iner, g_iner, μ_iner, K_iner, "inertial", n_iner; G0=1.0, Y=[1,2,3,4,5,6,7,8])

            @test size(B8) == (4, 8)
            @test rank(B8) == 4
            @test B8 * Ic8 ≈ zeros(ComplexF64, 4, size(Ic8, 2)) atol=1e-10
        end

        @testset "low-frequency algebraic branch" begin
            ω_iner = 1e-10  # << the is_low_freq switch for these parameters; exercises the "if" branch

            Ic = solid1d_mush_relax.common.get_Ic(ω_iner, r_iner, ρ_iner, g_iner, μ_iner, K_iner, "inertial", n_iner; G0=1.0)
            B  = solid1d_mush_relax.get_core_bc!(ω_iner, r_iner, ρ_iner, g_iner, μ_iner, K_iner, "inertial", n_iner; G0=1.0)

            @test size(B) == (3, 6)
            @test rank(B) == 3
            @test B * Ic ≈ zeros(ComplexF64, 3, size(Ic, 2)) atol=1e-10
        end
    end

    @testset "3. Layer Propagation Routines" begin
        Nr = 5
        rs = collect(range(0.5, 1.0, length=Nr))
        ρs = fill(1.0, Nr)
        gs = fill(1.0, Nr)
        μs = fill(complex(1.0), Nr)
        Ks = fill(complex(2.0), Nr)
        
        # Mushy vectors
        ρₗs = fill(0.9, Nr)
        Kls = fill(1.5, Nr)
        Kds = fill(complex(1.2), Nr)
        αs  = fill(complex(0.8), Nr)
        ηₗs = fill(1.0, Nr)
        ϕs  = fill(0.1, Nr)
        ks  = fill(1e-4, Nr)

        R = [zeros(precc, 8, 8) for _ in 1:Nr]
        idx_6 = [1, 2, 3, 5, 6, 7]
        R6_view = [view(R[i], idx_6, idx_6) for i in 1:Nr]
        R8_view = [view(R[i], 1:8, 1:8) for i in 1:Nr]

        # Test solid layer propagation engine
        Cn_l_6 = zeros(precc, 3, 6)
        Dnp_l_6 = zeros(precc, 3, 6)
        Cn_l_out6, Dnp_l_out6 = solid1d_mush_relax.propagate_solid(
            R6_view, Cn_l_6, Dnp_l_6, (2, 4), rs, ρs, gs, μs, Ks, ω, n; G0=1.0, Y=[1,2,4,5,3,6]
        )
        @test size(Cn_l_out6)  == (3, 6)
        @test size(Dnp_l_out6) == (3, 6)
        @test any(R[2] .!= 0.0)

        # Test mushy layer propagation engine
        Cn_l_8 = zeros(precc, 4, 8)
        Dnp_l_8 = zeros(precc, 4, 8)
        Cn_l_out8, Dnp_l_out8 = solid1d_mush_relax.propagate_mush(
            R8_view, Cn_l_8, Dnp_l_8, (2, 4), rs, ρs, gs, μs, Ks, ω, ρₗs, Kls, Kds, αs, ηₗs, ϕs, ks, n; G0=1.0, Y=[1,2,5,6,3,7,4,8]
        )
        @test size(Cn_l_out8)  == (4, 8)
        @test size(Dnp_l_out8) == (4, 8)
    end

    @testset "4. Interface Transitions Execution Loops" begin
        Nr = 5
        rs = collect(range(0.5, 1.0, length=Nr))
        ρs = fill(1.0, Nr)
        gs = fill(1.0, Nr)
        μs = fill(complex(1.0), Nr)
        Ks = fill(complex(2.0), Nr)
        
        ρₗs = fill(0.9, Nr)
        Kls = fill(1.5, Nr)
        Kds = fill(complex(1.2), Nr)
        αs  = fill(complex(0.8), Nr)
        ηₗs = fill(1.0, Nr)
        ϕs  = fill(0.1, Nr)
        ks  = fill(1e-4, Nr)

        R = [zeros(precc, 8, 8) for _ in 1:Nr]
        for i in 1:Nr
            R[i] .= Matrix{precc}(I, 8, 8)
        end
        R8_view = [view(R[i], 1:8, 1:8) for i in 1:Nr]

        # FIXED: Mushy layer solutions have 4 rows and 8 columns 
        Cn_l_in = zeros(precc, 4, 8)
        Dnp_l_in = zeros(precc, 4, 8)
        
        # Test Interface Transition: Mushy -> Solid
        Cn_l_out, Dnp_l_out = solid1d_mush_relax.interface_mush_solid(
            R8_view, Cn_l_in, Dnp_l_in, (2, 3), rs, ρs, gs, μs, Ks, ω, ρₗs, Kls, Kds, αs, ηₗs, ϕs, ks, n; G0=1.0, Y=[1,2,5,6,3,7,4,8]
        )
        @test size(Cn_l_out)  == (3, 6)
        @test size(Dnp_l_out) == (3, 6)

        # Test Interface Transition: Solid -> Mushy
        Cn_l_out2, Dnp_l_out2 = solid1d_mush_relax.interface_solid_mush(
            R8_view, Cn_l_out, Dnp_l_out, (3, 4), rs, ρs, gs, μs, Ks, ω, ρₗs, Kls, Kds, αs, ηₗs, ϕs, ks, n; G0=1.0, Y=[1,2,5,6,3,7,4,8]
        )
        @test size(Cn_l_out2)  == (4, 8)
        @test size(Dnp_l_out2) == (4, 8)
    end

    @testset "5. Surface Boundaries & System Resolutions" begin
        Nr = 5
        rs = collect(range(0.5, 1.0, length=Nr))
        ρs = fill(1.0, Nr)
        gs = fill(1.0, Nr)
        μs = fill(complex(1.0), Nr)
        Ks = fill(complex(2.0), Nr)

        # Initialize propagation history matrices with Identity properties
        R = [Matrix{precc}(I, 8, 8) for _ in 1:Nr]
        idx_6 = [1, 2, 3, 5, 6, 7]
        R6_view = [view(R[i], idx_6, idx_6) for i in 1:Nr]
        R8_view = [view(R[i], 1:8, 1:8) for i in 1:Nr]

        # FIXED: Populated with rand() instead of zeros to prevent SingularException(1)
        CNm_l6 = rand(precc, 3, 6)
        DN_l6  = rand(precc, 3, 6)
        y_t6, y_l6 = solid1d_mush_relax.surface_boundary(
            R6_view, CNm_l6, DN_l6, (Nr-1, Nr), rs, ρs, gs, μs, Ks, ω, n; G0=1.0, Y=[1,2,4,5,3,6]
        )
        @test length(y_t6) == 6
        @test length(y_l6) == 6

        # FIXED: Random complex population for the mushy surface boundary run
        CNm_l8 = rand(precc, 4, 8)
        DN_l8  = rand(precc, 4, 8)
        y_t8, y_l8 = solid1d_mush_relax.surface_boundary_mush(
            R8_view, CNm_l8, DN_l8, (Nr-1, Nr), rs, ρs, gs, μs, Ks, ω, n; G0=1.0, Y=[1,2,5,6,3,7,4,8]
        )
        @test length(y_t8) == 8
        @test length(y_l8) == 8
    end

    @testset "6. Physical limit: vanishing melt fraction recovers the pure-elastic solution" begin
        # Regression test: a mush layer with phi -> 0 (Biot coefficient alongside it)
        # is physically indistinguishable from an ordinary solid layer, so 
        # solid1d_mush_relax must reduce continuously to the pure-elastic solid1d_relax
        # Love numbers as phi shrinks. 

        R_planet_lim = 6.0e6
        r_core_lim   = 3.0e6
        radius_lim   = Float64[r_core_lim, 4.0e6, 5.0e6, R_planet_lim]

        rho_lim   = Float64[4500.0, 4200.0, 3800.0]
        visc_lim  = Float64[1e21, 1e21, 1e19]
        shear_lim = ComplexF64[6.0e10, 6.0e10, 5.0e10]
        bulk_lim  = ComplexF64[1.3e11, 1.3e11, 1.1e11]
        bulkd_lim = copy(bulk_lim)

        gravity_dummy_lim = zeros(Float64, length(rho_lim))

        mcore_lim   = 4/3*pi*r_core_lim^3*1.0e4
        rhocore_lim = 1.0e4
        mucore_lim  = ComplexF64(0.0)
        kcore_lim   = ComplexF64(1.3e11)

        omega_lim = 2.05e-5
        ndeg_lim  = 2
        drmin_lim, drmax_lim = 20000, 80000

        # Pure-elastic reference: no mush anywhere.
        _, _, _, kT_ref_lim, kL_ref_lim = Obliqua.run_solid1d_relax(
            omega_lim, rho_lim, radius_lim, gravity_dummy_lim, visc_lim, shear_lim, bulk_lim,
            R_planet_lim, mcore_lim, rhocore_lim, mucore_lim, kcore_lim;
            dr_min=drmin_lim, dr_max=drmax_lim, n=ndeg_lim, m=2, core="liquid",
            inertial_terms=false, optimize_scales=false, patch=false
        )

        function mush_love_numbers_lim(phi_arr, alpha_arr, perm_arr)
            _, _, _, _, kT, kL = Obliqua.run_solid1d_mush_relax(
                omega_lim, rho_lim, radius_lim, gravity_dummy_lim, visc_lim, shear_lim, bulk_lim, bulkd_lim,
                phi_arr, alpha_arr, perm_arr,
                R_planet_lim, mcore_lim, rhocore_lim, mucore_lim, kcore_lim;
                dr_min=drmin_lim, dr_max=drmax_lim, n=ndeg_lim, m=2, core="liquid",
                inertial_terms=false, visc_l=1e2, bulk_l=1e9,
                porosity_thresh=0.0, optimize_scales=false, patch=false
            )
            return kT, kL
        end

        # A single mush layer placed at three positions, so that each interface
        # function is exercised on its own: core-adjacent exercises
        # core_boundary_mush + interface_mush_solid, surface-adjacent exercises
        # interface_solid_mush + surface_boundary_mush, and mid-mantle exercises both.
        @testset "core-adjacent mush -> elastic limit" begin
            phi_lo, alpha_lo, perm_lo = Float64[1e-2,0.0,0.0], ComplexF64[1e-2,0.0,0.0], Float64[1e-14,0.0,0.0]
            phi_hi, alpha_hi, perm_hi = Float64[1e-6,0.0,0.0], ComplexF64[1e-6,0.0,0.0], Float64[1e-14,0.0,0.0]

            kT_lo, kL_lo = mush_love_numbers_lim(phi_lo, alpha_lo, perm_lo)
            kT_hi, kL_hi = mush_love_numbers_lim(phi_hi, alpha_hi, perm_hi)

            dT_lo, dL_lo = abs(kT_lo - kT_ref_lim), abs(kL_lo - kL_ref_lim)
            dT_hi, dL_hi = abs(kT_hi - kT_ref_lim), abs(kL_hi - kL_ref_lim)

            @test dT_hi < 1e-6
            @test dL_hi < 1e-6
            # Discriminating check: shrinking phi by four orders of magnitude must
            # shrink the residual by a comparable factor, ruling out a phi-independent
            # (structural) closure error rather than just pinning one lucky value.
            @test dT_hi < dT_lo / 100
            @test dL_hi < dL_lo / 100
        end

        @testset "mid-mantle mush -> elastic limit" begin
            phi_lo, alpha_lo, perm_lo = Float64[0.0,1e-2,0.0], ComplexF64[0.0,1e-2,0.0], Float64[0.0,1e-14,0.0]
            phi_hi, alpha_hi, perm_hi = Float64[0.0,1e-6,0.0], ComplexF64[0.0,1e-6,0.0], Float64[0.0,1e-14,0.0]

            kT_lo, kL_lo = mush_love_numbers_lim(phi_lo, alpha_lo, perm_lo)
            kT_hi, kL_hi = mush_love_numbers_lim(phi_hi, alpha_hi, perm_hi)

            dT_lo, dL_lo = abs(kT_lo - kT_ref_lim), abs(kL_lo - kL_ref_lim)
            dT_hi, dL_hi = abs(kT_hi - kT_ref_lim), abs(kL_hi - kL_ref_lim)

            @test dT_hi < 1e-6
            @test dL_hi < 1e-6
            @test dT_hi < dT_lo / 100
            @test dL_hi < dL_lo / 100
        end

        @testset "surface-adjacent mush -> elastic limit" begin
            phi_lo, alpha_lo, perm_lo = Float64[0.0,0.0,1e-2], ComplexF64[0.0,0.0,1e-2], Float64[0.0,0.0,1e-14]
            phi_hi, alpha_hi, perm_hi = Float64[0.0,0.0,1e-6], ComplexF64[0.0,0.0,1e-6], Float64[0.0,0.0,1e-14]

            kT_lo, kL_lo = mush_love_numbers_lim(phi_lo, alpha_lo, perm_lo)
            kT_hi, kL_hi = mush_love_numbers_lim(phi_hi, alpha_hi, perm_hi)

            dT_lo, dL_lo = abs(kT_lo - kT_ref_lim), abs(kL_lo - kL_ref_lim)
            dT_hi, dL_hi = abs(kT_hi - kT_ref_lim), abs(kL_hi - kL_ref_lim)

            @test dT_hi < 1e-6
            @test dL_hi < 1e-6
            @test dT_hi < dT_lo / 100
            @test dL_hi < dL_lo / 100
        end
    end
end