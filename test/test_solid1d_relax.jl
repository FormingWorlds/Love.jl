using Test
using Obliqua
using Obliqua.solid1d_relax
using Obliqua.solid1d_relax.common
using LinearAlgebra
using DoubleFloats
using Obliqua.constants


ROOT_DIR = abspath(joinpath(dirname(abspath(@__FILE__)),"../"))
RES_DIR         = joinpath(ROOT_DIR,"res/")
OUT_DIR         = joinpath(ROOT_DIR,"out/")
TEST_DIR        = joinpath(ROOT_DIR,"test/")


# ==========================================
# 2. UNIT TEST DEFINITIONS
# ==========================================

@testset "solid1d_relax Module Tests" begin

    G = 6.67430e-11 # Gravitational constant [m^3 kg^-1 s^-2]
    prec = Float64
    precc = Complex{prec}

    @testset "Mathematical Helpers: doublefactorial" begin
        # Base/Edge cases
        @test solid1d_relax.doublefactorial(0) == 1
        @test solid1d_relax.doublefactorial(1) == 1
        
        # Odd integers (5!! = 5 * 3 * 1 = 15)
        @test solid1d_relax.doublefactorial(5) == 15
        # Even integers (6!! = 6 * 4 * 2 = 48)
        @test solid1d_relax.doublefactorial(6) == 48
        
        # Exception handling for negative inputs
        @test_throws ErrorException solid1d_relax.doublefactorial(-1)
        @test_throws ErrorException solid1d_relax.doublefactorial(-5)
    end

    @testset "Gravity Profiling: get_g" begin
        # Setup a simple 2-boundary layer (1 cell center)
        r = [1.0, 2.0]
        ρ = [10.0]
        m_core = 5.0
        
        g, M_tot = solid1d_relax.get_g(r, ρ, m_core)
        
        # Check mass enclosed analytical verification
        # dm = 4/3 * π * (2^3 - 1^3) * 10 = 4/3 * π * 7 * 10 = 280/3 * π
        expected_dm = (4.0 / 3.0) * π * (2.0^3 - 1.0^3) * 10.0
        expected_M_tot = expected_dm + m_core
        @test M_tot ≈ expected_M_tot
        
        # Check gravity calculation at the outer boundary: g = G * M_enc / r[2]^2
        expected_g = [G * expected_M_tot / (2.0^2)]
        @test g ≈ expected_g
    end

    @testset "Grid Generation: resample_profiles" begin
        # Setup realistic planetary scale profiles (e.g., in meters/SI units)
        radius = [1000.0, 2000.0, 3000.0]
        rho = [3000.0, 4000.0]        # kg/m^3
        visc = [1e21, 1e22]           # Pa s
        shear = complex.([1e10, 2e10]) # Pa
        bulk = complex.([5e10, 6e10])  # Pa
        m_core = 1e20                 # kg
        dr_min = 10                   # 10 meter minimum spacing
        dr_max = 50                   # 50 meter maximum spacing

        r_new_b, ρ_new, η_new, μ_new, κ_new, g_new, M_tot = solid1d_relax.resample_profiles(
            radius, rho, visc, shear, bulk, m_core, dr_min, dr_max
        )

        # Calculate expected number of elements based on code logic
        α = log(dr_max / dr_min)
        expected_N = Int(ceil((radius[end] - radius[1]) / dr_min * α / (exp(α) - 1)))

        # Size verifications (N will now be > 1, preventing BoundsErrors)
        @test expected_N > 1
        @test length(r_new_b) == expected_N
        @test length(ρ_new) == expected_N - 1
        @test length(η_new) == expected_N - 1
        @test length(μ_new) == expected_N - 1
        @test length(κ_new) == expected_N - 1
        @test length(g_new) == expected_N - 1

        # Boundary constraints checks
        @test r_new_b[1] ≈ radius[1]
        @test r_new_b[end] ≈ radius[end]
    end

    @testset "Boundary Conditions: get_surface_bc!" begin
        R_planet = 2.0
        g_surface = 9.8
        n = 2
        
        # Test Tidal configuration parameters: (U=1, U_prime=0, tau=0, P=0)
        B, b = solid1d_relax.get_surface_bc!(R_planet, g_surface, n, 1, 0, 0, 0; G0=1.0)
        
        @test size(B) == (3, 6)
        @test length(b) == 6
        @test b[4] == 0.0 # Radial stress should be zero when U_prime and P are 0
        @test b[5] == 0.0 # Tangential stress
        @test b[6] ≈ ((2 * n + 1) / R_planet) * 1.0 # Pure analytical tidal evaluation

        # Test Load configuration parameters: (U=0, U_prime=1, tau=0, P=0)
        B_l, b_l = solid1d_relax.get_surface_bc!(R_planet, g_surface, n, 0, 1, 0, 0; G0=1.0)
        @test b_l[3] != complex(0.0) # Radial stress should change with surface mass load (U_prime=1)
    end

    @testset "Boundary Conditions: get_core_bc!" begin
        ω, r, ρ, g, μ, K, n = 1.0, 1.0, 1.0, 1.0, complex(1.0), complex(1.0), 2
        
        Ic = solid1d_relax.get_Ic(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0)
        B  = solid1d_relax.get_core_bc!(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0)
        
        # Check dimensions
        @test size(B) == (3, 6)
        
        # Check linear independence of constraint rows
        @test rank(B) == 3

        # Check left-nullspace orthogonality constraint: B * Ic ≈ 0
        @test B * Ic ≈ zeros(ComplexF64, 3, size(Ic, 2)) atol=1e-12
    end

    @testset "Relaxation Steps: Core, Propagate, and Surface execution loops" begin
        # Setup small controlled vectors for matrix generation loops
        r = [1.0, 1.5, 2.0]
        ρ = [1.0, 1.1, 1.2]
        g = [1.0, 1.2, 1.4]
        μ = complex.([2.0, 2.1, 2.2])
        K = complex.([3.0, 3.1, 3.2])
        ω = 0.5
        n = 2
        
        R = Vector{Matrix{precc}}(undef, length(r))
        
        # 1. Core boundary processing
        Cn_l, Dnp_l = solid1d_relax.core_boundary(R, (1, 2), r, ρ, g, μ, K, ω, 1.0, complex(0.0), complex(1.0), "liquid", n; G0=1.0)
        @test size(Cn_l) == (3, 6)
        @test size(Dnp_l) == (3, 6)
        @test isassigned(R, 1)

        # 2. Solid matrix propagation engine loop execution
        Cn_l, Dnp_l = solid1d_relax.propagate_solid(R, Cn_l, Dnp_l, (2, 2), r, ρ, g, μ, K, ω, n; G0=1.0)
        @test isassigned(R, 2)

        # 3. Final Surface resolution integration
        y_t, y_l = solid1d_relax.surface_boundary(R, Cn_l, Dnp_l, (2, 3), r, ρ, g, μ, K, ω, n; G0=1.0)
        @test length(y_t) == 6
        @test length(y_l) == 6
    end

    @testset "Full Solver System Integration: compute_y" begin
        r = [1.0, 1.5, 2.0]
        ρ = [1.0, 1.1, 1.2]
        g = [1.0, 1.2, 1.4]
        μ = complex.([2.0, 2.1, 2.2])
        K = complex.([3.0, 3.1, 3.2])
        ω, n = 0.5, 2
        ρ_core, μ_core, κ_core, M_tot = 1.0, complex(0.0), complex(1.0), 10.0

        y_t, y_l = solid1d_relax.compute_y(
            r, ρ, g, μ, K, ω, n, ρ_core, μ_core, κ_core, [r[end], M_tot, G]; core="liquid"
        )

        # Shape verifications
        @test size(y_t) == (6, length(r))
        @test size(y_l) == (6, 1)
        
        # Datatype execution checks
        @test eltype(y_t) == ComplexF64
        @test eltype(y_l) == ComplexF64
    end
end