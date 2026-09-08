using Test
using Obliqua
using Obliqua.solid1d_equil_relax
using LinearAlgebra
using DoubleFloats
using Obliqua.constants

ROOT_DIR = abspath(joinpath(dirname(abspath(@__FILE__)),"../"))
RES_DIR         = joinpath(ROOT_DIR,"res/")
OUT_DIR         = joinpath(ROOT_DIR,"out/")
TEST_DIR        = joinpath(ROOT_DIR,"test/")

# ==========================================
# Analytical reference: Love (1911), homogeneous sphere of rigidity μc.
# In the fluid/equilibrium limit (μc -> 0):
#   k2_T -> 3/(2(n-1))   (n=2 gives the textbook k = 3/2)
#   k2_L -> -1           (degree-independent)
# ==========================================
function love_reference(μc, mass_tot, R, n)
    An     = 4.0 * (2n^2 + 4n + 3.0) / (3.0 * n * G * mass_tot^2) * π * R^4
    μc_n   = An * μc
    factor = 1.0 / (1.0 + μc_n)
    k2_T   = n == 1 ? zero(factor) : factor * 3.0 / (2.0 * (n - 1.0))
    k2_L   = -factor
    return k2_T, k2_L
end

@testset "solid1d_equil_relax Module Tests" begin

    G = 6.67430e-11
    prec  = Float64
    precc = Complex{prec}

    @testset "get_A!: Saito (1974) reduced fluid matrix" begin
        r, ρ, g, n = 2.0e6, 3500.0, 4.2, 3
        G0 = G
        Gnorm = G / G0

        A = solid1d_equil_relax.get_A(r, ρ, g, n; G0=G0)

        @test size(A) == (2, 2)
        @test A[1,1] ≈ 4π*Gnorm*ρ/g - (n+1)/r
        @test A[1,2] ≈ 1.0
        @test A[2,1] ≈ 2*(n-1)/r * 4π*Gnorm*ρ/g
        @test A[2,2] ≈ (n-1)/r - 4π*Gnorm*ρ/g
    end

    @testset "get_Ic: regular solution at the innermost grid point" begin
        r, n = 1.0e4, 2
        Ic = solid1d_equil_relax.get_Ic(0.0, r, 3000.0, 1.0, complex(0.0), complex(0.0), "liquid", n)

        @test size(Ic) == (2, 1)
        @test Ic[1,1] ≈ r^n
        @test Ic[2,1] ≈ 2*(n-1)*r^(n-1)

        @test_throws ArgumentError solid1d_equil_relax.get_Ic(0.0, r, 3000.0, 1.0, complex(0.0), complex(0.0), "solid", n)
    end
    
    @testset "Boundary Conditions: get_core_bc!" begin
        ω, r, ρ, g, μ, K, n = 1.0, 1.0, 1.0, 1.0, complex(1.0), complex(1.0), 2
        
        Ic = solid1d_equil_relax.get_Ic(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0)
        B  = solid1d_equil_relax.get_core_bc!(ω, r, ρ, g, μ, K, "liquid", n; G0=1.0)
        
        # Check dimensions
        @test size(B) == (1, 2)
        
        # Check linear independence of constraint rows
        @test rank(B) == 1

        # Check left-nullspace orthogonality constraint: B * Ic ≈ 0
        @test B * Ic ≈ zeros(ComplexF64, 1, size(Ic, 2)) atol=1e-12
    end

    @testset "get_surface_bc!: tidal vs load forcing" begin
        R_planet, g_surface, n = 2.0e6, 4.2, 2
        G0 = G
        Gnorm = G / G0

        # tidal: U=1, U'=0, tau=0, P=0 -> free surface, unit external tidal potential
        B_t, b_t, y2_t, y6_t = solid1d_equil_relax.get_surface_bc!(R_planet, g_surface, n, 1, 0, 0, 0; G0=G0, Y=[1,2])
        @test y2_t ≈ 0.0
        @test y6_t ≈ (2n+1)/R_planet
        @test b_t[2] ≈ y6_t + 4π*Gnorm/g_surface*y2_t
        @test B_t[1, 2] ≈ 1.0

        # load: U=0, U'=1, tau=0, P=0
        B_l, b_l, y2_l, y6_l = solid1d_equil_relax.get_surface_bc!(R_planet, g_surface, n, 0, 1, 0, 0; G0=G0, Y=[1,2])
        @test y2_l ≈ -(2n+1)*g_surface/(4π*R_planet^2)
        @test y6_l ≈ (2n+1)/R_planet * (Gnorm/R_planet)

        # Exact algebraic identity: the y6- and y2- contributions to the y7
        # target cancel completely for the load case, independent of g, ρ, R, n:
        #   y6 + (4πG/g) y2 = (2n+1)G/R^2 - (2n+1)G/R^2 = 0
        # This is what ultimately forces k_L = -1 below.
        @test isapprox(b_l[2], 0.0; atol=1e-12)
    end

    @testset "Relaxation Steps: Core, Propagate, and Surface execution loops (smoke test)" begin
        r = [1.0, 1.5, 2.0, 2.5]
        ρ = [1.0, 1.05, 1.1, 1.15]
        g = [1.0, 1.2, 1.4, 1.6]
        μ = complex.(zeros(4))
        K = complex.(zeros(4))
        ω = 0.0
        n = 2

        R = Vector{Matrix{precc}}(undef, length(r))

        C1l, D2l = solid1d_equil_relax.core_boundary(R, (1, 2), r, ρ, g, μ, K, ω, ρ[1], complex(0.0), complex(0.0), "liquid", n; G0=1.0)
        @test isassigned(R, 1)
        @test size(R[1]) == (2, 2)

        C1l, D2l = solid1d_equil_relax.propagate_solid(R, C1l, D2l, (2, 3), r, ρ, g, μ, K, ω, n; G0=1.0)
        @test isassigned(R, 3)

        y_t, y_l = solid1d_equil_relax.surface_boundary(R, C1l, D2l, (3, 4), r, ρ, g, μ, K, ω, n; G0=1.0)
        @test length(y_t) == 6
        @test length(y_l) == 6
    end

    @testset "Load Love number: universal k_L = -1 (arbitrary, non-uniform profile)" begin
        # This does NOT rely on a homogeneous sphere -- see the exact-cancellation
        # identity checked above. Using a deliberately non-uniform density profile
        # here to demonstrate that generality.
        Nr = 400
        R_planet = 3.0e6
        r  = collect(range(R_planet*1e-3, R_planet, length=Nr))
        ρ  = 2000.0 .+ 1500.0 .* (1.0 .- r ./ R_planet)   # arbitrary, decreasing outward

        M_enc = similar(r)
        M_enc[1] = 4/3*π*ρ[1]*r[1]^3
        for i in 2:Nr
            M_enc[i] = M_enc[i-1] + 4/3*π*ρ[i]*(r[i]^3 - r[i-1]^3)
        end
        g = G .* M_enc ./ r.^2
        mass_tot = M_enc[end]

        μ = complex.(zeros(Nr))
        K = complex.(zeros(Nr))
        ω, n = 0.0, 2

        y_t, y_l = solid1d_equil_relax.compute_y(
            r, ρ, g, μ, K, ω, n, ρ[1], complex(0.0), complex(0.0), [R_planet, mass_tot, G]; core="liquid"
        )

        k_L = real(y_l[5]) - 1.0

        @test isapprox(k_L, -1.0; atol=1e-6)
        @test isapprox(imag(y_l[5]), 0.0; atol=1e-9)
    end

    @testset "Tidal Love number vs Love (1911) fluid limit (homogeneous sphere)" begin
        R_planet = 1.0e6
        ρ0       = 3000.0
        mass_tot = 4/3*π*ρ0*R_planet^3

        Nr = 2000
        r = exp.(range(log(R_planet*1e-6), log(R_planet), length=Nr))
        ρ  = fill(ρ0, Nr)
        g  = 4/3*π*G .* ρ .* r
        μ  = complex.(zeros(Nr))
        K  = complex.(zeros(Nr))
        ω  = 0.0

        for n in (2, 3, 4)
            y_t, y_l = solid1d_equil_relax.compute_y(
                r, ρ, g, μ, K, ω, n, ρ0, complex(0.0), complex(0.0), [R_planet, mass_tot, G]; core="liquid"
            )

            k_T = real(y_t[5]) - 1.0
            k_L = real(y_l[5]) - 1.0
            k2_T, k2_L = love_reference(0.0, mass_tot, R_planet, n)

            @test isapprox(k_T, k2_T; rtol=1e-3)
            @test isapprox(k_L, k2_L; atol=1e-6)

            @test isapprox(imag(y_t[5]), 0.0; atol=1e-9)

            # Equipotential identity for the free (tidal, no-load) surface:
            # y2(a)=0 => y1(a) = y5(a)/g(a)
            g_surface_phys = G * mass_tot / R_planet^2
            @test isapprox(real(y_t[1]), real(y_t[5]) / g_surface_phys; rtol=1e-3)
        end
    end

end