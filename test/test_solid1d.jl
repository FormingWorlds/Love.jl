using Test
using LinearAlgebra
using StaticArrays
using AssociatedLegendrePolynomials
using Obliqua
using Obliqua.solid1d

@testset "solid1d Module Tests" begin

    @testset "1. Layer Grid Discretization (expand_layers)" begin
        r_primary = [1000.0, 2000.0, 5000.0] # 3 primary boundaries (2 layers)
        nr = 10                             # 10 secondary intervals per layer
        
        rs = solid1d.expand_layers(r_primary; nr=nr)
        
        @test size(rs) == (11, 2)
        @test rs[1, 1] == 1000.0
        @test rs[end, 1] == 2000.0
        @test rs[1, 2] == 2000.0
        @test rs[end, 2] == 5000.0
        @test all(diff(rs[:, 1]) .> 0)
    end

    @testset "2. Gravity Profiles Calculations (get_g)" begin
        r_grid = [1000.0 2000.0; 1500.0 3500.0; 2000.0 5000.0] # Size (3, 2)
        ρ = [3000.0, 4500.0]
        m_core = 1e18
        
        g = solid1d.get_g(r_grid, ρ, m_core)
        
        @test size(g) == size(r_grid)
        @test all(g[2:end, :] .> 0.0) # Gravity must be positive outside the origin boundary
        @test g[1, 2] == g[end, 1]    # Internal boundary continuity verification
    end

    @testset "3. Spherical Harmonics (Ynm)" begin
        theta = [0.1, 0.5, 1.0]
        phi = [0.0, 0.5, 1.0]'
        n, m = 2, 1
        
        Y = solid1d.Ynm(n, m, theta, phi)
        
        @test size(Y) == (3, 3)
        @test eltype(Y) == ComplexF64
    end

    @testset "4. Spherical Grid Allocations (define_spherical_grid)" begin
        res = 45.0 # Large resolution step for lightweight test performance
        n, m = 2, 0
        
        solid1d.define_spherical_grid(res, n, m)
        
        @test solid1d.res == 45.0
        @test length(solid1d.clats) == length(0:45:180)
        @test length(solid1d.lons) == length(0:45:360-0.001)
        
        # Verify internal structural allocations match dimensional tensors
        nlats = length(solid1d.clats)
        nlons = length(solid1d.lons)
        @test size(solid1d.Y) == (1, nlats, nlons)
        @test size(solid1d.dYdθ) == (1, nlats, nlons)
        @test size(solid1d.Z) == (1, nlats, nlons)
        @test size(solid1d.X) == (1, nlats, nlons)
    end

    @testset "5. Precision Casts (convert_params_to_prec)" begin
        r = rand(Float64, 4, 2)
        ρ = rand(Float64, 2)
        g = rand(Float64, 4, 2)
        μ = rand(Float64, 2)
        κs = rand(Float64, 2)
        
        r_p, ρ_p, g_p, μ_p, κs_p = solid1d.convert_params_to_prec(r, ρ, g, μ, κs)
        
        @test eltype(r_p) == prec
        @test eltype(ρ_p) == prec
        @test eltype(g_p) == prec
        @test eltype(μ_p) == precc
        @test eltype(κs_p) == precc
    end

    @testset "6. RK4 Integrator Matrices (get_B & get_B_product!)" begin
        ω = 1e-4
        r1, r2 = 1000.0, 1050.0
        g1, g2 = 1.5, 1.6
        ρ = 3000.0
        μ = complex(1e10, 1e8)
        K = complex(5e10, 1e8)
        n = 2
        
        # Test individual step matrix generation
        B = solid1d.get_B(ω, r1, r2, g1, g2, ρ, μ, K, n)
        @test size(B) == (6, 6)
        @test eltype(B) == precc
        
        # Test layer product propagation
        r_arr = [1000.0, 1050.0, 1100.0]
        g_arr = [1.5, 1.55, 1.6]
        # Create views mock matching the execution context
        r_view = view(reshape(r_arr, 3, 1), :, 1)
        g_view = view(reshape(g_arr, 3, 1), :, 1)
        
        Bprod2 = zeros(precc, 6, 6, 2)
        solid1d.get_B_product!(Bprod2, ω, r_view, ρ, g_view, μ, K, n)
        
        @test all(Bprod2 .!= 0.0)
    end

    @testset "7. Global System Propagators (compute_M & compute_y)" begin
        ω = 1e-4
        r = [1000.0 2000.0; 1500.0 3500.0; 2000.0 5000.0]
        ρ = [3000.0, 4000.0]
        g = [1.0 2.0; 1.5 3.0; 2.0 4.0]
        μ = [complex(1e10), complex(2e10)]
        K = [complex(5e10), complex(6e10)]
        n = 2
        ρ_core, μ_core, κ_core = 7000.0, complex(0.0), complex(1e11)
        
        M, y1_4 = solid1d.compute_M(ω, r, ρ, g, μ, K, n, ρ_core, μ_core, κ_core, ones(prec, 3); core="liquid")
        
        @test size(M) == (3, 3)
        @test size(y1_4) == (6, 3, 2, 2)
        
        I6 = Matrix{prec}(I, 6, 6)

        # Test linear system resolution for structural displacements
        y = solid1d.compute_y(r, g, M, y1_4, n, I6, ones(prec, 7); load=false)
        @test size(y) == (6, 2, 2)
        @test eltype(y) == ComplexF64
    end

    @testset "8. Tensors and Volumetric Heating Profiles" begin
        # Setup spherical grids for structural evaluations
        solid1d.define_spherical_grid(90.0, 2, 0)
        
        # Mocking values for a single point boundary calculation
        nlats = length(solid1d.clats)
        nlons = length(solid1d.lons)
        ϵ_mock = zeros(ComplexF64, nlats, nlons, 6)
        ϵ_view = view(ϵ_mock, :, :, :)
        
        y_mock = zeros(ComplexF64, 6)
        y_mock[1:4] .= [0.01, 0.005, 1e5, 2e5]
        y_view = view(y_mock, :)
        
        solid1d.compute_strain_ten!(ϵ_view, y_view, 2, 2000.0, 3000.0, 2.5, complex(1e10), complex(5e10))
        @test any(ϵ_mock .!= 0.0)
        
        # Comprehensive Integration Test: heating profile distribution
        y_profile = rand(ComplexF64, 6, 2, 2)
        r = [1000.0 2000.0; 1500.0 3500.0; 2000.0 5000.0]
        ρ = [3000.0, 4000.0]
        g = [1.0 2.0; 1.5 3.0; 2.0 4.0]
        μ = [complex(1e10, 1e8), complex(2e10, 1e8)]
        κ = [complex(5e10, 1e8), complex(6e10, 1e8)]
        
        shear_out, comp_out = solid1d.get_heating_profile(y_profile, r, ρ, g, μ, κ, 2, 1e-4)
        
        # Verify Shear Outputs
        @test length(shear_out[1]) == 2 # Eμ_tot length matches nlayers
        @test size(shear_out[2]) == (3, 2) # Eμ_vol dimensions match r Matrix
        
        # Verify Compaction Outputs
        @test length(comp_out[1]) == 2 # Eκ_tot length matches nlayers
        @test size(comp_out[2]) == (3, 2) # Eκ_vol dimensions match r Matrix
    end
end