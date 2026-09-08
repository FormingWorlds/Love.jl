using Test
using LinearAlgebra
using StaticArrays
using DoubleFloats
using AssociatedLegendrePolynomials
using Obliqua
using Obliqua.solid1d_mush
using Obliqua.constants


@testset "solid1d_mush Module Tests" begin

    # -------------------------------------------------------------------------
    # Test 1: expand_layers
    # -------------------------------------------------------------------------
    @testset "expand_layers" begin
        r_primary = [0.0, 10.0, 30.0]
        nr = 4
        rs = solid1d_mush.expand_layers(r_primary; nr=nr)
        
        # Expecting a 2D matrix: (nr+1) rows by (length(r)-1) columns
        @test size(rs) == (5, 2)
        @test rs[1, 1] == 0.0
        @test rs[end, 1] == 10.0
        @test rs[1, 2] == 10.0
        @test rs[end, 2] == 30.0
        @test rs[3, 1] ≈ 5.0 # Midpoint of the first layer discretization
    end

    # -------------------------------------------------------------------------
    # Test 2: get_g
    # -------------------------------------------------------------------------
    @testset "get_g" begin
        r = [0.0 10.0; 5.0 20.0; 10.0 30.0]
        ρ = [1000.0, 3000.0]
        m_core = 1e20
        
        g = solid1d_mush.get_g(r, ρ, m_core)
        
        @test size(g) == size(r)
        # Verify cross-boundary copy condition: g[1, 2:end] == g[end, 1:end-1]
        @test g[1, 2] == g[end, 1]
        @test all(g[2:end, :] .>= 0.0) # Gravity magnitudes should be non-negative
    end

    # -------------------------------------------------------------------------
    # Test 3: Ynm
    # -------------------------------------------------------------------------
    @testset "Ynm" begin
        theta = [0.0, π/4, π/2]
        phi = [0.0, π/2, π]'
        n = 2
        m = 1
        
        y_harmonics = solid1d_mush.Ynm(n, m, theta, phi)
        
        @test size(y_harmonics) == (3, 3)
        @test eltype(y_harmonics) <: ComplexF64
    end

    # -------------------------------------------------------------------------
    # Test 4: define_spherical_grid
    # -------------------------------------------------------------------------
    @testset "define_spherical_grid" begin
        res_deg = 45.0
        n = 2
        m = 2
        
        solid1d_mush.define_spherical_grid(res_deg, n, m)
        
        @test solid1d_mush.res == res_deg
        @test length(solid1d_mush.clats) == length(0.0:res_deg:180.0)
        @test length(solid1d_mush.lons) == length(0.0:res_deg:360.0-0.001)
        @test size(solid1d_mush.Y, 2) == length(solid1d_mush.clats)
        @test size(solid1d_mush.Y, 3) == length(solid1d_mush.lons)
    end

    # -------------------------------------------------------------------------
    # Test 5: convert_params_to_prec
    # -------------------------------------------------------------------------
    @testset "convert_params_to_prec" begin
        # Dummy Float64 inputs
        r = zeros(2, 2); ρ = zeros(2); g = zeros(2, 2); μ = zeros(2)
        κs = zeros(2); ω = 1.0; ρl = zeros(2); κl = zeros(2)
        κd = zeros(2); α = zeros(2); ηl = zeros(2); ϕ = zeros(2); k = zeros(2)
        
        converted = solid1d_mush.convert_params_to_prec(r, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k)
        
        @test eltype(converted[1]) == solid1d_mush.prec  # r
        @test eltype(converted[4]) == solid1d_mush.precc # μ
        @test eltype(converted[5]) == solid1d_mush.precc # κs
        @test typeof(converted[6]) == solid1d_mush.prec  # ω
    end

    # -------------------------------------------------------------------------
    # Test 6: get_B and get_B! (Solid 6x6 & Two-Phase 8x8)
    # -------------------------------------------------------------------------
    @testset "Integrator Matrices (get_B & get_B!)" begin
        ω = 1e-3; r1 = 10.0; r2 = 12.0; g1 = 1.0; g2 = 1.1
        ρ = 3000.0; μ = 1e10 + 0im; K = 2e10 + 0im; n = 2
        
        # 6x6 Solid Integration Matrix
        B6 = solid1d_mush.get_B(ω, r1, r2, g1, g2, ρ, μ, K, n)
        @test size(B6) == (6, 6)
        
        # 8x8 Two-Phase Integration Matrix
        ρₗ = 1000.0; Kl = 2e9; Kd = 1e10+0im; α = 0.5+0im; ηₗ = 1.0; ϕ = 0.1; k = 1e-12
        B8 = solid1d_mush.get_B(ω, r1, r2, g1, g2, ρ, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n)
        @test size(B8) == (8, 8)
    end

    # -------------------------------------------------------------------------
    # Test 7: get_B_product!
    # -------------------------------------------------------------------------
    @testset "get_B_product!" begin
        nr = 4
        Bprod = zeros(precc, 8, 8, nr-1)
        r_sub = view([10.0, 11.0, 12.0, 13.0], :)
        g_sub = view([1.0, 1.1, 1.2, 1.3], :)
        
        ω = 1e-3; ρ = 3000.0; μ = 1e10+0im; K = 2e10+0im
        ρₗ = 1000.0; Kl = 2e9; Kd = 1e10+0im; α = 0.5+0im; ηₗ = 1.0; ϕ = 0.1; k = 1e-12; n = 2
        
        solid1d_mush.get_B_product!(Bprod, ω, r_sub, ρ, g_sub, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n)
        @test !all(Bprod .== 0.0) # Matrix shouldn't remain entirely zeros
    end

    # -------------------------------------------------------------------------
    # Test 8: compute_M and compute_y
    # -------------------------------------------------------------------------
    @testset "Interior Boundary Matrix & Solutions Solver" begin
        r = [0.1 10.0; 5.0 20.0; 10.0 30.0]
        g = [0.01 1.0; 0.5 1.5; 1.0 2.0]
        ρ = [3000.0, 3200.0]; μ = [1e10+0im, 1.2e10+0im]; K = [2e10+0im, 2.5e10+0im]
        ρₗ = [1000.0, 1000.0]; Kl = [2e9, 2e9]; Kd = [1e10+0im, 1.1e10+0im]
        α = [0.5+0im, 0.6+0im]; ηₗ = [1.0, 1.0]; ϕ = [0.0, 0.1]; k = [1e-12, 1e-12]
        ω = 1e-3; n = 2; ρ_core = 5000.0; μ_core = complex(0.0); κ_core = complex(1e11)
        
        M_mat, y1_4 = solid1d_mush.compute_M(ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n, ρ_core, μ_core, κ_core, ones(prec, 3); core="liquid")
        
        @test size(M_mat) == (4, 4)
        @test size(y1_4) == (8, 4, size(r, 1)-1, size(r, 2))
        
        I8 = Matrix{prec}(I, 8, 8)

        # Test full solution vector calculation
        y_sol = solid1d_mush.compute_y(r, g, M_mat, y1_4, n, I8, ones(prec, 7); load=false)
        @test size(y_sol) == (8, size(r, 1)-1, size(r, 2))
    end

    # -------------------------------------------------------------------------
    # Test 9: Physical Property Tensors (Strain, Displacement, Pore Pressure)
    # -------------------------------------------------------------------------
    @testset "Physical Properties Evaluators" begin
        # Setup grid configuration dynamically
        solid1d_mush.define_spherical_grid(45.0, 2, 2)
        n_lats = length(solid1d_mush.clats)
        n_lons = length(solid1d_mush.lons)
        
        # Allocations
        ϵ = zeros(ComplexF64, n_lats, n_lons, 6)
        dis_rel = zeros(ComplexF64, n_lats, n_lons, 3)
        p = zeros(ComplexF64, n_lats, n_lons)
        
        # Slice views
        ϵ_view = view(ϵ, :, :, :)
        dis_view = view(dis_rel, :, :, :)
        p_view = view(p, :, :)
        
        y_dummy = view(ones(ComplexF64, 8), :)
        
        # Evaluate Strain Tensor
        solid1d_mush.compute_strain_ten!(ϵ_view, y_dummy, 2, 15.0, 3000.0, 1.5, 1e10+0im, 2e10+0im, 1e-3, 1000.0, 2e9, 1e10+0im, 0.5+0im, 1.0, 0.1, 1e-12)
        @test any(ϵ .!= 0.0)
        
        # Evaluate Darcy Displacement Vector
        solid1d_mush.compute_darcy_displacement!(dis_view, y_dummy, 2, 15.0, 1e-3, 0.1, 1.0, 1e-12, 1.5, 1000.0)
        @test any(dis_rel .!= 0.0)
        
        # Evaluate Fluid Pore Pressure
        solid1d_mush.compute_pore_pressure!(p_view, y_dummy, 2)
        @test any(p .!= 0.0)
    end
end