

module solid1d_relax
    
    include("constants.jl")
    include("common.jl")
    using .constants
    using .common

    using LinearAlgebra
    using DoubleFloats
    using AssociatedLegendrePolynomials    
    using StaticArrays
    using SpecialFunctions
    using SparseArrays


    """
        solve_radial_system(r, ρ, g, μ, K, ω, n, R_planet, ρ_core, μ_core, κ_core, scales; core="liquid", inertial_terms=true, patch=false)

    Solve the radial system of ODEs for the solid-body problem using a relaxation method. This function 
    implements the forward-backward relaxation scheme described in the main text of N. Kobayashi (2006).
    
    # Arguments
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `n::Int`                          : Tidal degree.
    - `ρ_core::prec`                    : Density of the core, used for core boundary conditions.
    - `μ_core::precc`                   : Shear modulus of the core, used for core boundary conditions.
    - `κ_core::precc`                   : Bulk modulus of the core, used for core boundary conditions.
    - `scales::Vector{prec}`            : Vector of scaling parameters for non-dimensionalization.

    # Keyword Arguments
    - `core::String="liquid"            : Type of core boundary condition to apply. Options are "liquid" for a fluid core, "solid" for a solid core, and "inertial" for a core with inertial response.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `patch::Bool=false`               : Whether to insert an infinitesimal solid shell around the core. This patches an issue where y2 and y4 become decoupled and cause the solution to diverge in fluid layers.

    # Returns
    - `y_t::Vector{precc}`              : Vector of length 6 representing the tidal solution at the surface (radius = R_planet). This includes the displacements, stresses, and potential at the surface.
    - `y_l::Vector{precc}`              : Vector of length 6 representing the load solution at the surface (radius = R_planet). This includes the displacements, stresses, and potential at the surface.
    - `R::Vector{Matrix{precc}}`        : Vector of 6x6 matrices representing the coefficients of the ODE system at each radial layer.
    - `S::Matrix{prec}`                 : 6x6 matrix representing the normalization.
    """    
    function solve_radial_system(r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int, ρ_core::prec, μ_core::precc, κ_core::precc, scales::Vector{prec}; core::String="liquid", inertial_terms::Bool=true, patch::Bool=false)

        Nr = length(r)

        # indices for the three components of the relaxation scheme: 
        #   1: core boundary
        #   2: forward propagation
        #   3: surface boundary
        ids = [(1,2), (2, Nr-1), (Nr-1, Nr)]   

        # non-dimensional scaling
        R0, M0, ω0, ρ0, G0, g0, μ0, S, Sinv = get_scales(scales[1], scales[2], scales[3])

        # insert solid shell around core
        if patch
            μ = copy(μ)   # break aliasing with caller's array before mutating
            K = copy(K)
            μ[1] = precc(1.47e11)   # these values are chosen to be representative of a solid shell
            K[1] = precc(6.58e10)
        end

        # Scale physical profiles to be dimensionless
        rs = r ./ R0
        ρs = ρ ./ ρ0
        gs = g ./ g0
        μs = μ ./ μ0
        Ks = K ./ μ0
        ωs = ω / ω0

        R = Vector{Matrix{precc}}(undef, Nr)

        # component 1: apply core boundary condition and get first solution
        C1l, D2l = core_boundary(R, ids[1], rs, ρs, gs, μs, Ks, ωs, ρ_core/ρ0, μ_core/μ0, κ_core/μ0, core, n; G0=G0, inertial_terms=inertial_terms)

        # component 2: propagate the solution up to the surface (6x6)
        C1l, D2l = propagate_solid(R, C1l, D2l, ids[2], rs, ρs, gs, μs, Ks, ωs, n; G0=G0, inertial_terms=inertial_terms)

        # component 3: apply surface boundary condition and solve for the final solution at the surface
        y_t, y_l = surface_boundary(R, C1l, D2l, ids[3], rs, ρs, gs, μs, Ks, ωs, n; G0=G0)

        return y_t, y_l, R, S

    end


    """
        core_boundary(R, ids, r, ρ, g, μ, K, ω, ρ_core, μ_core, κ_core, core, n; G0=1, inertial_terms=true)

    Perform the forward-backward relaxation step at the core boundary. This function implements the recursion described 
    in N. Kobayashi (2007) for the initial step of the relaxation scheme, where we apply the core boundary condition and 
    get the first solution for the first layer above the core.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}                 : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρ_core::prec`                    : Density of the core, used for core boundary conditions.
    - `μ_core::precc`                   : Shear modulus of the core, used for core boundary conditions.
    - `κ_core::precc`                   : Bulk modulus of the core, used for core boundary conditions.
    - `core::String`                    : Type of core boundary condition to apply.
    - `n::Int`                          : Tidal degree.

    Keyword Arguments
    - `G0::prec=1`                      : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.

    # Returns
    - `C1l::Matrix{precc}`              : 3x6 matrix representing the "stored" lower half of the C1 matrix for the next iteration.
    - `D2l::Matrix{precc}`              : 3x6 matrix representing the "stored" lower half of the D2 matrix for the next iteration.
    """
    function core_boundary(R::Vector, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρ_core::prec, μ_core::precc, κ_core::precc, core::String, n::Int; G0=1, inertial_terms=true)

        start_id, end_id = ids

        # boundary conditions
        B1 = get_core_bc!(ω, r[start_id], ρ_core, g[start_id], μ_core, κ_core, core, n; G0=G0)        
        
        # first layer (n = 1)
        dr = r[end_id] - r[start_id]

        A1 = get_A(ω, r[start_id], ρ[start_id], g[start_id], μ[start_id], K[start_id], n; G0=G0, inertial_terms=inertial_terms)
        A2 = get_A(ω, r[end_id], ρ[end_id], g[end_id], μ[end_id], K[end_id], n; G0=G0, inertial_terms=inertial_terms)

        I6 = Matrix{precc}(I, 6, 6)

        C1 =  I6 + 0.5 * dr * A1
        D2 = -I6 + 0.5 * dr * A2

        # split matrices
        C1u, C1l = C1[1:3, :], C1[4:6, :]
        D2u, D2l = D2[1:3, :], D2[4:6, :]

        # build S1 and Q1
        S1 = [B1; C1u]              # 6×6
        Q1 = [zeros(3,6); D2u]      # 6×6

        # initial recursion
        R[start_id] = -S1 \ Q1

        return C1l, D2l
    end

    
    """
        propagate_solid(R, B, Cn_l, Dnp_l, ids, r, ρ, g, μ, K, ω, n; G0=1, inertial_terms=true)

    Perform the forward-backward relaxation step for the solid propagation segments. This function implements the 
    recursion described in N. Kobayashi (2007) for the segments of the radial grid that correspond to the solid 
    layers, where we propagate the solution up to the surface using the 6x6 system of equations.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `B::Vector{Matrix{precc}}`        : Vector of 8x1 matrices representing the inhomogeneous terms of the ODE system at each radial layer.
    - `Cn_l::Matrix{precc}`             : 3x6 matrix representing the "stored" lower half of the Cn matrix from the previous step.
    - `Dnp_l::Matrix{precc}`            : 3x6 matrix representing the "stored" lower half of the Dnp matrix from the previous step.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `n::Int`                          : Tidal degree.

    Keyword Arguments
    - `G0::prec=1`                      : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.

    # Returns
    - `Cn_l::Matrix{precc}`             : Updated 3x6 matrix representing the "stored" lower half of the Cn matrix for the next iteration.
    - `Dnp_l::Matrix{precc}`            : Updated 3x6 matrix representing the "stored" lower half of the Dnp matrix for the next iteration.
    """
    function propagate_solid(R::Vector, Cn_l::Matrix{precc}, Dnp_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int; G0=1, inertial_terms=true)

        start_id, end_id = ids

        I6 = Matrix{precc}(I, 6, 6)

        Cn_u = zeros(3,6)
        Dnp_u = zeros(3,6)

        # forward recursion
        for i in start_id:end_id

            dr = r[i+1] - r[i]

            # Calculate A at current and next step
            A_n  = get_A(ω, r[i],   ρ[i],   g[i],   μ[i],   K[i],   n; G0=G0, inertial_terms=inertial_terms)
            A_np = get_A(ω, r[i+1], ρ[i+1], g[i+1], μ[i+1], K[i+1], n; G0=G0, inertial_terms=inertial_terms)

            Cn  =  I6 + 0.5 * dr * A_n
            Dnp = -I6 + 0.5 * dr * A_np

            # 1. Use the "stored" lower halves from the previous step 
            # to fill the upper blocks of P and S.
            Pn_u = Cn_l
            Sn_u = Dnp_l
            Qn_u = zeros(precc, 3, 6)

            # 2. Get the upper halves of the NEWLY calculated Cn and Dnp
            Cn_u  = Cn[1:3, :]
            Dnp_u = Dnp[1:3, :]

            # 3. Build the 6x6 blocks
            Pn = [Pn_u; zeros(precc, 3, 6)]
            Sn = [Sn_u; Cn_u]
            Qn = [Qn_u; Dnp_u]

            # 4. Perform recursion
            Xn = Pn * R[i-1] + Sn
            R[i] = -Xn \ Qn

            # 5. Update the "stored" lower halves for the next iteration
            Cn_l  = Cn[4:6, :]
            Dnp_l = Dnp[4:6, :]
        end

        return Cn_l, Dnp_l
    end


    """
        surface_boundary(R, CNm_l, DN_l, ids, r, ρ, g, μ, K, ω, n; G0=1)

    Perform the forward-backward relaxation step at the surface boundary. This function implements the recursion described 
    in N. Kobayashi (2007) for the final step of the relaxation scheme, where we apply the surface boundary condition and 
    solve for the final solution at the surface, using the 6x6 system of equations.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `CNm_l::Matrix{precc}`            : 3x6 matrix representing the "stored" lower half of the CNm matrix from the previous step.
    - `DN_l::Matrix{precc}`             : 3x6 matrix representing the "stored" lower half of the DN matrix from the previous step.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `n::Int`                          : Tidal degree.

    Keyword Arguments
    - `G0::prec=1`                      : Gravitational constant used for non-dimensional scaling.

    # Returns
    - `y_t::Matrix{precc}`              : 6x1 matrix representing the solution at the surface for the tidal problem.
    - `y_l::Matrix{precc}`              : 6x1 matrix representing the solution at the surface for the load problem.
    """
    function surface_boundary(R::Vector, CNm_l::Matrix{precc}, DN_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int; G0=1)

        start_id, end_id = ids

        # tidal surface boundary condition
        BN_t, b_t = get_surface_bc!(r[end], g[end], n, 1, 0, 0, 0; G0=G0)
        # load surface boundary condition
        BN_l, b_l = get_surface_bc!(r[end], g[end], n, 0, 1, 0, 0; G0=G0)

        # permute the boundary condition vector
        b_t = b_t[ [1,2,5,3,4,6] ]
        b_l = b_l[ [1,2,5,3,4,6] ]

        PN = [CNm_l; zeros(3,6)]
        SN_t = [DN_l; BN_t]
        SN_l = [DN_l; BN_l]

        XN_t = PN * R[start_id] + SN_t
        XN_l = PN * R[start_id] + SN_l

        # solve outer (tides)
        y_t = XN_t \ b_t
        # solve outer (load)
        y_l = XN_l \ b_l

        return y_t, y_l

    end


    """
        get_core_bc!(ω, r, ρ, g, μ, K, type, n; G0=1)

    Get the core boundary condition matrix `B` for the solid-body problem. The core boundary 
    conditions are derived from the requirement that the radial stress at the core-mantle 
    boundary must balance the tidal potential, and that the tangential stresses must vanish.

    # Arguments
    - `ω::prec`                           : Forcing frequency.
    - `r::prec`                           : Radial position of the core-mantle boundary.
    - `ρ::prec`                           : Average core density.
    - `g::prec`                           : Gravity at the core-mantle boundary.
    - `μ::precc`                          : Average core shear modulus.
    - `K::precc`                          : Average core bulk modulus.
    - `type::String`                      : Type of core boundary condition to apply ("liquid", "solid", or "inertial").
    - `n::Int`                            : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                        : Gravitational constant scale for non-dimensionalization.

    # Returns
    - `B::Array{precc,2}`                 : 3x6 matrix representing the linear constraint B * y = 0 at the core.
    """
    function get_core_bc!(ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, type::String, n::Int; G0=1)

        Ic = get_Ic(ω, r, ρ, g, μ, K, type, n; G0=G0)

        # Compute a basis for the left nullspace of Ic (i.e. x where xᵀ * Ic = 0).
        # This automatically generates the 3 linearly independent constraint rows 
        # for B * y = 0 without inverting Ms or assuming non-singularity.
        Bt = nullspace(transpose(Ic))  # 6x3 matrix

        # Return as 3x6 constraint matrix B
        return permutedims(Bt)
    end


    """
        compute_y(r, ρ, g, μ, K, ω, n, R, ρ_core, μ_core, κ_core, M_tot; core="liquid", inertial_terms=true, patch=false)

    Compute the solution `y` to the solid-body problem using a relaxation method. This function performs the 
    forward-backward relaxation scheme described in the main text of N. Kobayashi (2006), where we first solve 
    the radial system of ODEs to obtain the solution at the surface, and then perform back-substitution to 
    compute the solution at all interior points. 

    # Arguments
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `n::Int`                          : Tidal degree.
    - `ρ_core::prec`                    : Density of the core, used for core boundary conditions.
    - `μ_core::precc`                   : Shear modulus of the core, used for core boundary conditions.
    - `κ_core::precc`                   : Bulk modulus of the core, used for core boundary conditions.
    - `scales::Vector{prec}`            : Vector of scaling parameters for non-dimensionalization.

    # Keyword Arguments
    - `core::String="liquid"            : Type of core boundary condition to apply. Options are "liquid" for a fluid core, "solid" for a solid core, and "inertial" for a core with inertial response.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `patch::Bool=false`               : Whether to insert an infinitesimal solid shell around the core. This patches an issue where y2 and y4 become decoupled and cause the solution to diverge in fluid layers.

    # Returns
    - `y_t::Matrix{ComplexF64}`         : 6xN matrix of the solution at all radial grid points, where N is the number of radial layers. Each column corresponds to a radial grid point, and each row corresponds to a state variable (displacements, stresses, potential).
    - `y_l::Matrix{ComplexF64}`         : 6x1 matrix of the solution at the surface for the load problem.
    """    
    function compute_y(r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int, ρ_core::prec, μ_core::precc, κ_core::precc, scales::Vector{prec}; core="liquid", inertial_terms=true, patch=false)

        # solve radial system to get surface solution and recursion matrices
        yN_t, yN_l, R, S = solve_radial_system(r, ρ, g, μ, K, ω, n, ρ_core, μ_core, κ_core, scales; core=core, inertial_terms=inertial_terms, patch=patch)

        Nr = length(r)
        T = eltype(yN_t)

        # allocate as 6 x N matrix 
        y_t = Matrix{T}(undef, 6, Nr)
        y_l = Matrix{T}(undef, 6, 1)

        # solve outer  (tides)
        y_t[:, Nr] = yN_t
        # solve outer (load)
        y_l[:, 1] = yN_l

        # back-substitution
        for i in Nr-1:-1:1
            y_t[:, i] = R[i] * y_t[:, i+1]
        end

        # scale the solution back to physical units
        for i in 1:Nr
            y_t[:, i] = S * y_t[:, i]
        end
        y_l[:, 1] = S * y_l[:, 1]

        # convert to ComplexF64
        y_t = ComplexF64.(y_t)
        y_l = ComplexF64.(y_l)

        return y_t, y_l
    end

end