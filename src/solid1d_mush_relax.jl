

module solid1d_mush_relax
    
    include("constants.jl")
    include("common.jl")
    using .constants
    using .common

    using LinearAlgebra
    import GenericLinearAlgebra
    using DoubleFloats
    using StaticArrays
    using SpecialFunctions
    using SparseArrays
    using Printf

    
    """
        solve_radial_system(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n, ρ_core, μ_core, κ_core, scales; core="liquid", inertial_terms=true, patch=false)

    Solve the radial system of ODEs for the solid-body problem using a relaxation method. This function 
    implements the forward-backward relaxation scheme described in the main text of N. Kobayashi (2006).
    
    # Arguments
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρₗ::Vector{prec}`                 : Vector of liquid densities at the layer centers.
    - `Kl::Vector{prec}`                : Vector of liquid bulk moduli at the layer centers.
    - `Kd::Vector{precc}`               : Vector of drained bulk moduli at the layer centers.
    - `α::Vector{precc}`                : Vector of Biot coefficients at the layer centers.
    - `ηₗ::Vector{prec}`                 : Vector of liquid viscosities at the layer centers.
    - `ϕ::Vector{prec}`                 : Vector of porosities at the layer centers.
    - `k::Vector{prec}`                 : Vector of permeabilities at the layer centers.
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
    - `y_t::Vector{precc}`              : Vector of length 8 representing the tidal solution at the top of the mantle. This includes the displacements, stresses, and potential at the surface.
    - `y_l::Vector{precc}`              : Vector of length 8 representing the load solution at the top of the mantle. This includes the displacements, stresses, and potential at the surface.
    - `R::Vector{Matrix{precc}}`        : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `Y_inv::Vector{Int}`              : Vector of inverse ordering indices for the 8x8 case.
    - `transitions::Vector{Int}`        : Indices of the interface layers (the ones closer to the core and surface).
    """
    function solve_radial_system(r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρₗ::Vector{prec}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηₗ::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, n::Int,
                                ρ_core::prec, μ_core::precc, κ_core::precc, scales::Vector{prec}; core="liquid", inertial_terms=true, patch=false)

        # Define ordering
        Y6 = [1,2,4,5,3,6]
        Y8 = [1,2,5,6,3,7,4,8]
        Y8_inv = [1,2,5,7,3,4,6,8]

        # Identify Regions
        is_mush = k .> 0
        Nr = length(r)

        # Dynamic Scaling
        R0, M0, ω0, ρ0, G0, g0, μ0, S, Sinv = get_scales(scales[1], scales[2], scales[3]; Y=Y8)

        # insert solid shell around core
        if patch
            μ  = copy(μ)   # break aliasing with caller's array before mutating
            K  = copy(K)
            Kd = copy(Kd)
            k  = copy(k)
            ϕ  = copy(ϕ)
            μ[1] = precc(1.47e11)   # these values are chosen to be representative of a solid shell
            K[1] = precc(6.58e10)
            Kd[1] = precc(1.47e11)
            k[1] = 0.0
            ϕ[1] = 0.0
        end

        # Scale physical profiles to be dimensionless
        rs = r ./ R0
        ρs = ρ ./ ρ0
        gs = g ./ g0
        μs = μ ./ μ0
        Ks = K ./ μ0
        ωs = ω / ω0 
        ρₗs = ρₗ./ρ0
        Kls = Kl./μ0
        Kds = Kd./μ0
        ηₗs = ηₗ./(μ0/ω0)
        ks = k./R0^2

        # Initialize Matrices
        R = [zeros(precc, 8, 8) for _ in 1:Nr]
        B = [zeros(precc, 8, 1) for _ in 1:Nr]
        idx_6 = [1, 2, 3, 5, 6, 7]
        R6_view = [view(R[i], idx_6, idx_6) for i in 1:Nr]
        R8_view = [view(R[i], 1:8, 1:8) for i in 1:Nr]

        @debug("\n--- Adaptive Solver (Direct Interfaces) ---")

        curr_idx = 1
        # Step 1: Core Boundary
        if !is_mush[1]
            @debug("STEP 1: [Core Boundary] Solid")
            C1l, D2l = core_boundary(R6_view, (1, 2), rs, ρs, gs, μs, Ks, ωs, ρ_core/ρ0, μ_core/μ0, κ_core/μ0, core, n; G0=G0, inertial_terms=inertial_terms,Y=Y6)
            curr_idx = 2
        else
            @debug("STEP 1: [Core Boundary] Mushy")
            C1l, D2l = core_boundary_mush(R8_view, (1, 2), rs, ρs, gs, μs, Ks, ωs, ρₗs, Kls, Kds, α, ηₗs, ϕ, ks, ρ_core/ρ0, μ_core/μ0, κ_core/μ0, core, n; G0=G0, inertial_terms=inertial_terms, Y=Y8)
            curr_idx = 2
        end

        # Step 2: Propagation and Transitions
        while curr_idx < Nr
            # Find how long the current material state lasts
            next_change = findnext(x -> x != is_mush[curr_idx], is_mush, curr_idx)
            segment_end = (next_change === nothing) ? Nr : next_change - 1
            
            # Propagate through the uniform segment
            if segment_end > curr_idx
                if !is_mush[curr_idx]
                    @debug("STEP: [Propagate Solid] | Range: ($curr_idx, $segment_end)")
                    C1l, D2l = propagate_solid(R6_view, C1l, D2l, (curr_idx, segment_end-1), 
                                            rs, ρs, gs, μs, Ks, ωs, n; G0=G0, inertial_terms=inertial_terms, Y=Y6)
                else
                    @debug("STEP: [Propagate Mushy] | Range: ($curr_idx, $segment_end)")
                    C1l, D2l = propagate_mush(R8_view, C1l, D2l, (curr_idx, segment_end-1), 
                                            rs, ρs, gs, μs, Ks, ωs, ρₗs, Kls, Kds, α, ηₗs, ϕ, ks, n; G0=G0, inertial_terms=inertial_terms, Y=Y8)
                end
            end

            # Handle the material interface if one exists
            if next_change !== nothing
                # The interface function now operates directly between segment_end and next_change
                trans_range = (segment_end, next_change)
                
                if !is_mush[segment_end] && is_mush[next_change]
                    @debug("STEP: [Interface] Solid -> Mushy | Indices: $trans_range")
                    C1l, D2l = interface_solid_mush(R8_view, C1l, D2l, trans_range, 
                                            rs, ρs, gs, μs, Ks, ωs, ρₗs, Kls, Kds, α, ηₗs, ϕ, ks, n; G0=G0, inertial_terms=inertial_terms, Y=Y8)
                else
                    @debug("STEP: [Interface] Mushy -> Solid | Indices: $trans_range")
                    C1l, D2l = interface_mush_solid(R8_view, C1l, D2l, trans_range, 
                                            rs, ρs, gs, μs, Ks, ωs, ρₗs, Kls, Kds, α, ηₗs, ϕ, ks, n; G0=G0, inertial_terms=inertial_terms, Y=Y8)
                end
                curr_idx = next_change
            else
                curr_idx = Nr
            end
        end

        # Step 3: Surface Boundary
        if !is_mush[Nr]
            @debug("STEP: [Surface Boundary] Solid")
            y_t, y_l = surface_boundary(R6_view, C1l, D2l, (Nr-1, Nr), rs, ρs, gs, μs, Ks, ωs, n; G0=G0, Y=Y6)
        else
            @debug("STEP: [Surface Boundary] Mushy")
            y_t, y_l = surface_boundary_mush(R8_view, C1l, D2l, (Nr-1, Nr), rs, ρs, gs, μs, Ks, ωs, n; G0=G0, Y=Y8)
        end
        
        @debug("--- Solver Complete ---\n")

        # Note: transitions returned here are based on the original grid for plotting/analysis
        return y_t, y_l, R, S, Y8, findall(diff(is_mush) .!= 0)
    end


    """
        interface_mush_solid(R, Cn_l, Dnp_l, ids, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

    Perform the forward-backward relaxation step at the interface between the mushy layer and the solid layer. This 
    function implements the recursion described in N. Kobayashi (2007) for the transition from the 8x8 system to the 6x6 system.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `Cn_l::Matrix{precc}`             : 3x6 matrix representing the "stored" lower half of the Cn matrix from the previous step.
    - `Dnp_l::Matrix{precc}`            : 3x6 matrix representing the "stored" lower half of the Dnp matrix from the previous step.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρₗ::Vector{prec}`                 : Vector of liquid densities at the layer centers.
    - `Kl::Vector{prec}`                : Vector of liquid bulk moduli at the layer centers.
    - `Kd::Vector{precc}`               : Vector of drained bulk moduli at the layer centers.
    - `α::Vector{precc}`                : Vector of Biot coefficients at the layer centers.
    - `ηₗ::Vector{prec}`                 : Vector of liquid viscosities at the layer centers.
    - `ϕ::Vector{prec}`                 : Vector of porosities at the layer centers.
    - `k::Vector{prec}`                 : Vector of permeabilities at the layer centers.

    # keyword arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `Y::Vector{Int}=1:8`              : Vector of column indices corresponding to the 6x6 system variables in the 8x8 system. This allows for

    # Returns
    - `Cn_l::Matrix{precc}`             : Updated 3x6 matrix representing the "stored" lower half of the Cn matrix for the next iteration.
    - `Dnp_l::Matrix{precc}`            : Updated 3x6 matrix representing the "stored" lower half of the Dnp matrix for the next iteration.
    """
    function interface_mush_solid(R::Vector, Cn_l::Matrix{precc}, Dnp_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρₗ::Vector{prec}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηₗ::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, n::Int; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

        start_id, end_id = ids
        i = start_id

        # define target columns for the 6x6 system variables in the 8x8 system
        target_cols = [1,2,3,5,6,7]

        I8 = Matrix{precc}(I, 8, 8)

        Cn  = I8
        Dnp = -I8
        Dnp[Y[7], Y[7]] = 0.0 
        Dnp[Y[8], Y[8]] = 0.0

        # forward recursion
        dr = r[i+1] - r[i]

        # Calculate A at current and next step
        A_n  = get_A(ω, r[i],   ρ[i],   g[i],   μ[i],   K[i],
                        ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

        A_np = get_A(ω, r[i+1], ρ[i+1], g[i+1], μ[i+1], K[i+1],
                    ρₗ[i+1], Kl[i+1], Kd[i+1], α[i+1], ηₗ[i+1], ϕ[i+1], k[i+1], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

        Cn  .+= 0.5 * dr * A_n
        Dnp .+= 0.5 * dr * A_np

        # 1. Use the "stored" lower halves from the previous step 
        # to fill the upper blocks of P and S.
        Pn_u = Cn_l
        Sn_u = Dnp_l
        Qn_u = zeros(precc, 4, 8)

        # 2. Get the upper halves of the NEWLY calculated Cn and Dnp
        Cn_u  = Cn[1:4, :]
        Dnp_u = Dnp[1:4, :]

        # 3. Build the 6x6 blocks
        Pn = [Pn_u; zeros(precc, 4, 8)]
        Sn = [Sn_u; Cn_u]
        Qn = [Qn_u; Dnp_u]
        Kn = zeros(precc, 8, 8)
        Kn[Y[8], Y[8]] = 1.0

        # 4. Perform recursion
        Xn = Pn * R[i-1] + Sn + Kn

        R_ifc = -pinv(Xn) * Qn

        # create a mask or list of all column indices EXCEPT Y[7] and Y[8]
        active_cols = [idx for idx in 1:8 if idx != Y[7] && idx != Y[8]]

        # update R only for the rows that are not the Darcy flux constraint
        # this ensures y8 remains zero at the interface
        R[i][:, active_cols] .= R_ifc[:, active_cols]
        R[i][:, Y[8]]        .= 0.0  # explicitly enforce the impermeable boundary

        # 5. Update the "stored" lower halves for the next iteration
        Cn_l  = Cn[5:8, :]
        Dnp_l = Dnp[5:8, :]

        # return as 3x6 for the next step
        Cn_l = Cn_l[1:3, target_cols]
        Dnp_l = Dnp_l[1:3, target_cols]

        return Cn_l, Dnp_l
    end


    """
        interface_solid_mush(R, Cn_l, Dnp_l, ids, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

    Perform the forward-backward relaxation step at the interface between the solid layer and the mushy layer. This 
    function implements the recursion described in N. Kobayashi (2007) for the transition from the 6x6 system to the 8x8 system.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `Cn_l::Matrix{precc}`             : 3x6 matrix representing the "stored" lower half of the Cn matrix from the previous step.
    - `Dnp_l::Matrix{precc}`            : 3x6 matrix representing the "stored" lower half of the Dnp matrix from the previous step.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρₗ::Vector{prec}`                 : Vector of liquid densities at the layer centers.
    - `Kl::Vector{prec}`                : Vector of liquid bulk moduli at the layer centers.
    - `Kd::Vector{precc}`               : Vector of drained bulk moduli at the layer centers.
    - `α::Vector{precc}`                : Vector of Biot coefficients at the layer centers.
    - `ηₗ::Vector{prec}`                 : Vector of liquid viscosities at the layer centers.
    - `ϕ::Vector{prec}`                 : Vector of porosities at the layer centers.
    - `k::Vector{prec}`                 : Vector of permeabilities at the layer centers.
    - `n::Int`                          : Tidal degree.

    # keyword arguments
    - `G0::prec=1`                      : Gravitational constant scale for non-dimensionalization.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `Y::Vector{Int}=1:8`              : Vector of column indices corresponding to the 6x6 system variables in the 8x8 system. This allows for

    # Returns
    - `Cn_l::Matrix{precc}`             : Updated 3x6 matrix representing the "stored" lower half of the Cn matrix for the next iteration.
    - `Dnp_l::Matrix{precc}`            : Updated 3x6 matrix representing the "stored" lower half of the Dnp matrix for the next iteration.    
    """
    function interface_solid_mush(R::Vector, Cn_l::Matrix{precc}, Dnp_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρₗ::Vector{prec}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηₗ::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, n::Int; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

        start_id, end_id = ids
        i = start_id

        # define target columns for the 6x6 system variables in the 8x8 system
        target_cols = [1,2,3,5,6,7]

        I8 = Matrix{precc}(I, 8, 8)

        Cn  = I8
        Cn[Y[7], Y[7]] = 0.0 
        Dnp = -I8
        
        # forward recursion
        dr = r[i+1] - r[i]

        # calculate A at current and next step
        A_n  = get_A(ω, r[i],   ρ[i],   g[i],   μ[i],   K[i],
                        ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

        A_np = get_A(ω, r[i+1], ρ[i+1], g[i+1], μ[i+1], K[i+1],
                    ρₗ[i+1], Kl[i+1], Kd[i+1], α[i+1], ηₗ[i+1], ϕ[i+1], k[i+1], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

        Cn  .+= 0.5 * dr * A_n
        Dnp .+= 0.5 * dr * A_np

        # use the "stored" lower halves from the previous step 
        # to fill the upper blocks of P and S.
        # expand the incoming 3x6 Solid Lower blocks to 4x8
        Pn_u = zeros(precc, 4, 8)
        Pn_u[1:3, target_cols] .= Cn_l
        
        Sn_u = zeros(precc, 4, 8)
        Sn_u[1:3, target_cols] .= Dnp_l
        Qn_u = zeros(precc, 4, 8)

        # get the upper halves of the NEWLY calculated Cn and Dnp
        Cn_u  = Cn[1:4, :]
        Dnp_u = Dnp[1:4, :]

        # build the 8x8 blocks
        Pn = [Pn_u; zeros(precc, 4, 8)]
        Sn = [Sn_u; Cn_u]
        Qn = [Qn_u; Dnp_u]
        Kn = zeros(precc, 8, 8)
        Kn[Y[8], Y[8]] = 1.0

        # perform recursion
        R_prev = R[i-1]
        Xn = Pn * R_prev + Sn + Kn

        R_ifc = -pinv(Xn) * Qn

        # create a mask or list of all row indices EXCEPT Y[8]
        active_rows = [idx for idx in 1:8 if idx != Y[8]]

        # update R only for the rows that are not the Darcy flux constraint
        # this ensures y8 remains zero at the interface
        R[i][active_rows, :] .= R_ifc[active_rows, :]
        R[i][Y[8], :]        .= 0.0  # explicitly enforce the impermeable boundary

        # update the "stored" lower halves for the next iteration
        Cn_l  = Cn[5:8, :]
        Dnp_l = Dnp[5:8, :]

        return Cn_l, Dnp_l
    end


    """
        core_boundary(R, ids, r, ρ, g, μ, K, ω, ρ_core, μ_core, κ_core, core, n; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6])

    Perform the forward-backward relaxation step at the core boundary. This function implements the recursion described 
    in N. Kobayashi (2007) for the initial step of the relaxation scheme, where we apply the core boundary condition and 
    get the first solution for the first layer above the core.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρ_core::prec`                    : Density of the core, used for core boundary conditions.
    - `μ_core::precc`                   : Complex shear modulus of the core, used for core boundary conditions.
    - `κ_core::precc`                   : Complex bulk modulus of the core, used for core boundary conditions.
    - `core::String`                    : Type of core boundary condition to apply.
    - `n::Int`                          : Tidal degree.

    Keyword Arguments
    - `G0::prec=1`                      : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `Y::Vector{Int}=[1,2,3,4,5,6]`   : Ordering of the solution vector components.

    # Returns
    - `C1l::Matrix{precc}`              : 3x6 matrix representing the "stored" lower half of the C1 matrix for the next iteration.
    - `D2l::Matrix{precc}`              : 3x6 matrix representing the "stored" lower half of the D2 matrix for the next iteration.
    """
    function core_boundary(R::Vector, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρ_core::prec, μ_core::precc, κ_core::precc, core::String, n::Int; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6])

        start_id, end_id = ids

        # boundary conditions
        B1 = get_core_bc!(ω, r[start_id], ρ_core, g[start_id], μ_core, κ_core, core, n; G0=G0, Y=Y)        
        
        # first layer (n = 1)
        dr = r[end_id] - r[start_id]

        A1 = get_A(ω, r[start_id], ρ[start_id], g[start_id], μ[start_id], K[start_id], n; G0=G0, inertial_terms=inertial_terms, Y=Y)
        A2 = get_A(ω, r[end_id], ρ[end_id], g[end_id], μ[end_id], K[end_id], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

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
        R[start_id] .= -S1 \ Q1

        return C1l, D2l
    end


    """
        core_boundary_mush(R, ids, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, ρ_core, μ_core, κ_core, core, n; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

    Perform the forward-backward relaxation step at the core boundary for the two-phase problem. This function implements 
    the recursion described in N. Kobayashi (2007) for the initial step of the relaxation scheme, where we apply the core 
    boundary condition and get the first solution for the first layer above the core, but now using the full 8x8 system 
    that includes the porous layer components.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρₗ::Vector{prec}`                 : Vector of liquid densities at the layer centers.
    - `Kl::Vector{prec}`                : Vector of liquid bulk moduli at the layer centers.
    - `Kd::Vector{precc}`               : Vector of drained bulk moduli at the layer centers.
    - `α::Vector{precc}`                : Vector of Biot coefficients at the layer centers.
    - `ηₗ::Vector{prec}`                 : Vector of liquid viscosities at the layer centers.
    - `ϕ::Vector{prec}`                 : Vector of porosities at the layer centers.
    - `k::Vector{prec}`                 : Vector of permeabilities at the layer centers.
    - `ρ_core::prec`                    : Density of the core, used for core boundary conditions.
    - `μ_core::precc`                   : Complex shear modulus of the core, used for core boundary conditions.
    - `κ_core::precc`                   : Complex bulk modulus of the core, used for core boundary conditions.
    - `core::String`                    : Type of core boundary condition to apply.
    - `n::Int`                          : Tidal degree.

    Keyword Arguments
    - `G0::prec=1`                      : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `Y::Vector{Int}=[1,2,3,4,5,6,7,8]`: Ordering of the solution vector components.

    # Returns
    - `C1l::Matrix{precc}`              : 4x8 matrix representing the "stored" lower half of the C1 matrix for the next iteration.
    - `D2l::Matrix{precc}`              : 4x8 matrix representing the "stored" lower half of the D2 matrix for the next iteration.
    """
    function core_boundary_mush(R::Vector, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρₗ::Vector{prec}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηₗ::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, ρ_core::prec, μ_core::precc, κ_core::precc, core::String, n::Int; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

        start_id, end_id = ids

        # boundary conditions
        B1 = get_core_bc!(ω, r[start_id], ρ_core, g[start_id], μ_core, κ_core, core, n; G0=G0, Y=Y)     
        
        # first layer (n = 1)
        dr = r[end_id] - r[start_id]

        A1 = get_A(ω, r[start_id], ρ[start_id], g[start_id], μ[start_id], K[start_id],
                    ρₗ[start_id], Kl[start_id], Kd[start_id], α[start_id], ηₗ[start_id], ϕ[start_id], k[start_id], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

        A2 = get_A(ω, r[end_id], ρ[end_id], g[end_id], μ[end_id], K[end_id],
                ρₗ[end_id], Kl[end_id], Kd[end_id], α[end_id], ηₗ[end_id], ϕ[end_id], k[end_id], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

        I8 = Matrix{precc}(I, 8, 8)

        C1 =  I8 + 0.5 * dr * A1
        D2 = -I8 + 0.5 * dr * A2

        # split matrices
        C1u, C1l = C1[1:4, :], C1[5:8, :]
        D2u, D2l = D2[1:4, :], D2[5:8, :]

        # build S1 and Q1
        S1 = [B1; C1u]              # 8×8
        Q1 = [zeros(4,8); D2u]      # 8×8

        # initial recursion
        R[start_id] .= -pinv(S1) * Q1

        return C1l, D2l
    end

    
    """
        propagate_solid(R, Cn_l, Dnp_l, ids, r, ρ, g, μ, K, ω, n; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6])

    Perform the forward-backward relaxation step for the solid propagation segments. This function implements the 
    recursion described in N. Kobayashi (2007) for the segments of the radial grid that correspond to the solid 
    layers, where we propagate the solution up to the surface using the 6x6 system of equations.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
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
    - `Y::Vector{Int}=[1,2,3,4,5,6]`    : Ordering of the solution vector components.

    # Returns
    - `Cn_l::Matrix{precc}`             : Updated 3x6 matrix representing the "stored" lower half of the Cn matrix for the next iteration.
    - `Dnp_l::Matrix{precc}`            : Updated 3x6 matrix representing the "stored" lower half of the Dnp matrix for the next iteration.
    """
    function propagate_solid(R::Vector, Cn_l::Matrix{precc}, Dnp_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6])

        start_id, end_id = ids

        I6 = Matrix{precc}(I, 6, 6)

        # Cn_u = zeros(3,6)
        # Dnp_u = zeros(3,6)

        # forward recursion
        for i in start_id:end_id

            dr = r[i+1] - r[i]

            # Calculate A at current and next step
            A_n  = get_A(ω, r[i],   ρ[i],   g[i],   μ[i],   K[i],   n; G0=G0, inertial_terms=inertial_terms, Y=Y)
            A_np = get_A(ω, r[i+1], ρ[i+1], g[i+1], μ[i+1], K[i+1], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

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
            
            if i == start_id
                # For the first step into the mush, we may need to use pinv if the system is not yet fully constrained by the solid boundary conditions.
                R[i] .= - pinv(Xn) * Qn
            else
                R[i] .= -Xn \ Qn
            end

            # 5. Update the "stored" lower halves for the next iteration
            Cn_l  = Cn[4:6, :]
            Dnp_l = Dnp[4:6, :]
        end

        return Cn_l, Dnp_l
    end

    
    """
        propagate_mush(R, Cn_l, Dnp_l, ids, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

    Perform the forward-backward relaxation step for the mushy layer propagation segment. This function implements the
    recursion described in N. Kobayashi (2007) for the segment of the radial grid that corresponds to the mushy layer, 
    where we propagate the solution up to the surface using the full 8x8 system of equations that includes the porous 
    layer components.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `Cn_l::Matrix{precc}`             : 4x8 matrix representing the "stored" lower half of the Cn matrix from the previous step.
    - `Dnp_l::Matrix{precc}`            : 4x8 matrix representing the "stored" lower half of the Dnp matrix from the previous step.
    - `ids::Tuple{Int, Int}`            : Tuple containing the start and end indices of the current segment in the radial grid.
    - `r::Vector{prec}`                 : Vector of radial grid points (layer centers).
    - `ρ::Vector{prec}`                 : Vector of densities at the layer centers.
    - `g::Vector{prec}`                 : Vector of gravity values at the layer centers.
    - `μ::Vector{precc}`                : Vector of shear moduli at the layer centers.
    - `K::Vector{precc}`                : Vector of bulk moduli at the layer centers.
    - `ω::prec`                         : Angular frequency of the tidal forcing.
    - `ρₗ::Vector{prec}`                 : Vector of liquid densities at the layer centers.
    - `Kl::Vector{prec}`                : Vector of liquid bulk moduli at the layer centers.
    - `Kd::Vector{precc}`               : Vector of drained bulk moduli at the layer centers.
    - `α::Vector{precc}`                : Vector of Biot coefficients at the layer centers.
    - `ηₗ::Vector{prec}`                 : Vector of liquid viscosities at the layer centers.
    - `ϕ::Vector{prec}`                 : Vector of porosities at the layer centers.
    - `k::Vector{prec}`                 : Vector of permeabilities at the layer centers.
    - `n::Int`                          : Tidal degree. 

    Keyword Arguments
    - `G0::prec=1`                      : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.  
    - `Y::Vector{Int}=[1,2,3,4,5,6,7,8]`: Indices for the state variables (default is for standard case).

    # Returns
    - `Cn_l::Matrix{precc}`             : Updated 4x8 matrix representing the "stored" lower half of the Cn matrix for the next iteration.
    - `Dnp_l::Matrix{precc}`            : Updated 4x8 matrix representing the "stored" lower half of the Dnp matrix for the next iteration.
    """
    function propagate_mush(R::Vector, Cn_l::Matrix{precc}, Dnp_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρₗ::Vector{prec}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηₗ::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, n::Int; G0=1, inertial_terms=true, Y=[1,2,3,4,5,6,7,8])

        start_id, end_id = ids

        I8 = Matrix{precc}(I, 8, 8)

        # Cn_u = zeros(4,8)
        # Dnp_u = zeros(4,8)

        # forward recursion
        for i in start_id:end_id

            dr = r[i+1] - r[i]

            # Calculate A at current and next step
            A_n  = get_A(ω, r[i],   ρ[i],   g[i],   μ[i],   K[i],
                            ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

            A_np = get_A(ω, r[i+1], ρ[i+1], g[i+1], μ[i+1], K[i+1],
                        ρₗ[i+1], Kl[i+1], Kd[i+1], α[i+1], ηₗ[i+1], ϕ[i+1], k[i+1], n; G0=G0, inertial_terms=inertial_terms, Y=Y)

            Cn  =  I8 + 0.5 * dr * A_n
            Dnp = -I8 + 0.5 * dr * A_np

            # 1. Use the "stored" lower halves from the previous step 
            # to fill the upper blocks of P and S.
            Pn_u = Cn_l
            Sn_u = Dnp_l
            Qn_u = zeros(precc, 4, 8)

            # 2. Get the upper halves of the NEWLY calculated Cn and Dnp
            Cn_u  = Cn[1:4, :]
            Dnp_u = Dnp[1:4, :]

            # 3. Build the 6x6 blocks
            Pn = [Pn_u; zeros(precc, 4, 8)]
            Sn = [Sn_u; Cn_u]
            Qn = [Qn_u; Dnp_u]

            # 4. Perform recursion
            Xn = Pn * R[i-1] + Sn

            if i == start_id
                # For the first step into the mush, we may need to use pinv if the system is not yet fully constrained by the solid boundary conditions.
                R[i] .= -pinv(Xn) * Qn
            else
                R[i] .= -Xn \ Qn
            end

            # 5. Update the "stored" lower halves for the next iteration
            Cn_l  = Cn[5:8, :]
            Dnp_l = Dnp[5:8, :]
        end

        return Cn_l, Dnp_l
    end


    """
        surface_boundary(R, CNm_l, DN_l, ids, r, ρ, g, μ, K, ω, n; G0=1, Y=[1,2,3,4,5,6])

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
    - `Y::Vector{Int}=[1,2,3,4,5,6,7,8]`: Ordering of the solution vector components.

    # Returns
    - `y_t::Matrix{precc}`              : 6x1 matrix representing the solution at the surface for the tidal problem.
    - `y_l::Matrix{precc}`              : 6x1 matrix representing the solution at the surface for the load problem.
    """
    function surface_boundary(R::Vector, CNm_l::Matrix{precc}, DN_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int; G0=1, Y=[1,2,3,4,5,6,7,8])

        start_id, end_id = ids

        # tidal surface boundary condition
        BN_t, b_t = get_surface_bc!(r[end], g[end], n, 1, 0, 0, 0; G0=G0, Y=Y)
        # load surface boundary condition
        BN_l, b_l = get_surface_bc!(r[end], g[end], n, 0, 1, 0, 0; G0=G0, Y=Y)

        PN = [CNm_l; zeros(3,6)]
        SN_t = [DN_l; BN_t]
        SN_l = [DN_l; BN_l]

        XN_t = PN * R[start_id] + SN_t
        XN_l = PN * R[start_id] + SN_l

        # solve outer  (tides)
        y_t = XN_t \ b_t
        # solve outer (load)
        y_l = XN_l \ b_l

        return y_t, y_l

    end


    """
        surface_boundary_mush(R, CNm_l, DN_l, ids, r, ρ, g, μ, K, ω, n; G0=1, Y=[1,2,3,4,5,6,7,8])

    Perform the forward-backward relaxation step at the surface boundary for the two-phase problem. This function implements 
    the recursion described in N. Kobayashi (2007) for the final step of the relaxation scheme, where we apply the surface 
    boundary condition and solve for the final solution at the surface, using the full 8x8 system of equations that includes 
    the porous layer components.

    # Arguments
    - `R::Vector`                       : Vector of 8x8 matrices representing the coefficients of the ODE system at each radial layer.
    - `CNm_l::Matrix{precc}`            : 4x8 matrix representing the "stored" lower half of the CNm matrix from the previous step.
    - `DN_l::Matrix{precc}`             : 4x8 matrix representing the "stored" lower half of the DN matrix from the previous step.
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
    - `Y::Vector{Int}=[1,2,3,4,5,6,7,8]`: Ordering of the solution vector components.

    # Returns
    - `y_t::Matrix{precc}`              : 8x1 matrix representing the solution at the surface for the tidal problem.
    - `y_l::Matrix{precc}`              : 8x1 matrix representing the solution at the surface for the load problem.
    """
    function surface_boundary_mush(R::Vector, CNm_l::Matrix{precc}, DN_l::Matrix{precc}, ids::Tuple{Int, Int}, r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int; G0=1, Y=[1,2,3,4,5,6,7,8])

        start_id, end_id = ids

        # tidal surface boundary condition
        BN_t, b_t = get_surface_bc!(r[end], g[end], n, 1, 0, 0, 0; G0=G0, Y=Y)
        # load surface boundary condition
        BN_l, b_l = get_surface_bc!(r[end], g[end], n, 0, 1, 0, 0; G0=G0, Y=Y)

        PN = [CNm_l; zeros(4,8)]
        SN_t = [DN_l; BN_t]
        SN_l = [DN_l; BN_l]

        XN_t = PN * R[start_id] + SN_t
        XN_l = PN * R[start_id] + SN_l

        # solve outer  (tides)
        y_t = pinv(XN_t) * b_t
        # solve outer (load)
        y_l = pinv(XN_l) * b_l

        return y_t, y_l

    end


    """
        get_core_bc!(ω, r, ρ, g, μ, K, type, n; G0=1, Y=[1,2,3,4,5,6])

    Get the core boundary condition matrix `B` for the solid-body problem. The core boundary 
    conditions are derived from the requirement that the radial stress at the core-mantle 
    boundary must balance the tidal potential, and that the tangential stresses must vanish.

    # Arguments
    - `ω::Float64`                        : Forcing frequency.
    - `r::prec`                           : Radial position of the core-mantle boundary.
    - `ρ::prec`                           : Average core density.
    - `g::prec`                           : Gravity at the core-mantle boundary.
    - `μ::precc`                          : Average core shear modulus.
    - `K::precc`                          : Average core bulk modulus.
    - `type::String`                      : Type of core boundary condition to apply ("liquid", "solid", or "inertial").
    - `n::Int`                            : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                        : Gravitational constant scale for non-dimensionalization.
    - `Y::Vector{Int}=[1,2,3,4,5,6]`     : Indices for the state variables (supports M=6 standard or M=8 mushing systems).

    # Returns
    - `B::Array{precc,2}`                 : N x M matrix (where M = length(Y), N = M/2) representing the core boundary constraint B * y = 0.
    """
    function get_core_bc!(ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, type::String, n::Int; G0=1, Y=[1,2,3,4,5,6])
        
        # Get the Initial Conditions matrix (M x N, where N = M/2 fundamental solutions)
        Ic = get_Ic(ω, r, ρ, g, μ, K, type, n; G0=G0, Y=Y)

        # Compute the left nullspace directly via SVD (xᵀ * Ic = 0)
        # Returns an 8 x 4 matrix where columns form an orthonormal basis for the nullspace
        Bt = nullspace(transpose(Ic))

        # Return as 4 x 8 constraint matrix B
        return permutedims(Bt)
    end


    """
        compute_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n, ρ_core, μ_core, κ_core, M_tot; core="liquid", patch=false)

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
    - `ρₗ::Vector{prec}`                 : Vector of liquid densities at the layer centers.
    - `Kl::Vector{prec}`                : Vector of liquid bulk moduli at the layer centers.
    - `Kd::Vector{precc}`               : Vector of drained bulk moduli at the layer centers.
    - `α::Vector{precc}`                : Vector of Biot coefficients at the layer centers.
    - `ηₗ::Vector{prec}`                 : Vector of liquid viscosities at the layer centers.
    - `ϕ::Vector{prec}`                 : Vector of porosities at the layer centers.
    - `k::Vector{prec}`                 : Vector of permeabilities at the layer centers.
    - `n::Int`                          : Tidal degree.
    - `ρ_core::prec`                    : Density of the core, used for core boundary conditions.
    - `μ_core::precc`                   : Complex shear modulus of the core, used for core boundary conditions.
    - `κ_core::precc`                   : Complex bulk modulus of the core, used for core boundary conditions.
    - `scales::Vector{prec}`            : Vector of scaling parameters for non-dimensionalization.

    # Keyword Arguments
    - `core::String="liquid"            : Type of core boundary condition to apply. Options are "liquid" for a fluid core, "solid" for a solid core, and "inertial" for a core with inertial response.
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `patch::Bool=false`               : Whether to insert an infinitesimal solid shell around the core. This patches an issue where y2 and y4 become decoupled and cause the solution to diverge in fluid layers.

    # Returns
    - `y_t::Matrix{ComplexF64}`         : 8xN matrix of the solution at all radial grid points, where N is the number of radial layers. Each column corresponds to a radial grid point, and each row corresponds to a state variable (displacements, stresses, potential).
    - `y_l::Matrix{ComplexF64}`         : 8x1 matrix of the solution at the surface for the load problem.
    """    
    function compute_y(r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, ρₗ::Vector{prec}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηₗ::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, n::Int, ρ_core::prec, μ_core::precc, κ_core::precc, scales::Vector{prec}; core="liquid", inertial_terms=true, patch=false)

        # solve radial system to get surface solution and recursion matrices
        yN_t, yN_l, R, S, Y8, transitions = solve_radial_system(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n, ρ_core, μ_core, κ_core, scales; core=core, inertial_terms=inertial_terms, patch=patch)

        Nr = length(r)
        T = eltype(yN_t)
        M = length(yN_t)

        # allocate as 8 x N matrix 
        y_t = zeros(T, 8, Nr)
        y_l = zeros(T, 8, 1)

        # solve outer  (tides)
        y_t[1:M, Nr] = yN_t
        # solve outer (load)
        y_l[1:M, 1] = yN_l

        # back-substitution
        for i in Nr-1:-1:1
            y_t[:, i] = R[i] * y_t[:, i+1]        
        end

        # reorder y-functions to standard ordering (U, V, X, Y, phi, psi, P, R)
        y_t = y_t[Y8, :]
        y_l = y_l[Y8, :]
        
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