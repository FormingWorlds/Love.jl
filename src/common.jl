


module common

    include("constants.jl")
    using .constants

    import GenericLinearAlgebra
    using DoubleFloats
    using AssociatedLegendrePolynomials    
    using StaticArrays
    using SpecialFunctions
    using SparseArrays
    using LinearAlgebra
    using Optim

    export expand_layers, resample_profiles, get_g, optimize_scales, Ynm, define_spherical_grid, get_scales, doublefactorial, sbesselj, get_Ic, get_A, get_A!, get_surface_bc!, compute_strain_ten!, compute_darcy_displacement!, compute_pore_pressure!, get_heating_profile, get_heating_map

    
    """
        expand_layers(r; nr::Int=80)

    Discretize the primary layers given by `r` into `nr` discrete secondary layers.

    Note: exclusively used with shooting method discretization.

    # Arguments
    - `r::Array{prec,1}`               : 1D array of primary layer boundaries.

    # Keyword Arguments
    - `nr::Int=80`                        : Number of secondary layers to discretize.

    # Returns
    - `rs::Array{prec,2}`              : 2D array of secondary layer boundaries/
    """
    function expand_layers(r::Array{prec,1}; nr::Int=80)
        
        rs = zeros(prec, (nr+1, length(r)-1))
        
        for i in 1:length(r)-1
            rfine = LinRange(r[i], r[i+1], nr+1)
            rs[:, i] .= rfine[1:end] 
        end
    
        return rs
    end


    """
        resample_profiles(radius, rho, visc, shear, bulk, m_core, dr_min, dr_max)

    Resample the input profiles onto a new grid with `ncalc` points. The new grid is generated using a 
    stretched and refined scheme, which allows for better resolution in regions of interest (e.g., near 
    layer boundaries). 

    Note: exclusively used with relaxation method discretization.

    # Arguments
    - `radius::Vector{prec}`              : Original radius profile (layer boundaries).
    - `rho::Vector{prec}`                 : Original density profile (defined at layer centers).
    - `visc::Vector{prec}`                : Original viscosity profile (defined at layer centers).
    - `shear::Vector{precc}`              : Original shear modulus profile (defined at layer centers).
    - `bulk::Vector{precc}`               : Original bulk modulus profile (defined at layer centers).
    - `m_core::prec`                      : Mass of the core, used for gravity calculations.
    - `Δr_min::Int64`                     : Minimum grid spacing for the new grid.
    - `Δr_max::Int64`                     : Maximum grid spacing for the new grid.

    # Returns
    Tuple of resampled profiles on the new grid:
    - `r_new_b::Vector{prec}`             : New radius profile at layer boundaries.
    - `ρ_new::Vector{prec}`               : New density profile at layer centers.
    - `η_new::Vector{prec}`               : New viscosity profile at layer centers.
    - `μ_new::Vector{precc}`              : New shear modulus profile at layer centers.
    - `κ_new::Vector{precc}`              : New bulk modulus profile at layer centers.
    - `g_new::Vector{prec}`               : New gravity profile at layer centers.
    - `M_tot::Float64`                    : Total mass enclosed within the outermost layer boundary.
    """ 
    function resample_profiles(radius::Vector{prec}, rho::Vector{prec}, visc::Vector{prec}, shear::Vector{precc}, bulk::Vector{precc}, m_core::prec, dr_min::Int64, dr_max::Int64)
        # setup grids
        α = log(dr_max / dr_min)

        N = Int(ceil((radius[end] - radius[1]) / dr_min * α / (exp(α) - 1)))

        # indices i = 1:N
        i = collect(1:N)

        # convert to BigFloat for consistency
        i_bf = prec.(i)
        N_bf = prec(N)

        # compute normalized coordinate (N - i)/(N - 1)
        ξ = (N_bf .- i_bf) ./ (N_bf - 1)

        # compute r_i
        r_new_b = radius[end] .+ (radius[1] - radius[end]) .* (
            (exp.(α .* ξ) .- 1) ./ (exp(α) - 1)
        )

        # cell centers
        r_new_c = 0.5 .* (r_new_b[1:end-1] .+ r_new_b[2:end])

        # obtain new profiles (Constant per original layer)
        ρ_new = similar(rho, N-1)
        η_new = similar(visc, N-1)
        μ_new = similar(shear, N-1)
        κ_new = similar(bulk, N-1)

        for i in 1:N-1
            # find index such that r_b[idx] <= r_new_c[i] < r_b[idx+1]
            idx = searchsortedfirst(radius, r_new_c[i]) - 1
            idx = clamp(idx, 1, length(rho)) # Safety clamp

            ρ_new[i] = rho[idx]
            η_new[i] = visc[idx]
            μ_new[i] = shear[idx]
            κ_new[i] = bulk[idx]
        end

        g_new, M_tot = get_g(r_new_b, ρ_new, m_core) 

        return r_new_b, ρ_new, η_new, μ_new, κ_new, g_new, M_tot
    end


    """
        resample_profiles(radius, rho, visc, shear, bulk_s, bulk_l, bulk_d, alpha, visc_l, phi, k, m_core, dr_min, dr_max)

    Resample the input profiles onto a new grid with `ncalc` points. The new grid is generated using a 
    stretched and refined scheme, which allows for better resolution in regions of interest (e.g., near 
    layer boundaries). 

    Note: exclusively used with relaxation method discretization.

    # Arguments
    - `radius::Vector{prec}`              : Original radius profile (layer boundaries).
    - `rho::Vector{prec}`                 : Original density profile (defined at layer centers).
    - `visc::Vector{prec}`                : Original viscosity profile (defined at layer centers).
    - `shear::Vector{precc}`              : Original shear modulus profile (defined at layer centers).
    - `bulk_s::Vector{precc}`             : Original solid bulk modulus profile (defined at layer centers).
    - `bulk_l::Vector{prec}`              : Original liquid bulk modulus profile (defined at layer centers).
    - `bulk_d::Vector{precc}`             : Original deep bulk modulus profile (defined at layer centers).
    - `alpha::Vector{precc}`              : Original alpha profile (defined at layer centers).
    - `visc_l::Vector{prec}`              : Original liquid viscosity profile (defined at layer centers).
    - `phi::Vector{prec}`                 : Original phi profile (defined at layer centers).
    - `m_core::prec`                      : Mass of the core, used for gravity calculations.
    - `Δr_min::Int64`                     : Minimum grid spacing for the new grid.
    - `Δr_max::Int64`                     : Maximum grid spacing for the new grid.

    # Returns
    Tuple of resampled profiles on the new grid:
    - `r_new_b::Vector{prec}`             : New radius profile at layer boundaries.
    - `ρ_new::Vector{prec}`               : New density profile at layer centers.
    - `η_new::Vector{prec}`               : New viscosity profile at layer centers.
    - `μ_new::Vector{precc}`              : New shear modulus profile at layer centers.
    - `κs_new::Vector{precc}`             : New solid bulk modulus profile at layer centers.
    - `κl_new::Vector{prec}`              : New liquid bulk modulus profile at layer centers.
    - `κd_new::Vector{precc}`             : New deep bulk modulus profile at layer centers.
    - `α_new::Vector{precc}`              : New alpha profile at layer centers.
    - `ηl_new::Vector{prec}`              : New liquid viscosity profile at layer centers.
    - `φ_new::Vector{prec}`               : New phi profile at layer centers.
    - `k_new::Vector{prec}`               : New k profile at layer centers.
    - `g_new::Vector{prec}`               : New gravity profile at layer centers.
    - `M_tot::prec`                       : Total mass of the body, used for non-dimensionalization.
    """ 
    function resample_profiles(radius::Vector{prec}, rho::Vector{prec}, visc::Vector{prec}, shear::Vector{precc}, bulk_s::Vector{precc}, bulk_l::Vector{prec}, bulk_d::Vector{precc}, alpha::Vector{precc}, visc_l::Vector{prec}, phi::Vector{prec}, k::Vector{prec}, m_core::prec, dr_min::Int64, dr_max::Int64)
        # setup grids
        α = log(dr_max / dr_min)

        N = Int(ceil((radius[end] - radius[1]) / dr_min * α / (exp(α) - 1)))

        # indices i = 1:N
        i = collect(1:N)

        # convert to prec for consistency
        i_bf = prec.(i)
        N_bf = prec(N)

        # compute normalized coordinate (N - i)/(N - 1)
        ξ = (N_bf .- i_bf) ./ (N_bf - 1)

        # compute r_i
        r_new_b = radius[end] .+ (radius[1] - radius[end]) .* (
            (exp.(α .* ξ) .- 1) ./ (exp(α) - 1)
        )

        # cell centers
        r_new_c = 0.5 .* (r_new_b[1:end-1] .+ r_new_b[2:end])

        # obtain new profiles (Constant per original layer)
        ρ_new = similar(rho, N-1)
        η_new = similar(visc, N-1)
        μ_new = similar(shear, N-1)
        κs_new = similar(bulk_s, N-1)
        κl_new = similar(bulk_l, N-1)
        κd_new = similar(bulk_d, N-1)
        α_new = similar(alpha, N-1)
        ηl_new = similar(visc_l, N-1)
        φ_new = similar(phi, N-1)
        k_new = similar(k, N-1)

        for i in 1:N-1
            # find index such that r_b[idx] <= r_new_c[i] < r_b[idx+1]
            idx = searchsortedfirst(radius, r_new_c[i]) - 1
            idx = clamp(idx, 1, length(rho)) # Safety clamp

            ρ_new[i] = rho[idx]
            η_new[i] = visc[idx]
            μ_new[i] = shear[idx]
            κs_new[i] = bulk_s[idx]
            κl_new[i] = bulk_l[idx]
            κd_new[i] = bulk_d[idx]
            α_new[i] = alpha[idx]
            ηl_new[i] = visc_l[idx]
            φ_new[i] = phi[idx]
            k_new[i] = k[idx]
        end

        g_new, M_tot = get_g(r_new_b, ρ_new, m_core) 

        return r_new_b, ρ_new, η_new, μ_new, κs_new, κl_new, κd_new, α_new, ηl_new, φ_new, k_new, g_new, M_tot
    end


    """
        get_g(r, ρ, m_core)

    Compute the radial gravity structure associated with a density profile `r` at intervals given by `r`.

    Note: exclusively used with shooting method discretization.

    # Arguments
    - `r::Array{prec,2}`               : 2D array of layer boundaries. 
    - `ρ::Array{prec,1}`               : 1D array of layer densities. The length of `ρ` must be equal to the number of columns in `r`.
    - `m_core::prec`                   : Mass of the core, which is used to compute the gravity at the core boundary.

    # Returns
    - `g::Array{prec,2}`               : 2D array of gravity values at the layer boundaries. The dimensions of `g` are the same as `r`.

    # Notes
    `r` must be be a 2D array, with index 1 representing the top radius of secondary layers, and index 2
    representing the top radius of primary layers. 
    """
    function get_g(r::Array{prec,2}, ρ::Array{prec,1}, m_core::prec)
        g = zeros(prec, size(r))
        M = zeros(prec, size(r))

        # Base mass enclosed at the inner boundary of the first layer (Core Mass)
        M[1, 1] = m_core

        # Shell mass incremental calculations
        for i in 1:size(r, 2)
            # Shell masses for elements 2:end in layer i
            M[2:end, i] = (4.0/3.0) * π .* diff(r[:, i].^3) .* ρ[i]
        end

        # Cumulative mass enclosed at every point in the grid
        M_enclosed = accumulate(+, M)

        # Gravity g = G * M_enclosed / r^2
        g .= G .* M_enclosed ./ (r.^2)

        return g
    end


    """
        get_g(r, ρ, m_core)

    Compute the radial gravity structure associated with a density profile `r` at intervals given by `r`.

    Note: exclusively used with relaxation method discretization.
    # Arguments
    - `r::Array{prec,1}`               : 1D array of layer boundaries. 
    - `ρ::Array{prec,1}`               : 1D array of layer densities. The length of `ρ` must be equal to the number of columns in `r`.
    - `m_core::prec`                   : Mass of the core.

    # Returns
    - `g::Array{prec,1}`               : 1D array of gravity values at the layer boundaries. The dimensions of `g` are the same as `r`.
    - `M_enc::prec`                    : Total mass enclosed within the outermost layer boundary.
    """
    function get_g(r::Vector{prec}, ρ::Vector{prec}, m_core::prec)

        dm = 4.0/3.0 * π .* diff(r.^3) .* ρ

        M_enc = cumsum(dm) .+ m_core
            
        g = G .* M_enc ./ r[2:end].^2

        return g, M_enc[end]
    end


    """
        optimize_scales(r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int, p0::Vector{prec})

    Optimize the scaling parameters (R0, M0, G0) to minimize the condition number of the system matrix A across the radial profile. This ensures numerical stability in solving the ODEs.

    # Arguments
    - `r::Vector{prec}`                  : Radial positions (m).
    - `ρ::Vector{prec}`                  : Density profile (kg/m^3).
    - `g::Vector{prec}`                  : Gravity profile (m/s^2).
    - `μ::Vector{precc}`                 : Shear modulus profile (Pa).
    - `K::Vector{precc}`                 : Bulk modulus profile (Pa).
    - `ω::prec`                          : Angular frequency (rad/s).
    - `n::Int`                           : Tidal degree.
    - `p0::Vector{prec}`                 : Initial guess for scales [R0, M0, G0].

    # Returns
    - `best_params::Vector{prec}`        : Optimized scales [R0, M0, G0].
    """
    function optimize_scales(r::Vector{prec}, ρ::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, K::Vector{precc}, ω::prec, n::Int, p0::Vector{prec})::Vector{prec}

        function objective_cond(scales)
            R0, M0, G0 = scales
            
            # Enforce strict positive parameters to avoid unphysical divisions/roots
            if R0 <= 0 || M0 <= 0 || G0 <= 0
                return Inf
            end
            
            # Recalculate derived scales matching `solid1d_relax.get_scales` internals
            ρ0 = M0 / (R0^3)
            g0 = G0 * M0 / (R0^2)
            ω0 = sqrt(G0 * ρ0)
            μ0 = g0 * ρ0 * R0  # Assuming standard reference stress scale
            
            # Scale profiles to dimensionless forms using current trial scales
            rs = r ./ R0
            ρs = ρ ./ ρ0
            gs = g ./ g0
            μs = μ ./ μ0
            Ks = K ./ μ0
            ωs = ω / ω0
            
            ntotal = length(rs)
            max_log_cond = -Inf
            
            tmp_A = zeros(precc, 6, 6)
            
            # Find worst condition number across the radial slice
            for i in 1:ntotal
                get_A!(tmp_A, ωs, rs[i], ρs[i], gs[i], μs[i], Ks[i], n; G0=G0)
                c_num = log10(cond(tmp_A))
                if c_num > max_log_cond
                    max_log_cond = c_num
                end
            end
            
            return max_log_cond
        end

        @info "Optimizing scales to minimize condition number of A..."

        # Run Nelder-Mead optimization
        res = optimize(objective_cond, p0, NelderMead(), Optim.Options(iterations=500))
        
        best_params = res.minimizer
        min_conds   = res.minimum
        if min_conds > 14.5
            @warn "Warning: Condition number of A is high (log10(cond(A)) = $min_conds). This may indicate numerical instability."
        end

        @info "Optimized scales: R0=$(best_params[1]), M0=$(best_params[2]), G0=$(best_params[3]), log10(cond(A))=$min_conds"

        return best_params
    end


    """
        optimize_scales(r::Vector{prec}, ρs::Vector{prec}, ρl::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, Ks::Vector{precc}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηl::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, ω::prec, n::Int, p0::Vector{prec})

    Optimize the scaling parameters (R0, M0, G0) to minimize the condition number of the system matrix A across the radial profile for a mushy mantle model. This ensures numerical stability in solving the ODEs.

    # Arguments
    - `r::Vector{prec}`                  : Radial positions (m).
    - `ρs::Vector{prec}`                 : Solid density profile (kg/m^3).
    - `ρl::Vector{prec}`                 : Liquid density profile (kg/m^3).
    - `g::Vector{prec}`                  : Gravity profile (m/s^2).
    - `μ::Vector{precc}`                 : Shear modulus profile (Pa).
    - `Ks::Vector{precc}`                : Solid bulk modulus profile (Pa).
    - `Kl::Vector{prec}`                 : Liquid bulk modulus profile (Pa).
    - `Kd::Vector{precc}`                : Drained bulk modulus profile (Pa).
    - `α::Vector{precc}`                 : Biot modulus profile (dimensionless).
    - `ηl::Vector{prec}`                 : Liquid viscosity profile (Pa·s).
    - `ϕ::Vector{prec}`                  : Porosity profile (dimensionless).
    - `k::Vector{prec}`                  : Permeability profile (m^2).
    - `ω::prec`                          : Forcing frequency (Hz).
    - `n::Int`                           : Tidal degree.
    - `p0::Vector{prec}`                 : Initial guess for scales [R0, M0, G0].

    # Returns
    - `best_params::Vector{prec}`        : Optimized scales [R0, M0, G0].
    """
    function optimize_scales(r::Vector{prec}, ρs::Vector{prec}, ρl::Vector{prec}, g::Vector{prec}, μ::Vector{precc}, Ks::Vector{precc}, Kl::Vector{prec}, Kd::Vector{precc}, α::Vector{precc}, ηl::Vector{prec}, ϕ::Vector{prec}, k::Vector{prec}, ω::prec, n::Int, p0::Vector{prec})::Vector{prec}

        function objective_cond(scales)
            R0, M0, G0 = scales
            
            # Enforce strict positive parameters to avoid unphysical divisions/roots
            if R0 <= 0 || M0 <= 0 || G0 <= 0
                return Inf
            end
            
            # Recalculate derived scales matching `solid1d_relax.get_scales` internals
            ρ0 = M0 / (R0^3)
            g0 = G0 * M0 / (R0^2)
            ω0 = sqrt(G0 * ρ0)
            μ0 = g0 * ρ0 * R0  # Assuming standard reference stress scale
            
            # Scale profiles to dimensionless forms using current trial scales
            rs = r ./ R0
            ρs = ρs ./ ρ0
            gs = g ./ g0
            μs = μ ./ μ0
            Kss = Ks ./ μ0
            ωs = ω / ω0 
            ρls = ρl./ρ0
            Kls = Kl./μ0
            Kds = Kd./μ0
            ηls = ηl./(μ0/ω0)
            ks = k./R0^2
            
            ntotal = length(rs)
            max_log_cond = -Inf
            
            tmp_A = zeros(precc, 8, 8)
            
            # Find worst condition number across the radial slice
            for i in 1:ntotal
                get_A!(tmp_A, ωs, rs[i], ρs[i], gs[i], μs[i], Kss[i], ρls[i], Kls[i], Kds[i], α[i], ηls[i], ϕ[i], ks[i], n; G0=G0)
                c_num = log10(cond(tmp_A))
                if c_num > max_log_cond
                    max_log_cond = c_num
                end
            end
            
            return max_log_cond
        end

        @info "Optimizing scales to minimize condition number of A..."

        # Run Nelder-Mead optimization
        res = optimize(objective_cond, p0, NelderMead(), Optim.Options(iterations=500))
        
        best_params = res.minimizer
        min_conds   = res.minimum
        if min_conds > 14.5
            @warn "Warning: Condition number of A is high (log10(cond(A)) = $min_conds). This may indicate numerical instability."
        end

        @info "Optimized scales: R0=$(best_params[1]), M0=$(best_params[2]), G0=$(best_params[3]), log10(cond(A))=$min_conds"

        return best_params
    end
        
    
    """
        Ynm(n::Int, m::Int, theta::Array{Float64,1}, phi::Array{Float64,1})

    Compute the spherical harmonic Ynm for given n, m, theta, and phi.

    # Arguments
    - `n::Int`                          : Tidal degree.
    - `m::Int`                          : Tidal order.
    - `theta::AbstractArray`            : Array of colatitudes in radians.
    - `phi::AbstractArray`              : Array of longitudes in radians.

    # Returns
    - `Ynm::Array{ComplexF64,2}`        : 2D array of spherical harmonic values for each combination of theta and phi.
    """
    function Ynm(n::Int, m::Int, theta::AbstractArray, phi::AbstractArray)::Array{ComplexF64,2}
        return Plm.(n, m, cos.(theta)) .* exp.(1im * m .* phi)
    end


    """
        define_spherical_grid(res::Float64, n::Int, m::Int)

    Create the spherical grid of angular resolution `res` in degrees. This is used for 
    numerical integrations over solid angle. A new grid can easily be defined by 
    recalling the function with a new `res`.

    # Arguments
    - `res::Float64`                     : Desired angular resolution in degrees.
    - `n::Int`                           : Tidal degree.
    - `m::Int`                           : Tidal order.

    # Returns
    - `SphericalGrid::NamedTuple`        : A named tuple containing the spherical grid information.
    """
    function define_spherical_grid(res::Float64, n::Int, m::Int)::NamedTuple
        # θ and φ grids
        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))
        clats[1] += 1e-6
        clats[end] -= 1e-6

        # allocate arrays
        Y    = zeros(ComplexF64, 1, length(clats), length(lons))
        dYdθ = similar(Y)
        dYdϕ = similar(Y)
        Z    = similar(Y)
        X    = similar(Y)

        sinθ = sin.(clats)
        cosθ = cos.(clats)
        cotθ = cosθ ./ sinθ
        cscθ = csc.(clats)

        # Normalization factor for spherical harmonics
        norm = sqrt((2*n+1)  * factorial(n-m) / (4π * factorial(n+m)))
        
        i = 1

        # Y
        Y[i,:,:] = Ynm(n,m,clats,lons)

        # ∂Y/∂θ
        Pn = Plm.(n, m, cosθ)
        if n > m
            Pn_1 = Plm.(n-1, m, cosθ)
            dPdθ = (n .* cosθ .* Pn .- (n + m) .* Pn_1) ./ (sinθ)
        else
            # m == n -> P_{n-1}^m = 0
            dPdθ = (n .* cosθ .* Pn) ./ (sinθ)
        end
        dYdθ[i,:,:] .= dPdθ .* exp.(1im .* m .* lons)

        # ∂Y/∂ϕ
        dYdϕ[i,:,:] .= 1im * m .* Y[i,:,:]

        # Z = 2 ((1/sinθ) ∂²Y/∂θ∂ϕ - cotθ cscθ ∂Y/∂ϕ)
        Z[i,:,:] .= 2 .* (1im * m ./ sinθ .* dYdθ[i,:,:] .- cotθ .* cscθ .* dYdϕ[i,:,:])

        # X = -2 (cotθ ∂Y/∂θ + csc²θ ∂²Y/∂ϕ²) - n(n+1)) Y
        X[i,:,:] .= -2 .* (cotθ .* dYdθ[i,:,:] .- cscθ.^2 .* m^2 .* Y[i,:,:]) .- n*(n+1) .* Y[i,:,:]

        # Normalize
        Y[i,:,:]    .*= norm
        dYdθ[i,:,:] .*= norm
        dYdϕ[i,:,:] .*= norm
        Z[i,:,:]    .*= norm
        X[i,:,:]    .*= norm

        SphericalGrid = (
            res = res,
            clats = clats,
            lons = lons,
            Y = Y,
            dYdθ = dYdθ,
            dYdϕ = dYdϕ,
            Z = Z,
            X = X
        )

        return SphericalGrid
    end


    """
        get_scales(R0::prec, M0::prec, G0::prec; Y::Vector{Int}=[1,2,3,4,5,6])

    Compute the characteristic scales for the problem based on a reference radius `R0`, mass `M0`, and gravitational constant scale 
    `G0`. These scales are used to non-dimensionalize the equations and ensure numerical stability.

    # Arguments
    - `R0::prec`                         : Reference radius scale (e.g., planetary radius).
    - `M0::prec`                         : Reference mass scale (e.g., planetary mass).
    - `G0::prec`                         : Reference gravitational constant scale.

    # Keyword Arguments
    - `Y::Vector{Int}=[1,2,3,4,5,6]`     : Ordering of the solution vector components. Merely used to determine the size of the scaling matrix S.

    # Returns
    Tuple of characteristic scales:
    - `R0::prec`                         : Length scale (m).
    - `M0::prec`                         : Mass scale (kg).
    - `ω0::prec`                         : Frequency scale (rad/s).
    - `ρ0::prec`                         : Density scale (kg/m^3).
    - `G0::prec`                         : Gravitational constant scale (m^3 kg^-1 s^-2).
    - `g0::prec`                         : Gravity scale (m/s^2).
    - `μ0::prec`                         : Shear modulus scale (Pa).
    - `S::Diagonal{prec}`                : Diagonal scaling matrix for state variables.
    - `Sinv::Diagonal{prec}`             : Inverse of the scaling matrix S.
    """
    function get_scales(R0::prec, M0::prec, G0::prec; Y=[1,2,3,4,5,6])::Tuple

        # Define the number of state variables based on the length of Y
        N = length(Y)

        # Derive dependent scales
        ρ0 = M0 / ((4/3) * π * R0^3)
        ω0 = sqrt(G0 * ρ0)
        g0 = G0 * ρ0 * R0
        μ0 = G0 * (ρ0^2) * (R0^2)
        P0 = G0 * ρ0 * (R0^2)

        # Define the scaling matrix S and its inverse
        S = zeros(prec, N, N)
        if N == 4
            S[1, 1] = 1.0/g0     # y1: radial displacement (m)
            S[2, 2] = μ0/(g0*R0) # y3: radial stress (Pa)
            S[3, 3] = 1.0        # y5: potential (m^2/s^2)
            S[4, 4] = g0/P0      # y6: potential gradient/gravity (m/s^2)
        elseif N == 6
            S[1, 1] = 1.0/g0     # y1: radial displacement (m)
            S[2, 2] = 1.0/g0     # y2: tangential displacement (m)
            S[3, 3] = μ0/(g0*R0) # y3: radial stress (Pa)
            S[4, 4] = μ0/(g0*R0) # y4: tangential stress (Pa)
            S[5, 5] = 1.0        # y5: potential (m^2/s^2)
            S[6, 6] = g0/P0      # y6: potential gradient/gravity (m/s^2)
        elseif N == 8
            S[1, 1] = 1.0/g0     # y1: radial displacement (m)
            S[2, 2] = 1.0/g0     # y2: tangential displacement (m)
            S[3, 3] = μ0/(g0*R0) # y3: radial stress (Pa)
            S[4, 4] = μ0/(g0*R0) # y4: tangential stress (Pa)
            S[5, 5] = 1.0        # y5: potential (m^2/s^2)
            S[6, 6] = g0/P0      # y6: potential gradient/gravity (m/s^2)
            S[7, 7] = μ0/(g0*R0) # y7: pore pressure (Pa)
            S[8, 8] = 1.0/g0     # y8: relative radial displacement (m)
        end

        Sinv = inv(S)
        return R0, M0, ω0, ρ0, G0, g0, μ0, S, Sinv
    end


    """
        doublefactorial(n::Int)

    Compute the double factorial of an integer n, defined as n!! = n * (n-2) * (n-4) * ... until 1 or 0.

    # Arguments
    - `n::Int`                      : The integer for which to compute the double factorial. Must be non-negative.

    # Returns
    - `result::Int`                 : The double factorial of n.
    """
    function doublefactorial(n::Int)::Int
        n < 0 && error("doublefactorial not defined for negative n")
        n == 0 && return one(n)
        n == 1 && return one(n)

        result = one(n)
        for k in n:-2:1
            result *= k
        end
        return result
    end

    
    """
        sbesselj(nu::Int, x::T) where T <: Number

    Compute the scaled spherical Bessel function of the first kind, defined as:
    sbesselj(nu, x) = sqrt(π / (2x)) * besselj(nu + 0.5, x)

    # Arguments
    - `nu::Int`                       : Order of the spherical Bessel function.
    - `x::T`                          : Argument of the Bessel function. Can be a real or complex number.

    # Returns
    - `result::T`                     : The value of the scaled spherical Bessel function of the first kind at the given order and argument.
    """
    function sbesselj(nu::Int, x::T) where T <: Number
        if T <: Complex{BigFloat} || T <: BigFloat
            # Cast to Float64 for SpecialFunctions.jl compatibility
            x64 = ComplexF64(x)
            val = sqrt(π / (2 * x64)) * besselj(nu + 0.5, x64)
            return parse(T, string(val)) # Robust way to bring back to BigFloat
        else
            return sqrt(π / (2 * x)) * besselj(nu + 0.5, x)
        end
    end


    """
        get_Ic(ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, type::String, n::Int; G0::prec=1, Y::Vector{Int}=[1,2,3,4,5,6])

    Core solution basis at the core-mantle boundary: the linearly independent
    starting vectors for radially integrating the tidal/Love-number ODEs
    through the overlying mantle.

    # Core types

    - `"solid"`: Static, incompressible, elastic solid core (Love 1911). Valid
      in the incompressible limit (K -> infinity); K does not enter the
      formula. All three columns are genuine elementary solutions, regular at
      r = 0.
    - `"liquid"`: Static (ω = 0) fluid core. A hydrostatic fluid responds to
      loading instantaneously, so this is not a continuous solution of the
      interior ODE but a boundary condition embedding the liquid-solid
      interface: two columns carry the physical degrees of freedom, and one
      column is a pure tangential-slip vector (Y[2] = 1, elsewhere 0),
      representing the discontinuity in horizontal displacement that a fluid
      core permits.
    - `"inertial"`: General, oscillating (ω != 0), compressible, finite-μ/K
      solid core (Takeuchi & Saito 1972; Kervazo et al. 2021). All three
      columns are genuine elementary solutions, regular at r = 0: one regular
      power-law solution, and two independent solutions built from spherical
      Bessel functions (the two roots of the dispersion relation for the
      radial wavenumber). A solid core has no fluid-solid interface, so —
      unlike `"liquid"` and `"inertial-liquid"` — none of its columns is a
      slip vector.
    - `"inertial-liquid"`: General, oscillating (ω != 0) fluid (μ = 0) core
      (Takeuchi & Saito 1972; Kervazo et al. 2021; Korenaga 2025). Two columns
      are genuine elementary solutions (one regular, one Bessel-based); the 
      third is the tangential-slip vector, as in `"liquid"`.

    # How the core types relate

    - `"solid"` (μ -> 0) reduces to `"liquid"` (ω = 0): incompressible, static
      elasticity without shear rigidity is a hydrostatic fluid.
    - `"inertial-liquid"` reduces to `"liquid"` as ω -> 0: a slowly
      oscillating fluid approaches hydrostatic equilibrium.
    - `"inertial"` reduces to `"inertial-liquid"` as μ -> 0: an oscillating
      solid without shear rigidity is an oscillating fluid.
    - `"inertial"` reduces to `"solid"` as ω -> 0 and K -> infinity: the
      static, incompressible limit of the general oscillating solid recovers
      the classical elastic solution.

    Use `"solid"`/`"liquid"` for the static limit, and
    `"inertial"`/`"inertial-liquid"` when frequency-dependent (dynamic)
    effects matter.

    https://academic.oup.com/gji/article/203/3/2150/2594863

    Note: Ordering in Y corresponds to the mapping of the ith element, hence the y-functions order
    
        Y = [1,2,5,3,4,6]

    would correspond to 
        
        Y = [1,2,4,5,3,6] # <==
    
    and similarly for

        Y = [1,2,5,7,3,4,6,8]

    one should pass 
        
        Y = [1,2,5,6,3,7,4,8] # <==

    # Arguments
    - `ω::prec`                          : Angular frequency.
    - `r::prec`                          : Radius of the core boundary.
    - `ρ::prec`                          : Density of the core.
    - `g::prec`                          : Gravity at the core boundary.
    - `μ::precc`                         : Shear modulus of the core.
    - `K::precc`                         : Bulk modulus of the core.
    - `type::String`                     : Type of core, either "liquid", "inertial", "inertial-liquid", or "solid".
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `Y::Vector{Int}=[1,2,3,4,5,6]`     : Ordering of the solution vector components. This allows for different conventions in the literature.

    # Returns
    - `Ic::Array{precc,2}`               : MxN array of linearly independent solutions at the core boundary. These are used as starting vectors for the numerical integration across the interior.
    """
    function get_Ic(ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, type::String, n::Int; G0::prec=prec(1.0), Y::Vector{Int}=[1,2,3,4,5,6])::Array{precc,2}
    
        M = length(Y)
        N = Int(M / 2)
        Ic = zeros(precc, M, N)
        G_norm = G / G0

        if type=="liquid"
            Ic[Y[1],1] = -r^n / g
            Ic[Y[1],3] = 1.0
            Ic[Y[2],2] = 1.0
            Ic[Y[3],3] = g*ρ
            Ic[Y[5],1] = -r^n
            Ic[Y[6],1] = -2(n-1)*r^(n-1)
            Ic[Y[6],3] = -4π * G_norm * ρ
        elseif type == "inertial"
            # General finite-mu, finite-K, inertial (omega != 0) solid core solution,
            # after Kervazo et al. (2021, A&A) Appendix B, Eqs. B.22-B.34, with the
            # index convention Y[2]<->Y[3] swapped relative to Kervazo's own y2(radial
            # stress)/y3(tangential displacement V) labels (confirmed against get_A!).
            #
            # 1. Define physical parameters
            γ = 4π * G_norm * ρ / 3
            λ = K - (2/3)*μ
            α = sqrt(μ / ρ)                 # shear wave speed (Kervazo B.29)
            β = sqrt((K + (4/3)*μ) / ρ)     # compressional wave speed (Kervazo B.29)

            # 2. Calculate wavenumber k and dimensionless x.
            # Kervazo's B.29 assigns alpha to the shear speed and beta to the
            # compressional speed. Their own B.30/B.32 dispersion relation for k^2
            # and f, however, was carried over unchanged from Takeuchi & Saito
            # (1972) Eq. 99, whose own caption assigns alpha to the COMPRESSIONAL
            # speed and beta to shear -- the opposite of B.29. Re-deriving k^2, f
            # from the momentum equation (get_A! rows 3/4) directly, independent of
            # either paper's labelling, confirms the T&S/Kervazo B.29 labelling is
            # the one that must appear in B.30/B.32: alpha and beta are transposed
            # relative to what Kervazo prints. This is a typo in Kervazo's paper.
            # Verified numerically: with alpha, beta swapped here 
            # (and in Y[5]/Y[6] below), the finite-difference residual on the 
            # radial/tangential stress rows drops from O(1)-O(1e3) to the 
            # floating-point/finite-difference noise floor (~1e-6 to 1e-11).
            disc2 = (ω^2/α^2 - (ω^2+4γ)/β^2)^2 + 4*n*(n+1)*γ^2/(α^2*β^2)
            disc = sqrt(Complex{BigFloat}(disc2))
            k2 = 0.5*((ω^2+4γ)/β^2 + ω^2/α^2 + disc)      # "+" root -> column 2
            k = sqrt(Complex{BigFloat}(k2))
            x = k * r

            f = α^2/γ * (k2 - ω^2/α^2)      # Kervazo B.32, alpha/beta corrected (see above)
            h = f - (n + 1)                 # Kervazo B.32

            k2m = 0.5*((ω^2+4γ)/β^2 + ω^2/α^2 - disc)      # "-" root -> column 3
            km = sqrt(Complex{BigFloat}(k2m))
            xm = km * r
            fm = α^2/γ * (k2m - ω^2/α^2)
            hm = fm - (n + 1)

            # 3. Third Elementary Solution (Takeuchi & Saito 1972, Eq. 100 -- the
            # k^2->0 degenerate/regular solution). 
            Ic[Y[1],1] = n * r^(n-1)
            Ic[Y[2],1] = r^(n-1)
            Ic[Y[3],1] = 2μ * n * (n-1) * r^(n-2)
            Ic[Y[4],1] = 2μ * (n-1) * r^(n-2)
            Ic[Y[5],1] = (n*γ - ω^2) * r^n
            Ic[Y[6],1] = (2*(n-1)*n*γ - (2*n + 1)*ω^2) * r^(n-1)

            # 4. Second Elementary Solution (Kervazo B.23-B.28, general finite-mu,K)
            # Scaled Analytical Solution: divided by j_n(x) to prevent numerical
            # overflow. sbesselj wraps SpecialFunctions.besselj, which is numerically
            # stable across the physically relevant range of x, so no separate
            # small-x/low-frequency branch is required.
            @info("Using full inertial solution for core boundary conditions.")
            jn = sbesselj(n, x)
            jnp1 = sbesselj(n+1, x)

            ϕl_scaled = doublefactorial(2n+1) / x^n
            ψl_scaled = 2*(2n+3)/x^2 * (1/jn - ϕl_scaled)
            ϕlp1_scaled = (doublefactorial(2n+3) / x^(n+1)) * (jnp1 / jn)

            pref = -r^(n+1) / (2n + 3)

            Ic[Y[1],2] = pref * (0.5 * n * h * ψl_scaled + f * ϕlp1_scaled)
            Ic[Y[2],2] = pref * (0.5 * h * ψl_scaled - ϕlp1_scaled)
            Ic[Y[3],2] = (-(λ + 2μ) * r^n * f * ϕl_scaled
                          + μ * r^n / (2n+3) * (-n*(n-1)*h*ψl_scaled + 2*(2*f + n*(n+1))*ϕlp1_scaled))
            Ic[Y[4],2] = μ * r^n * (ϕl_scaled - 1/(2n+3) * ((n-1)*h*ψl_scaled + 2*(f+1)*ϕlp1_scaled))
            # Takeuchi & Saito (1972) Eq. 102's y5 has a bare (non-Bessel) term
            # -(n+1)*(shear speed)^2/r^2 in addition to (compressional speed)^2*f/r^2
            Ic[Y[5],2] = r^(n+2) * (((β^2*f - (n+1)*α^2) / r^2) * (1/jn) - (3γ*f / (2*(2n+3))) * ψl_scaled)
            Ic[Y[6],2] = r^(n+1) * ((((2n+1)*(β^2*f - (n+1)*α^2)) / r^2) * (1/jn) - (3γ*((2n+1)*f - n*h) / (2*(2n+3))) * ψl_scaled)

            # 5. Third Elementary Solution (Kervazo B.23-B.28, "-" root of B.30).
            # A genuinely solid (elastic, finite mu) core has no fluid-solid
            # interface, so there is no physical tangential-slip surface to
            # embed here (unlike "liquid"/"inertial-liquid", whose third column
            # IS that slip vector) -- the correct third basis vector is instead
            # this second, independent Bessel elementary solution. 
            jnm = sbesselj(n, xm)
            jnp1m = sbesselj(n+1, xm)

            ϕl_scaled_m = doublefactorial(2n+1) / xm^n
            ψl_scaled_m = 2*(2n+3)/xm^2 * (1/jnm - ϕl_scaled_m)
            ϕlp1_scaled_m = (doublefactorial(2n+3) / xm^(n+1)) * (jnp1m / jnm)

            pref_m = -r^(n+1) / (2n + 3)

            Ic[Y[1],3] = pref_m * (0.5 * n * hm * ψl_scaled_m + fm * ϕlp1_scaled_m)
            Ic[Y[2],3] = pref_m * (0.5 * hm * ψl_scaled_m - ϕlp1_scaled_m)
            Ic[Y[3],3] = (-(λ + 2μ) * r^n * fm * ϕl_scaled_m
                          + μ * r^n / (2n+3) * (-n*(n-1)*hm*ψl_scaled_m + 2*(2*fm + n*(n+1))*ϕlp1_scaled_m))
            Ic[Y[4],3] = μ * r^n * (ϕl_scaled_m - 1/(2n+3) * ((n-1)*hm*ψl_scaled_m + 2*(fm+1)*ϕlp1_scaled_m))
            Ic[Y[5],3] = r^(n+2) * (((β^2*fm - (n+1)*α^2) / r^2) * (1/jnm) - (3γ*fm / (2*(2n+3))) * ψl_scaled_m)
            Ic[Y[6],3] = r^(n+1) * ((((2n+1)*(β^2*fm - (n+1)*α^2)) / r^2) * (1/jnm) - (3γ*((2n+1)*fm - n*hm) / (2*(2n+3))) * ψl_scaled_m)

            # 6. Column-wise Normalization (Brings vectors to unit length)
            # Uses LinearAlgebra's norm. 
            for col in 1:3
                col_norm = norm(view(Ic, :, col))
                if col_norm > 0.0
                    Ic[:, col] ./= col_norm
                end
            end

            # print rank for debugging
            @debug("Rank of Ic for inertial core: ", rank(Ic))
            @debug("Condition number of Ic for inertial core: ", cond(Ic))

        elseif type == "inertial-liquid"
            # Fluid-core (mu=0) inertial (omega != 0) solid core solution, after
            # Korenaga (2025, Icarus) Appendix B, Eqs. B.4-B.12 (the oscillating
            # homogeneous fluid sphere solution of Love 1911 / Takeuchi & Saito
            # 1972, in Sabadini et al. 2016 notation). A sign flip on Y[5]/Y[6] 
            # (Phi/Psi) is required, since Korenaga's sign convention for those 
            # two components is the opposite of Obliqua's (confirmed numerically 
            # against get_A!).
            #
            # 1. Define physical parameters (K only: mu=0 identically in this branch)
            γ = 4π * G_norm * ρ / 3
            α = sqrt(K / ρ)                 # Korenaga's alpha: fluid compressional speed
            f = -ω^2 / γ                    # Korenaga B.8
            h = f - (n + 1)                 # Korenaga B.9

            # 2. Calculate wavenumber k and dimensionless x (Korenaga B.7)
            k2 = (ω^2 + 4γ - n*(n+1)*γ^2 / ω^2) / α^2
            k = sqrt(Complex{BigFloat}(k2))
            x = k * r

            # 3. Third Elementary Solution (Takeuchi & Saito 1972, Eq. 100 -- the
            # k^2->0 degenerate/regular solution). Y[3],Y[4] vanish here because
            # mu=0 identically in this branch (Korenaga B.4 is the mu=0 special
            # case of the same formula; verified exactly against get_A!).
            Ic[Y[1],1] = n * r^(n-1)
            Ic[Y[2],1] = r^(n-1)
            Ic[Y[3],1] = 2μ * n * (n-1) * r^(n-2)
            Ic[Y[4],1] = 2μ * (n-1) * r^(n-2)
            Ic[Y[5],1] = (n*γ - ω^2) * r^n
            Ic[Y[6],1] = (2*(n-1)*n*γ - (2*n + 1)*ω^2) * r^(n-1)

            # 4. Second Elementary Solution (Korenaga B.5, fluid mu=0)
            # Scaled Analytical Solution: divided by j_n(x) to prevent numerical
            # overflow (sbesselj wraps SpecialFunctions.besselj, numerically stable
            # across the physically relevant range of x).
            @info("Using full inertial-liquid solution for core boundary conditions.")
            jn = sbesselj(n, x)
            jnp1 = sbesselj(n+1, x)

            ϕl_scaled = doublefactorial(2n+1) / x^n
            ψl_scaled = 2*(2n+3)/x^2 * (1/jn - ϕl_scaled)
            ϕlp1_scaled = (doublefactorial(2n+3) / x^(n+1)) * (jnp1 / jn)

            pref = -r^(n+1) / (2n + 3)

            Ic[Y[1],2] = pref * (0.5 * n * h * ψl_scaled + f * ϕlp1_scaled)
            Ic[Y[2],2] = pref * (0.5 * h * ψl_scaled - ϕlp1_scaled)
            Ic[Y[3],2] = -K * r^n * f * ϕl_scaled
            Ic[Y[4],2] = 0.0
            Ic[Y[5],2] = r^(n+2) * ((α^2 * f / r^2) * (1/jn) - (3γ*f / (2*(2n+3))) * ψl_scaled)
            Ic[Y[6],2] = r^(n+1) * (((2n+1)*α^2*f / r^2) * (1/jn) - (3γ*((2n+1)*f - n*h) / (2*(2n+3))) * ψl_scaled)

            # 5. Boundary Condition Column
            Ic[:,3] .= 0.0
            Ic[Y[2],3] = 1.0 # tangential slip at CMB

            # 6. Column-wise Normalization (Brings vectors to unit length)
            for col in 1:3
                col_norm = norm(view(Ic, :, col))
                if col_norm > 0.0
                    Ic[:, col] ./= col_norm
                end
            end

            @debug("Rank of Ic for inertial-liquid core: ", rank(Ic))
            @debug("Condition number of Ic for inertial-liquid core: ", cond(Ic))

        elseif type == "solid"
            # First column
            Ic[Y[1], 1] = n*r^( n+1 ) / ( 2*( 2n + 3) )
            Ic[Y[2], 1] = ( n+3 )*r^( n+1 ) / ( 2*( 2n+3 ) * ( n+1 ) )
            Ic[Y[3], 1] = ( n*ρ*g*r + 2*( n^2 - n - 3)*μ ) * r^n / ( 2*( 2n + 3) )
            Ic[Y[4], 1] = n *( n+2 ) * μ * r^n / ( ( 2n + 3 )*( n+1 ) )
            Ic[Y[6], 1] = -2π*G_norm*ρ*n*r^( n+1 ) / ( 2n + 3 )

            # Second column
            Ic[Y[1], 2] = r^( n-1 )
            Ic[Y[2], 2] = r^( n-1 ) / n
            Ic[Y[3], 2] = ( ρ*g*r + 2*( n-1 )*μ ) * r^( n-2 )
            Ic[Y[4], 2] = 2*( n-1 ) * μ * r^( n-2 ) / n
            Ic[Y[6], 2] = -4π*G_norm*ρ*r^( n-1 )

            # Third column
            Ic[Y[3], 3] = -ρ * r^n
            Ic[Y[5], 3] = r^n
            Ic[Y[6], 3] = ( 2n + 1) * r^( n-1 )
        else
            error("Invalid core type: $type. Must be 'liquid', 'inertial', 'inertial-liquid', or 'solid'.")
        end

        # Non-zero pore pressure and zero radial Darcy flux
        if M == 8
            Ic[Y[7], 4] = 1.0
        end

        return Ic
    end


    """
        get_A(ω, r, ρ, g, n; G0=1, Y=[1,2])

    Compute the 2x2 `A` matrix in the ODE for the equilibrium-(fluid)-body problem.
    This is a simplified version of the A matrix used for the fluid-body problem, where 
    shear and bulk moduli are assumed zero, as well as forcing frequency ω is zero.

    The formulation is based on the work of Saito 1974, and reduces to the Love 1911
    formulation for the tidal response of a homogeneous fluid body. Note, this model 
    does apply to inhomogeneous fluid bodies, as the density and gravity can vary with radius.

    # Arguments
    - `ω::prec`                          : Forcing frequency of the tidal forcing.
    - `r::prec`                          : Radius at which to compute the A matrix.
    - `ρ::prec`                          : Density at radius r.
    - `g::prec`                          : Gravity at radius r.
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `Y::Vector{Int}=[1,2]`             : Ordering of the solution vector components. This allows for different conventions in the literature.

    # Returns
    - `A::Array{precc,2}`               : 2x2 A matrix at radius r, which is used in the ODE for the equilibrium-(fluid)-body problem.
    """
    function get_A(r::prec, ρ::prec, g::prec, n::Int; G0::prec=prec(1.0), Y::Vector{Int}=[1,2])::Array{precc,2}
        M = length(Y)
        A = zeros(precc, M, M) 
        get_A!(A, r, ρ, g, n; G0=G0, Y=Y)
        return A
    end


    """
        get_A(ω, r, ρ, g, μ, K, n; G0=1, inertial_terms=false, λ=nothing, Y=[1,2,3,4,5,6])

    Compute the 6x6 `A` matrix in the ODE for the solid-body problem.

    # Arguments
    - `ω::prec`                          : Forcing frequency of the tidal forcing.
    - `r::prec`                          : Radius at which to compute the A matrix.
    - `ρ::prec`                          : Density at radius r.
    - `g::prec`                          : Gravity at radius r.
    - `μ::precc`                         : Shear modulus at radius r.
    - `K::precc`                         : Bulk modulus at radius r.
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix.
    - `λ::prec=nothing`                  : Lamé's first parameter at radius r. If not provided, it is computed as λ = K - 2μ/3.
    - `Y::Vector{Int}=[1,2,3,4,5,6]`     : Ordering of the solution vector components. This allows for different conventions in the literature.

    # Returns
    - `A::Array{precc,2}`               : 6x6 A matrix at radius r, which is used in the ODE for the solid-body problem.
    """
    function get_A(ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, n::Int; G0::prec=prec(1.0), inertial_terms::Bool=false, λ::Union{Nothing, precc}=nothing, Y::Vector{Int}=[1,2,3,4,5,6])::Array{precc,2}
        M = length(Y)
        A = zeros(precc, M, M) 
        get_A!(A, ω, r, ρ, g, μ, K, n; G0=G0, inertial_terms=inertial_terms, λ=λ, Y=Y)
        return A
    end

    
    """
        get_A(ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=false, λ=nothing, Y=[1,2,3,4,5,6,7,8])

    Compute the 8x8 `A` matrix in the ODE for the two-phase problem. These correspond to 
    the coefficients given in Equation S4.6 in Hay et al., (2025).

    # Arguments
    - `ω::prec`                          : Forcing frequency.
    - `r::prec`                          : Radius at which to compute the A matrix.
    - `ρ::prec`                          : Density at radius r.
    - `g::prec`                          : Gravity at radius r.
    - `μ::precc`                         : Shear modulus at radius r.
    - `K::precc`                         : Bulk modulus at radius r.
    - `ρₗ::prec`                          : Liquid density at radius r.
    - `Kl::prec`                         : Liquid bulk modulus at radius r.
    - `Kd::precc`                        : Drained bulk modulus at radius r.
    - `α::precc`                         : Biot coefficient at radius r.
    - `ηₗ::prec`                          : Liquid viscosity at radius r.
    - `ϕ::prec`                          : Porosity at radius r.
    - `k::prec`                          : Permeability at radius r.    
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix.
    - `λ::precc=nothing`                 : Lamé's first parameter at radius r. If not provided, it is computed as λ = K - 2μ/3.
    - `Y::Vector{Int}=[1,2,3,4,5,6,7,8]` : Ordering of the solution vector components. This allows for different conventions in the literature.

    # Returns
    - `A::Array{precc,2}`               : 8x8 A matrix at radius r, which is used in the ODE for the solid-body problem.

    See also [`get_A!`](@ref)
    """
    function get_A(ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, ρₗ::prec, Kl::prec, Kd::precc, α::precc, ηₗ::prec, ϕ::prec, k::prec, n::Int; G0::prec=1, inertial_terms::Bool=false, λ::Union{Nothing, precc}=nothing, Y::Vector{Int}=[1,2,3,4,5,6,7,8])
        A = zeros(precc, 8, 8)
        get_A!(A, ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=G0, inertial_terms=inertial_terms, λ=λ, Y=Y)
        return A
    end


    """
        get_A!(A, ω, r, ρ, g, n; G0=1, Y=[1,2])

    Compute the 2x2 `A` matrix in the ODE for the fluid-body problem. These correspond to 
    the coefficients given in Saito 1974 Eq. 18.

    # Arguments
    - `A::Array{precc,2}`                : 2x2 A matrix at radius r, which is used in the ODE for the fluid-body problem.
    - `ω::prec`                          : Forcing frequency of the tidal forcing.
    - `r::prec`                          : Radius at which to compute the A matrix.
    - `ρ::prec`                          : Density at radius r.
    - `g::prec`                          : Gravity at radius r.
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `Y::Vector{Int}=[1,2]`             : Ordering of the solution vector components. This allows for different conventions in the literature.
    """
    function get_A!(A::Matrix, r::prec, ρ::prec, g::prec, n::Int; G0::prec=prec(1.0), Y::Vector{Int}=[1,2])
        G_norm = G / G0
        r_inv = 1.0/r

        A[Y[1],Y[1]] = 4π*G_norm*ρ/g - (n+1)*r_inv
        A[Y[1],Y[2]] = 1

        A[Y[2],Y[1]] = 2*(n-1)*r_inv * 4π*G_norm*ρ/g
        A[Y[2],Y[2]] = (n-1)*r_inv - 4π*G_norm*ρ/g
    end


    """
        get_A!(A, ω, r, ρ, g, μ, K, n; G0=1, inertial_terms=false, λ=nothing, Y=[1,2,3,4,5,6])

    Compute the 6x6 `A` matrix in the ODE for the solid-body problem. These correspond to 
    the coefficients given in Equation S4.6 in Hay et al., (2025) when α=φ=0, as well as Sabadini and Vermeersen 
    (2016) Eq. 1.95.

    # Arguments
    - `A::Array{precc,2}`                : 6x6 A matrix at radius r, which is used in the ODE for the solid-body problem.
    - `ω::prec`                          : Forcing frequency of the tidal forcing.
    - `r::prec`                          : Radius at which to compute the A matrix.
    - `ρ::prec`                          : Density at radius r.
    - `g::prec`                          : Gravity at radius r.
    - `μ::precc`                         : Shear modulus at radius r.
    - `K::precc`                         : Bulk modulus at radius r.
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix.
    - `λ::precc=nothing`                 : Lamé's first parameter at radius r. If not provided, it is computed as λ = K - 2μ/3.
    - `Y::Vector{Int}=[1,2,3,4,5,6]`     : Ordering of the solution vector components. This allows for different conventions in the literature.
    """
    function get_A!(A::Matrix, ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, n::Int; G0::prec=prec(1.0), inertial_terms::Bool=false, λ::Union{Nothing, precc}=nothing, Y::Vector{Int}=[1,2,3,4,5,6])
        if isnothing(λ)
            λ = K - 2μ/3
        end

        G_norm = G / G0

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)
        rβ_inv = r_inv * β_inv

        A[Y[1],Y[1]] = -2λ * r_inv*β_inv
        A[Y[2],Y[1]] = -r_inv
        A[Y[3],Y[1]] = 4r_inv * (3K*μ*r_inv*β_inv - ρ*g)
        A[Y[4],Y[1]] = -r_inv * (6K*μ*r_inv*β_inv - ρ*g)
        A[Y[5],Y[1]] = 4π * G_norm * ρ
        A[Y[6],Y[1]] = 4π*(n+1)*G_norm*ρ*r_inv

        A[Y[1],Y[2]] = n*(n+1) * λ * r_inv*β_inv
        A[Y[2],Y[2]] = r_inv
        A[Y[3],Y[2]] = -n*(n+1)*r_inv * (6K*μ*r_inv*β_inv - ρ*g) 
        A[Y[4],Y[2]] = 2μ*r_inv^2 * (n*(n+1)*(1 + λ*β_inv) - 1.0)
        A[Y[6],Y[2]] = -4π*n*(n+1)*G_norm*ρ*r_inv

        A[Y[1],Y[3]] = β_inv
        A[Y[3],Y[3]] = r_inv*β_inv * (-4μ)
        A[Y[4],Y[3]] = -λ * r_inv*β_inv
        
        A[Y[2],Y[4]] = 1.0 / μ
        A[Y[3],Y[4]] = n*(n+1)*r_inv
        A[Y[4],Y[4]] = -3r_inv

        A[Y[3],Y[5]] = ρ * (n+1)*r_inv
        A[Y[4],Y[5]] = -ρ*r_inv
        A[Y[5],Y[5]] = -(n+1)r_inv     

        A[Y[3],Y[6]] = -ρ
        A[Y[5],Y[6]] = 1.0
        A[Y[6],Y[6]] = (n-1)r_inv

        if inertial_terms
            A[Y[3],Y[1]] -= ω^2 * ρ
            A[Y[4],Y[2]] -= ω^2 * ρ
        end
    end


    """
        get_A!(A, ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=false, λ=nothing, Y=[1,2,3,4,5,6,7,8])

    Compute the 8x8 `A` matrix in the ODE for the two-phase problem. These correspond to 
    the coefficients given in Equation S4.6 in Hay et al., (2025).

    # Arguments
    - `A::Array{precc,2}`                : 8x8 A matrix at radius r, which is used in the ODE for the two-phase problem.
    - `ω::prec`                          : Forcing frequency.
    - `r::prec`                          : Radius at which to compute the A matrix.
    - `ρ::prec`                          : Density at radius r.
    - `g::prec`                          : Gravity at radius r.
    - `μ::precc`                         : Shear modulus at radius r.
    - `K::precc`                         : Bulk modulus at radius r.
    - `ρₗ::prec`                          : Liquid density at radius r.
    - `Kl::prec`                         : Liquid bulk modulus at radius r.
    - `Kd::precc`                        : Drained bulk modulus at radius r.
    - `α::precc`                         : Biot coefficient at radius r.
    - `ηₗ::prec`                          : Liquid viscosity at radius r.
    - `ϕ::prec`                          : Porosity at radius r.
    - `k::prec`                          : Permeability at radius r.
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix.
    - `λ::precc=nothing`                 : Lamé's first parameter at radius r. If not provided, it is computed as λ = K - 2μ/3.
    - `Y::Vector{Int}=[1,2,3,4,5,6,7,8]` : Ordering of the solution vector components. This allows for different conventions in the literature.

    # Notes
    See also [`get_A`](@ref)
    """
    function get_A!(A::Matrix, ω::prec, r::prec, ρ::prec, g::prec, μ::precc, K::precc, ρₗ::prec, Kl::prec, Kd::precc, α::precc, ηₗ::prec, ϕ::prec, k::prec, n::Int; G0::prec=prec(1.0), inertial_terms::Bool=false, λ::Union{Nothing, precc}=nothing, Y::Vector{Int}=[1,2,3,4,5,6,7,8])
        λ = Kd .- 2μ/3       # Lame's second param, which uses the drained compaction modulus
        S = ϕ/Kl + (α - ϕ)/K # Storavity, which uses liquid and solid grain bulk moduli  

        # First add the solid-body coefficients, but using drained moduli. 
        get_A!(A, ω, r, ρ, g, μ, Kd, n; λ=λ, G0=G0, inertial_terms=inertial_terms, Y=Y)    # Note that here we replace the bulk modulus with the compaction modulus

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)

        G_norm = G / G0

        # If there is a porous layer, now add the two-phase components
        if !iszero(ϕ)

            A[Y[1],Y[7]] = α * β_inv

            A[Y[3],Y[1]] += 1im * k*ρₗ^2 *g^2 * n*(n+1) / (ω*ηₗ) * r_inv^2
            A[Y[3],Y[5]] += -(n+1)r_inv * 1im *(k*ρₗ^2*g*n)/(ω*ηₗ) * r_inv                               
            A[Y[3],Y[7]] = 1im * (k*ρₗ*g*n*(n+1))/(ω*ηₗ)*r_inv^2 - 4μ*α*β_inv*r_inv 
            A[Y[3],Y[8]] =  1im * (k*ρₗ^2*g^2*n*(n+1))/(ω*ηₗ)*r_inv^2 - 4ϕ*ρₗ*g*r_inv 
        
            A[Y[4],Y[7]] = 2α*μ*r_inv * β_inv
            A[Y[4],Y[8]] = ϕ*ρₗ*g*r_inv 
            
            A[Y[5],Y[8]] = 4π*G_norm*ρₗ*ϕ

            A[Y[6],Y[1]] += -1im * 4π*G_norm*n*(n+1)*r_inv * (k*ρₗ^2*g/(ω*ηₗ)*r_inv)
            A[Y[6],Y[5]] = 1im*4π*n*(n+1)G_norm*(ρₗ)^2*k*r_inv^2 / (ω*ηₗ)  
            A[Y[6],Y[7]] = -1im *4π*n*(n+1)G_norm*ρₗ*k*r_inv^2 / ( ω*ηₗ) 
            A[Y[6],Y[8]] = 4π*G_norm*(n+1)*r_inv * (ϕ*ρₗ - 1im * n*k*ρₗ^2*g/(ω*ηₗ)*r_inv) 
            
            A[Y[7],Y[1]] = ρₗ*g*r_inv * ( 4 - 1im *(k*ρₗ*g*n*(n+1)/(ω*ϕ*ηₗ))*r_inv)  
            A[Y[7],Y[2]] = -ρₗ*n*(n+1)*r_inv*g
            A[Y[7],Y[5]] = -ρₗ*(n+1)r_inv * (1 - 1im*(k*ρₗ*g*n)/(ω*ϕ*ηₗ)*r_inv)  
            A[Y[7],Y[6]] = ρₗ 
            A[Y[7],Y[7]] = - 1im*(k*ρₗ*g*n*(n+1))/(ω*ϕ*ηₗ)*r_inv^2
            A[Y[7],Y[8]] = -1im*ω*ϕ*ηₗ/k - 4π*G_norm*(ρ - ϕ*ρₗ)*ρₗ + ρₗ*g*r_inv*(4 - 1im*(k*ρₗ*g*n*(n+1))/(ω*ϕ*ηₗ)*r_inv) 
        
            A[Y[8],Y[1]] = r_inv*( 1im * k*ρₗ*g*n*(n+1)/(ω*ϕ*ηₗ)*r_inv - α/ϕ * 4μ*β_inv) 
            A[Y[8],Y[2]] = α/ϕ * 2n*(n+1)*μ *β_inv * r_inv
            A[Y[8],Y[3]] = -α/ϕ * β_inv 
            A[Y[8],Y[5]] = -1im * k *ρₗ *n*(n+1) / (ω*ϕ*ηₗ)*r_inv^2 
            A[Y[8],Y[7]] = 1im*k*n*(n+1)/(ω*ϕ*ηₗ)*r_inv^2 - 1/ϕ * (S + α^2 * β_inv) # If solid and liquid are compressible, keep the 1/M term
            A[Y[8],Y[8]] = 1im * k *ρₗ*g *n*(n+1) / (ω*ϕ*ηₗ)*r_inv^2  - 2r_inv 
        end

    end


    """
        get_surface_bc!(R, g, n, U, U_prime, tau, P; G0=1, Y=[1,2,3,4,5,6])

    Get the surface boundary condition vector `b` and matrix `BN` for the solid-body problem. The surface 
    boundary conditions are determined by setting, respectively (U, U', tau, P) to (1,0,0,0) for tidal Love 
    number and (0,1,0,0) for load Love number in system.

    https://hal.science/hal-03421553/document

    # Arguments
    - `R::prec`                          : Planetary radius, used for surface boundary conditions.
    - `g::prec`                          : Gravity at the surface, used for surface boundary conditions.
    - `n::Int`                           : Tidal degree.
    - `U::Int`                           : Tidal potential at the surface.
    - `U_prime::Int`                     : Radial derivative of the tidal potential at the surface.
    - `tau::Int`                         : Tangential tidal stress at the surface.
    - `P::Int`                           : Surface mass load at the surface.

    # Keyword Arguments
    - `G0::prec=1`                       : Gravitational constant scale for non-dimensionalization.
    - `Y::Vector{Int}=[1,2,3,4,5,6]`     : Indices for the state variables (default is for standard case).

    # Returns
    - `B::Array{precc,2}`                : NxM matrix representing the linear relationship between the state variables at the surface and the boundary conditions.
    - `b::Vector{precc}`                 : Vector of length M representing the inhomogeneous part of the surface boundary conditions.
    """
    function get_surface_bc!(R::prec, g::prec, n::Int, U::Int, U_prime::Int, tau::Int, P::Int; G0=1, Y=[1,2,3,4,5,6])
        
        M = length(Y)
        N = Int(M / 2)

        # b vector (Right Hand Side of the B*y = b system)
        b = zeros(precc, M) 
        
        y2 = -(2 * n + 1) * g / (4 * pi * R^2) * U_prime - P
        y4 = tau
        y6 = ((2 * n + 1) / R) * (U + G/G0 / R * U_prime)
        y8 = 0

        if M == 8
            # radial Stress y3
            b[Y[3]] = y2
            
            # tangential Stress y4
            b[Y[4]] = y4
            
            # potential Stress y6
            b[Y[6]] = y6
            
            # darcy flux boundary
            b[Y[8]] = y8
        elseif M == 6
            # radial Stress y3
            b[Y[3]] = y2
            
            # tangential Stress y4
            b[Y[4]] = y4
            
            # potential Stress y6
            b[Y[6]] = y6
        elseif M == 2
            # helper y7 = y6 + 4πG/g y2
            b[Y[2]] = y6 + (4 * pi * G/G0 / g) * y2
        else
            error("Unsupported M value. M should be either 6 or 8.")
        end
        
        # construct the 4x8 B matrix
        # this matrix extracts y3, y4, and y6
        B = zeros(precc, N, M)

        if M == 8
            # stress components
            B[1, Y[3]] = 1.0  # radial stress y3
            B[2, Y[4]] = 1.0  # tangential stress y4
            # potential component
            B[3, Y[6]] = 1.0
            B[4, Y[8]] = 1.0
        elseif M == 6
            # stress components
            B[1, Y[3]] = 1.0  # radial stress y3
            B[2, Y[4]] = 1.0  # tangential stress y4
            # potential component
            B[3, Y[6]] = 1.0        
        elseif M == 2
            # only potential component
            B[1, Y[2]] = 1.0  # helper y7
        else
            error("Unsupported M value. M should be either 2, 6, or 8.")
        end
        
        if M == 2
            return B, b, y2, y6
        else
            return B, b
        end
    end


    """
        compute_strain_ten!(ϵ, y, n, rr, ρr, gr, μr, Ksr, SphericalGrid)

    Calculate the strain tensor ϵ at a particular radial level. 

    # Arguments
    - `ϵ::Array{ComplexF64,3}`          : 3D array to store the strain tensor at a particular radial level, with dimensions corresponding to latitude, longitude, and the 6 independent components of the strain tensor.
    - `y::Array{ComplexF64,1}`          : 1D array of the solution vector y at a particular radial level, with 6 components.
    - `n::Int`                          : Tidal degree.
    - `rr::Float64`                     : Radius at which to compute the strain tensor.
    - `ρr::Float64`                     : Density at radius rr.
    - `gr::Float64`                     : Gravity at radius rr.
    - `μr::ComplexF64`                  : Shear modulus at radius rr.
    - `Ksr::ComplexF64`                 : Bulk modulus at radius rr.
    - `SphericalGrid::NamedTuple`       : A struct containing the spherical grid information, including latitudes, longitudes, and the associated spherical harmonic functions.
    """
    function compute_strain_ten!(ϵ::Array{ComplexF64,3}, y::Array{ComplexF64,1}, n::Int, rr::Float64, ρr::Float64, gr::Float64, μr::ComplexF64, Ksr::ComplexF64, SphericalGrid::NamedTuple)
               
        i = 1

        @views clats = SphericalGrid.clats
        @views Y    = SphericalGrid.Y[i,:,:]
        @views dYdθ = SphericalGrid.dYdθ[i,:,:]
        @views dYdϕ = SphericalGrid.dYdϕ[i,:,:]
        @views Z    = SphericalGrid.Z[i,:,:]
        @views X    = SphericalGrid.X[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]

        λr = Ksr .- 2μr/3
        βr = λr + 2μr

        # Compute strain tensor
        ϵ[:,:,1] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3)/(βr*rr)  * Y
        ϵ[:,:,2] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
        ϵ[:,:,3] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
        ϵ[:,:,4] = 0.5/μr * y4 * dYdθ        
        ϵ[:,:,5] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,6] = 0.5 * y2/rr * Z
    end

    
    """
        compute_strain_ten!(ϵ, y, n, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr, SphericalGrid)

    Calculate the strain tensor ϵ at a particular radial level. 

    # Arguments
    - `ϵ::Array{ComplexF64,4}`          : 4D array to store the strain tensor at a particular radial level, with dimensions corresponding to latitude, longitude, and the 6 independent components of the strain tensor.
    - `y::Array{ComplexF64,1}`          : 1D array of the solution vector y at a particular radial level, with 6 components.
    - `n::Int`                          : Tidal degree.
    - `rr::Float64`                     : Radius at which to compute the strain tensor.
    - `ρr::Float64`                     : Density at radius rr.
    - `gr::Float64`                     : Gravity at radius rr.
    - `μr::ComplexF64`                  : Shear modulus at radius rr.
    - `Ksr::ComplexF64`                 : Bulk modulus at radius rr.
    - `ω::Float64`                      : Forcing frequency.
    - `ρlr::Float64`                    : Liquid density at radius rr.
    - `Klr::Float64`                    : Liquid bulk modulus at radius rr.
    - `Kdr::ComplexF64`                 : Drained bulk modulus at radius rr.
    - `αr::ComplexF64`                  : Biot coefficient at radius rr.
    - `ηlr::Float64`                    : Liquid viscosity at radius rr.
    - `ϕr::Float64`                     : Porosity at radius rr.
    - `kr::Float64`                     : Permeability at radius rr.
    - `SphericalGrid::NamedTuple`       : A struct containing the spherical grid information (Y, dYdθ, dYdϕ) for the current radial level.
    """
    function compute_strain_ten!(ϵ::Array{ComplexF64,3}, y::Array{ComplexF64,1}, n::Int, rr::Float64, ρr::Float64, gr::Float64, μr::ComplexF64, Ksr::ComplexF64, ω::Float64, ρlr::Float64, Klr::Float64, Kdr::ComplexF64, αr::ComplexF64, ηlr::Float64, ϕr::Float64, kr::Float64, SphericalGrid::NamedTuple)
        i = 1

        @views clats = SphericalGrid.clats
        @views Y    = SphericalGrid.Y[i,:,:]
        @views dYdθ = SphericalGrid.dYdθ[i,:,:]
        @views dYdϕ = SphericalGrid.dYdϕ[i,:,:]
        @views Z    = SphericalGrid.Z[i,:,:]
        @views X    = SphericalGrid.X[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]
        y7 = y[7]

        λr = Kdr - 2μr/3
        βr = λr + 2μr

        # Compute strain tensor
        ϵ[:,:,1] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3 + rr*αr*y7)/(βr*rr)  * Y     # e_rr
        ϵ[:,:,2] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)                      # e_
        ϵ[:,:,3] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)                      # e_
        ϵ[:,:,4] = 0.5/μr * y4 * dYdθ                                           # e_rθ
        ϵ[:,:,5] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats)                     # e_rϕ
        ϵ[:,:,6] = 0.5 * y2/rr * Z                                              # e_        
    end


    """
        compute_darcy_displacement!(dis_rel, y, n, r, ω, ϕ, ηl, k, g, ρl, SphericalGrid)

    Calculate the Darcy displacement vector at a particular radial level. 

    # Arguments
    - `dis_rel::Array{ComplexF64}`      : Array to store the Darcy displacement vector at a particular radial level, with dimensions corresponding to latitude, longitude, and the 3 components of the Darcy displacement vector.
    - `y::Matrix`                       : Solution vector y at a particular radial level, with 8 components.
    - `n::Int`                          : Tidal degree. 
    - `r::Float64`                      : Radius at which to compute the Darcy displacement vector.
    - `ω::Float64`                      : Forcing frequency.
    - `ϕ::Float64`                      : Porosity at radius r.
    - `ηl::Float64`                     : Liquid viscosity at radius r.
    - `k::Float64`                      : Permeability at radius r.
    - `g::Float64`                      : Gravity at radius r.
    - `ρl::Float64`                     : Liquid density at radius r.
    - `SphericalGrid::NamedTuple`       : A struct containing the spherical grid information (Y, dYdθ, dYdϕ) for the current radial level.
    """
    function compute_darcy_displacement!(dis_rel::Array{ComplexF64}, y::Array{ComplexF64,1}, n::Int, r::Float64, ω::Float64, ϕ::Float64, ηl::Float64, k::Float64, g::Float64, ρl::Float64, SphericalGrid::NamedTuple)
        i = 1

        @views clats = SphericalGrid.clats
        @views Y    = SphericalGrid.Y[i,:,:]
        @views dYdθ = SphericalGrid.dYdθ[i,:,:]
        @views dYdϕ = SphericalGrid.dYdϕ[i,:,:]
        
        y1 = y[1]
        y5 = y[5]
        y7 = y[7]
        y8 = y[8]
        y9 = 1im * k / (ω*ϕ*ηl*r) * (ρl*g*y1 - ρl * y5 + ρl*g*y8 + y7)
        
        dis_rel[:,:,1] = Y * y8 
        dis_rel[:,:,2] = dYdθ * y9
        dis_rel[:,:,3] = dYdϕ * y9 ./ sin.(clats)
    end


    """
        compute_pore_pressure!(p, y, n, SphericalGrid)

    Calculate the fluid pore pressur at a particular radial level. 

    # Arguments
    - `p::Array{ComplexF64,4}`          : 4D array to store the pore pressure at a particular radial level, with dimensions corresponding to latitude and longitude.
    - `y::Matrix`                       : Solution vector y at a particular radial level, with 8 components.
    - `n::Int`                          : Tidal degree.
    - `SphericalGrid::NamedTuple`       : A struct containing the spherical grid information (Y, dYdθ, dYdϕ) for the current radial level.
    """
    function compute_pore_pressure!(p::Array{ComplexF64,4}, y::Matrix, n::Int, SphericalGrid::NamedTuple)
        i = 1

        @views Y    = SphericalGrid.Y[i,:,:]
        @views dYdθ = SphericalGrid.dYdθ[i,:,:]
        @views dYdϕ = SphericalGrid.dYdϕ[i,:,:]
        
        y7 = y[7]
        
        p[:,:] = Y * y7 
    end
   

    """
        function get_heating_profile(y, r, ρ, g, μ, κ, n, ω, SphericalGrid)

    Get the radial volumetric heating for solid-body tides and eccentricity forcing,
    assuming synchronous rotation. Heating rate is computed with numerical integration 
    using the solution `y`, using Eq. 2.39a/b integrated over solid angle.

    # Arguments
    - `y::Matrix`                        : Solution vector y across the interior, returned by `compute_y`.
    - `r::AbstractVector`                : 1D vector of radial coordinates or shell boundaries.  
    - `ρ::AbstractVector`                : 1D vector of densities at each radial shell.  
    - `g::AbstractVector`                : 1D vector of gravitational acceleration values at each radial shell.  
    - `μ::AbstractVector`                : 1D vector of complex shear moduli at each radial shell.  
    - `κ::AbstractVector`                : 1D vector of complex bulk moduli at each radial shell.  
    - `n::Int`                           : Tidal degree.  
    - `ω::prec`                          : Tidal frequency in radians per second.  
    - `SphericalGrid::NamedTuple`        : A struct containing the spherical grid information, including latitudes, longitudes, and the associated spherical harmonic functions.

    # Returns
    - `Eμ_tot::Array{Float64,1}`         : 1D array of total power dissipated in each radial shell due to shear, in W.
    - `Eκ_tot::Array{Float64,1}`         : 1D array of total power dissipated in each radial shell due to compaction, in W.
    """
    function get_heating_profile(y::Matrix, r::AbstractVector, ρ::AbstractVector, g::AbstractVector, μ::AbstractVector, κ::AbstractVector, n::Int, ω::prec, SphericalGrid::NamedTuple)

        # convert to Float64 or ComplexF64 for heating calculations
        r = Float64.(r)
        ρ = Float64.(ρ)
        g = Float64.(g)
        μ = ComplexF64.(μ)
        κ = ComplexF64.(κ)
        ω = Float64(ω)

        dres = deg2rad(SphericalGrid.res)

        clats = SphericalGrid.clats
        lons  = SphericalGrid.lons

        Nr = length(r)
        nlats = length(clats)
        nlons = length(lons)

        # strain tensor per radius
        ϵ = zeros(ComplexF64, nlats, nlons, 6)

        # outputs
        Eμ_tot = zeros(Float64, Nr-1)
        Eκ_tot = zeros(Float64, Nr-1)

        for i in 1:Nr-1

            rr = r[i]
            dr = r[i+1] - r[i]

            dvol = 4π/3 * (r[i+1]^3 - r[i]^3)

            yrr = y[:, i]

            compute_strain_ten!(ϵ, yrr, n, rr, ρ[i], g[i], μ[i], κ[i], SphericalGrid)

            # shear heating
            Eμ_loc = sum(abs.(ϵ[:,:,1:3]).^2, dims=3) .+
                    2sum(abs.(ϵ[:,:,4:6]).^2, dims=3) .-
                    1/3 .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2

            Eμ_loc .*= ω * imag(μ[i])

            # bulk heating
            Eκ_loc = ω/2 * imag(κ[i]) .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2

            # angular integration
            weight = sin.(clats)

            Eμ_tot[i] = sum(weight .* Eμ_loc * dres^2) * rr^2 * dr / dvol
            Eκ_tot[i] = sum(weight .* Eκ_loc * dres^2) * rr^2 * dr / dvol

        end

        return Eμ_tot, Eκ_tot
    end


    """
        get_heating_profile(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, n, SphericalGrid; lay=nothing)

    Get the radial volumetric heating for two-phase tides and eccentricity forcing,
    assuming synchronous rotation. Heating rate is computed with numerical integration 
    using the solution `y`, using Eq. 2.39a/b/c integrated over solid angle. 

    # Arguments
    - `y::Matrix`                        : Solution vector.
    - `r::AbstractVector`                : 1D vector of radial boundaries/shell coordinates.
    - `ρ::AbstractVector`                : 1D vector of layer densities.
    - `g::AbstractVector`                : 1D vector of gravity values.
    - `μ::AbstractVector`                : 1D vector of complex shear moduli.
    - `Ks::AbstractVector`               : 1D vector of bulk moduli for shear dissipation.
    - `ω::prec`                          : Tidal frequency in radians per second.
    - `ρl::AbstractVector`               : 1D vector of liquid densities.
    - `Kl::AbstractVector`               : 1D vector of liquid bulk moduli.
    - `Kd::AbstractVector`               : 1D vector of drained bulk moduli.
    - `α::AbstractVector`                : 1D vector of Biot coefficients.
    - `ηl::AbstractVector`               : 1D vector of liquid viscosities.
    - `ϕ::AbstractVector`                : 1D vector of porosities.
    - `k::AbstractVector`                : 1D vector of permeabilities.
    - `n::Int`                           : Tidal degree.
    - `SphericalGrid::NamedTuple`        : A struct containing the spherical grid information (Y, dYdθ, dYdϕ) for the current radial level.

    # Returns
    - `Eμ_tot::Vector{Float64}`          : Total power dissipated due to shear (W/m³).
    - `Eκ_tot::Vector{Float64}`          : Total power dissipated due to compaction (W/m³).
    - `El_tot::Vector{Float64}`          : Total power dissipated due to Darcy flow (W/m³).
    """    
    function get_heating_profile(y::Matrix, r::AbstractVector, ρ::AbstractVector, g::AbstractVector, μ::AbstractVector, Ks::AbstractVector, ω::prec, ρl::AbstractVector, Kl::AbstractVector, Kd::AbstractVector, α::AbstractVector, ηl::AbstractVector, ϕ::AbstractVector, k::AbstractVector, n::Int, SphericalGrid::NamedTuple)

        # convert to Float64 or ComplexF64 for heating calculations
        r = Float64.(r)
        ρ = Float64.(ρ)
        g = Float64.(g)
        μ = ComplexF64.(μ)
        Ks = ComplexF64.(Ks)
        ω = Float64(ω)
        ρl = Float64.(ρl)
        Kl = Float64.(Kl)
        Kd = ComplexF64.(Kd)
        α = ComplexF64.(α)
        ηl = Float64.(ηl)
        ϕ = Float64.(ϕ)
        k = Float64.(k)

        dres = deg2rad(SphericalGrid.res)
        clats = SphericalGrid.clats
        lons = SphericalGrid.lons
        
        # Reference spherical harmonic Y if needed globally
        # (Assuming the logic uses the same spatial resolution as the first example)
        Y = SphericalGrid.Y[1, :, :] 

        Nr = length(r)
        nlats = length(clats)
        nlons = length(lons)

        # Pre-allocate spatial buffers
        ϵ = zeros(ComplexF64, nlats, nlons, 6)
        d_disp = zeros(ComplexF64, nlats, nlons, 3)
        p = zeros(ComplexF64, nlats, nlons)

        # Output vectors (per radial shell)
        Eμ_tot = zeros(Float64, Nr - 1)
        Eκ_tot = zeros(Float64, Nr - 1)
        El_tot = zeros(Float64, Nr - 1)

        for i in 1:Nr-1
            rr = r[i]
            dr = r[i+1] - r[i]
            dvol = 4π/3 * (r[i+1]^3 - r[i]^3)
            yrr = y[:, i]

            # 1. Compute Tensors/Fields
            compute_strain_ten!(ϵ, yrr, n, rr, ρ[i], g[i], μ[i], Ks[i], ω, ρl[i], Kl[i], Kd[i], α[i], ηl[i], ϕ[i], k[i], SphericalGrid)
            
            if ϕ[i] > 0
                compute_darcy_displacement!(d_disp, yrr, n, rr, ω, ϕ[i], ηl[i], k[i], g[i], ρl[i], SphericalGrid)
                p .= yrr[7] .* Y 
            else
                p .= 0.0
            end

            # 2. Shear Heating (Solid)
            Eμ_loc = sum(abs.(ϵ[:,:,1:3]).^2, dims=3) .+ 
                     2sum(abs.(ϵ[:,:,4:6]).^2, dims=3) .- 
                     1/3 .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2
            Eμ_loc .*= ω * imag(μ[i])

            # 3. Compaction Heating (Bulk)
            Eκ_loc = ω/2 * imag(Kd[i]) .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2
            if ϕ[i] > 0
                Eκ_loc .+= (ω/2 * imag(Kd[i]) .* abs.(p ./ Ks[i]).^2)       # <-- Fixed typo present in Love.jl :)
            end

            # 4. Darcy Dissipation (Liquid)
            El_loc = zeros(nlats, nlons)
            if ϕ[i] > 0
                El_loc .= 0.5 * ϕ[i]^2 * ω^2 * (ηl[i] / k[i]) .* (abs.(d_disp[:,:,1]).^2 .+ abs.(d_disp[:,:,2]).^2 .+ abs.(d_disp[:,:,3]).^2)
            end

            # 5. Angular Integration and Volume Averaging
            weight = sin.(clats)
            
            Eμ_tot[i] = sum(weight .* Eμ_loc .* dres^2) * rr^2 * dr / dvol
            Eκ_tot[i] = sum(weight .* Eκ_loc .* dres^2) * rr^2 * dr / dvol
            El_tot[i] = sum(weight .* El_loc .* dres^2) * rr^2 * dr / dvol
        end

        return Eμ_tot, Eκ_tot, El_tot
    end


    """
        function get_heating_map(y, r, ρ, g, μ, κ, n, ω, SphericalGrid)

    Get the surface heating map for solid-body tides, assuming synchronous rotation. 
    The heating rate is computed by radially integrating the local volumetric 
    dissipation rates determined from the solution `y` and Eq. 2.39a/b.

    # Arguments
    - `y::Matrix`                        : Solution vector.
    - `r::AbstractVector`                : 1D vector of radial shell boundaries.  
    - `ρ::AbstractVector`                : 1D vector of densities at each radial shell.  
    - `g::AbstractVector`                : 1D vector of gravity values.  
    - `μ::AbstractVector`                : 1D vector of complex shear moduli.  
    - `κ::AbstractVector`                : 1D vector of complex bulk moduli.  
    - `n::Int`                           : Tidal degree.  
    - `ω::prec`                          : Tidal frequency in radians per second.  
    - `SphericalGrid::NamedTuple`        : A struct formed by `define_spherical_grid`, containing 
                                           the geographic grid and spherical harmonic derivatives.

    # Returns
    - `Eμ_3d::Array{Float64,3}`          : 3D map of shear dissipation (W/m²).
    - `Eκ_3d::Array{Float64,3}`          : 3D map of compaction/bulk dissipation (W/m²).
    """
    function get_heating_map(y::Matrix, r::AbstractVector, ρ::AbstractVector, g::AbstractVector, μ::AbstractVector, κ::AbstractVector, n::Int, ω::prec, SphericalGrid::NamedTuple)

        # convert to Float64 or ComplexF64 for heating calculations
        r = Float64.(r)
        ρ = Float64.(ρ)
        g = Float64.(g)
        μ = ComplexF64.(μ)
        κ = ComplexF64.(κ)
        ω = Float64(ω)

        clats = SphericalGrid.clats
        lons  = SphericalGrid.lons

        Nr = length(r)
        nlats = length(clats)
        nlons = length(lons)

        # Pre-allocate spatial buffers for local shell calculation
        ϵ = zeros(ComplexF64, nlats, nlons, 6)

        # Initialize output maps (Integrated over radius)
        Eμ_3d = zeros(Float64, nlats, nlons, Nr-1)
        Eκ_3d = zeros(Float64, nlats, nlons, Nr-1)

        for i in 1:Nr-1

            rr = r[i]
            dr = r[i+1] - r[i]
            dvol = 4π/3 * (r[i+1]^3 - r[i]^3)
            yrr = y[:, i]

            # 1. Compute the strain tensor for the current shell
            compute_strain_ten!(ϵ, yrr, n, rr, ρ[i], g[i], μ[i], κ[i], SphericalGrid)

            # 2. Local Volumetric Shear Heating (W/m³)
            Eμ_loc = sum(abs.(ϵ[:,:,1:3]).^2, dims=3) .+
                     2sum(abs.(ϵ[:,:,4:6]).^2, dims=3) .-
                     1/3 .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2

            Eμ_loc .*= ω * imag(μ[i])

            # 3. Local Volumetric Bulk Heating (W/m³)
            Eκ_loc = ω/2 * imag(κ[i]) .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2

            # 4. Accumulate the heat flux from each shell into the final map (W/m³)
            Eμ_3d[:, :, i] = Eμ_loc * rr^2 * dr / dvol
            Eκ_3d[:, :, i] = Eκ_loc * rr^2 * dr / dvol

        end

        return Eμ_3d, Eκ_3d
    end


    """
        function get_heating_map(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, n, SphericalGrid)

    Get the surface heating map for two-phase tides, assuming synchronous rotation. 
    The local volumetric heating rates (W/m³) are computed from the solution `y` 
    using the rheological parameters for both the solid matrix and the melt, then 
    radially integrated to obtain surface maps in W/m².

    # Arguments
    - `y::Matrix`                        : Solution vector.
    - `r::AbstractVector`                : 1D vector of radial boundaries/shell coordinates.
    - `ρ::AbstractVector`                : 1D vector of layer densities.
    - `g::AbstractVector`                : 1D vector of gravity values.
    - `μ::AbstractVector`                : 1D vector of complex shear moduli.
    - `Ks::AbstractVector`               : 1D vector of bulk moduli for shear dissipation.
    - `ω::prec`                          : Tidal frequency in radians per second.
    - `ρl::AbstractVector`               : 1D vector of liquid densities.
    - `Kl::AbstractVector`               : 1D vector of liquid bulk moduli.
    - `Kd::AbstractVector`               : 1D vector of drained bulk moduli.
    - `α::AbstractVector`                : 1D vector of Biot coefficients.
    - `ηl::AbstractVector`               : 1D vector of liquid viscosities.
    - `ϕ::AbstractVector`                : 1D vector of porosities.
    - `k::AbstractVector`                : 1D vector of permeabilities.
    - `n::Int`                           : Tidal degree.
    - `SphericalGrid::NamedTuple`        : A struct containing geographic grid and spherical harmonic data.

    # Returns
    - `Eμ_3d::Array{Float64,3}`          : 3D map of shear dissipation (W/m²).
    - `Eκ_3d::Array{Float64,3}`          : 3D map of compaction dissipation (W/m²).
    - `El_3d::Array{Float64,3}`          : 3D map of Darcy (percolation) dissipation (W/m²).
    """
    function get_heating_map(y::Matrix, r::AbstractVector, ρ::AbstractVector, g::AbstractVector, μ::AbstractVector, Ks::AbstractVector, ω::prec, ρl::AbstractVector, Kl::AbstractVector, Kd::AbstractVector, α::AbstractVector, ηl::AbstractVector, ϕ::AbstractVector, k::AbstractVector, n::Int, SphericalGrid::NamedTuple)

        # convert to Float64 or ComplexF64 for heating calculations
        r = Float64.(r)
        ρ = Float64.(ρ)
        g = Float64.(g)
        μ = ComplexF64.(μ)
        Ks = ComplexF64.(Ks)
        ρl = Float64.(ρl)
        Kl = Float64.(Kl)
        Kd = ComplexF64.(Kd)
        α = ComplexF64.(α)
        ηl = Float64.(ηl)
        ϕ = Float64.(ϕ)
        ω = Float64(ω)
        k = Float64.(k)

        clats = SphericalGrid.clats
        lons  = SphericalGrid.lons
        Y     = SphericalGrid.Y[1, :, :] 

        Nr = length(r)
        nlats = length(clats)
        nlons = length(lons)

        # Pre-allocate spatial buffers for local shell calculation
        ϵ = zeros(ComplexF64, nlats, nlons, 6)
        d_disp = zeros(ComplexF64, nlats, nlons, 3)
        p = zeros(ComplexF64, nlats, nlons)

        # Initialize output 3D maps (Dimensions: lat × lon × radial_shell)
        Eμ_3d = zeros(Float64, nlats, nlons, Nr-1)
        Eκ_3d = zeros(Float64, nlats, nlons, Nr-1)
        El_3d = zeros(Float64, nlats, nlons, Nr-1)

        for i in 1:Nr-1

            rr = r[i]
            dr = r[i+1] - r[i]
            dvol = 4π/3 * (r[i+1]^3 - r[i]^3)
            yrr = y[:, i]

            # 1. Compute Tensors/Fields for the current shell
            compute_strain_ten!(ϵ, yrr, n, rr, ρ[i], g[i], μ[i], Ks[i], ω, ρl[i], Kl[i], Kd[i], α[i], ηl[i], ϕ[i], k[i], SphericalGrid)
            
            if ϕ[i] > 0
                compute_darcy_displacement!(d_disp, yrr, n, rr, ω, ϕ[i], ηl[i], k[i], g[i], ρl[i], SphericalGrid)
                p .= yrr[7] .* Y 
            else
                p .= 0.0
            end

            # 2. Local Volumetric Shear Heating (Solid)
            Eμ_loc = sum(abs.(ϵ[:,:,1:3]).^2, dims=3) .+ 
                     2sum(abs.(ϵ[:,:,4:6]).^2, dims=3) .- 
                     1/3 .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2
            Eμ_loc .*= ω * imag(μ[i])

            # 3. Local Volumetric Compaction Heating (Bulk)
            Eκ_loc = ω/2 * imag(Kd[i]) .* abs.(sum(ϵ[:,:,1:3], dims=3)).^2
            if ϕ[i] > 0
                Eκ_loc .+= (ω/2 * imag(Kd[i]) .* abs.((p) ./ Ks[i]).^2)
            end

            # 4. Local Volumetric Darcy Dissipation (Liquid)
            El_loc = zeros(nlats, nlons)
            if ϕ[i] > 0
                El_loc .= 0.5 * ϕ[i]^2 * ω^2 * (ηl[i] / k[i]) .* (abs.(d_disp[:,:,1]).^2 .+ abs.(d_disp[:,:,2]).^2 .+ abs.(d_disp[:,:,3]).^2)
            end

            # 5. Accumulate the heat flux from each shell into the final map (W/m³)
            Eμ_3d[:, :, i] = Eμ_loc * rr^2 * dr / dvol
            Eκ_3d[:, :, i] = Eκ_loc * rr^2 * dr / dvol
            El_3d[:, :, i] = El_loc * rr^2 * dr / dvol

        end

        return Eμ_3d, Eκ_3d, El_3d
    end

end