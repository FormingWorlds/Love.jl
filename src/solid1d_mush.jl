

module solid1d_mush
    
    include("constants.jl")
    include("common.jl")
    using .constants
    using .common

    using LinearAlgebra
    using DoubleFloats
    using AssociatedLegendrePolynomials    
    using StaticArrays


    clats = 0.0
    lons  = 0.0
    Y     = 0.0
    dYdθ  = 0.0
    dYdϕ  = 0.0
    Z     = 0.0
    X     = 0.0
    res   = 20.0
    

    """
        Ynm(n, m, theta, phi)

    Compute the spherical harmonic Ynm for given n, m, theta, and phi.

    # Arguments
    - `n::Int`                          : Tidal degree.
    - `m::Int`                          : Tidal order.
    - `theta::AbstractArray`            : Array of colatitudes in radians.
    - `phi::LinearAlgebra.Adjoint`      : Array of longitudes in radians.

    # Returns
    - `Ynm::Array{ComplexF64,2}`        : 2D array of spherical harmonic values for each combination of theta and phi.
    """
    function Ynm(n::Int, m::Int, theta::AbstractArray, phi::LinearAlgebra.Adjoint)
        return Plm.(n, m, cos.(theta)) .* exp.(1im * m .* phi)
    end


    """
        define_spherical_grid(res, n, m)

    Create the spherical grid of angular resolution `res` in degrees. This is used for 
    numerical integrations over solid angle. A new grid can easily be defined by 
    recalling the function with a new `res`.

    # Arguments
    - `res::Float64`                     : Desired angular resolution in degrees.
    - `n::Int`                           : Tidal degree.
    - `m::Int`                           : Tidal order.

    # Notes
    The grid is internal to solid1d_mush, but can be accessed with 

        solid1d_mush.clats[:] # colatitude grid
        solid1d_mush.lons[:]  # longitude grid
    """
    function define_spherical_grid(res::Float64, n::Int, m::Int)
        solid1d_mush.res = res

        # θ and φ grids
        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))
        clats[1] += 1e-6
        clats[end] -= 1e-6

        # allocate arrays
        solid1d_mush.Y    = zeros(ComplexF64, 1, length(clats), length(lons))
        solid1d_mush.dYdθ = similar(solid1d_mush.Y)
        solid1d_mush.dYdϕ = similar(solid1d_mush.Y)
        solid1d_mush.Z    = similar(solid1d_mush.Y)
        solid1d_mush.X    = similar(solid1d_mush.Y)

        sinθ = sin.(clats)
        cosθ = cos.(clats)
        cotθ = cosθ ./ sinθ
        cscθ = csc.(clats)

        # Normalization factor for spherical harmonics
        norm = sqrt((2*n+1)  * factorial(n-m) / (4π * factorial(n+m)))
        
        i = 1

        # Y
        solid1d_mush.Y[i,:,:] = Ynm(n,m,clats,lons)

        # ∂Y/∂θ
        Pn = Plm.(n, m, cosθ)
        if n > m
            Pn_1 = Plm.(n-1, m, cosθ)
            dPdθ = (n .* cosθ .* Pn .- (n + m) .* Pn_1) ./ (sinθ)
        else
            # m == n -> P_{n-1}^m = 0
            dPdθ = (n .* cosθ .* Pn) ./ (sinθ)
        end
        solid1d_mush.dYdθ[i,:,:] .= dPdθ .* exp.(1im .* m .* lons)

        # ∂Y/∂ϕ
        solid1d_mush.dYdϕ[i,:,:] .= 1im * m .* solid1d_mush.Y[i,:,:]

        # Z = 2 ((1/sinθ) ∂²Y/∂θ∂ϕ - cotθ cscθ ∂Y/∂ϕ)
        solid1d_mush.Z[i,:,:] .= 2 .* (1im * m ./ sinθ .* solid1d_mush.dYdθ[i,:,:] .- cotθ .* cscθ .* solid1d_mush.dYdϕ[i,:,:])

        # X = -2 (cotθ ∂Y/∂θ + csc²θ ∂²Y/∂ϕ²) - n(n+1)) Y
        solid1d_mush.X[i,:,:] .= -2 .* (cotθ .* solid1d_mush.dYdθ[i,:,:] .- cscθ.^2 .* m^2 .* solid1d_mush.Y[i,:,:]) .- n*(n+1) .* solid1d_mush.Y[i,:,:]

        # Normalize
        solid1d_mush.Y[i,:,:]    .*= norm
        solid1d_mush.dYdθ[i,:,:] .*= norm
        solid1d_mush.dYdϕ[i,:,:] .*= norm
        solid1d_mush.Z[i,:,:]    .*= norm
        solid1d_mush.X[i,:,:]    .*= norm

        # save grids
        solid1d_mush.clats = clats
        solid1d_mush.lons  = lons
    end


    """
        convert_params_to_prec(r, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k)

    Convert input parameters into the required precision.
    
    # Arguments
    - `r::Array{Float64,2}`               : 2D array of layer boundaries.
    - `ρ::Array{Float64,1}`               : 1D array of layer densities. 
    - `g::Array{Float64,2}`               : 2D array of gravity values at the layer boundaries. 
    - `μ::Array{Float64,1}`               : 1D array of layer shear moduli.
    - `κs::Array{Float64,1}`              : 1D array of layer bulk moduli. 
    - `ω::Float64`                        : Forcing frequency.
    - `ρl::Array{Float64,1}`              : 1D array of layer liquid densities.
    - `κl::Array{Float64,1}`              : 1D array of layer liquid bulk moduli.
    - `κd::Array{Float64,1}`              : 1D array of layer drained bulk moduli.
    - `α::Array{Float64,1}`               : 1D array of layer Biot coefficients.
    - `ηl::Array{Float64,1}`              : 1D array of layer liquid viscosities.
    - `ϕ::Array{Float64,1}`               : 1D array of layer porosities.
    - `k::Array{Float64,1}`               : 1D array of layer permeabilities.

    # Returns
    Tuple of converted parameters in the required precision.
    """
    function convert_params_to_prec(r, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k)
        r_prec = convert(Array{prec}, r)
        ρ_prec = convert(Array{prec}, ρ)
        g_prec = convert(Array{prec}, g)
        μ_prec = convert(Array{precc}, μ)
        κs_prec = convert(Array{precc}, κs)
        ρl_prec = convert(Array{prec}, ρl)
        κl_prec = convert(Array{prec}, κl)
        κd_prec = convert(Array{precc}, κd)
        α_prec = convert(Array{precc}, α)
        ηl_prec = convert(Array{prec}, ηl)
        ϕ_prec = convert(Array{prec}, ϕ)
        k_prec = convert(Array{prec}, k)
        ω_prec = convert(prec, ω)

        return (r_prec,  ρ_prec, g_prec, μ_prec, κs_prec, ω_prec, ρl_prec, κl_prec, κd_prec, α_prec, ηl_prec, ϕ_prec, k_prec)
    end

    
    """
        get_B(ω, r1, r2, g1, g2, ρ, μ, K, n; G0=1, inertial_terms=false)

    Compute the 6x6 numerical integrator matrix, which integrates dy/dr from `r1` to `r2` for the solid-body problem.

    # Arguments
    - `ω::prec`                          : Forcing frequency.
    - `r1::prec`                         : Starting radius for integration.
    - `r2::prec`                         : Ending radius for integration.
    - `g1::prec`                         : Gravity at radius r1.
    - `g2::prec`                         : Gravity at radius r2.
    - `ρ::prec`                          : Density at radius r.
    - `μ::precc`                         : Shear modulus at radius r.
    - `K::precc`                         : Bulk modulus at radius r.
    - `n::Int`                           : Tidal degree.

    Keyword Arguments
    - `G0=1`                             : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.

    # Returns
    - `B::Array{precc,2}`               : 6x6 numerical integrator matrix for integrating dy/dr from r1 to r2 for the solid-body problem.

    # Notes
    See 'get_B!' for definition.
    """ 
    function get_B(ω::prec, r1::prec, r2::prec, g1::prec, g2::prec, ρ::prec, μ::precc, K::precc, n::Int; G0=one(prec), inertial_terms::Bool=false)
        B = zeros(precc, 6, 6)
        get_B!(B, ω, r1, r2, g1, g2, ρ, μ, K, n; G0=G0, inertial_terms=inertial_terms)
        return B
    end


    """
        get_B!(B, ω, r1, r2, g1, g2, ρ, μ, K, n; G0=1, inertial_terms=false)

    Compute the 6x6 numerical integrator matrix, which integrates dy/dr from `r1` to `r2` for the solid-body problem.
    `B` here represnts the RK4 integrator, given by Eq. S5.5 in Hay et al., (2025).

    # Arguments
    - `B::Array{precc,2}`                : 6x6 numerical integrator matrix for integrating dy/dr from r1 to r2 for the solid-body problem.
    - `ω::prec`                          : Forcing frequency.
    - `r1::prec`                         : Starting radius for integration.
    - `r2::prec`                         : Ending radius for integration.
    - `g1::prec`                         : Gravity at radius r1.
    - `g2::prec`                         : Gravity at radius r2.
    - `ρ::prec`                          : Density at radius r.
    - `μ::precc`                         : Shear modulus at radius r.
    - `K::precc`                         : Bulk modulus at radius r.
    - `n::Int`                           : Tidal degree.

    Keyword Arguments
    - `G0=1`                             : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.

    # Notes
    See also [`get_B`](@ref)
    """
    function get_B!(B::Array{precc,2}, ω::prec, r1::prec, r2::prec, g1::prec, g2::prec, ρ::prec, μ::precc, K::precc, n::Int; G0=one(prec), inertial_terms::Bool=false)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        A1 = get_A(ω, r1, ρ, g1, μ, K, n; G0=G0, inertial_terms=inertial_terms)
        Ahalf = get_A(ω, rhalf, ρ, ghalf, μ, K, n; G0=G0, inertial_terms=inertial_terms)
        A2 = get_A(ω, r2, ρ, g2, μ, K, n; G0=G0, inertial_terms=inertial_terms)

        k16 = zeros(precc, 6, 6)
        k26 = zeros(precc, 6, 6)
        k36 = zeros(precc, 6, 6)
        k46 = zeros(precc, 6, 6)

        k16 .= dr * A1 
        k26 .= dr * Ahalf * (I + 0.5k16)
        k36 .= dr * Ahalf * (I + 0.5k26)
        k46 .= dr * A2 * (I + k36) 

        # Only compute over the first six indices
        B[1:6,1:6] .= (I + 1.0/6.0 * (k16 + 2k26 + 2k36 + k46))
    end


    """
        get_B(ω, r1, r2, g1, g2, ρ, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=false)

    Compute the 8x8 numerical integrator matrix, which integrates dy/dr from `r1` to `r2` for the two-phase problem.

    # Arguments
    - `ω::prec`                          : Forcing frequency.
    - `r1::prec`                         : Starting radius for integration.
    - `r2::prec`                         : Ending radius for integration.
    - `g1::prec`                         : Gravity at radius r1.
    - `g2::prec`                         : Gravity at radius r2.
    - `ρ::prec`                          : Density at radius r.
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

    Keyword Arguments
    - `G0=1`                             : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.

    # Returns
    - `B::Array{precc,2}`               : 8x8 numerical integrator matrix for integrating dy/dr from r1 to r2 for the two-phase problem.

    # Notes
    See 'get_B!' for definition.
    """ 
    function get_B(ω::prec, r1::prec, r2::prec, g1::prec, g2::prec, ρ::prec, μ::precc, K::precc, ρₗ::prec, Kl::prec, Kd::precc, α::precc, ηₗ::prec, ϕ::prec, k::prec, n::Int; G0=one(prec), inertial_terms::Bool=false)
        B = zeros(precc, 8, 8)
        get_B!(B, ω, r1, r2, g1, g2, ρ, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=G0, inertial_terms=inertial_terms)

        return B
    end


    """
        get_B!(B, ω, r1, r2, g1, g2, ρ, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=false)

    Compute the 8x8 numerical integrator matrix, which integrates dy/dr from `r1` to `r2` for the two-phase problem.
    `B` here represnts the RK4 integrator, given by Eq. S5.5 in Hay et al., (2025).

    # Arguments
    - `B::Array{precc,2}`                : 8x8 numerical integrator matrix for integrating dy/dr from r1 to r2 for the two-phase problem.
    - `ω::prec`                          : Forcing frequency.
    - `r1::prec`                         : Starting radius for integration.
    - `r2::prec`                         : Ending radius for integration.
    - `g1::prec`                         : Gravity at radius r1.
    - `g2::prec`                         : Gravity at radius r2.
    - `ρ::prec`                          : Density at radius r.
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

    Keyword Arguments
    - `G0=1`                             : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.

    # Notes
    See also [`get_B`](@ref)
    """
    function get_B!(B::Array{precc,2}, ω::prec, r1::prec, r2::prec, g1::prec, g2::prec, ρ::prec, μ::precc, K::precc, ρₗ::prec, Kl::prec, Kd::precc, α::precc, ηₗ::prec, ϕ::prec, k::prec, n::Int; G0=one(prec), inertial_terms::Bool=false)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        Abot_p = zeros(precc, 8, 8)
        Amid_p = zeros(precc, 8, 8)
        Atop_p = zeros(precc, 8, 8)

        get_A!(Abot_p, ω, r1, ρ, g1, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=G0, inertial_terms=inertial_terms)
        get_A!(Amid_p, ω, rhalf, ρ, ghalf, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=G0, inertial_terms=inertial_terms)
        get_A!(Atop_p, ω, r2, ρ, g2, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=G0, inertial_terms=inertial_terms)

        k18 = zeros(precc, 8, 8)
        k28 = zeros(precc, 8, 8)
        k38 = zeros(precc, 8, 8)
        k48 = zeros(precc, 8, 8)

        k18 .= dr * Abot_p 
        k28 .= dr *  (Amid_p .+ 0.5Amid_p *k18) 
        k38 .= dr *  (Amid_p .+ 0.5*Amid_p *k28)
        k48 .= dr *  (Atop_p .+ Atop_p*k38) 

        I8 = SMatrix{8,8,precc}(I)

        B .= (I8 + 1.0/6.0 .* (k18 .+ 2*(k28 .+ k38) .+ k48))
    end


    """
        get_B_product!(Bprod2, ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=1, inertial_terms=false)

    Compute the product of the 8x8 B matrices within a primary layer. This is used to propgate the
    y solution across a single two-phase primary layer. Bprod is denoted by D in Eq. S5.14 in 
    Hay et al., (2025).

    # Arguments
    - `Bprod2::Array{precc}`             : 8x8x(nr-1)x(nlayers-1) array to store the B products across each secondary layer within each primary layer. 
    - `ω::prec`                          : Forcing frequency.
    - `r::SubArray{prec}`                : 1D array of layer boundaries.
    - `ρ::prec`                          : 1D array of layer densities. 
    - `g::SubArray{prec}`                : 1D array of gravity values at the layer boundaries. 
    - `μ::precc`                         : 1D array of layer shear moduli.
    - `K::precc`                         : 1D array of layer bulk moduli.
    - `ρₗ::prec`                          : 1D array of liquid densities at layer boundaries.
    - `Kl::prec`                         : 1D array of liquid bulk moduli at layer boundaries.
    - `Kd::precc`                        : 1D array of drained bulk moduli at layer boundaries.
    - `α::precc`                         : 1D array of Biot coefficients at layer boundaries.
    - `ηₗ::prec`                          : 1D array of liquid viscosities at layer boundaries.
    - `ϕ::prec`                          : 1D array of porosities at layer boundaries.
    - `k::prec`                          : 1D array of permeabilities at layer boundaries.
    - `n::Int`                           : Tidal degree.    

    Keyword Arguments
    - `G0=1`                             : Gravitational constant used for non-dimensional scaling.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.
    """
    function get_B_product!(Bprod2::Array{precc}, ω::prec, r::SubArray{prec}, ρ::prec, g::SubArray{prec}, μ::precc, K::precc, ρₗ::prec, Kl::prec, Kd::precc, α::precc, ηₗ::prec, ϕ::prec, k::prec, n::Int; G0=one(prec), inertial_terms::Bool=false)
        # Check dimensions of Bprod2

        nr = size(r)[1]

        Bstart = zeros(precc, 8, 8)
        B = zeros(precc, 8, 8)

        for i in 1:6
            Bstart[i,i] = 1
        end

        # if layer is porous, 
        # don't filter out y7 and y8 components
        if ϕ>0
            Bstart[7,7] = 1
            Bstart[8,8] = 1  
        end

        r1 = r[1]
        g1 = g[1]
        for j in 1:nr-1
            r2 = r[j+1]
            g2 = g[j+1]

            if ϕ>0 
                get_B!(B, ω, r1, r2, g1, g2, ρ, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n; G0=G0, inertial_terms=inertial_terms)
            else
                get_B!(B, ω, r1, r2, g1, g2, ρ, μ, K, n; G0=G0, inertial_terms=inertial_terms)
            end

            Bprod2[:,:,j] .= B * (j==1 ? Bstart : @view(Bprod2[:,:,j-1])) 

            r1 = r2
            g1 = g2 
        end
    end


    """
        compute_M(ω, r, ρ, g, μ, K, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, n, ρ_core, μ_core, κ_core, scales; core="liquid", inertial_terms=false)

    Compute the 4x4 M matrix, which relates the solution at the surface and porous layer interface to the core solution.
     
    # Arguments
    - `ω::prec`                          : Forcing frequency.
    - `r::Array{prec,2}`                 : 2D array of layer boundaries.
    - `ρ::Array{prec,1}`                 : 1D array of layer densities. 
    - `g::Array{prec,2}`                 : 2D array of gravity values at the layer boundaries. 
    - `μ::Array{precc,1}`                : 1D array of layer shear moduli.
    - `K::Array{precc,1}`                : 1D array of layer bulk moduli.
    - `ρₗ::Array{prec,1}`                 : 1D array of liquid densities at layer boundaries.
    - `Kl::Array{prec,1}`                : 1D array of liquid bulk moduli at layer boundaries.
    - `Kd::Array{precc,1}`               : 1D array of drained bulk moduli at layer boundaries.
    - `α::Array{precc,1}`                : 1D array of Biot coefficients at layer boundaries.
    - `ηₗ::Array{prec,1}`                 : 1D array of liquid viscosities at layer boundaries.
    - `ϕ::Array{prec,1}`                 : 1D array of porosities at layer boundaries.
    - `k::Array{prec,1}`                 : 1D array of permeabilities at layer boundaries.
    - `n::Int`                           : Tidal degree.
    - `ρ_core::prec`                     : Density of the core.
    - `μ_core::precc`                    : Shear modulus of the core.
    - `κ_core::precc`                    : Bulk modulus of the core.
    - `scales::Vector{prec}`             : Vector of scaling parameters for non-dimensionalization.

    # Keyword Arguments
    - `core::String="liquid"`            : Type of core, either "liquid" or "solid". This is used to compute the starting vector for the numerical integration across the interior.
    - `inertial_terms::Bool=false`       : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.

    # Returns
    - `M::Array{precc,2}`                : 4x4 M matrix, which is used to propagate the solution across the entire interior. 
    - `y1_4::Array{precc,4}`             : 4D array of the y solutions across each layer, which is used in the `compute_y` function to compute the solution vector across the interior.
    - `S::Vector{Matrix{precc}}`         : Vector of 8x8 matrices representing the normalization.
    - `scale::Vector{prec}`              : Vector of scaling parameters for non-dimensionalization.
    """
    function compute_M(ω::prec, r::Array{prec,2}, ρ::Array{prec,1}, g::Array{prec,2}, μ::Array{precc,1}, K::Array{precc,1}, ρₗ::Array{prec,1}, Kl::Array{prec,1}, Kd::Array{precc,1}, α::Array{precc,1}, ηₗ::Array{prec,1}, ϕ::Array{prec,1}, k::Array{prec,1}, n::Int, ρ_core::prec, μ_core::precc, κ_core::precc, scales::Vector{prec}; core::String="liquid", inertial_terms::Bool=false)
        porous_layer = ϕ .> 0.0

        ## Convert parameters to the precision of precc:
        r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k = convert_params_to_prec(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        sum(porous_layer) > 1.0 && error("Can only handle one porous layer for now!")

        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        # non-dimensional scaling
        R0, M0, ω0, ρ0, G0, g0, μ0, S, Sinv = get_scales(scales[1], scales[2], scales[3]; Y=[1,2,3,4,5,6,7,8])

        scale = [R0, M0, ω0, ρ0, G0, g0, μ0]

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

        # Define starting vector as the core solution matrix, Y_r_C (Eq. S5.15)
        if porous_layer[1]
            y_start = get_Ic(ωs, rs[end,1], ρ_core/ρ0, gs[end,1], μ_core/μ0, κ_core/μ0, core, n; G0=G0, Y=[1,2,3,4,5,6,7,8])
        else
            y_start = zeros(precc, 8, 4)
            y_start[1:6, 1:3] .= get_Ic(ωs, rs[end,1], ρ_core/ρ0, gs[end,1], μ_core/μ0, κ_core/μ0, core, n; G0=G0, Y=[1,2,3,4,5,6])
        end

        y1_4 = zeros(precc, 8,   4, nsublayers-1, nlayers)  # Four linearly independent y solutions
        
        for i in 1:nlayers
            Bprod = zeros(precc, 8, 8, nsublayers-1) # D matrix from Eq. S5.13
            @views get_B_product!(Bprod, ωs, rs[:,i], ρs[i], gs[:,i], μs[i], Ks[i], ρₗs[i], Kls[i], Kds[i], α[i], ηₗs[i], ϕ[i], ks[i], n; G0=G0, inertial_terms=inertial_terms)

            # Modify starting vector if the layer is porous
            # If a new porous layer (i.e., sitting on a non-porous layer)
            # reset the pore pressure and darcy flux
            if porous_layer[i] && (i > 1 && !porous_layer[i-1])
                y_start[7,4] = 1.0          # Non-zero pore pressure
                y_start[8,4] = 0.0          # Zero radial Darcy flux
            elseif !porous_layer[i]
                y_start[7:8, :] .= 0.0      # Pore pressure and flux undefined
            end

            for j in 1:nsublayers-1
                y1_4[:,:,j,i] = @view(Bprod[:,:,j]) * y_start 
            end

            y_start[:,:] .= @view(y1_4[:,:,end,i])
        end

        # Get solution at the surface and porous layer interface (Eq. S5.20)
        M = zeros(precc, 4,4)
        
        M[1, :] .= y1_4[3,:,end,end] # Row 1 - Radial Stress
        M[2, :] .= y1_4[4,:,end,end] # Row 2 - Tangential Stress
        M[3, :] .= y1_4[6,:,end,end] # Row 3 - Potential Stress
        
        has_porous = false
        for i in 1:nlayers
            if porous_layer[i]
                M[4, :] .= y1_4[8, :, end, i] # Row 4 - Darcy flux
                has_porous = true
            end
        end

        # If no porous layer present, set an isolated dummy identity condition
        if !has_porous
            M[4, :] .= 0.0
            M[4, 4] = 1.0  # Decouples C[4] from the 3x3 system
        end
        
        return M, y1_4, S, scale
    end


    """
        compute_y(r, g, M, y1_4, n; load=false)

    Compute the 8x1 solution vector at the surface and porous layer interface, which is used to compute the strain, 
    Darcy flux, and pore pressure at a particular radial level. This is given by Eq. S5.20 in Hay et al., (2025).

    # Arguments
    - `r::Array{prec,2}`                 : 2D array of layer boundaries.
    - `g::Array{prec,2}`                 : 2D array of gravity values at the layer boundaries. 
    - `M::Array{precc,2}`                : 4x4 M matrix, which is used to propagate the solution across the entire interior. 
    - `y1_4::Array{precc,4}`             : 4D array of the y solutions across each layer, which is used in the `compute_y` function to compute the solution vector across the interior.
    - `n::Int`                           : Tidal degree.
    - `S::Matrix{prec}`                  : 8x8 matrix representing the normalization.
    - `scale::Array{prec,1}`             : Vector of scaling parameters for non-dimensionalization.

    # Keyword Arguments
    - `load::Bool=false`                 : If true, compute the solution for a loaded problem.
    
    # Returns
    - `y::Array{ComplexF64,4}`           : 4D array of the solution vector y across the interior.
    """
    function compute_y(r::Array{prec,2}, g::Array{prec,2}, M::Array{precc,2}, y1_4::Array{precc,4}, n::Int, S::Matrix{prec}, scale::Array{prec,1}; load::Bool=false)

        tau = 0
        P = 0
        U_prime = 0
        U = 0
        if load
            U_prime = 1
        elseif !load
            U = 1
        end

        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        # Compute the boundary conditions at the surface
        _, b = get_surface_bc!(r[end,end]/scale[1], g[end,end]/scale[6], n, U, U_prime, tau, P; G0=scale[5], Y=[1,2,3,4,5,6,7,8])
        
        # Compute the weights of integration for the four linearly independent solutions
        C = M \ b[collect((3, 4, 6, 8))]

        # Compute the modal solution vector y across the interior
        y = zeros(ComplexF64, 8, nsublayers-1, nlayers)

        for i in 1:nlayers
            for j in 1:nsublayers-1
                # modal solution = dimensional scaling * elementary solutions * weights of integration
                y[:,j,i] = S*@view(y1_4[:,:,j,i])*C
            end
        end

        return y
    end


    """
        compute_strain_ten!(ϵ, y, n, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)

    Calculate the strain tensor ϵ at a particular radial level. 

    # Arguments
    - `ϵ::SubArray`                     : 4D array to store the strain tensor at a particular radial level, with dimensions corresponding to latitude, longitude, and the 6 independent components of the strain tensor.
    - `y::SubArray`                     : 1D array of the solution vector y at a particular radial level, with 6 components.
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
    """
    function compute_strain_ten!(ϵ::SubArray, y::SubArray, n::Int, rr::Float64, ρr::Float64, gr::Float64, μr::ComplexF64, Ksr::ComplexF64, ω::Float64, ρlr::Float64, Klr::Float64, Kdr::ComplexF64, αr::ComplexF64, ηlr::Float64, ϕr::Float64, kr::Float64)
        i = 1

        @views Y    = solid1d_mush.Y[i,:,:]
        @views dYdθ = solid1d_mush.dYdθ[i,:,:]
        @views dYdϕ = solid1d_mush.dYdϕ[i,:,:]
        @views Z    = solid1d_mush.Z[i,:,:]
        @views X    = solid1d_mush.X[i,:,:]

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
        compute_darcy_displacement!(dis_rel, y, n, r, ω, ϕ, ηl, k, g, ρl)

    Calculate the Darcy displacement vector at a particular radial level. 

    # Arguments
    - `dis_rel::SubArray`               : 4D array to store the Darcy displacement vector at a particular radial level, with dimensions corresponding to latitude, longitude, and the 3 components of the Darcy displacement vector.
    - `y::SubArray`                     : 1D array of the solution vector y at a particular radial level, with 8 components.
    - `n::Int`                          : Tidal degree. 
    - `r::Float64`                      : Radius at which to compute the Darcy displacement vector.
    - `ω::Float64`                      : Forcing frequency.
    - `ϕ::Float64`                      : Porosity at radius r.
    - `ηl::Float64`                     : Liquid viscosity at radius r.
    - `k::Float64`                      : Permeability at radius r.
    - `g::Float64`                      : Gravity at radius r.
    - `ρl::Float64`                     : Liquid density at radius r.
    """
    function compute_darcy_displacement!(dis_rel::SubArray, y::SubArray, n::Int, r::Float64, ω::Float64, ϕ::Float64, ηl::Float64, k::Float64, g::Float64, ρl::Float64)
        i = 1

        @views Y    = solid1d_mush.Y[i,:,:]
        @views dYdθ = solid1d_mush.dYdθ[i,:,:]
        @views dYdϕ = solid1d_mush.dYdϕ[i,:,:]
        
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
        compute_pore_pressure!(p, y, n)

    Calculate the fluid pore pressur at a particular radial level. 

    # Arguments
    - `p::SubArray`                     : 4D array to store the pore pressure at a particular radial level, with dimensions corresponding to latitude and longitude.
    - `y::SubArray`                     : 1D array of the solution vector y at a particular radial level, with 8 components.
    - `n::Int`                          : Tidal degree.
    """
    function compute_pore_pressure!(p::SubArray, y::SubArray, n::Int)
        i = 1

        @views Y    = solid1d_mush.Y[i,:,:]
        @views dYdθ = solid1d_mush.dYdθ[i,:,:]
        @views dYdϕ = solid1d_mush.dYdϕ[i,:,:]
        
        y7 = y[7]
        
        p[:,:] = Y * y7 
    end


    """
        get_heating_profile(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, n; lay=nothing)

    Get the radial volumetric heating for two-phase tides and eccentricity forcing,
    assuming synchronous rotation.
    Heating rate is computed with numerical integration using the solution 
    `y` returned by [`compute_y`](@ref), 
    using Eq. 2.39a/b/c integrated over solid angle. 
    The heating profile for a specific layer is specified with `lay`, otherwise all 
    layers will be caclulated.

    # Arguments
    - `y::Array{ComplexF64}`             : 4D array of the solution vector y across the interior, returned by `compute_y`.
    - `r::Matrix`                        : 2D array of layer boundaries.
    - `ρ::AbstractVector`                : 1D array of layer densities.
    - `g::Matrix`                        : 2D array of gravity values at the layer boundaries.
    - `μ::AbstractVector`                : 1D array of layer shear moduli.
    - `Ks::AbstractVector`               : 1D array of layer bulk moduli for shear dissipation.
    - `ω::prec`                          : Tidal frequency in radians per second.
    - `ρl::AbstractVector`               : 1D array of liquid densities at layer boundaries.
    - `Kl::AbstractVector`               : 1D array of liquid bulk moduli at layer boundaries.
    - `Kd::AbstractVector`               : 1D array of drained bulk moduli at layer boundaries.
    - `α::AbstractVector`                : 1D array of Biot coefficients at layer boundaries.
    - `ηl::AbstractVector`               : 1D array of liquid viscosities at layer boundaries.
    - `ϕ::AbstractVector`                : 1D array of porosities at layer boundaries.
    - `k::AbstractVector`                : 1D array of permeabilities at layer boundaries.
    - `n::Int`                           : Tidal degree.

    # Keyword Arguments
    - `lay::Int=nothing`                 : If specified, compute the heating profile for only the layer corresponding to this index. Otherwise, compute for all layers.

    # Returns
    - `Eμ_tot::Array{Float64,1}`         : 1D array of total power dissipated in each primary layer due to shear, in W.
    - `Eμ_vol::Array{Float64,2}`         : 2D array of angular averaged volumetric heating profiles in W/m^3 for dissipation due to shear, with dimensions corresponding to the secondary layer and primary layer indices.
    - `Eκ_tot::Array{Float64,1}`         : 1D array of total power dissipated in each primary layer due to compaction, in W.
    - `Eκ_vol::Array{Float64,2}`         : 2D array of angular averaged volumetric heating profiles in W/m^3 for dissipation due to compaction, with dimensions corresponding to the secondary layer and primary layer indices.
    - `El_tot::Array{Float64,1}`         : 1D array of total power dissipated in each primary layer due to Darcy flow, in W.
    - `El_vol::Array{Float64,2}`         : 2D array of angular averaged volumetric heating profiles in W/m^3 for dissipation due to Darcy flow, with dimensions corresponding to the secondary layer and primary layer indices.
    """
    function get_heating_profile(y::Array{ComplexF64}, r::Matrix, ρ::AbstractVector, g::Matrix, μ::AbstractVector, Ks::AbstractVector, ω::prec, ρl::AbstractVector, Kl::AbstractVector, Kd::AbstractVector, α::AbstractVector, ηl::AbstractVector, ϕ::AbstractVector, k::AbstractVector, n::Int; lay::Union{Int,Nothing}=nothing)
        
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

        dres = deg2rad(solid1d_mush.res)

        @views clats = solid1d_mush.clats[:]
        @views lons = solid1d_mush.lons[:]

        nsublay = size(r)[1]
        nlay = size(r)[2]
        nlats = length(clats)
        nlons = length(lons)

        ϵ = zeros(ComplexF64, nlats, nlons, 6, nsublay-1, nlay)
        d_disp = zeros(ComplexF64, nlats, nlons, 3, nsublay-1, nlay)
        p = zeros(ComplexF64, nlats, nlons, nsublay-1, nlay)

        ϵs = zero(ϵ)
        d_disps = zero(d_disp)
        ps = zero(p)

        # # Retrieve stain tensor 
        x = 1
        @views Y    = solid1d_mush.Y[x,:,:]

        for i in 1:nlay # Loop of layers
            ρr = ρ[i]
            Ksr = Ks[i]
            μr = μ[i]
            ρlr = ρl[i]
            Klr = Kl[i]
            Kdr = Kd[i]
            αr = α[i]
            ηlr = ηl[i]
            ϕr = ϕ[i]
            kr = k[i]

            for j in 1:nsublay-1 # Loop over sublayers 
                @views yrr = y[:,j,i]
                (y1, y2, y3, y4, y5, y6, y7, y8) = yrr
                
                rr = r[j,i]
                gr = g[j,i]

                compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, n, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
                
                if ϕ[i] > 0 
                    compute_darcy_displacement!(@view(d_disp[:,:,:,j,i]), yrr, n, rr, ω, ϕr, ηlr, kr, gr, ρlr)
                    compute_pore_pressure!(@view(p[:,:,j,i]), yrr, n)
                end

                p[:,:,j,i] .= y7 * Y    # pore pressure

            end
        end

        ϵs .+= ϵ
        d_disps .+= d_disp
        ps .+= p

        # Shear heating in the solid
        Eμ = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        Eμ_tot = zeros(  (nlay) )
        Eμ_vol = zeros(  size(r) )

        # Darcy dissipation in the liquid
        El = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        El_tot = zeros(  (nlay) )
        El_vol = zeros(  size(r) )

        # Bulk dissipation in the solid
        Eκ = zero(Eμ)
        Eκ_tot = zero( Eμ_tot )
        Eκ_vol = zero( Eμ_vol )


        if isnothing(lay)
            rstart = 1
            rend = nlay
        else
            rstart = lay
            rend = lay
        end

        for j in rstart:rend   # loop from CMB to surface          
            for i in 1:nsublay-1
                dr = (r[i+1, j] - r[i, j])
                dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

                @views Eμ[:,:,i, j] = sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 
                                    2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 
                                    1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                Eμ[:,:,i, j] .*= ω * imag(μ[j])

                @views Eκ[:,:,i, j] = ω/2 *imag(Kd[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                if ϕ[j] > 0
                    @views Eκ[:,:,i, j] .+= ω/2 *imag(Kd[j]) * (abs.(ps[:,:,i,j]) ./ Ks[j]).^2
                end

                # Integrate across r to find dissipated energy per unit area
                @views Eμ_vol[i,j] = sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol

                @views Eκ_vol[i,j] = sum(sin.(clats) .* (Eκ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
        
                if ϕ[j] > 0            
                    @views El[:,:,i, j] = 0.5 *  ϕ[j]^2 * ω^2 * ηl[j]/k[j] * (abs.(d_disps[:,:,1,i,j]).^2 + abs.(d_disps[:,:,2,i,j]).^2 + abs.(d_disps[:,:,3,i,j]).^2)
                    @views El_vol[i,j] = sum(sin.(clats) .* (El[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol

                end

            end
            
            # Average over sublayers to get layer volumetric heating (W/m³)
            Eμ_tot[j] = sum(Eμ_vol[1:nsublay-1,j] .* diff(r[1:nsublay,j])) / (r[nsublay,j] - r[1,j])
            Eκ_tot[j] = sum(Eκ_vol[1:nsublay-1,j] .* diff(r[1:nsublay,j])) / (r[nsublay,j] - r[1,j])
            El_tot[j] = sum(El_vol[1:nsublay-1,j] .* diff(r[1:nsublay,j])) / (r[nsublay,j] - r[1,j])

        end

        return (Eμ_tot, Eμ_vol), (Eκ_tot, Eκ_vol), (El_tot, El_vol)
    end

end