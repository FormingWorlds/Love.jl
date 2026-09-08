

module fluid0d

    include("constants.jl")
    using .constants

    export compute_fluid_lovenumbers, mean_rho

    
    """
        compute_fluid_lovenumbers(omega, R, H_magma, g, ρ_ratio, n, σ_R)    
    
    Calculate kn lovenumbers in fluid.

    # Arguments
    - `omega::prec`                     : Forcing frequency range.
    - `R::prec`                         : Outer radius of fluid segment in mantle.
    - `H_magma::prec`                   : Height of fluid segment in mantle.
    - `g::prec`                         : Surface gravity at top of fluid segment in mantle.
    - `ρ_ratio::prec`                   : Density contrast between current (fluid) and lower (non-fluid) layer.
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `sigma_R::Float64=1e-3`           : Rayleigh drag coefficient.

    # Returns
    - `kn_T::precc`                     : Complex Tidal kn Lovenumber.
    - `kn_L::precc`                     : Complex Load kn Lovenumber.
    """
    function compute_fluid_lovenumbers(
            omega::Float64,
            R::prec,
            H_magma::prec,
            g::prec,
            ρ_ratio::prec,
            n::Int,
            σ_R::Float64
        )::Tuple{precc,precc}
                
        # calculate parameters
        μ_n  = n * (n + 1)
        ξ_n = 3.0 / (2.0*n + 1.0) * ρ_ratio
        σ_n = sqrt(μ_n * g * H_magma / R^2) # characterestic frequency

        σ_T = omega - im*σ_R

        # Tidal love number
        kn_T = -ξ_n * σ_n^2 / (omega*σ_T - σ_n^2)

        # Load love numbers
        kn_L = 0. 

        return kn_T, kn_L
    end


    """
        mean_rho(ρ, r)

    Calculate the mean density in segment.

    # Arguments
    - `ρ::Array{prec,1}`                : Density profile of the planet.
    - `r::Array{prec,1}`                : Radial positions of layers, from bottom to top of segment.

    # Returns
    - `mean_density::prec`              : Mean density in segment.
    """
    function mean_rho( 
            ρ::Array{prec,1},
            r::Array{prec,1}
        )::prec

        # calculate mean density of region
        V = 4/3 * π * (r[2:end].^3 .- r[1:end-1].^3) # Volume of each layer (m^3)
        M = sum(V .* ρ[1:end])                       # Total mass (kg)
        mean_density = M / sum(V)                    # Mean density (kg/m^3)

        return mean_density

    end

end