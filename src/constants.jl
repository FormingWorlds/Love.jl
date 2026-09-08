

module constants

    using DoubleFloats

    export prec, precc
    export ROOT_DIR, RES_DIR, OUT_DIR
    export AU, G, M_Earth
    export res, lat_vals, nlats, nlons, dres, Ω_total

    # Set the precision for the entire module. You can switch between different types as needed.
    # const prec  = BigFloat
    # const precc = Complex{BigFloat}

    const prec  = Float64
    const precc = Complex{Float64}

    # const prec = Double64
    # const precc = Complex{Double64}

    # Define directory paths
    const ROOT_DIR::String = abspath(dirname(abspath(@__FILE__)), "../")
    const RES_DIR::String  = joinpath(ROOT_DIR,"res/")
    const OUT_DIR::String  = joinpath(ROOT_DIR,"out/")


    # Define physical constants
    const AU::prec      = prec(1.495978707e11)   # m
    const G::prec       = prec(6.6743e-11)       # m^3 kg^-1 s^-2
    const M_Earth::prec = prec(5.9724e24)        # kg


    # Define spherical grid resolution
    const res::Float64 = 60.0                    # angular resolution in degrees
    const lat_vals     = deg2rad.(0:res:180)
    const nlats        = length(lat_vals)
    const nlons        = length(0:res:360-0.001)
    const dres         = deg2rad(res)
    const Ω_total      = sum(sin.(lat_vals)) * nlons * dres^2

end
