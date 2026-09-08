# Core file containing functions for running the model

# Not for direct execution
if (abspath(PROGRAM_FILE) == @__FILE__)
    thisfile = @__FILE__
    error("The file '$thisfile' is not for direct execution")
end

module Obliqua

    # Include system libraries
    using LoggingExtras
    using Printf
    import TOML:parsefile
    using Interpolations
    using LinearAlgebra
    using SpecialFunctions
    using AssociatedLegendrePolynomials
    using NCDatasets
    
    # Include local jl files
    include("constants.jl")
    include("solid0d.jl")
    include("solid1d.jl")
    include("solid1d_mush.jl")
    include("solid1d_relax.jl")
    include("solid1d_mush_relax.jl")
    include("solid1d_equil_relax.jl")
    include("fluid0d.jl")
    include("Hansen.jl")
    include("load.jl")
    include("interior.jl")
    include("plotting.jl")

    # Import submodules
    using  .constants
    import .solid0d
    import .solid1d
    import .solid1d_mush
    import .solid1d_relax
    import .solid1d_mush_relax
    import .solid1d_equil_relax
    import .fluid0d
    import .Hansen
    import .load
    import .interior
    import .plotting

    # Export submodules (mostly for autodoc purposes)
    export solid0d
    export solid1d
    export solid1d_mush
    export solid1d_relax
    export solid1d_mush_relax
    export solid1d_equil_relax
    export fluid0d
    export Hansen
    export load
    export interior
    export plotting

    export run_tides


    """
        Create a logger object and return it.

    Arguments:
    - `outpath::String`                 : Output file (empty to disable file logging).

    Optional arguments:
    - `to_term::Bool`                   : Log to terminal?

    Returns:
    - `logger_both`                     : Logger object.
    """
    function make_logger(outpath::String; to_term::Bool=true)

        # Formatting
        color::Int = 39
        level::String = "UNSET"
        term_io::IO = stdout

        # Setup file logger
        to_file::Bool = !isempty(outpath)
        if to_file
            # remove old file
            if isfile(outpath)
                rm(outpath)
            end

            # configure
            logger_file = FormatLogger(outpath; append=false) do io, args
                if args.level == LoggingExtras.Info
                    level = "INFO"
                elseif args.level == LoggingExtras.Warn
                    level = "WARN"
                elseif args.level == LoggingExtras.Debug
                    level = "DEBUG"
                elseif args.level == LoggingExtras.Error
                    level = "ERROR"
                end
                @printf(io, "[ %-5s ] %s \n", level, args.message)
            end;
        end

        # Setup terminal logger
        if to_term
            logger_term = FormatLogger() do io, args
                if args.level == LoggingExtras.Info
                    color = 32
                    level = "INFO"
                elseif args.level == LoggingExtras.Warn
                    color = 93
                    level = "WARN"
                elseif args.level == LoggingExtras.Debug
                    color = 96
                    level = "DEBUG"
                elseif args.level == LoggingExtras.Error
                    color = 91
                    term_io = stderr
                    level = "ERROR"
                end
                # Set color, set bold, print level, unset bold, unset color, message
                @printf(term_io, "[\033[%dm\033[1m %-5s \033[21m\033[0m] %s\n",
                                    color, level, args.message)
            end;
        end

        # Return logger object
        if to_file && to_term
            return TeeLogger(logger_file, logger_term)
        elseif to_term
            return logger_term
        elseif to_file
            return logger_file
        else
            println(stderr, "Warning: using NullLogger to log all messages")
            return NullLogger()
        end
    end


    """
        Setup terminal logging and file logging.

    Arguments:
    - `outpath::String`                 : Output file (empty to disable file logging)
    - `verbosity::Int`                  : Verbosity (0: silent, 1: normal, 2: debug)
    """
    function setup_logging(outpath::String, verbosity::Int)

        # If silent
        if verbosity==0
            global_logger(MinLevelLogger(current_logger(), Logging.Error))
            return nothing
        end

        # Make the logger
        logger_both = make_logger(outpath)
        global_logger(logger_both)

        # Disable debug
        if verbosity == 1
            disable_logging(Logging.Debug)
        end

        return nothing
    end


    """
        Open and validate config file.

    Arguments:
    - `cfg_path::String`                : Path to configuration file

    Returns:
    - `cfg_dict::Dict`                  : Dictionary containing the configuration
    """
    function open_config(cfg_path::String)::Dict

        # open file
        cfg_dict = parsefile(cfg_path)

        # check headers
        headers = ["params", "orbit", "struct", "interior_energetics", "title", "version"]
        for h in headers
            if !haskey(cfg_dict, h)
                error("Key $h is missing from configuration file at '$cfg_path'")
            end
        end

        # check that output dir is named
        if !haskey(cfg_dict["params"]["out"],"path") || (cfg_dict["params"]["out"]=="")
            error("Output directory is missing from configuration file at '$cfg_path'")
        end
        out_path = abspath(cfg_dict["params"]["out"]["path"])

        # check if this is a dangerous path
        if ispath(joinpath(out_path, ".git")) || (joinpath(out_path) == pwd()) || samefile(out_path, ROOT_DIR)
            error("Output directory is unsafe")
        end

        # looks good
        return cfg_dict
    end


    """
        run_tides(omega, axial, ecc, sma, S_mass, rho, radius, visc, shear, bulk, bulkd, phi, perm, cfg)

    Compute the tidal heating profile of a planetary interior considering solid and fluid layers.

    # Arguments
    - `omega::prec`                 : Orbital frequency of the body.
    - `axial::prec`                 : Axial (spin) frequency of the body.
    - `ecc::Float64`                : Orbital eccentricity.
    - `sma::Float64`                : Semi-major axis of the orbit.
    - `S_mass::Float64`             : Mass of the central body (e.g., star) inducing tides.
    - `rho::Array{prec,1}`          : Radial density profile of the planet, from core to surface.
    - `radius::Array{prec,1}`       : Radial positions of layers, from core to surface.
    - `visc::Array{prec,1}`         : Viscosity profile of the planet.
    - `shear::Array{prec,1}`        : Shear modulus profile of the solid layers.
    - `bulk::Array{prec,1}`         : Bulk modulus profile of the solid layers.
    - `bulkd::Array{prec,1}`        : Drained bulk modulus profile of the fluid layers.
    - `phi::Array{prec,1}`          : Porosity profile of the fluid layers.
    - `perm::Array{prec,1}`         : Permeability profile of the fluid layers.
    - `cfg::Dict`                   : Configuration parameters from dictionary.

    # Returns
    - `power_prf::Array{Float64,1}`      : Radial profile of tidal heating (W/m³).
    - `power_blk::Float64`               : Total tidal power integrated over the interior (W).
    - `nmk::Array{Tuple{Int,Int,Int},1}` : List of tidal modes (n, m, k) considered in the calculation.
    - `σ_range::Array{Float64,1}`        : Frequencies at which the Love number `k_n` was evaluated.
    - `Hansen::Array{Float64,1}`         : Hansen coefficients corresponding to the tidal modes.
    - `knms_total::Array{ComplexF64,1}`  : Complex Love number `k_n` for the planet.
    """
    function run_tides(omega::prec,
                        axial::prec,
                        ecc::Float64,
                        sma::Float64,
                        S_mass::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        visc::Array{prec,1},
                        shear::Array{prec,1},
                        bulk::Array{prec,1},
                        bulkd::Array{prec,1},
                        phi::Array{prec,1},
                        perm::Array{prec,1},
                        cfg::Dict
                        )::Tuple{Vector{Float64}, Float64, Vector{Tuple{Int,Int,Int}}, Vector{Float64}, Vector{ComplexF64}}
      
        # Read configuration options from dict
        @info "Using configuration '$(cfg["title"])'"
        
        # Check that config has these always-required keys
        req_keys = Dict(
            "params.out" => ["path"],
            "orbit.obliqua" => [
                "store_3D", "min_frac","visc_l","visc_lus","visc_sus",
                "n","m","spectrum", "material_mu", "material_k",
                "module_solid", "module_fluid", "module_mushy"
            ],
            "orbit.obliqua.solid" => [
                "ncalc"
            ],
            "orbit.obliqua.fluid" => [
                "sigma_R"
            ],
            "struct" => [
                "core_density"
            ]
        )
        for (section, keys) in req_keys
            path = split(section, ".")
            node = cfg

            # walk down the nested Dict
            for p in path
                if !(node isa AbstractDict)
                    @error "Config: expected Dict at `$(join(path, "."))`, got $(typeof(node))"
                    break
                end
                if !haskey(node, p)
                    @error "Config: missing required section `$(join(path, "."))`"
                    break
                end
                node = node[p]
            end

            @debug "Tested config section `$(join(path, "."))`"

            # check required keys at this level
            for k in keys
                @debug "Testing config key `$(join(path, "."))::$k`"
                if !haskey(node, k)
                    @error "Config: missing required key `$(join(path, "."))::$k`"
                end
            end
        end

        @debug "Config file validation complete."

        # collection of config params 
        outpath      = cfg["params"]["out"]["path"]
        if outpath == "out"
            outpath = OUT_DIR
        end
        time         = cfg["params"]["out"]["time"]

        store_3D     = cfg["orbit"]["obliqua"]["store_3D"]
        enforce_ec   = cfg["orbit"]["obliqua"]["enforce_ec"]
        optimize_scales = cfg["orbit"]["obliqua"]["optimize_scales"]
        solid_shell  = cfg["orbit"]["obliqua"]["solid_shell"]

        min_frac     = cfg["orbit"]["obliqua"]["min_frac"]

        visc_l       = cfg["orbit"]["obliqua"]["visc_l"]
        visc_lus     = cfg["orbit"]["obliqua"]["visc_lus"]
        visc_s       = cfg["orbit"]["obliqua"]["visc_s"]
        visc_sus     = cfg["orbit"]["obliqua"]["visc_sus"]

        n            = cfg["orbit"]["obliqua"]["n"]
        m            = cfg["orbit"]["obliqua"]["m"]

        spectrum = cfg["orbit"]["obliqua"]["spectrum"]
        if spectrum == "adaptive"
            s_min    = cfg["orbit"]["obliqua"]["s_min"]
            s_max    = cfg["orbit"]["obliqua"]["s_max"]
            s_min    = nothing_if_none(s_min)
            s_max    = nothing_if_none(s_max)
        elseif spectrum == "full"
            N_σ      = cfg["orbit"]["obliqua"]["N_sigma"]
            p_min    = cfg["orbit"]["obliqua"]["p_min"]
            p_max    = cfg["orbit"]["obliqua"]["p_max"]
        end

        material_μ   = cfg["orbit"]["obliqua"]["material_mu"]
        material_κ   = cfg["orbit"]["obliqua"]["material_k"]
        alpha        = cfg["orbit"]["obliqua"]["alpha"]

        module_solid = cfg["orbit"]["obliqua"]["module_solid"]
        ncalc        = cfg["orbit"]["obliqua"]["solid"]["ncalc"]
        dr_min       = cfg["orbit"]["obliqua"]["solid"]["dr_min"]
        dr_max       = cfg["orbit"]["obliqua"]["solid"]["dr_max"]
        core         = cfg["orbit"]["obliqua"]["solid"]["core"]
        inertial_terms = cfg["orbit"]["obliqua"]["solid"]["inertial_terms"]
        bulk_l       = cfg["orbit"]["obliqua"]["solid"]["bulk_l"]
        porosity_thresh = cfg["orbit"]["obliqua"]["solid"]["porosity_thresh"]

        module_fluid = cfg["orbit"]["obliqua"]["module_fluid"]
        sigma_R      = cfg["orbit"]["obliqua"]["fluid"]["sigma_R"]
        sigma_R_inf  = cfg["orbit"]["obliqua"]["fluid"]["sigma_R_inf"]
        sigma_R_prf  = cfg["orbit"]["obliqua"]["fluid"]["sigma_R_prf"]
        H_R          = cfg["orbit"]["obliqua"]["fluid"]["H_R"]
        efficiency   = cfg["orbit"]["obliqua"]["fluid"]["efficiency"]

        module_mushy = cfg["orbit"]["obliqua"]["module_mushy"]
        if module_mushy == "interp"
            b_width  = cfg["orbit"]["obliqua"]["mushy"]["b_width"]
            t_width  = cfg["orbit"]["obliqua"]["mushy"]["t_width"]
        end

        # convert "true" to true and "false" to false
        store_3D   = true_if_true(store_3D)
        enforce_ec = true_if_true(enforce_ec)
        optimize_scales = true_if_true(optimize_scales)
        solid_shell = true_if_true(solid_shell)

        # convert "none" to nothing
        module_solid = nothing_if_none(module_solid)
        module_fluid = nothing_if_none(module_fluid)
        module_mushy = nothing_if_none(module_mushy)

        # convert interior profiles to prec                 
        ρ = convert(Vector{prec}, rho)
        r = convert(Vector{prec}, radius)
        η = convert(Vector{prec}, visc)
        μ = convert(Vector{precc},shear)
        κ = convert(Vector{precc},bulk)
        κd = convert(Vector{precc},bulkd)
        ϕ = convert(Vector{prec}, phi)
        K = convert(Vector{prec}, perm)
        
        # 1d tidal models have singularity at r=0, so we add a small offset to avoid this
        # 0d models only consider r[end] and remain unaffected.
        if iszero(r[1])
            # Add small offset to avoid singularity at r=0
            r[1] = 1e-6 * r[2]
        end

        # number of layers
        N_layers = length(r)-1

        # liquidus and solidus viscosity
        η_l = visc_lus
        η_s = visc_sus

        # get smoothed out region masks and segments
        mask_l, mask_s, mask_c, is_seg, segments = get_layers(r, η, η_l, η_s; min_frac)

        # check if CMB is at the bottom of the mantle
        if any(r[1] .>= r[2:end])
            throw("CMB radius not at bottom of mantle, did you properly order the interior arrays?")
        end

        # check which core properties to use for CMB boundary condition
        #   "core" uses the core density to compute the core mass
        #   "mantle" uses the lowest mantle layer density to compute the core mass
        core_props = cfg["orbit"]["obliqua"]["solid"]["core_props"]

        # get core mass from core density and radius
        if core_props == "core"
            ρ_core = convert(prec, cfg["struct"]["core_density"])
        elseif core_props == "mantle"
            ρ_core = ρ[1]
        else
            throw("Invalid core_props value: $core_props. Must be 'core' or 'mantle'.")
        end
        m_core = (4/3)*π*r[1]^3*ρ_core

        # find planet radius (m)
        R = maximum(r)

        # shell masses
        dv = 4/3 * π * (r[2:end].^3 .- r[1:end-1].^3) 
        dm = dv .* ρ

        # cumulative enclosed mass, including core mass
        M_enc = cumsum(dm) .+ m_core

        # gravity at each layer radius
        g = G .* M_enc ./ r[2:end].^2

        # collect tidal modes (n, m) 
        nm  = [(n_i, m_i) for n_i in n, m_i in m if n_i >= m_i]
        # initiate container for (n, m, k) combinations
        nmk = Vector{Tuple{Int,Int,Int}}()

        # Initialize containers that will hold data per (n, m, k) mode
        σ_range  = Vector{Float64}()
        X_hansen = Vector{Float64}()
        μc       = Matrix{ComplexF64} # layer x frequency
        κc       = Matrix{ComplexF64} # layer x frequency
        κdc      = Matrix{ComplexF64} # layer x frequency

        # orbital and axial frequencies
        if spectrum == "adaptive"            

            for (n_i, m_i) in nm
                # avoid overwriting inputs by using local scope variables
                s_min_i = s_min === nothing ? nothing : s_min
                s_max_i = s_max === nothing ? nothing : s_max

                # get s range for proper convergence for given eccentricity
                s_min_ecc, s_max_ecc = Hansen.get_k_range(ecc, n_i, m_i)

                if s_min_i === nothing 
                    s_min_i = s_min_ecc
                elseif s_min_i > s_min_ecc
                    @warn "Provided s_min=$s_min_i is larger than estimated s_min=$s_min_ecc for eccentricity $(round(ecc, digits=2))."
                end
                if s_max_i === nothing
                    s_max_i = s_max_ecc
                elseif s_max_i < s_max_ecc
                    @warn "Provided s_max=$s_max_i is smaller than estimated s_max=$s_max_ecc for eccentricity $(round(ecc, digits=2))."
                end

                @info "Using adaptive spectrum with s range [$s_min_i, $s_max_i] for eccentricity $(round(ecc, digits = 2)) and tide (n, m) = ($n_i, $m_i)."

                # build s range for this mode
                s_range = collect(s_min_i:s_max_i)

                # get hansen coefficients for this specific mode
                _, X_hansen_i = Hansen.get_hansen(ecc, n_i, m_i, s_min_i, s_max_i)
                
                # calculate forcing frequencies only for region of interest
                σ_range_i = m_i .* axial .- s_range .* omega
                σ_range_i = Float64.(σ_range_i)

                for i in 1:length(s_range)
                    push!(nmk, (n_i, m_i, s_range[i]))
                    push!(X_hansen, X_hansen_i[i])
                    push!(σ_range, σ_range_i[i])
                end
            end

            N_σ = length(σ_range)

        elseif spectrum == "full"
            # Maintain backward compatibility for the fallback plotting spectrum
            t_range = 10 .^ range(p_min, stop=p_max, length=N_σ)       
            σ_range_i = 2π ./ (t_range .* 1e3 .* 365.25 .* 24 .* 3600)    
            σ_range_i = reshape(σ_range_i, :)

            # set particular nmk for full spectrum
            for i in 1:N_σ
                push!(nmk, (nm[1][1], nm[1][2], 1)) 
                push!(σ_range, σ_range_i[i])
            end
            @info "Using (n, m, k) = ($(nm[1][1]), $(nm[1][2]), 1) for full spectrum."
        end

        # get frequency dependent complex shear modulus per mode
        μc = complex_modulus(σ_range, μ, η; material=material_μ, α=alpha)

        # get frequency dependent complex bulk modulus per mode
        if module_solid === "solid1d-mush" || module_solid === "solid1d-mush-relax"
            κc = complex_modulus(σ_range, κ, η./(ϕ.+1e-15); material="elastic", α=alpha)
            κdc = complex_modulus(σ_range, κd, η./(ϕ.+1e-15); material=material_κ, α=alpha)
            
            # calculate Biot's modulus
            α  = precc(1.0) .- (κdc ./ κc)
        else
            κc = complex_modulus(σ_range, κ, η./(ϕ.+1e-15); material=material_κ, α=alpha)
            κdc = complex_modulus(σ_range, κd, η./(ϕ.+1e-15); material="elastic", α=alpha)

            α = fill!(similar(κc, precc), 1)
        end

        # smooth complex moduli for mushy layers
        μc, κc = smooth_complex_modulus!(μc, κc, r, η)

        μ_core = zeros(precc, N_σ)
        κ_core = zeros(precc, N_σ)
        # update core properties based on config
        if core_props == "core"
            μ_core = fill(convert(precc, cfg["struct"]["core_shear"]), N_σ)
            κ_core = fill(convert(precc, cfg["struct"]["core_bulk"]), N_σ)
        elseif core_props == "mantle"
            μ_core = μc[1, :]
            κ_core = κc[1, :]
        else
            throw("Invalid core_props value: $core_props. Must be 'core' or 'mantle'.")
        end

        # allocate outputs for this specific mode's frequency count
        # initiate forcing frequency dependent k Love numbers (one spectrum for each segment)
        knms_T = zeros(ComplexF64, N_σ, length(segments))
        knms_L = zeros(ComplexF64, N_σ, length(segments))
        # initiate forcing frequency dependent heating profile
        prf_total = zeros(Float64, N_σ, N_layers)
        map_total_μ = zeros(Float64, N_σ, length(0:res:180), length(0:res:360-0.001), N_layers)     # shear
        map_total_κ = zeros(Float64, N_σ, length(0:res:180), length(0:res:360-0.001), N_layers)     # bulk
        map_total_l = zeros(Float64, N_σ, length(0:res:180), length(0:res:360-0.001), N_layers)     # darcy
        map_total_f = zeros(Float64, N_σ, length(0:res:180), length(0:res:360-0.001), N_layers)     # friction
        map_total_d = zeros(Float64, N_σ, length(0:res:180), length(0:res:360-0.001), N_layers)     # drag

        # core density for bottom boundary
        ρ_mean_lower = ρ_core
        # Rayleigh drag efficiency at fluid core
        efficiency_seg = efficiency

        # interpolation activity boolean, used with the interp function for mushy layers
        interp_active = false
        interp_previous = false

        # loop over segments, starting at CMB
        for (iseg, seg) in pairs(segments)
            # check if interpolation is active
            if interp_active
                interp_previous = true
                interp_active = false
            end

            # get start and stop index for segment
            i_start, i_end = is_seg[iseg]

            # perform slices
            r_seg  = r[i_start:i_end+1]
            g_seg  = g[i_start:i_end]
            ρ_seg  = ρ[i_start:i_end]
            η_seg  = η[i_start:i_end]                              
            μc_seg = μc[i_start:i_end, :] 
            κc_seg = κc[i_start:i_end, :]
            κdc_seg = κdc[i_start:i_end, :]
            ϕ_seg  = ϕ[i_start:i_end]
            g_seg  = g[i_start:i_end]
            α_seg  = α[i_start:i_end, :]
            K_seg  = K[i_start:i_end]

            # mean density in current segment
            if length(ρ_seg) == 1
                ρ_mean = ρ_seg[1]
            else
                ρ_mean = fluid0d.mean_rho(ρ_seg, r_seg)
            end

            # density ratio
            ρ_ratio = ρ_mean / ρ_mean_lower

            # add patch if solid_shell=true and segment is fluid-mush modeled by solid1d_relax or solid1d_mush_relax
            patch = false
            if solid_shell && (seg == "solid") && (module_solid == "solid1d-relax" || module_solid == "solid1d-mush-relax")
                if !any(η_seg .> 1e17)
                    patch = true
                end
            end

            # get k2 spectrum for segment
            for iss in 1:N_σ
                # specify mode
                n_i, m_i, s_i = nmk[iss]

                # specify forcing frequency
                σ = σ_range[iss]

                # if forcing frequency is zero, switch any solid1d variant to equil-relax model
                if abs(σ) < 1e-18 && occursin("solid1d", module_solid)
                    module_solid = "solid1d-equil-relax"
                end

                # if segment is solid
                if seg == "solid"
                    # don't model solid tides
                    if module_solid===nothing
                        knms_T[iss, iseg], knms_L[iss, iseg] = 0., 0.
                    # 0D interior but no heating profile in segment 
                    elseif module_solid=="solid0d"
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_solid0d( 
                            μc_seg[:, iss],
                            r_seg,
                            M_enc[i_end];
                            n=n_i
                        )
                    # elseif 1D interior and heating profile from strain tensor
                    elseif module_solid=="solid1d"
                        prf_total[iss, i_start:i_end], 
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_solid1d( 
                            σ, ρ_seg,
                            r_seg, η_seg,                               
                            μc_seg[:, iss], 
                            κc_seg[:, iss], R, 
                            m_core, ρ_core, 
                            μ_core[iss], κ_core[iss];
                            ncalc=ncalc, n=n_i, m=m_i, core=core,
                            inertial_terms=inertial_terms,
                            optimize_scales=optimize_scales
                        )
                    # elseif 1D interior and heating profile from strain tensor
                    elseif module_solid=="solid1d-relax"
                        prf_total[iss, i_start:i_end], 
                        map_total_μ[iss, :, :, i_start:i_end], 
                        map_total_κ[iss, :, :, i_start:i_end], 
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_solid1d_relax( 
                            σ, ρ_seg, r_seg, g_seg,
                            η_seg, μc_seg[:, iss], 
                            κc_seg[:, iss], R, 
                            m_core, ρ_core, 
                            μ_core[iss], κ_core[iss];
                            dr_min=dr_min, dr_max=dr_max, 
                            n=n_i, m=m_i, core=core, 
                            inertial_terms=inertial_terms,
                            optimize_scales=optimize_scales, patch=patch
                        )
                    # elseif 1D interior with mush interface and heating profile from strain tensor
                    elseif module_solid=="solid1d-mush"
                        prf_total[iss, i_start:i_end], 
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_solid1d_mush( 
                            σ, ρ_seg, r_seg,
                            η_seg, μc_seg[:, iss], 
                            κc_seg[:, iss], κdc_seg[:, iss], 
                            ϕ_seg, α_seg[:, iss], K_seg, R, 
                            m_core, ρ_core, 
                            μ_core[iss], κ_core[iss];
                            ncalc=ncalc, n=n_i, m=m_i, core=core, 
                            inertial_terms=inertial_terms,
                            visc_l=visc_l, bulk_l=bulk_l, 
                            porosity_thresh=porosity_thresh, 
                            optimize_scales=optimize_scales
                        )
                    elseif module_solid=="solid1d-mush-relax"
                        prf_total[iss, i_start:i_end], 
                        map_total_μ[iss, :, :, i_start:i_end], 
                        map_total_κ[iss, :, :, i_start:i_end], 
                        map_total_l[iss, :, :, i_start:i_end], 
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_solid1d_mush_relax( 
                            σ, ρ_seg, r_seg, g_seg,
                            η_seg, μc_seg[:, iss], 
                            κc_seg[:, iss], κdc_seg[:, iss], 
                            ϕ_seg, α_seg[:, iss], K_seg, R, 
                            m_core, ρ_core,
                            μ_core[iss], κ_core[iss];
                            dr_min=dr_min, dr_max=dr_max, 
                            n=n_i, m=m_i, core=core, 
                            inertial_terms=inertial_terms, 
                            visc_l=visc_l, bulk_l=bulk_l,
                            porosity_thresh=porosity_thresh, 
                            optimize_scales=optimize_scales, patch=patch
                        )
                    # elseif 1D interior and heating profile from strain tensor
                    elseif module_solid=="solid1d-equil-relax"
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_solid1d_equil_relax( 
                            σ, ρ_seg, r_seg, g_seg,
                            η_seg, μc_seg[:, iss], 
                            κc_seg[:, iss], R, 
                            m_core, ρ_core, 
                            μ_core[iss], κ_core[iss];
                            dr_min=dr_min, dr_max=dr_max, 
                            n=n_i, m=m_i, core=core, 
                            optimize_scales=optimize_scales, patch=patch
                        )

                        # reset solid module to original value for next frequency/segment
                        module_solid = cfg["orbit"]["obliqua"]["module_solid"]
                    else
                        throw("No compatible solid tides module: $module_solid.")
                    end

                # if segment is fluid
                elseif seg == "fluid"
                    # don't model fluid tides
                    if module_fluid===nothing
                        knms_T[iss, iseg], knms_L[iss, iseg] = 0., 0.
                    # 0D interior but no heating profile in segment 
                    elseif module_fluid=="fluid0d"
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_fluid0d(
                            σ, ρ_seg, 
                            r_seg, ρ_ratio;
                            n=n_i, 
                            sigma_R=sigma_R
                        ) 
                    # elseif 1D interior and heating profile from density-contrast/Rayleigh-drag
                    elseif module_fluid=="fluid1d"
                        # get bottom shear/bulk/darcy heating rates for fluid segment
                        if iseg > 1
                            # get previous segment's upper interface heating
                            i_sp, i_ep = is_seg[iseg-1]
                            P_b = [map_total_μ[iss, :, :, i_ep], 
                                   map_total_κ[iss, :, :, i_ep], 
                                   map_total_l[iss, :, :, i_ep]]
                        else 
                            # no previous segment, so set bottom heating to zero
                            P_b = [zeros(Float64, nlats, nlons) for _ in 1:3]
                        end

                        prf_total[iss, i_start:i_end], 
                        map_total_μ[iss, :, :, i_start:i_end], 
                        map_total_κ[iss, :, :, i_start:i_end], 
                        map_total_l[iss, :, :, i_start:i_end], 
                        map_total_f[iss, :, :, i_start:i_end], 
                        map_total_d[iss, :, :, i_start:i_end], 
                        knms_T[iss, iseg], knms_L[iss, iseg] = run_fluid1d(
                            σ, ρ_seg, r_seg, 
                            η_seg, ρ_ratio,
                            P_b, R; n=n_i,
                            sigma_R=sigma_R,
                            sigma_inf=sigma_R_inf,
                            sigma_R_prf=sigma_R_prf,
                            H_R=H_R, efficiency=efficiency_seg,
                            visc_l=visc_l
                        )
                    else
                        throw("No compatible fluid tides module: $module_fluid.")
                    end
                
                # if segment is mush
                elseif seg == "mush"
                    # don't model mush tides
                    if module_mushy===nothing
                        knms_T[iss, iseg], knms_L[iss, iseg] = 0., 0.
                    elseif module_mushy=="interp"
                        # turn on interpolation mode
                        interp_active = true

                        # get heating in previous layers
                        if i_start > 1 && iseg > 1
                            i_sp, i_ep = is_seg[iseg-1]
                            P_b = prf_total[iss, i_ep]

                            # first solve heating spectrum for lower interface
                            prf_total[iss, i_start:i_end], 
                            knms_T[iss, iseg], knms_L[iss, iseg] = run_interp(
                                σ, r_seg, R, 0., P_b;
                                t_width=t_width, b_width=b_width
                            )
                        end
                        # Then model next segment to get heating at the upper interface P_t
                    else
                        throw("No compatible mush tides module: $module_mushy.")
                    end

                # if segment is ice
                elseif seg == "ice"
                    # calculate ice tides in ice region 
                    knms_T[iss, iseg], knms_L[iss, iseg] = 0., 0. # no expression for this yet
                    @warn "Ice layers are currently not supported. Skipping this segment..."
                    
                # if segment is water
                elseif seg == "water"
                    # calculate water tides in water region 
                    knms_T[iss, iseg], knms_L[iss, iseg] = 0., 0. # no expression for this yet
                    @warn "Water layers are currently not supported. Skipping this segment..."    
                end

                if interp_previous
                    # Solve heating spectrum for upper interface in previous segment
                    P_t = prf_total[iss, i_start] # get heating in bottom layer of current segment
                    i_sp, i_ep = is_seg[iseg-1]
                    Δprf, ΔkT, ΔkL = run_interp(
                        σ, r[i_sp:i_ep+1], R, P_t, 0.;
                        t_width=t_width, b_width=b_width
                    )
                    prf_total[iss, i_sp:i_ep] .+= Δprf
                    knms_T[iss, iseg-1]        += ΔkT
                    knms_L[iss, iseg-1]        += ΔkL
                end

                # repeat for all probe forcing frequencies
            end
        
            # enforce energy conservation if selected in config file
            if enforce_ec && any(prf_total[:, i_start:i_end] .!= 0.0)
                @views enforce_energy_conservation!(
                    prf_total[:, i_start:i_end], 
                    knms_T[:, iseg],
                    σ_range,
                    Float64(R), 
                    Float64.(dv[i_start:i_end]), 
                    nmk, 
                    map_total_μ[:, :, :, i_start:i_end], 
                    map_total_κ[:, :, :, i_start:i_end], 
                    map_total_l[:, :, :, i_start:i_end],
                    map_total_f[:, :, :, i_start:i_end],
                    map_total_d[:, :, :, i_start:i_end]
                )
            end

            # update previous segment mean density before moving to next segment
            ρ_mean_lower = ρ_mean
            # update Rayleigh drag efficiency away from core
            efficiency_seg = 1.

            # turn off interpolator after completion
            interp_previous = false

            # step to next segment
        end  

        # initialize total k2 with the contribution from the top layer
        knms_total = copy(knms_T[:, end])  

        # loop from top (surface) to just above CMB
        # This implementation is currently only valid for CMB -> Solid -> Fluid -> Surface layering, 
        # and would need to be adapted for more complex layering (e.g., fluid under solid).
        for iseg in reverse(1:length(segments)-1)

            # get the indices for the current segment and the previous segment
            # current segemnt
            i_start, _ = is_seg[iseg]
            # previous segment
            _, i_end_ini = is_seg[iseg+1]

            load = segments[iseg] == "solid" && segments[iseg+1] == "fluid" ? true : false

            if load
                for i in 1:N_σ
                    # calculate the load distribution factor
                    factor = 1 + (imag(knms_L[i, iseg] * knms_total[i])) / (imag(knms_T[i, iseg]) + imag(knms_total[i]) + 1e-40) # add small number to avoid divide by zero

                    if factor < 0. || factor > 1.1
                        @warn "Load distribution factor is negative for frequency index $i and segment $iseg. Setting $factor to 1."                        
                        # A known issue where for particularly low forcing frequencies (< 1e-10 Hz) with fluid layers, the load distribution factor can 
                        # become negative. This is a known issue and is not physically meaningful. The code will set the load Love number to its analytical
                        # limit in this case, and recalculate the load distribution factor. This is a temporary fix until a more robust solution is implemented.

                        knms_L[i, iseg] = -2/3 * (2-1) * knms_T[i, iseg]        # analytical limit (incompressible homogeneous sphere; n = 2)
                        factor = 1 + (imag(knms_L[i, iseg] * knms_total[i])) / (imag(knms_T[i, iseg]) + imag(knms_total[i]) + 1e-40) 
                    end

                    # rescale heating profile with the load distribution factor
                    prf_total[i, i_start:i_end_ini] .*= factor

                    # rescale global maps with the load distribution factor
                    map_total_μ[i, :, :, i_start:i_end_ini] .*= factor
                    map_total_κ[i, :, :, i_start:i_end_ini] .*= factor
                    map_total_l[i, :, :, i_start:i_end_ini] .*= factor
                    map_total_f[i, :, :, i_start:i_end_ini] .*= factor
                    map_total_d[i, :, :, i_start:i_end_ini] .*= factor

                    # update the total Lovenumber
                    knms_total[i] = knms_T[i, iseg] + (1.0 + knms_L[i, iseg]) * knms_total[i]
                end
            else
                for i in 1:N_σ
                    # update the total Lovenumber
                    knms_total[i] = knms_T[i, iseg] + knms_total[i]
                end
            end
        end

        # extract imaginary part of complex global k2 spectrum
        imag_kn = .-imag.(knms_total)

        # if using full spectrum, then calculate heating profile and bulk heating at each 
        # frequency and return the full spectrum of heating profiles and bulk heating for plotting
        # the code assumes s=1 to find a solution for the Hansen coefficients and normalization.
        if spectrum == "full"
            # get (n, m, k) mode
            n, m, k = nmk[1]

            # Hansen coefficient at s=1
            _, X = Hansen.get_hansen(ecc, n, m, 1, 1)

            A = 2 * sqrt(4π * factorial(n-m) / ((2*n+1) * factorial(n+m))) * Plm(n, m, 0.) * X
                        
            U = (G*S_mass/sma) * (R/sma)^n * A

            prefactor = Float64.((2*n + 1) * R / (8π*G) .* σ_range)

            U2 = Float64.(abs2.(U))

            # return bulk heating at each frequency
            P_T_1_blk = prefactor .* imag_kn .* U2

            # return power profile at each frequency
            P_T_1_prf = zeros(Float64, N_σ, length(r)-1)

            # return global map at each frequency
            P_T_1_glb_μ = map_total_μ .* U2
            P_T_1_glb_κ = map_total_κ .* U2
            P_T_1_glb_l = map_total_l .* U2
            P_T_1_glb_f = map_total_f .* U2
            P_T_1_glb_d = map_total_d .* U2

            ratios = zeros(Float64, N_σ)

            for iss in 1:N_σ
                unorm_prf = prf_total[iss, :]
                
                P_T_1_prf[iss, :] = Float64.(unorm_prf .* U2)

                # integrate the radial profile over the volume for this frequency
                integrated_profile_power = sum(dv .* P_T_1_prf[iss, :])
                
                # calculate ratio
                ratio = P_T_1_blk[iss] / integrated_profile_power
                ratios[iss] = ratio

                # Print formatted results
                @debug("Freq Index: %d | Freq: %.6f | Ratio: %.6f\n", iss, σ_range[iss], ratio)
            end          

            P_T_blk = Float64(sum(P_T_1_blk)) # W

            # get radial heating profile W/m^3
            P_T_prf = Float64.([sum(P_T_1_prf[:,j]) for j in 1:size(P_T_1_prf,2)])

            # determine the total heat input from heating profile
            P_T_prf_blk = Float64(sum(dv .* P_T_prf))

            # integrate spatially if not storing 3D maps
            if !store_3D
                # setup spherical grid used for angular averaging (`res` is defined in `constants.jl`)
                lat_vals = deg2rad.(collect(0:res:180))
                dres   = deg2rad.(res)

                # create a weight array for the latitude dimension (sin(lat))
                weight = reshape(sin.(lat_vals), 1, :, 1, 1)
                
                # perform area-weighted integration
                # the sum collapses dims 2 (lat) and 3 (lon)
                P_T_1_glb_μ = sum(P_T_1_glb_μ .* weight .* dres^2, dims=(2, 3))     # Wm-3
                P_T_1_glb_κ = sum(P_T_1_glb_κ .* weight .* dres^2, dims=(2, 3))     # Wm-3
                P_T_1_glb_l = sum(P_T_1_glb_l .* weight .* dres^2, dims=(2, 3))     # Wm-3
                P_T_1_glb_f = sum(P_T_1_glb_f .* weight .* dres^2, dims=(2, 3))     # Wm-3
                P_T_1_glb_d = sum(P_T_1_glb_d .* weight .* dres^2, dims=(2, 3))     # Wm-3
            end

            # convert to Float32 to save space
            P_T_1_glb_μ = Float32.(P_T_1_glb_μ)
            P_T_1_glb_κ = Float32.(P_T_1_glb_κ)
            P_T_1_glb_l = Float32.(P_T_1_glb_l)
            P_T_1_glb_f = Float32.(P_T_1_glb_f)
            P_T_1_glb_d = Float32.(P_T_1_glb_d)

            # define data file path
            filename = @sprintf("%.0f_obliqua.nc", time)
            datafile_path = joinpath(outpath, filename)
            
            # store results in netcdf file
            data_to_nc(
                    nmk, is_seg, segments, knms_total, knms_T, knms_L, 
                    σ_range, P_T_blk, P_T_prf_blk, P_T_1_prf, Float64.(r),
                    P_T_1_glb_μ, P_T_1_glb_κ, P_T_1_glb_l, P_T_1_glb_f, P_T_1_glb_d, datafile_path
                )

            @info("Expected bulk heating: $P_T_blk")
            @info("Obtained bulk heating: $P_T_prf_blk")

            # convert to mass heating rate (W/kg)
            P_T_prf ./ ρ

            # return full Imk2 spectrum for plotting
            return Float64.(P_T_prf), Float64.(P_T_blk), nmk, Float64.(σ_range), ComplexF64.(knms_total)
        end
            
        # alternatively, if using adaptive spectrum, then calculate heating profile and
        # bulk heating only at the frequencies of interest and return the spectrum of heating 
        # profiles and bulk heating for plotting. Here the s_range is properly chosen to find
        # the true solution for the Hansen coefficients and normalization given the orbit.

        # calculate tidal heating
        # initialize frequency dependent quentities
        A_nms_e   = zeros(Float64, N_σ)
        U_nms_e   = zeros(Float64, N_σ)

        # initialize frequency dependent total heating
        P_T_s_blk = zeros(Float64,  N_σ)

        # initialize frequency dependent heating profile
        P_T_s_prf = zeros(Float64,  N_σ, length(shear))     # W/m^3
        P_T_s_glb_μ = zeros(Float64,  N_σ, length(collect(0:res:180)), length(collect(0:res:360-0.001)), length(shear))     # W
        P_T_s_glb_κ = zeros(Float64,  N_σ, length(collect(0:res:180)), length(collect(0:res:360-0.001)), length(shear))     # W
        P_T_s_glb_l = zeros(Float64,  N_σ, length(collect(0:res:180)), length(collect(0:res:360-0.001)), length(shear))     # W
        P_T_s_glb_f = zeros(Float64,  N_σ, length(collect(0:res:180)), length(collect(0:res:360-0.001)), length(shear))     # W
        P_T_s_glb_d = zeros(Float64,  N_σ, length(collect(0:res:180)), length(collect(0:res:360-0.001)), length(shear))     # W
        # loop over tidal modes 
        for iss in 1:N_σ
            # specify mode
            n_i, m_i, s_i = nmk[iss]

            # calculate physical forcing frequency
            σ = m_i*axial - s_i*omega

            # if forcing frequency is zero, then skip to next frequency (no heating)
            iszero(σ) && continue

            # calculate coefficients
            a = (abs(m_i) == 0 && s_i == 0) ? 1.0 : 0.0
            b = (abs(m_i) == 0 && s_i < 0)  ? 1.0 : 0.0
            
            A_nms_e[iss] = (2. - a) * (1. - b) * sqrt(4π * factorial(n_i-m_i) / ((2*n_i+1) * factorial(n_i+m_i))) * Plm.(n_i, m_i, 0.) * X_hansen[iss]   
            U_nms_e[iss] = Float64.( (G*S_mass/sma) * (R/sma)^n_i * A_nms_e[iss] )

            # get imaginary part of complex k2 love number from global spectrum at forcing frequency
            img_full_knm = imag_kn[iss] 

            # calculate prefactor and total availible heat
            prefactor = Float64.( (2*n_i+1) * R * σ / (8π*G) )
            U2 = abs2(U_nms_e[iss])

            # calculate total heat input at forcing frequency
            P_T_s_blk[iss] = prefactor * img_full_knm  * U2
            
            # Global heating profile & Hansen renorm
            @views begin
                P_T_s_prf[iss, :] .= prf_total[iss, :] .* U2
                P_T_s_glb_μ[iss, :, :, :] .= map_total_μ[iss, :, :, :] .* U2
                P_T_s_glb_κ[iss, :, :, :] .= map_total_κ[iss, :, :, :] .* U2
                P_T_s_glb_l[iss, :, :, :] .= map_total_l[iss, :, :, :] .* U2
                P_T_s_glb_f[iss, :, :, :] .= map_total_f[iss, :, :, :] .* U2
                P_T_s_glb_d[iss, :, :, :] .= map_total_d[iss, :, :, :] .* U2
            end

            # For debugging purposes log the ratio between expected and radially integrated heating
            ratio = P_T_s_blk[iss] / sum(P_T_s_prf[iss, :] .* dv)
            @debug "Forcing Frequency: $σ, Global heating ratio (blk/prf): $ratio"

        end

        # total tidal heating
        P_T_blk = sum(P_T_s_blk) # W

        # get radial heating profile W/m^3
        P_T_prf = [sum(P_T_s_prf[:,j]) for j in 1:size(P_T_s_prf,2)]

        # determine the total heat input from heating profile
        P_T_prf_blk = Float64.(sum(P_T_prf .* dv))

        # integrate spatially if not storing 3D maps
        if !store_3D
            # setup spherical grid used for angular averaging (`res` is defined in `constants.jl`)
            lat_vals = deg2rad.(collect(0:res:180))
            dres   = deg2rad.(res)
            
            # create a weight array for the latitude dimension (sin(lat))
            weight = reshape(sin.(lat_vals), 1, :, 1, 1)
                        
            # perform area-weighted integration
            # the sum collapses dims 2 (lat) and 3 (lon)
            P_T_s_glb_μ = sum(P_T_s_glb_μ .* weight .* dres^2, dims=(2, 3))     # Wm-3
            P_T_s_glb_κ = sum(P_T_s_glb_κ .* weight .* dres^2, dims=(2, 3))     # Wm-3
            P_T_s_glb_l = sum(P_T_s_glb_l .* weight .* dres^2, dims=(2, 3))     # Wm-3
            P_T_s_glb_f = sum(P_T_s_glb_f .* weight .* dres^2, dims=(2, 3))     # Wm-3
            P_T_s_glb_d = sum(P_T_s_glb_d .* weight .* dres^2, dims=(2, 3))     # Wm-3
        end

        # convert to Float32 to save space
        P_T_s_glb_μ = Float32.(P_T_s_glb_μ)
        P_T_s_glb_κ = Float32.(P_T_s_glb_κ)
        P_T_s_glb_l = Float32.(P_T_s_glb_l)
        P_T_s_glb_f = Float32.(P_T_s_glb_f)
        P_T_s_glb_d = Float32.(P_T_s_glb_d)

        # define data file path
        filename = @sprintf("%.0f_obliqua.nc", time)
        datafile_path = joinpath(outpath, filename)

        # store results in netcdf file
        data_to_nc(
                nmk, is_seg, segments, knms_total, knms_T, knms_L, 
                σ_range, P_T_blk, P_T_prf_blk, P_T_s_prf, Float64.(r),
                P_T_s_glb_μ, P_T_s_glb_κ, P_T_s_glb_l, P_T_s_glb_f, P_T_s_glb_d, datafile_path
            )

        @info("Expected bulk heating: $P_T_blk")
        @info("Obtained bulk heating: $P_T_prf_blk")

        # convert to mass heating rate (W/kg)
        P_T_prf ./= ρ

        # convert everything to Float64
        return Float64.(P_T_prf), Float64(P_T_blk), nmk, Float64.(σ_range), ComplexF64.(knms_total)

    end


    """Convert 'true'/'false' string into boolean literal."""
    function true_if_true(val)
        if val == "true" || val == true
            return true
        elseif val == "false" || val == false
            return false
        end
    end


    """Convert 'none' string into nothing literal."""
    function nothing_if_none(val)
        if val == "none"
            return nothing
        else 
            return val 
        end
    end


    """
        get_layers(r, η, η_l, η_s; min_frac=0.02)

    Determine the phase profile of a planetary interior considering solid, fluid, and mush layers.

    # Arguments
    - `r::Array{prec,1}`                : Radial positions of layers, from core to surface.
    - `η::Array{prec,1}`                : Viscosity profile of the planet.
    - `η_l::Float64`                    : Liquidus viscosity.
    - `η_s::Float64`                    : Solidus viscosity.

    # Keyword Arguments
    - `min_frac::Float64=0.02`          : Minimal segment radius fraction before smoothing.

    # Returns
    - `mask_s::Vector{Bool}`            : Solid region mask.
    - `mask_l::Vector{Bool}`            : Fluid region mask.
    - `mask_c::Vector{Bool}`            : Mush region mask.
    - `is_seg::Vector{Tuple{Int,Int}}`  : Segment [start, stop] index array.
    - `segments::Vector{String}`        : Segment phase array.

    # Smoothing behaviour
    Two passes are applied to `phase_prf` before segments are finalised, and
    repeated to a fixed point (segment count only ever decreases, so this always
    terminates):

    1. **Interior islands**: a segment strictly between two neighbours of the
    *same* phase (a "sandwich"), and thinner than `min_frac`, is absorbed into
    that surrounding phase. Phase-agnostic: applies equally to a thin solid/mush
    sliver in fluid, or a thin fluid sliver between two solid/mush segments.
    2. **Edge layers**: a segment touching the CMB (r[1]) or the surface (r[end])
    that is thinner than `min_frac` is absorbed into its single neighbour,
    but *only* if that neighbour is itself "substantial" (thickness/H ≥
    min_frac). This is the CMB-touching residual-melt-layer case: a thin
    fluid sliver sitting directly on the core, next to a thick solid/mush
    layer, gets folded into the solid/mush segment once the latter is no
    longer negligible. If the neighbour is also thin (e.g. early in
    crystallisation, before the solid/mush layer has grown past threshold),
    no merge happens yet — pass 1 handles that regime instead.
    """
    function get_layers(r::Array{prec,1},
                        η::Array{prec,1},
                        η_l::Float64,
                        η_s::Float64;
                        min_frac::Float64=0.02
                        )::Tuple{Vector{Bool},Vector{Bool},Vector{Bool},Vector{Tuple{Int,Int}},Vector{String}}

        # masks for liquid and solid regions
        mask_l = η .< η_l
        mask_s = η_s .< η

        # total mantle thickness
        H = r[end] - r[1]
        N = length(r) - 1 # subtract CMB layer

        # phase encoding:
        #   0 = mush / other
        #   1 = solid
        #   2 = liquid
        phase_prf = zeros(Int, N)

        # build phase profile
        for i in 1:N
            if mask_s[i]
                phase_prf[i] = 1
            elseif mask_l[i]
                phase_prf[i] = 2
            end
        end

        # ---- helper: contiguous same-phase runs of phase_prf[1:N] ----
        # NOTE: this mirrors the segment-building loop further below (i <= N,
        # no extra increment after the inner while). The original smoothing loop
        # used `i < N` plus a stray `i += 1` after the inner while, which
        # (a) never visits index N as a segment start, and (b) skips the first
        # index of every segment after the first when reading it back for a
        # neighbour-phase comparison. Both are fixed here.
        function segment_bounds(pf::Vector{Int})
            segs = Vector{Tuple{Int,Int}}()
            j = 1
            while j <= N
                pj = pf[j]
                j_start = j
                while j <= N && pf[j] == pj
                    j += 1
                end
                push!(segs, (j_start, j - 1))
            end
            return segs
        end

        # ---- smooth phase profile, to a fixed point ----
        # Bounded above by N: each sweep that changes anything strictly reduces
        # the segment count, which is bounded below by 1, so this cannot loop
        # more than (initial segment count - 1) times. N is a safe, non-arbitrary
        # cap; if it's ever hit without convergence something upstream broke the
        # monotonic-decrease assumption, so warn rather than fail silently.
        converged = false
        for _ in 1:N
            changed = false

            # --- Pass 1: interior islands (sandwich) ---
            segs = segment_bounds(phase_prf)
            for idx in 1:length(segs)
                i_start, i_end = segs[idx]
                if idx > 1 && idx < length(segs)
                    p = phase_prf[i_start]
                    p_below = phase_prf[segs[idx-1][1]]
                    p_above = phase_prf[segs[idx+1][1]]
                    if p_below == p_above && p_below != p
                        dr = r[i_end+1] - r[i_start]
                        if dr / H < min_frac
                            phase_prf[i_start:i_end] .= p_below
                            changed = true
                        end
                    end
                end
            end

            # --- Pass 2: edge layers (CMB and surface) ---
            segs = segment_bounds(phase_prf)
            if length(segs) > 1
                # bottom (CMB-touching) segment
                i_start, i_end = segs[1]
                i_start2, i_end2 = segs[2]
                p = phase_prf[i_start]
                p_nb = phase_prf[i_start2]
                dr = r[i_end+1] - r[i_start]
                dr_nb = r[i_end2+1] - r[i_start2]
                if p != p_nb && dr / H < min_frac && dr_nb / H >= min_frac
                    phase_prf[i_start:i_end] .= p_nb
                    changed = true
                end

                # re-fetch: the bottom merge above may have changed segment count
                segs = segment_bounds(phase_prf)
                if length(segs) > 1
                    i_start, i_end = segs[end]
                    i_start2, i_end2 = segs[end-1]
                    p = phase_prf[i_start]
                    p_nb = phase_prf[i_start2]
                    dr = r[i_end+1] - r[i_start]
                    dr_nb = r[i_end2+1] - r[i_start2]
                    if p != p_nb && dr / H < min_frac && dr_nb / H >= min_frac
                        phase_prf[i_start:i_end] .= p_nb
                        changed = true
                    end
                end
            end

            if !changed
                converged = true
                break
            end
        end
        converged || @warn "get_layers: phase-profile smoothing did not converge within N sweeps; check for an unexpected oscillation."

        # update masks
        mask_s .= phase_prf .== 1
        mask_l .= phase_prf .== 2

        # build segments + bottom mask
        segments = String[]
        mask_c = falses(length(r))

        # build boundary indices array
        is_seg = Vector{Tuple{Int,Int}}()

        i = 1
        while i <= N
            p = phase_prf[i]

            # mark bottom of this
            if i > 1
                mask_c[i-1] = true # "bottom" radius is r[i-1]
            end

            i_start = i

            # step forward through this segment
            while i <= N && phase_prf[i] == p
                i += 1
            end

            i_end = i - 1

            # store boundary indices
            push!(is_seg, (i_start, i_end))

            # store segments
            push!(segments,
                p == 1 ? "solid" :
                p == 2 ? "fluid" :
                        "mush"
            )
        end

        return mask_s, mask_l, mask_c, is_seg, segments
    end


    """
        complex_modulus(σ_range, x_profile, η_profile; material="andrade", α=0.3)

    Return the complex shear modulus μ̃(σ) and complex bulk modulus κ̃(σ) for Elastic, Maxwell, or Andrade rheology.

    # Arguments
    - `σ_range::AbstractVector`         : Forcing frequency range.
    - `x_profile::Array{precc,1}`       : Shear or Bulk profile of the planet (aka unrelaxed rigidity, unrelaxed bulk).
    - `η_profile::Array{prec,1}`        : Viscosity profile of the planet.
    
    # Keyword Arguments
    - `material::String="andrade"`      : Material for which to find complex shear modulus.
    - `α::Float64=0.3"`                 : Power-law exponent (free parameter).

    # Returns
    - `xc::Matrix{precc}`               : Complex shear modulus profile at all forcing frequencies.
    """
    function complex_modulus(σ_range::AbstractVector,
                        x_profile::Array{precc,1},
                        η_profile::Array{prec,1};
                        material::String="andrade", 
                        α::Float64=0.3
                        )::Matrix{precc}

        nlayer = length(x_profile)
        nfreq  = length(σ_range)

        # initialize output array
        xc = zeros(precc, nlayer, nfreq)

        if material == "maxwell"
            @inbounds for i in 1:nlayer
                x = x_profile[i]
                if iszero(x)
                    for j in 1:nfreq
                        xc[i,j] = zero(precc)
                    end
                    continue
                end

                η = η_profile[i]
                μ_over_η = x / η
                for j in 1:nfreq
                    σ = σ_range[j]
                    im_σ = 1im * σ
                    xc[i,j] = im_σ * x / (im_σ + μ_over_η)
                end
            end
        elseif material == "andrade"
            gamma_factor = gamma(1 + α)
            
            @inbounds for i in 1:nlayer
                x = x_profile[i]
                if iszero(x)
                    for j in 1:nfreq
                        xc[i,j] = zero(precc)
                    end
                    continue
                end

                η = η_profile[i]
                τM = η / x  # Maxwell time
                
                for j in 1:nfreq
                    σ = σ_range[j]
                    
                    if iszero(σ)
                        xc[i,j] = zero(precc)
                    else
                        im_σ_τM = 1im * σ * τM
                        
                        # Rewritten fraction to avoid division by zero
                        denom = 1 + im_σ_τM + gamma_factor * (im_σ_τM)^(1 - α)
                        xc[i,j] = x * im_σ_τM / denom
                    end
                end
            end
        
        elseif material == "elastic" || material == "none"
            @inbounds for i in 1:nlayer
                x = x_profile[i]
                for j in 1:nfreq
                    xc[i,j] = x
                end
            end

        else
            throw(ArgumentError("Material type for complex shear modulus not defined; options: 'maxwell' or 'andrade'"))
        end

        return xc
    end


    """
        smooth_complex_modulus!(μc, κc, r, η)

    Return the smoothed complex shear modulus μ̃(σ) and complex bulk modulus κ̃(σ).

    # Arguments
    - `μc::Matrix{precc}`               : Complex shear modulus profile at all forcing frequencies.
    - `κc::Matrix{precc}`               : Complex bulk modulus profile at all forcing frequencies.
    - `r::Array{prec,1}`                : Radial positions of layers, from core to surface.
    - `η_profile::Array{prec,1}`        : Viscosity profile of the planet.
    
    # Returns
    - `μc::Matrix{precc}`               : Smoothed complex shear modulus profile at all forcing frequencies.
    - `κc::Matrix{precc}`               : Smoothed complex bulk modulus profile at all forcing frequencies.  
    """
    function smooth_complex_modulus!(μc::Matrix{precc}, κc::Matrix{precc}, r::Array{prec,1}, η::Array{prec,1})
        # assumes r is a 1D Vector of cell centers or boundaries corresponding to the layers.

        @info "Smoothing complex modulus profiles to avoid sharp jumps in viscosity..."

        # compute the log-viscosity gradient between adjacent layers
        log_η = log10.(η)
        dlog_η = diff(log_η)

        # identify where the properties jump too aggressively (> 3 orders of magnitude)
        jump_indices = findall(.-dlog_η .> 3.0)

        if isempty(jump_indices)
            @info "No problematic layers found. No smoothing applied."
            return μc, κc 
        end

        # define how many layers outside the jump to include in the smoothing blend
        pad = 2 
        
        # find the total index bounds of the problematic region
        idx_start = max(1, minimum(jump_indices) - pad)
        idx_end   = min(length(η), maximum(jump_indices) + 1 + pad)

        # ensure we have a valid window to blend between
        if idx_start >= idx_end
            @info "Smoothing window is invalid. No smoothing applied."
            return μc, κc
        end

        # extract bounding vectors across all frequencies
        μ_low, μ_high = μc[idx_start, :], μc[idx_end, :]
        κ_low, κ_high = κc[idx_start, :], κc[idx_end, :]

        @info "Smoothing between indices $idx_start and $idx_end (r = $(round(r[idx_start], digits=2)) to $(round(r[idx_end], digits=2)) m)"

        # pre-calculate natural logs element-by-element
        log_μ_low  = log.(μ_low)
        log_μ_high = log.(μ_high)
        log_κ_low  = log.(κ_low)
        log_κ_high = log.(κ_high)

        # create a smooth log-linear interpolation across all columns
        for i in idx_start:idx_end
            dr = (r[i] - r[idx_start]) / (r[idx_end] - r[idx_start])  # normalized weight (0.0 to 1.0)
            
            # linearly blend in log-space, then exponentiate back
            μc[i, :] .= exp.((1.0 - dr) .* log_μ_low .+ dr .* log_μ_high)
            κc[i, :] .= exp.((1.0 - dr) .* log_κ_low .+ dr .* log_κ_high)
        end

        return μc, κc
    end
    

    """
        enforce_energy_conservation!(prf_slice, knms_T_slice, ω, R, dv_slice, nmk, map_μ_slice, map_κ_slice, map_l_slice, map_f_slice, map_d_slice)

    Enforce energy conservation by scaling the heating profile and global maps to match the expected bulk heating from the k Lovenumber.

    # Arguments
    - `prf_slice::Array{Float64,2}`         : Heating profile slice for the current segment.
    - `knms_T_slice::Array{ComplexF64,1}`   : Complex Tidal k2 Lovenumber slice for the current segment.
    - `ω::Array{Float64,1}`                 : Forcing frequency range.
    - `R::Float64`                          : Planet radius.
    - `dv_slice::Array{Float64,1}`          : Volume elements for the current segment.
    - `nmk::Vector{Tuple{Int, Int, Int}}`   : Vector of (n, m, k) tuples for each harmonnic mode.
    - `map_μ_slice::Array{Float64,4}`       : Global map slice for shear heating.
    - `map_κ_slice::Array{Float64,4}`       : Global map slice for bulk heating.
    - `map_l_slice::Array{Float64,4}`       : Global map slice for darcy heating.
    - `map_f_slice::Array{Float64,4}`       : Global map slice for frictional heating.
    - `map_d_slice::Array{Float64,4}`       : Global map slice for drag heating.
    """
    function enforce_energy_conservation!(prf_slice::SubArray{Float64,2}, knms_T_slice::SubArray{ComplexF64,1}, ω::Array{Float64,1}, R::Float64, dv_slice::Array{Float64,1}, nmk::Vector{Tuple{Int,Int,Int}}, map_μ_slice::SubArray{Float64,4}, map_κ_slice::SubArray{Float64,4}, map_l_slice::SubArray{Float64,4}, map_f_slice::SubArray{Float64,4}, map_d_slice::SubArray{Float64,4})
        # extract the first element (n_i) from each tuple in the nmk vector
        n_elements = getindex.(nmk, 1)

        # compute the prefactor
        prefactor = (2 .* reshape(n_elements, :, 1) .+ 1) .* R ./ (8π * G) .* ω
        
        # knms_T_slice is a 1D vector across frequencies
        norm_blk = - prefactor .* imag.(knms_T_slice)

        # compute the sum of the heating profile slice weighted by the volume elements
        slice_sum = sum(prf_slice .* dv_slice'; dims=2)
        
        # calculate ratio vector for all frequencies
        ratio = norm_blk ./ slice_sum
        ratio_4d = reshape(ratio, :, 1, 1, 1)

        # apply updates directly to the views
        prf_slice    .*= ratio
        map_μ_slice  .*= ratio_4d
        map_κ_slice  .*= ratio_4d
        map_l_slice  .*= ratio_4d
        map_f_slice  .*= ratio_4d
        map_d_slice  .*= ratio_4d
    end


    """
        run_solid0d(μc, radius, mass_tot; n=2)

    Calculate kn Lovenumbers in the 0D solid.

    # Arguments
    - `μc::Array{precc,1}`              : Forcing frequency range.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `mass_tot::Float64`               : Total mass of planet.
    
    # Keyword Arguments
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).

    # Returns
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_solid0d( μc::Array{precc,1},
                        radius::Array{prec,1},
                        mass_tot::prec;
                        n::Int64=2
                        )::Tuple{ComplexF64,ComplexF64}

        # internal structure arrays
        μc = convert(Vector{precc}, μc)
        r  = convert(Vector{prec}, radius)

        # get mean complex shear modulus in segment
        μc_mean = solid0d.mean_cmu(μc, r)

        # surface properties
        R = maximum(r)  # Segment outer radius (m)

        # get kn Lovenumbers
        kn_T, kn_L = solid0d.compute_solid_lovenumbers(μc_mean, mass_tot, R, n)

        return ComplexF64(kn_T), ComplexF64(kn_L)

    end


    """
        run_solid1d(omega, rho, radius, visc, shear, bulk, R, m_core, ρ_core, μ_core, κ_core; ncalc=2000, n=2, m=2, core="liquid", inertial_terms=false, optimize_scales=false)

    Use 1D solid tides model to calculate kn Lovenumbers, and compute 1D heating profile from strain tensor.
    This method ignores inertia effects, since they break the numerical stability.

    # Arguments
    - `omega::prec`                     : Forcing frequency range.
    - `rho::Array{prec,1}`              : Density profile of the planet.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `visc::Array{prec,1}`             : Viscosity profile of the planet.
    - `μ_profile::Array{precc,1}`       : Complex shear modulus profile of the planet.
    - `bulk::Array{precc,1}`            : Complex bulk modulus profile of the planet.
    - `R::prec`                         : Planet radius.
    - `m_core::prec`                    : Core mass.
    - `ρ_core::prec`                    : Core density.
    - `μ_core::precc`                   : Core shear modulus.
    - `κ_core::precc`                   : Core bulk modulus.
    
    # Keyword Arguments
    - `ncalc::Int=2000`                 : Number of sublayers.
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `m::Int=2`                        : Harmonic of the true anomaly. m=2 corresponds to the semidiurnal tide, m=1 diurnal tide.
    - `core::String="liquid"`           : Core state, either "liquid" or "solid".
    - `inertial_terms::Bool=false`      : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.
    - `optimize_scales::Bool=false`     : Whether to optimize non-dimensionalization scales.

    # Returns
    - `power_prf::Array{Float64,1}`     : Heating profile.
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_solid1d( omega::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        visc::Array{prec,1},
                        shear::Array{precc,1},
                        bulk::Array{precc,1},
                        R::prec,
                        m_core::prec,
                        ρ_core::prec,
                        μ_core::precc,
                        κ_core::precc;
                        ncalc::Int=2000,
                        n::Int=2,
                        m::Int=2,
                        core::String="liquid",
                        inertial_terms::Bool=false,
                        optimize_scales::Bool=false
                        )::Tuple{Array{Float64,1},ComplexF64,ComplexF64}

        # internal structure arrays.
        # first element is the innermost layer, last element is the outermost layer
        omega = prec(omega)
        ρ = convert(Vector{prec}, rho)
        r = convert(Vector{prec}, radius)
        η = convert(Vector{prec}, visc)
        μc = convert(Vector{precc},shear)
        κc = convert(Vector{precc},bulk)

        # subdivide input layers such that we have ~ncalc in total
        rr = solid1d.expand_layers(r, nr=convert(Int,div(ncalc,length(η))))

        # check for zero values in the expanded radius array
        if iszero.(rr) |> any
            throw(ArgumentError("Expanded radius array contains zero values. Check input radius and ncalc."))
        end

        # get gravity at each layer
        g = solid1d.get_g(rr, ρ, m_core)

        # compute mass of each layer and total mass of planet
        dm    = m_core + sum(4/3 * π * ρ .* diff(r.^3))
        M_enc = cumsum(dm) .+ m_core    # cumulative mass of planet
        M_tot = M_enc[end]              # total mass of planet

        # get non-dimensionalization scales
        if optimize_scales
            scales = solid1d.common.optimize_scales(r[2:end], ρ, g, μc, κc, ω, n, [r[end], M_tot, G])
        else
            scales = [r[end], M_tot, G] # default scales if not optimizing
        end

        # create grid
        solid1d.define_spherical_grid(res, n, m)

        # get y-functions
        M, y1_4, S, scale = solid1d.compute_M(omega, rr, ρ, g, μc, κc, n, ρ_core, μ_core, κ_core, scales; core=core, inertial_terms=inertial_terms)
        # Tidal
        tidal_solution_T = solid1d.compute_y(rr, g, M, y1_4, n, S, scale; load=false)
        # Load
        tidal_solution_L = solid1d.compute_y(rr, g, M, y1_4, n, S, scale; load=true)
        
        # get kn tidal Love Number (complex-valued)
        kn_T = tidal_solution_T[5, end, end] - 1
        kn_L = tidal_solution_L[5, end, end] - 1
        
        # Get profile power output (W m-3), converted to W/kg
        Eμ, Eκ = solid1d.get_heating_profile(tidal_solution_T, rr, ρ, g, μc, κc, n, omega; lay=nothing)

        Eμ_tot, _ = Eμ   # shear       (W), (W/m3)
        Eκ_tot, _ = Eκ   # compaction  (W), (W/m3)

        # Renormalization factor
        power_prf = (Eμ_tot .+ Eκ_tot) .* (R ./ maximum(r)) # Compute total volumetric heating (W/m3)

        return Float64.(power_prf), ComplexF64(kn_T), ComplexF64(kn_L)
    end
    

    """
        run_solid1d_relax(omega, rho, radius, gravity, visc, shear, bulk, R, m_core, ρ_core, μ_core, κ_core; dr_min=300, dr_max=3000, n=2, m=2, core="liquid", inertial_terms=true, optimize_scales=false, patch=false)

    Use 1D solid tides model with relaxation method to calculate kn Lovenumbers, and compute 1D heating profile from strain tensor.
    This method includes inertia effects, but is more computationally expensive. 
    
    # Arguments
    - `omega::prec`                     : Forcing frequency range.
    - `rho::Array{prec,1}`              : Density profile of the planet.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `gravity::Array{prec,1}`          : Gravity profile of the planet.
    - `visc::Array{prec,1}`             : Viscosity profile of the planet.
    - `shear::Array{precc,1}`           : Complex shear modulus profile of the planet.
    - `bulk::Array{precc,1}`            : Complex bulk modulus profile of the planet.
    - `R::prec`                         : Planet radius.
    - `m_core::prec`                    : Core mass.
    - `ρ_core::prec`                    : Core density.
    - `μ_core::precc`                   : Core shear modulus.
    - `κ_core::precc`                   : Core bulk modulus.

    # Keyword Arguments
    - `dr_min::Int=300`                 : Minimum layer thickness in m.
    - `dr_max::Int=3000`                : Maximum layer thickness in m.
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `m::Int=2`                        : Harmonic of the true anomaly. m=2 corresponds to the semidiurnal tide, m=1 diurnal tide.
    - `core::String="liquid"`           : Core state, either "liquid", "solid", or "inertial".
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `optimize_scales::Bool=false`     : Whether to optimize non-dimensionalization scales.
    - `patch::Bool=false`               : Whether to insert an infinitesimal solid shell around the core. This patches an issue where y2 and y4 become decoupled and cause the solution to diverge in fluid layers.

    # Returns
    - `power_prf::Array{Float64,1}`     : Heating profile.
    - `Eμ_glb_itp::Array{prec,4}`       : Heating map (colatitude, longitude, radius).
    - `Eκ_glb_itp::Array{prec,4}`       : Heating map (colatitude, longitude, radius).
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_solid1d_relax( omega::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        gravity::Array{prec,1},
                        visc::Array{prec,1},
                        shear::Array{precc,1},
                        bulk::Array{precc,1},
                        R::prec,
                        m_core::prec,
                        ρ_core::prec,
                        μ_core::precc,
                        κ_core::precc;
                        dr_min::Int=300,
                        dr_max::Int=3000,
                        n::Int=2,
                        m::Int=2,
                        core::String="liquid",
                        inertial_terms::Bool=true,
                        optimize_scales::Bool=false,
                        patch::Bool=false
                        )::Tuple{Array{Float64,1},Array{Float64, 3},Array{Float64, 3},ComplexF64,ComplexF64}

        # convert inputs
        ω = prec(omega)
        ρ = convert(Vector{prec}, rho)
        r = convert(Vector{prec}, radius)
        g = convert(Vector{prec}, gravity)
        η = convert(Vector{prec}, visc)
        μc = convert(Vector{precc}, shear)
        κc = convert(Vector{precc}, bulk)

        # resample profiles onto new grid
        r_grid, ρ, η, μc, κc, g_grid, M_tot = solid1d_relax.resample_profiles(r, ρ, η, μc, κc, m_core, dr_min, dr_max)

        # use cell centers
        r_centers = 0.5 .* (r_grid[1:end-1] .+ r_grid[2:end])

        # check for zero values in the expanded radius array
        if iszero.(r_centers) |> any
            throw(ArgumentError("Expanded radius array contains zero values. Check input radius and ncalc."))
        end

        # get non-dimensionalization scales
        if optimize_scales
            scales = solid1d_relax.common.optimize_scales(r[2:end], ρ, g, μc, κc, ω, n, [r[end], M_tot, G])
        else
            scales = [r[end], M_tot, G] # default scales if not optimizing
        end

        # define angular grid
        SphericalGrid = solid1d_relax.define_spherical_grid(res, n, m)

        # solve y functions across grid
        y_t, y_l = solid1d_relax.compute_y(r_centers, ρ, g_grid, μc, κc, ω, n, ρ_core, μ_core, κ_core, scales; core=core, inertial_terms=inertial_terms, patch=patch)

        # Love numbers
        kn_T = y_t[5, end] - 1
        kn_L = y_l[5, 1]   - 1

        # heating profile
        Eμ_tot, Eκ_tot = solid1d_relax.get_heating_profile(
            y_t, r_grid, ρ, g_grid, μc, κc, n, ω, SphericalGrid
        )

        Eμ_glb, Eκ_glb = solid1d_mush_relax.get_heating_map(
            y_t, r_grid, ρ, g_grid, μc, κc, n, ω, SphericalGrid
        )

        power_prf = abs.(Eμ_tot .+ Eκ_tot) 

        # interpolate from grid back to original radius points 
        itp = linear_interpolation(r_centers, power_prf, extrapolation_bc=Line())
        
        # original centers
        r_orig_centers = 0.5 .* (r[1:end-1] .+ r[2:end])
        scale_factor = R / maximum(r)

        power_prf = Float64.(itp.(r_orig_centers) .* scale_factor)

        # radial interpolation for global maps
        nlats = length(Eμ_glb[:, 1, 1])
        nlons = length(Eμ_glb[1, :, 1])
        Nr_orig = length(r_orig_centers)
        
        # pre-allocate separate 3D grids for each source on the original spacing
        Eμ_glb_itp = zeros(Float64, nlats, nlons, Nr_orig)
        Eκ_glb_itp = zeros(Float64, nlats, nlons, Nr_orig)

        # loop over geographic coordinates, treating each lat/lon as a 1D radial column
        for lon_idx in 1:nlons
            for lat_idx in 1:nlats
                # shear component interpolation
                shear_col = @view Eμ_glb[lat_idx, lon_idx, :]
                itp_shear = linear_interpolation(r_centers, shear_col, extrapolation_bc=Line())
                Eμ_glb_itp[lat_idx, lon_idx, :] .= Float64.(itp_shear.(r_orig_centers) .* scale_factor)

                # compaction component interpolation
                comp_col = @view Eκ_glb[lat_idx, lon_idx, :]
                itp_comp = linear_interpolation(r_centers, comp_col, extrapolation_bc=Line())
                Eκ_glb_itp[lat_idx, lon_idx, :] .= Float64.(itp_comp.(r_orig_centers) .* scale_factor)
            end
        end

        return power_prf, Eμ_glb_itp, Eκ_glb_itp, kn_T, kn_L
    end
    

    """
        run_solid1d_mush(omega, rho, radius, visc, shear, bulk, bulkd, phi, alpha, perm, R, ρ_core, μ_core, κ_core; ncalc=2000, n=2, m=2, core="liquid", inertial_terms=false, visc_l=1e2, bulk_l=1e9, permea=1e-7, porosity_thresh=1e-5, optimize_scales=false)

    Use 1D solid tides model with mush interface to calculate kn Lovenumbers, and compute 1D heating profile from strain tensor.

    # Arguments
    - `omega::Float64`                  : Forcing frequency range.
    - `rho::Array{prec,1}`              : Density profile of the planet.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `visc::Array{prec,1}`             : Viscosity profile of the planet.
    - `shear::Array{precc,1}`           : Complex shear modulus profile of the planet.
    - `bulk::Array{precc,1}`            : Complex bulk modulus profile of the planet.
    - `bulkd::Array{precc,1}`           : Complex drained bulk modulus profile of the planet.
    - `phi::Array{prec,1}`              : Melt fraction (porosity) profile of the planet.
    - `alpha::Array{precc,1}`           : Biot's modulus profile of the planet.
    - `perm::Array{prec,1}`             : Permeability profile of the planet.
    - `R::prec`                         : Planet radius.
    - `m_core::prec`                    : Core mass.
    - `ρ_core::prec`                    : Core density.
    - `μ_core::precc`                   : Core shear modulus.
    - `κ_core::precc`                   : Core bulk modulus.

    # Keyword Arguments
    - `ncalc::Int=2000`                 : Number of sublayers.
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `m::Int=2`                        : Harmonic of the true anomaly. m=2 corresponds to the semidiurnal tide, m=1 diurnal tide.
    - `core::String="liquid"`           : Core state, either "liquid" or "solid".
    - `inertial_terms::Bool=false`      : Whether to include inertial terms in the motion matrix. This is not recommended for the shooting method, as it can break numerical stability.
    - `visc_l::Float64=1e2`             : Liquid viscosity.
    - `bulk_l::Float64=1e9`             : Liquid bulk modulus.
    - `porosity_thresh::Float64=1e-5`   : Porosity threshold, below this value no mush.
    - `optimize_scales::Bool=false`     : Whether to optimize non-dimensionalization scales.

    # Returns
    - `power_prf::Array{Float64,1}`     : Heating profile.
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_solid1d_mush( omega::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        visc::Array{prec,1},
                        shear::Array{precc,1},
                        bulk::Array{precc,1},
                        bulkd::Array{precc,1},
                        phi::Array{prec,1},
                        alpha::Array{precc,1},
                        perm::Array{prec,1},
                        R::prec,
                        m_core::prec,
                        ρ_core::prec,
                        μ_core::precc,
                        κ_core::precc;
                        ncalc::Int=2000,
                        n::Int=2,
                        m::Int=2,
                        core::String="liquid",
                        visc_l::Float64=1e2,
                        bulk_l::Float64=1e9,
                        porosity_thresh::Float64=1e-5,
                        inertial_terms::Bool=false,
                        optimize_scales::Bool=false
                        )::Tuple{Array{Float64,1},ComplexF64,ComplexF64}

        # internal structure arrays.
        # first element is the innermost layer, last element is the outermost layer
        omega = prec(omega)
        ρ  = copy(convert(Vector{prec}, rho))
        r  = copy(convert(Vector{prec}, radius))
        η  = copy(convert(Vector{prec}, visc))
        μc = copy(convert(Vector{precc}, shear))
        κs = copy(convert(Vector{precc}, bulk))
        κd = copy(convert(Vector{precc}, bulkd))
        ϕ  = copy(convert(Vector{prec}, phi))
        α  = copy(convert(Vector{precc}, alpha))
        k  = copy(convert(Vector{prec}, perm))

        # allocate zero arrays with same length and precision as r
        κl = zeros(prec, length(r))
        ηl = zeros(prec, length(r))

        # implicitely the mush interface occurs at the top of the solid
        # the mush layer index is therefore
        ii = length(ϕ)

        # update the liquid arrays
        κl[ii] = prec(bulk_l)   # liquid bulk modulus
        ηl[ii] = prec(visc_l)   # liquid viscosity

        # set porosity to zero outside mush region (otherwise code cannot solve system)
        ϕ[1:ii-1]   .= 0.0      # zero below ii
        # limit the mush layer melt fraction to ϕ=<0.1, as the model becomes unstable otherwise
        ϕ[ii] = min(ϕ[ii], 0.3)

        ρs = ρ.*(1.0.-ϕ)        # solid density 
        ρl = ρ.*ϕ               # liquid density

        # subdivide input layers such that we have ~ncalc in total
        rr = solid1d_mush.expand_layers(r, nr=convert(Int,div(ncalc,length(η))))

        # check for zero values in the expanded radius array
        if iszero.(rr) |> any
            throw(ArgumentError("Expanded radius array contains zero values. Check input radius and ncalc."))
        end

        # get gravity at each layer
        g = solid1d_mush.get_g(rr, ρ, m_core)

        # compute mass of each layer and total mass of planet
        dm    = m_core + sum(4/3 * π * ρ .* diff(r.^3))
        M_enc = cumsum(dm) .+ m_core    # cumulative mass of planet
        M_tot = M_enc[end]              # total mass of planet

        # get non-dimensionalization scales
        if optimize_scales
            scales = solid1d_mush.common.optimize_scales(r[2:end], ρ, g, μc, κc, ω, n, [r[end], M_tot, G])
        else
            scales = [r[end], M_tot, G] # default scales if not optimizing
        end

        # create grid
        solid1d_mush.define_spherical_grid(res, n, m)

        # get y-functions
        M, y1_4, S, scale = solid1d_mush.compute_M(omega, rr, ρs, g, μc, κs, ρl, κl, κd, α, ηl, ϕ, k, n, ρ_core, μ_core, κ_core, scales; core=core, inertial_terms=inertial_terms)
        #   Tidal
        tidal_solution_T = solid1d_mush.compute_y(rr, g, M, y1_4, n, S, scale; load=false)
        #   Load
        tidal_solution_L = solid1d_mush.compute_y(rr, g, M, y1_4, n, S, scale; load=true)

        # get kn tidal Love Number (complex-valued)
        kn_T = tidal_solution_T[5, end, end] - 1
        kn_L = tidal_solution_L[5, end, end] - 1

        # Get profile power output (W m-3), converted to W/kg
        Eμ, Eκ, El = solid1d_mush.get_heating_profile(tidal_solution_T,
                               rr, ρs, g, μc, κs,
                               omega, ρl, κl, κd, 
                               α, ηl, ϕ, k, n)

        Eμ_tot, _ = Eμ   # shear       (W), (W/m3)
        Eκ_tot, _ = Eκ   # compaction  (W), (W/m3)
        El_tot, _ = El   # fluid       (W), (W/m3)

        # Renormalization factor
        power_prf = (Eμ_tot .+ Eκ_tot .+ El_tot) .* (R ./ maximum(r)) # Compute total volumetric heating (W/m3)

        return Float64.(power_prf), ComplexF64(kn_T), ComplexF64(kn_L)
    end


    """
        run_solid1d_mush_relax(omega, rho, radius, visc, shear, bulk, bulkd, phi, alpha, perm, R, m_core, ρ_core, μ_core, κ_core; dr_min=300, dr_max=3000, n=2, m=2, core="liquid", inertial_terms=true, visc_l=1e2, bulk_l=1e9, porosity_thresh=1e-5, optimize_scales=false, patch=false)

    Use 1D solid tides model with relaxation method to calculate kn Lovenumbers, and compute 1D heating profile from strain tensor.
    This method includes inertia effects, but is more computationally expensive. 
    
    # Arguments
    - `omega::prec`                     : Forcing frequency range.
    - `rho::Array{prec,1}`              : Density profile of the planet.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `gravity::Array{prec,1}`          : Gravity profile of the planet.
    - `visc::Array{prec,1}`             : Viscosity profile of the planet.
    - `shear::Array{precc,1}`           : Complex shear modulus profile of the planet.
    - `bulk::Array{precc,1}`            : Complex bulk modulus profile of the planet.
    - `bulkd::Array{precc,1}`           : Complex drained bulk modulus profile of the planet.
    - `phi::Array{prec,1}`              : Melt fraction (porosity) profile of the planet.
    - `alpha::Array{precc,1}`           : Biot's modulus profile of the planet.
    - `perm::Array{prec,1}`             : Permeability profile of the planet.
    - `R::prec`                         : Planet radius.
    - `m_core::prec`                    : Core mass.
    - `ρ_core::prec`                    : Core density.
    - `μ_core::precc`                   : Core shear modulus.
    - `κ_core::precc`                   : Core bulk modulus.

    # Keyword Arguments
    - `dr_min::Int=300`                 : Minimum layer thickness in m.
    - `dr_max::Int=3000`                : Maximum layer thickness in m.
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `m::Int=2`                        : Harmonic of the true anomaly. m=2 corresponds to the semidiurnal tide, m=1 diurnal tide.
    - `core::String="liquid"`           : Core state, either "liquid", "solid", or "inertial".
    - `inertial_terms::Bool=true`       : Whether to include inertial terms in the motion matrix.
    - `visc_l::Float64=1e2`             : Liquid viscosity.
    - `bulk_l::Float64=1e9`             : Liquid bulk modulus.
    - `porosity_thresh::Float64=1e-5`   : Porosity threshold, below this value no mush.
    - `optimize_scales::Bool=false`     : Whether to optimize non-dimensionalization scales.
    - `patch::Bool=false`               : Whether to insert an infinitesimal solid shell around the core. This patches an issue where y2 and y4 become decoupled and cause the solution to diverge in fluid layers.

    # Returns
    - `power_prf::Array{prec,1}`        : Heating profile.
    - `Eμ_glb_itp::Array{prec,4}`       : Heating map (colatitude, longitude, radius).
    - `Eκ_glb_itp::Array{prec,4}`       : Heating map (colatitude, longitude, radius).
    - `El_glb_itp::Array{prec,4}`       : Heating map (colatitude, longitude, radius).
    - `kn_T::precc`                     : Complex Tidal kn Lovenumber.
    - `kn_L::precc`                     : Complex Load kn Lovenumber.
    """
    function run_solid1d_mush_relax( omega::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        gravity::Array{prec,1},
                        visc::Array{prec,1},
                        shear::Array{precc,1},
                        bulk::Array{precc,1},
                        bulkd::Array{precc,1},
                        phi::Array{prec,1},
                        alpha::Array{precc,1},
                        perm::Array{prec,1},
                        R::prec,
                        m_core::prec,
                        ρ_core::prec,
                        μ_core::precc,
                        κ_core::precc;
                        dr_min::Int=300,
                        dr_max::Int=3000,
                        n::Int=2,
                        m::Int=2,
                        core::String="liquid",
                        inertial_terms::Bool=true,
                        visc_l::Float64=1e2,
                        bulk_l::Float64=1e9,
                        porosity_thresh::Float64=1e-5,
                        optimize_scales::Bool=false,
                        patch::Bool=false
                        )::Tuple{Array{Float64,1},Array{Float64, 3},Array{Float64, 3},Array{Float64, 3},ComplexF64,ComplexF64}

        # internal structure arrays.
        # first element is the innermost layer, last element is the outermost layer
        ω  = prec(omega)
        ρ  = copy(convert(Vector{prec}, rho))
        r  = copy(convert(Vector{prec}, radius))
        g  = convert(Vector{prec}, gravity)
        η  = copy(convert(Vector{prec}, visc))
        μc = copy(convert(Vector{precc}, shear))
        κs = copy(convert(Vector{precc}, bulk))
        κd = copy(convert(Vector{precc}, bulkd))
        ϕ  = copy(convert(Vector{prec}, phi))
        α  = copy(convert(Vector{precc}, alpha))
        k  = copy(convert(Vector{prec}, perm))
        
        # apply thresholding to porosity array
        ϕ[ϕ .< porosity_thresh] .= 0.0
        
        # allocate and fill arrays 
        κl = fill(prec(bulk_l), length(r))
        ηl = fill(prec(visc_l), length(r))
        
        # resample profiles onto new grid
        r_grid, ρ, η, μc, κs, κl, κd, α, ηl, ϕ, k, g_grid, M_tot = solid1d_mush_relax.resample_profiles(r, ρ, η, μc, κs, κl, κd, α, ηl, ϕ, k, m_core, dr_min, dr_max)

        ρs = ρ.*(1.0.-ϕ)        # solid density 
        ρl = ρ.*ϕ               # liquid density

        # use cell centers
        r_centers = 0.5 .* (r_grid[1:end-1] .+ r_grid[2:end])

        # check for zero values in the expanded radius array
        if iszero.(r_centers) |> any
            throw(ArgumentError("Expanded radius array contains zero values. Check input radius and ncalc."))
        end

        # get non-dimensionalization scales
        if optimize_scales
            scales = solid1d_mush_relax.common.optimize_scales(r[2:end], ρs, ρl, g, μc, κs, κl, κd, α, ηl, ϕ, k, ω, n, [r[end], M_tot, G])
        else
            scales = [r[end], M_tot, G] # default scales if not optimizing
        end

        # define angular grid
        SphericalGrid = solid1d_mush_relax.define_spherical_grid(res, n, m)
        
        # solve y functions across grid
        y_t, y_l = solid1d_mush_relax.compute_y(r_centers, ρ, g_grid, μc, κs, ω, ρl, κl, κd, α, ηl, ϕ, k, n, ρ_core, μ_core, κ_core, scales; core=core, inertial_terms=inertial_terms, patch=patch)

        # Love numbers
        kn_T = y_t[5, end] - 1
        kn_L = y_l[5, 1]   - 1

        # heating profile
        Eμ_tot, Eκ_tot, El_tot = solid1d_mush_relax.get_heating_profile(
            y_t, r_grid, ρ, g_grid, μc, κs, ω, ρl, κl, κd, α, ηl, ϕ, k, n, SphericalGrid
        )

        Eμ_glb, Eκ_glb, El_glb = solid1d_mush_relax.get_heating_map(
            y_t, r_grid, ρ, g_grid, μc, κs, ω, ρl, κl, κd, α, ηl, ϕ, k, n, SphericalGrid
        )

        power_prf = abs.(Eμ_tot .+ Eκ_tot .+ El_tot) 

        # interpolate from grid back to original radius points 
        itp = linear_interpolation(r_centers, power_prf, extrapolation_bc=Line())

        # original centers
        r_orig_centers = 0.5 .* (radius[1:end-1] .+ radius[2:end])
        scale_factor = Float64(R) / maximum(radius)
        
        power_prf = Float64.(itp.(r_orig_centers) .* scale_factor)
        
        # radial interpolation for global maps
        nlats = length(Eμ_glb[:, 1, 1])
        nlons = length(Eμ_glb[1, :, 1])
        Nr_orig = length(r_orig_centers)
        
        # pre-allocate separate 3D grids for each source on the original spacing
        Eμ_glb_itp = zeros(Float64, nlats, nlons, Nr_orig)
        Eκ_glb_itp = zeros(Float64, nlats, nlons, Nr_orig)
        El_glb_itp = zeros(Float64, nlats, nlons, Nr_orig)

        # loop over geographic coordinates, treating each lat/lon as a 1D radial column
        for lon_idx in 1:nlons
            for lat_idx in 1:nlats
                # shear component interpolation
                shear_col = @view Eμ_glb[lat_idx, lon_idx, :]
                itp_shear = linear_interpolation(r_centers, shear_col, extrapolation_bc=Line())
                Eμ_glb_itp[lat_idx, lon_idx, :] .= Float64.(itp_shear.(r_orig_centers) .* scale_factor)

                # compaction component interpolation
                comp_col = @view Eκ_glb[lat_idx, lon_idx, :]
                itp_comp = linear_interpolation(r_centers, comp_col, extrapolation_bc=Line())
                Eκ_glb_itp[lat_idx, lon_idx, :] .= Float64.(itp_comp.(r_orig_centers) .* scale_factor)

                # darcy / liquid component interpolation
                darcy_col = @view El_glb[lat_idx, lon_idx, :]
                itp_darcy = linear_interpolation(r_centers, darcy_col, extrapolation_bc=Line())
                El_glb_itp[lat_idx, lon_idx, :] .= Float64.(itp_darcy.(r_orig_centers) .* scale_factor)
            end
        end

        return power_prf, Eμ_glb_itp, Eκ_glb_itp, El_glb_itp, kn_T, kn_L
    end


    """
        run_solid1d_relax(omega, rho, radius, gravity, visc, shear, bulk, R, m_core, ρ_core, μ_core, κ_core; dr_min=300, dr_max=3000, n=2, m=2, core="liquid", optimize_scales=false, patch=false)

    Use 1D solid tides model with relaxation method to calculate kn Lovenumbers, and compute 1D heating profile from strain tensor.
    This method includes inertia effects, but is more computationally expensive. 
    
    # Arguments
    - `omega::prec`                     : Forcing frequency range.
    - `rho::Array{prec,1}`              : Density profile of the planet.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `gravity::Array{prec,1}`          : Gravity profile of the planet.
    - `visc::Array{prec,1}`             : Viscosity profile of the planet.
    - `shear::Array{precc,1}`           : Complex shear modulus profile of the planet.
    - `bulk::Array{precc,1}`            : Complex bulk modulus profile of the planet.
    - `R::prec`                         : Planet radius.
    - `m_core::prec`                    : Core mass.
    - `ρ_core::prec`                    : Core density.
    - `μ_core::precc`                   : Core shear modulus.
    - `κ_core::precc`                   : Core bulk modulus.

    # Keyword Arguments
    - `dr_min::Int=300`                 : Minimum layer thickness in m.
    - `dr_max::Int=3000`                : Maximum layer thickness in m.
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `m::Int=2`                        : Harmonic of the true anomaly. m=2 corresponds to the semidiurnal tide, m=1 diurnal tide.
    - `core::String="liquid"`           : Core state, either "liquid", "solid", or "inertial".
    - `optimize_scales::Bool=false`     : Whether to optimize non-dimensionalization scales.
    - `patch::Bool=false`               : Whether to insert an infinitesimal solid shell around the core. This patches an issue where y2 and y4 become decoupled and cause the solution to diverge in fluid layers.

    # Returns
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_solid1d_equil_relax( omega::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        gravity::Array{prec,1},
                        visc::Array{prec,1},
                        shear::Array{precc,1},
                        bulk::Array{precc,1},
                        R::prec,
                        m_core::prec,
                        ρ_core::prec,
                        μ_core::precc,
                        κ_core::precc;
                        dr_min::Int=300,
                        dr_max::Int=3000,
                        n::Int=2,
                        m::Int=2,
                        core::String="liquid",
                        optimize_scales::Bool=false,
                        patch::Bool=false
                        )::Tuple{ComplexF64,ComplexF64}

        # convert inputs
        ω = prec(omega)
        ρ = convert(Vector{prec}, rho)
        r = convert(Vector{prec}, radius)
        g = convert(Vector{prec}, gravity)
        η = convert(Vector{prec}, visc)
        μc = convert(Vector{precc}, shear)
        κc = convert(Vector{precc}, bulk)

        # resample profiles onto new grid
        r_grid, ρ, η, μc, κc, g_grid, M_tot = solid1d_equil_relax.resample_profiles(r, ρ, η, μc, κc, m_core, dr_min, dr_max)

        # use cell centers
        r_centers = 0.5 .* (r_grid[1:end-1] .+ r_grid[2:end])

        # check for zero values in the expanded radius array
        if iszero.(r_centers) |> any
            throw(ArgumentError("Expanded radius array contains zero values. Check input radius and ncalc."))
        end
        
        # get non-dimensionalization scales
        if optimize_scales
            scales = solid1d_equil_relax.common.optimize_scales(r[2:end], ρ, g, μc, κc, ω, n, [r[end], M_tot, G])
        else
            scales = [r[end], M_tot, G] # default scales if not optimizing
        end

        # solve y functions across grid
        y_t, y_l = solid1d_equil_relax.compute_y(r_centers, ρ, g_grid, μc, κc, ω, n, ρ_core, μ_core, κ_core, scales; core=core, patch=patch)

        # Love numbers
        kn_T = y_t[5, end] - 1
        kn_L = y_l[5, 1]   - 1

        return kn_T, kn_L
    end


    """
        run_fluid0d(omega, rho, radius, ρ_ratio; n=2, sigma_R=1e-3)

    Calculate kn Lovenumbers in the 0D fluid.

    # Arguments
    - `omega::Float64`                  : Forcing frequency.
    - `rho::Array{prec,1}`              : Density profile of the planet.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `ρ_ratio::prec`                   : Density contrast between current (fluid) and lower (non-fluid) layer.
    
    # Keyword Arguments
    - `n::Int=2`                        : Power of the radial factor (goes with (r/a)^{n}, since r<<a only n=2 contributes significantly).
    - `sigma_R::Float64=1e-3`           : Rayleigh drag coefficient.

    # Returns
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_fluid0d( omega::Float64,
                        rho::Array{prec,1},
                        radius::Array{prec,1},
                        ρ_ratio::prec;
                        n::Int64=2,
                        sigma_R::Float64=1e-3
                        )::Tuple{ComplexF64,ComplexF64}

        # internal structure arrays
        ρ = convert(Vector{prec}, rho)
        r = convert(Vector{prec}, radius)

        # surface properties
        R = maximum(r)  # Segment outer radius (m)
        g = G * sum(4/3 * π * (r[2:end].^3 .- r[1:end-1].^3) .* ρ[1:end]) / R^2  # Surface gravity (m/s^2)

        # fluid magma ocean bottom and height
        r_b = r[1]
        H_magma = r[end] - r_b
             
        # get kn Lovenumbers
        kn_T, kn_L = fluid0d.compute_fluid_lovenumbers(omega, R, H_magma, g, ρ_ratio, n, sigma_R)

        return ComplexF64(kn_T), ComplexF64(kn_L)

    end


    """
        run_fluid1d(omega, rho, radius, visc, ρ_ratio, P_b, R; n=2, sigma_R=1e-3, sigma_inf=1e-7, sigma_R_prf="uniform", H_R=1e3, efficiency=0.3, visc_l=1e2)

    Compute tidal heating profile and Love numbers for a 1D fluid model. The radial structure is defined by the density and 
    viscosity profiles, and the heating at the lower interface. The model includes Rayleigh drag at the interface and in the
    fluid interior. The users can specify the drag profile type and scale height. The function returns the heating profile,
    heating maps for different heating mechanisms, and the tidal and load Love numbers.

    # Arguments
    - `omega::Float64`                  : Forcing frequency
    - `rho::Vector{prec}`               : Density profile
    - `radius::Vector{prec}`            : Radial grid (core → surface)
    - `visc::Vector{prec}`              : Viscosity profile
    - `ρ_ratio::prec`                   : Density ratio of lower layer
    - `P_b::Vector{<:AbstractMatrix{Float64}}`: Heating at lower interface
    - `R::prec`                         : Planet radius

    # Keyword Arguments
    - `n::Int=2`                        : Radial power (dominant term n=2)
    - `sigma_R::Float64=1e-3`           : Rayleigh drag at interface
    - `sigma_inf::Float64=1e-7`         : Drag in fluid interior
    - `sigma_R_prf::String="uniform"`   : Drag profile type
    - `H_R::Float64=1e3`                : Drag scale height
    - `efficiency::Float64=0.3`         : Interface efficiency factor
    - `visc_l::Float64=1e2`             : Liquid viscosity

    # Returns
    - `power_prf::Vector{Float64}`      : Heating profile
    - `map_μ::Array{Float64,3}`         : Shear heating map (colatitude, longitude, radius)
    - `map_κ::Array{Float64,3}`         : Compaction heating map (colatitude, longitude, radius)
    - `map_l::Array{Float64,3}`         : Darcy heating map (colatitude, longitude, radius)
    - `map_f::Array{Float64,3}`         : Friction heating map (colatitude, longitude, radius)
    - `map_d::Array{Float64,3}`         : Drag heating map (colatitude, longitude, radius)
    - `kn_T::ComplexF64`                : Tidal Love number
    - `kn_L::ComplexF64`                : Load Love number
    """
    function run_fluid1d(   omega::Float64,
                            rho::Vector{prec},
                            radius::Vector{prec},
                            visc::Vector{prec},
                            ρ_ratio::prec,
                            P_b::Vector{<:AbstractMatrix{Float64}},
                            R::prec;
                            n::Int = 2,
                            sigma_R::Float64 = 1e-3,
                            sigma_inf::Float64 = 1e-7,
                            sigma_R_prf::String = "uniform",
                            H_R::Float64 = 1e3,
                            efficiency::Float64 = 0.3,
                            visc_l::Float64 = 1e2
                        )::Tuple{Vector{Float64},Array{Float64,3},Array{Float64,3},Array{Float64,3},Array{Float64,3},Array{Float64,3}, ComplexF64, ComplexF64}

        r = convert(Vector{prec}, radius)
        
        Nz = length(r) - 1
        
        # determine angular mean of a lat/lon map (for anisotropic maps)
        weight = reshape(sin.(lat_vals), :, 1)
        angular_mean(M::AbstractMatrix) = sum(weight .* M .* dres^2)

        # build isotropic map (for `fluid1d` output) from a 1D radial profile
        function isotropic_map(target::Vector{Float64})
            out = zeros(Float64, nlats, nlons, Nz)
            for i in 1:Nz
                out[:, :, i] .= target[i] ./ Ω_total
            end
            return out
        end

        # build anisotropic map with optional depth weight profile
        function propagate_map(base::AbstractMatrix{Float64}, profile::Vector{Float64} = ones(Nz))
            out = zeros(Float64, nlats, nlons, Nz)
            for i in 1:Nz
                out[:, :, i] .= base .* profile[i]
            end
            return out
        end

        # determine shell volumes (explicitly cast geometric dimensions derived from `r` to Float64)
        Vs  = Float64((4/3) * π * (r[end]^3 - r[1]^3))
        dVs = Float64.((4/3) * π .* (r[2:end].^3 .- r[1:end-1].^3))

        # compute kn Lovenumbers for the fluid layer with Rayleigh drag at the interface and in the interior
        kn_T, kn_L  = run_fluid0d(omega, rho, r, ρ_ratio; n=n, sigma_R=efficiency * sigma_R)    # total
        kn_T_inf, _ = run_fluid0d(omega, rho, r, ρ_ratio; n=n, sigma_R=sigma_inf)               # fluid baseline (no friction)

        # compute dimensionless volumetric heating from the imaginary part of the tidal Love number
        prefactor = Float64((2n + 1) * R * omega / (8π * G))
        E_total = Float64(prefactor * -imag(kn_T))     
        E_inf   = Float64(prefactor * -imag(kn_T_inf))
        
        # compute shell centers and height for depth-dependent drag profile
        r_mid = 0.5 .* (r[1:end-1] .+ r[2:end])
        z     = Float64.(abs.(r_mid .- r[1]))

        # extract shear, bulk, and darcy contributions to the total heating (for smooth transition from solid-mush to fluid)
        μ_ini, κ_ini, l_ini = P_b
        # obtain 1D equivalent of the 3D heating map by angular averaging
        shear_bulk_darcy = Float64(angular_mean(μ_ini) + angular_mean(κ_ini) + angular_mean(l_ini))

        # initiate depth-dependent decay factor for shear, bulk, and darcy heating profiles
        decay_factor = ones(Float64, Nz)

        # `dynamic_interp` profile: Gaussian shoulder for shear/bulk/darcy, Gaussian for friction, and sigmoid for drag       
        if sigma_R_prf == "dynamic_interp"
            # power coefficient for Gaussian shoulder profile
            p_decay = 1.5       # kept as 1.5 for smooth shoulder

            # build baseline Gaussian shoulder profile using reference scale height H_R
            decay_factor = @. exp(-((z / H_R)^p_decay))
            E_sbd_base = sum(shear_bulk_darcy .* decay_factor .* dVs)
            E_sbd_max  = 0.25 * E_inf       # limit shear/bulk/darcy to 25% of total energy

            # check if the shear/bulk/darcy energy is below the cap
            if E_sbd_base <= E_sbd_max || shear_bulk_darcy == 0.0
                # below cap: keep Gaussian shoulder with reference scale height H_R
                E_sbd = E_sbd_base
            else
                # above cap: bisect H_decay (< H_R) so integrated shear/bulk/darcy energy hits E_sbd_max
                V_target = E_sbd_max / shear_bulk_darcy
                H_low  = 1e-12
                H_high = H_R        # H_R is guaranteed upper bound since E_sbd_base > E_sbd_max

                # bisect to find H_decay such that the integrated shear/bulk/darcy energy equals E_sbd_max
                for _ in 1:60
                    H_mid = 0.5 * (H_low + H_high)
                    if sum(exp.(-((z ./ H_mid).^p_decay)) .* dVs) < V_target
                        H_low = H_mid
                    else
                        H_high = H_mid
                    end
                end

                # compute final decay factor with the bisected H_decay
                H_decay = 0.5 * (H_low + H_high)
                decay_factor = @. exp(-((z / H_decay)^p_decay))
                E_sbd = E_sbd_max
            end

            # shear/bulk/darcy component: dedicated budget (E_sbd)
            sbd_prf = shear_bulk_darcy .* decay_factor

            # friction component: dedicated budget (E_total - E_inf)
            E_gauss = max(E_total - E_inf, 0.0)

            # define mean and standard deviation for Gaussian profile
            σg = H_R / 4
            μg = 2 * H_R

            # compute Gaussian profile for friction, ensuring it is zero at the surface (z=0)
            f_at_zero = exp(-0.5 * (μg / σg)^2)

            # compute raw Gaussian profile and normalize to match the energy budget E_gauss
            friction_raw = @. max(0.0, exp(-0.5 * ((z - μg) / σg)^2) - f_at_zero)
            norm_g = sum(friction_raw .* dVs)
            friction = friction_raw .* (norm_g > 0 ? E_gauss / norm_g : 0.0)

            # drag component: dedicated budget (E_inf - E_sbd)
            E_drag = max(E_inf - E_sbd, 0.0)

            # determine the depth where the viscosity drops below the liquid viscosity threshold (visc_l)
            idx = findfirst(visc .<= visc_l)

            if isnothing(idx)
                # if no viscosity is below the threshold, set z_visc to the maximum depth
                z_visc = maximum(z)
            elseif idx == 1
                # if the first layer is below the threshold, set z_visc to the depth of the first layer
                z_visc = z[1]
            else
                # interpolate to find the depth where viscosity equals visc_l
                μ1, μ2 = Float64(visc[idx-1]), Float64(visc[idx])
                f = clamp((visc_l - μ1)/(μ2 - μ1), 0.0, 1.0)
                z_visc = z[idx-1] + f*(z[idx]-z[idx-1])
            end

            # define a smooth transition for the drag profile using a hyperbolic tangent function
            width = max(z_visc / 8, H_R / 10)
            
            # compute the drag profile, ensuring it is zero at the surface (z=0) and transitions to 1 at z_visc
            d_at_zero = 0.5 * (1 + tanh((-z_visc / 2) / width))
            drag_raw = @. 0.5 * (1 + tanh((z - z_visc / 2) / width))
            drag_raw[z .>= z_visc] .= 1.0
            drag_raw .= max.(0.0, drag_raw .- d_at_zero)

            # normalize the drag profile to match the energy budget E_drag
            norm_d = sum(drag_raw .* dVs)
            drag = drag_raw .* (norm_d > 0 ? E_drag / norm_d : 0.0)
            
            # combine all components to form the total power profile
            power_prf = sbd_prf .+ drag .+ friction

            # create isotropic maps for friction and drag components
            map_f = isotropic_map(friction)
            map_d = isotropic_map(drag)

        # `uniform`, `exp`, `linear`, or `quadratic` profile: simple depth-dependent drag profile
        else
            # compute the dedicated budget for the depth-dependent profile
            D_power_blk   = max((E_total - E_inf) / Vs, 0.0)

            # compute the depth-dependent shape factor based on the specified profile type
            shape = ones(Float64, length(z))

            if sigma_R_prf == "exp"
                # exponential decay profile
                shape .= exp.(-z ./ H_R)

            elseif sigma_R_prf == "linear"
                # linear decay profile
                shape .= max.(0.0, 1.0 .- z ./ H_R)

            elseif sigma_R_prf == "quadratic"
                # quadratic decay profile
                shape .= max.(0.0, 1.0 .- z ./ H_R).^2

            elseif sigma_R_prf == "dynamic"
                # dynamic profile: exponential decay with a minimum cutoff to avoid numerical issues
                l_mix = max.(min.(z, H_R), 1e-12)
                shape .= exp.(-z ./ l_mix)

            elseif sigma_R_prf != "uniform"
                # unknown profile type: throw an error
                error("Unknown sigma_R_prf: $sigma_R_prf")
            end

            # compute the final power profile by combining the baseline and depth-dependent components
            power_prf = E_inf / Vs .+ D_power_blk .* shape

            # normalize the power profile to match the total energy budget
            unorm = sum(power_prf .* dVs) / Vs
            if unorm > 0
                power_prf .*= (E_total / Vs) / unorm
            end

            # compute the shear/bulk/darcy component for the depth-dependent profile
            drag_component = max.(power_prf .- shear_bulk_darcy, 0.0)

            # create isotropic maps for friction and drag components
            map_f = zeros(Float64, nlats, nlons, Nz)        # friction is not explicitly computed in this case
            map_d = isotropic_map(drag_component)
        end     

        # propagate anisotropic maps with potential depth decay applied
        map_μ = propagate_map(μ_ini, decay_factor)
        map_κ = propagate_map(κ_ini, decay_factor)
        map_l = propagate_map(l_ini, decay_factor)

        return Vector{Float64}(power_prf), map_μ, map_κ, map_l, map_f, map_d, ComplexF64(kn_T), ComplexF64(kn_L)
    end


    """
        run_interp(omega, radius, P_t, P_b; t_width=0.1, b_width=0.1)

    Interpolate dissipation and kn Lovenumbers in a 1D region without active tides.

    # Arguments
    - `omega::Float64`                  : Forcing frequency.
    - `radius::Array{prec,1}`           : Radial positions of layers, from core to surface.
    - `R::prec`                         : Planet Radius.
    - `P_t::prec`                       : Heating at upper interface.
    - `P_b::prec`                       : Heating at lower interface.
    
    # Keyword Arguments
    - `t_width::Float64=0.1`            : Fraction of segment height as standard deviation for upper dissipation peak.
    - `b_width::Float64=0.1`            : Fraction of segment height as standard deviation for lower dissipation peak.

    # Returns
    - `power_prf::Array{Float64,1}`     : Heating profile.
    - `kn_T::ComplexF64`                : Complex Tidal kn Lovenumber.
    - `kn_L::ComplexF64`                : Complex Load kn Lovenumber.
    """
    function run_interp(omega::Float64,
                        radius::Vector{prec},
                        R::prec,
                        P_t::Real,
                        P_b::Real;
                        t_width::Real = 0.1,
                        b_width::Real = 0.1
                    )::Tuple{Vector{Float64}, ComplexF64, ComplexF64}

        # radial grid
        r = convert(Vector{prec}, radius)
        r_mid = 0.5 .* (r[1:end-1] .+ r[2:end])

        # shell volumes
        Vs = 4/3 * π * (r[2:end].^3 .- r[1:end-1].^3)

        # interface locations
        r_top = r[end]
        r_bot = r[1]

        # decay lengths
        l_t = t_width * (r[end] - r[1])
        l_b = b_width * (r[end] - r[1])

        # exponential decay profiles
        G_top = exp.(-abs.(r_mid .- r_top) ./ l_t)
        G_bot = exp.(-abs.(r_mid .- r_bot) ./ l_b)

        # normalize peaks to unity
        if maximum(G_top) > 0
            G_top ./= maximum(G_top)
        end
        if maximum(G_bot) > 0
            G_bot ./= maximum(G_bot)
        end

        # total power profile
        power_prf = P_t .* G_top .+ P_b .* G_bot

        # allocate Love numbers
        kn_T = zeros(precc, length(r_mid))
        kn_L = zeros(precc, length(r_mid))

        # tidal prefactor
        prefactor = 5 * R * omega / (8π * G)

        # infer complex kn from dissipation
        @inbounds for is in eachindex(r_mid)
            kn_T[is] = -im * power_prf[is] * Vs[is] / prefactor
        end

        return Float64.(power_prf), ComplexF64(sum(kn_T)), ComplexF64(sum(kn_L))
    end


    """
        data_to_nc(nmk, is_seg, segments, knms_T, knms_L, σ_range, P_T_s_prf, P_T_prf, P_T_blk, P_T_prf_blk, radius, P_T_s_glb_μ, P_T_s_glb_κ, P_T_s_glb_l, datafile_path)
    
    Write model results to a NetCDF file.

    # Arguments
    - `nmk::Vector{Tuple{Int,Int,Int}}`         : Array of (n,m,k) tuples for each segment.
    - `is_seg::Array{Tuple{Int,Int},1}`         : Array of (il,it) tuples indicating segment indices.
    - `segments::Array{String,1}`               : Array of segment labels.
    - `knms_total::Array{ComplexF64,1}`         : Total kn Lovenumbers for each (n,m,k).
    - `knms_T::Matrix{ComplexF64}`              : Tidal kn Lovenumbers for each (n,m,k) and segment.
    - `knms_L::Matrix{ComplexF64}`              : Load kn Lovenumbers for each (n,m,k) and segment.
    - `σ_range::Array{Float64,1}`               : Array of forcing frequencies.
    - `P_T_blk::Float64`                        : Tidal heating (bulk).
    - `P_T_prf_blk::Float64`                    : Tidal heating (bulk from profile).
    - `P_T_s_prf::Matrix{Float64}`              : Tidal heating profile for each (n,m,k) and spatial location.
    - `radius::Vector{Float64}`                 : Radial grid points.
    - `P_T_s_glb_μ::Array{Float64,4}`           : Shear tidal heating map for each (n,m,k) and spatial location.
    - `P_T_s_glb_κ::Array{Float64,4}`           : Bulk tidal heating map for each (n,m,k) and spatial location.
    - `P_T_s_glb_l::Array{Float64,4}`           : Darcy tidal heating map for each (n,m,k) and spatial location.
    - `P_T_s_glb_f::Array{Float64,4}`           : Friction tidal heating map for each (n,m,k) and spatial location.
    - `P_T_s_glb_d::Array{Float64,4}`           : Drag tidal heating map for each (n,m,k) and spatial location.
    - `datafile_path::String`                   : Path to the output NetCDF file.
    """
    function data_to_nc(nmk::Vector{Tuple{Int,Int,Int}}, 
                        is_seg::Array{Tuple{Int,Int},1}, 
                        segments::Array{String,1},  
                        knms_total::Array{ComplexF64,1},
                        knms_T::Matrix{ComplexF64}, 
                        knms_L::Matrix{ComplexF64}, 
                        σ_range::Array{Float64,1}, 
                        P_T_blk::Float64, 
                        P_T_prf_blk::Float64, 
                        P_T_s_prf::Matrix{Float64},
                        radius::Vector{Float64},
                        P_T_s_glb_μ::Array{Float32,4},
                        P_T_s_glb_κ::Array{Float32,4},
                        P_T_s_glb_l::Array{Float32,4},
                        P_T_s_glb_f::Array{Float32,4},
                        P_T_s_glb_d::Array{Float32,4},
                        datafile_path::String
                       )
        
        # helper function to convert Complex arrays to Float64 arrays with an extra dimension
        function pack_complex(A::AbstractArray{Complex{T}}) where T
            # Creates an array with an extra trailing dimension of size 2
            return cat(real.(A), imag.(A), dims=ndims(A)+1)
        end

        @info "Writing results to $datafile_path"

        n_vals = [t[1] for t in nmk]
        m_vals = [t[2] for t in nmk]
        k_vals = [t[3] for t in nmk]

        il_vals = [t[1] for t in is_seg]
        it_vals = [t[2] for t in is_seg]

        NCDataset(datafile_path, "c") do ds
            
            # DEFINE DIMENSIONS
            defDim(ds, "nmk",      length(nmk))
            defDim(ds, "segments", length(segments))
            defDim(ds, "r",        length(radius))
            defDim(ds, "z",        length(P_T_s_glb_μ[1, 1, 1, :]))
            defDim(ds, "lon",      length(P_T_s_glb_μ[1, 1, :, 1]))
            defDim(ds, "lat",      length(P_T_s_glb_μ[1, :, 1, 1]))
            defDim(ds, "complex",  2)

            # PART 1
            defVar(ds, "n", n_vals, ("nmk",))
            defVar(ds, "m", m_vals, ("nmk",))
            defVar(ds, "k", k_vals, ("nmk",))

            defVar(ds, "il", il_vals, ("segments",))
            defVar(ds, "it", it_vals, ("segments",))
            
            defVar(ds, "segment_lbl", segments, ("segments",))
            
            # data grids, packed into ("nmk", "segments", "complex")
            defVar(ds, "knms_T", pack_complex(knms_T), ("nmk", "segments", "complex"))
            defVar(ds, "knms_L", pack_complex(knms_L), ("nmk", "segments", "complex"))
            
            defVar(ds, "sigma_range", σ_range, ("nmk",))
            defVar(ds, "knms_total", pack_complex(knms_total), ("nmk", "complex"))

            defVar(ds, "P_T_blk",     P_T_blk,     ()) 
            defVar(ds, "P_T_prf_blk", P_T_prf_blk, ()) 

            defVar(ds, "radius", radius, ("r",))
            defVar(ds, "P_T_s_prf", P_T_s_prf, ("nmk", "z"))
            defVar(ds, "lon", P_T_s_glb_μ[1, 1, :, 1], ("lon",))
            defVar(ds, "lat", P_T_s_glb_μ[1, :, 1, 1], ("lat",))
            defVar(ds, "P_T_s_glb_μ", P_T_s_glb_μ, ("nmk", "lat", "lon", "z"))
            defVar(ds, "P_T_s_glb_κ", P_T_s_glb_κ, ("nmk", "lat", "lon", "z"))
            defVar(ds, "P_T_s_glb_l", P_T_s_glb_l, ("nmk", "lat", "lon", "z"))
            defVar(ds, "P_T_s_glb_f", P_T_s_glb_f, ("nmk", "lat", "lon", "z"))
            defVar(ds, "P_T_s_glb_d", P_T_s_glb_d, ("nmk", "lat", "lon", "z"))

            ds.attrib["title"] = "Obliqua Model Run Data"
        end
    end
        

end