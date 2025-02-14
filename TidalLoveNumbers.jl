# Julia code to calculate tidal deformation, Love numbers, and heating
# Author: H. Hay

# μ: solid shear modulus
# ρ: solid density 
# κ: solid bulk modulus 
# λ: Lame's First Parameter
# η: solid viscosity 
# g: gravity 
# ω: rotation rate


module TidalLoveNumbers

    using LinearAlgebra
    using DoubleFloats
    using AssociatedLegendrePolynomials
    include("SphericalHarmonics.jl")
    using .SphericalHarmonics

    export get_g, get_A!, get_A, get_B_product, get_Ic, get_B
    export expand_layers, set_G, calculate_y
    export get_displacement, get_solution
    export get_bulk_heating, get_heating_profile

    # If results seem odd, you can increase the numerical precision with
    # one of the options below. These get quite a bit slower...
    prec = Float64 
    precc = ComplexF64 

    # prec = Double64     
    # precc = ComplexDF64 

    # prec = BigFloat
    # precc = Complex{BigFloat}

    G = prec(6.6743e-11)
    n = 2

    M = 6              # Matrix size: 6x6 if only solid material, 8x8 for two-phases
    nr = 80            # Number of sub-layers in each layer (TODO: change to an array)

    Abot = zeros(precc, 6, 6)
    Amid = zeros(precc, 6, 6)
    Atop = zeros(precc, 6, 6)

    # Overwrite Gravitional constant for non-dimensional 
    # calculations
    function set_G(new_G)
        TidalLoveNumbers.G = new_G
    end

    function set_nr(new_nr)
        TidalLoveNumbers.nr = new_nr
    end

    function get_g(r, ρ)
        g = zeros(prec, size(r))
        M = zeros(prec, size(r))

        for i in 1:size(r)[2]
            M[2:end,i] = 4.0/3.0 * π .* diff(r[:,i].^3) .* ρ[i]
        end

        g[2:end,:] .= G*accumulate(+,M[2:end,:]) ./ r[2:end,:].^2
        g[1,2:end] = g[end,1:end-1]

        return g

    end

    function get_A(r, ρ, g, μ, κ)
        A = zeros(precc, 6, 6) 
        get_A!(A, r, ρ, g, μ, κ)
        return A
    end

    function get_A!(A::Matrix, r, ρ, g, μ, κ, λ=nothing)
        if isnothing(λ)
            λ = κ - 2μ/3
        end

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)

        A[1,1] = -2λ * β_inv * r_inv
        A[1,2] = n*(n+1) * λ * β_inv * r_inv
        A[1,3] = β_inv

        A[2,1] = -r_inv
        A[2,2] = r_inv
        A[2,4] = 1.0 / μ

        A[3,1] = 4r_inv * (3κ*μ*r_inv*β_inv - ρ*g)
        A[3,2] = -n*(n+1)*r_inv * (6κ*μ*r_inv*β_inv - ρ*g ) #+
        A[3,3] = β_inv * (-4μ*r_inv )
        A[3,4] = n*(n+1)*r_inv
        A[3,5] = -ρ * (n+1)*r_inv
        A[3,6] = ρ

        A[4,1] = -r_inv * (6κ*μ*r_inv*β_inv - ρ*g )
        A[4,2] = 2μ*r_inv^2 * (n*(n+1)*(1 + λ*β_inv) - 1.0 )
        A[4,3] = -r_inv * λ * β_inv # changed to match sabadini    
        A[4,4] = -3r_inv
        A[4,5] = ρ*r_inv

        A[5,1] = -4π * G * ρ
        A[5,5] = -(n+1)r_inv
        A[5,6] = 1.0

        A[6,1] = -4π*(n+1)*G*ρ*r_inv
        A[6,2] = -A[6,1]*n
        A[6,6] = (n-1)r_inv

    end

    function get_B(r1, r2, g1, g2, ρ, μ, κ)
        B = zeros(precc, 6, 6)
        get_B!(B, r1, r2, g1, g2, ρ, μ, κ)
        return B
    end

    function get_B!(B, r1, r2, g1, g2, ρ, μ, κ)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        A1 = get_A(r1, ρ, g1, μ, κ)
        Ahalf = get_A(rhalf, ρ, ghalf, μ, κ)
        A2 = get_A(r2, ρ, g2, μ, κ)
        
        k1 = dr * A1 
        k2 = dr * Ahalf * (I + 0.5k1)
        k3 = dr * Ahalf * (I + 0.5k2)
        k4 = dr * A2 * (I + k3) 

        B[1:6,1:6] .= (I + 1.0/6.0 * (k1 + 2k2 + 2k3 + k4))
    end

    # first method: solid layer -- for a specific layer?
    function get_B_product2!(Bprod2, r, ρ, g, μ, κ)
        # Check dimensions of Bprod2

        Bstart = zeros(precc, 6, 6)
        B = zeros(precc, 6, 6)

        for i in 1:6
            Bstart[i,i,1] = 1
        end

        nr = size(r)[1]

        r1 = r[1]
        for j in 1:nr-1
            r2 = r[j+1]
            g1 = g[j]
            g2 = g[j+1]

            get_B!(B, r1, r2, g1, g2, ρ, μ, κ)
            Bprod2[:,:,j] .= B * (j==1 ? Bstart : Bprod2[:,:,j-1])

            r1 = r2
        end
    end

    function calculate_y(r, ρ, g, μ, κ; core="liquid")
        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core, 6, 3)

        y1_3 = zeros(precc, 6, 3, nsublayers-1, nlayers) # Three linearly independent y solutions
        y = zeros(ComplexF64, 6, nsublayers-1, nlayers)
        
        for i in 2:nlayers
            # Product of B matrices in sublayer
            Bprod = zeros(precc, 6, 6, nsublayers-1)
            get_B_product2!(Bprod, r[:, i], ρ[i], g[:, i], μ[i], κ[i])

            # Set B matrix product into y_solutions
            for j in 1:nsublayers-1
                y1_3[:,:,j,i] = Bprod[:,:,j] * y_start 
            end

            # Set y solutions for the start of the next layer
            y_start[:,:] .= y1_3[:,:,end,i]  
        end

        # The ODE has now been integrated. Next, collect the constrained
        # parts of the solution in order to apply BCs.

        M = zeros(precc, 3,3)

        # Row 1 - Radial Stress
        M[1, :] .= y1_3[3,:,end,end]

        # Row 2 - Tangential Stress
        M[2, :] .= y1_3[4,:,end,end]
    
        # Row 3 - Potential Stress
        M[3, :] .= y1_3[6,:,end,end]
         
        # Create boundary condition vector
        b = zeros(precc, 3)
        b[3] = (2n+1)/r[end,end] # non-zero potential stress for tidal forcing
        
        # Solve system of linear equations to find integration constants, C
        C = M \ b

        # Apply integration constants vector to integrated solution to obtain
        # the true solution
        for i in 2:nlayers
            for j in 1:nsublayers-1
                y[:,j,i] = y1_3[:,:,j,i]*C
            end
        end

        return y
    end

    # Get the total heating rate across the entire body
    function get_total_heating(y, ω, R, ecc)
        k2 = y[5, end,end] - 1.0    # Get k2 Love number at surface
        total_power = -21/2 * imag(k2) * (ω*R)^5/G * ecc^2

        return total_power
    end

    function get_solution(y, n, m, r, ρ, g, μ, κ, ω, ρₗ, κₗ, ηₗ, ϕ, k, res=5.0)
        #κ is the bulk modulus of the solid! The drained bulk modulus
        # is (1-α)*κ

        λ = κ .- 2μ/3
        κₛ = κ

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))

        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        Y = m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)
        S = m < 0 ? Snmc(n,abs(m),clats,lons) : Snm(n,abs(m),clats,lons)

        # Better way to do this? (Analytical expression?)
        if iszero(abs(m))
            # d2Ydθ2 = -3cos.(2clats) * exp.(1im * m * lons)
            dYdθ = -1.5sin.(2clats) * exp.(1im * m * lons)
            dYdϕ = Y * 1im * m

            Z = 0.0 * Y
            X = -6cos.(2clats)*exp.(1im *m * lons) .+ n*(n+1)*Y
        elseif  abs(m) == 2
            # d2Ydθ2 = 6cos.(2clats) * exp.(1im * m * lons)
            dYdθ = 3sin.(2clats) * exp.(1im * m * lons)
            dYdϕ = Y * 1im * m
            
            Z = 6 * 1im * m * cos.(clats) * exp.(1im * m * lons)
            X = 12cos.(2clats)* exp.(1im * m * lons) .+ n*(n+1)*Y 
        end

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        q_flux = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)
        p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])
        ζ = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])

        for i in 2:size(r)[2] # Loop of layers
            ηₗr = ηₗ[i]
            ρₗr = ρₗ[i]
            ρr = ρ[i]
            kr  = k[i]
            κₗr = κₗ[i]
            κr = κ[i]
            μr = μ[i]
            ϕr = ϕ[i]
            λr = λ[i]

            if ϕr > 0
                κₛ = κ[i]
                κd = (1-α)κₛ
                κu = κd + κₗr .*κₛ .*α^2 ./ (ϕ[i] .* κₛ + (α-ϕ[i]) .* κₗr)
                λr = κd .- 2μr/3
            end
            # λr = λ[i]

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                (y1, y2, y3, y4, y5, y6, y7, y8) = ComplexF64.(y[:,j,i])
                
                rr = r[j,i]
                gr = g[j,i]
                
                disp[:,:,:,j,i]   .= get_displacement(y1, y2, Y, S)
                if ϕ[i] > 0
                    q_flux[:,:,:,j,i] .= get_darcy_velocity(y5, y7, y8, kr, rr, ηₗr, ρₗr, Y, S)
                end

                A = ComplexF64.(get_A(rr, ρr, gr, μr, κr, ω, ρₗr, κₗr, ηₗr, ϕr, kr))
                dy1dr = dot(A[1,:], y[:,j,i])
                dy2dr = dot(A[2,:], y[:,j,i])

                ϵ[:,:,1,j,i] = dy1dr * Y
                ϵ[:,:,2,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
                ϵ[:,:,3,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
                ϵ[:,:,4,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdθ
                ϵ[:,:,5,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdϕ .* 1.0 ./ sin.(clats) 
                ϵ[:,:,6,j,i] = 0.5 * y2/rr * Z
                ϵV = dy1dr .+ 2/rr * y1 .- n*(n+1)/rr * y2

                σ[:,:,1,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,1,j,i] .- (ϕ[i]>0 ? α*y7*Y : 0.0)
                σ[:,:,2,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,2,j,i] .- (ϕ[i]>0 ? α*y7*Y : 0.0)
                σ[:,:,3,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,3,j,i] .- (ϕ[i]>0 ? α*y7*Y : 0.0)
                σ[:,:,4,j,i] .= 2μr * ϵ[:,:,4,j,i]
                σ[:,:,5,j,i] .= 2μr * ϵ[:,:,5,j,i]
                σ[:,:,6,j,i] .= 2μr * ϵ[:,:,6,j,i]

                if ϕ[i] > 0
                    p[:,:,j,i] .= y7 * Y
                    ζ[:,:,j,i] .= (α*ϵV + α^2/(κu - κd) * y7) * Y
                end

            end
        end

        return disp, q_flux, ϵ, σ, p, ζ
    end

    function get_solution(y, n, m, r, ρ, g, μ, κ, res=10.0)
        #κ is the bulk modulus of the solid! The drained bulk modulus
        # is (1-α)*κ

        λ = κ .- 2μ/3
        # κₛ = κ

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))

        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        Y = m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)
        S = m < 0 ? Snmc(n,abs(m),clats,lons) : Snm(n,abs(m),clats,lons)

        # Better way to do this? (Analytical expression?)
        if iszero(abs(m))
            # d2Ydθ2 = -3cos.(2clats) * exp.(1im * m * lons)
            dYdθ = -1.5sin.(2clats) * exp.(1im * m * lons)
            dYdϕ = Y * 1im * m

            Z = 0.0 * Y
            X = -6cos.(2clats)*exp.(1im *m * lons) .+ n*(n+1)*Y
        elseif  abs(m) == 2
            # d2Ydθ2 = 6cos.(2clats) * exp.(1im * m * lons)
            dYdθ = 3sin.(2clats) * exp.(1im * m * lons)
            dYdϕ = Y * 1im * m
            
            Z = 6 * 1im * m * cos.(clats) * exp.(1im * m * lons)
            X = 12cos.(2clats)* exp.(1im * m * lons) .+ n*(n+1)*Y 
        end

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        # q_flux = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)
        p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])
        ζ = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])

        for i in 2:size(r)[2] # Loop of layers
            # ηₗr = ηₗ[i]
            # ρₗr = ρₗ[i]
            ρr = ρ[i]
            # kr  = k[i]
            # κₗr = κₗ[i]
            κr = κ[i]
            μr = μ[i]
            # ϕr = ϕ[i]
            λr = λ[i]

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                (y1, y2, y3, y4, y5, y6) = y[:,j,i]
                
                rr = r[j,i]
                gr = g[j,i]
                
                disp[:,:,:,j,i]   .= get_displacement(y1, y2, Y, S)

                A = get_A(rr, ρr, gr, μr, κr)
                dy1dr = dot(A[1,:], y[:,j,i])
                dy2dr = dot(A[2,:], y[:,j,i])

                ϵ[:,:,1,j,i] = dy1dr * Y
                ϵ[:,:,2,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
                ϵ[:,:,3,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
                ϵ[:,:,4,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdθ
                ϵ[:,:,5,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdϕ .* 1.0 ./ sin.(clats) 
                ϵ[:,:,6,j,i] = 0.5 * y2/rr * Z
                ϵV = dy1dr .+ 2/rr * y1 .- n*(n+1)/rr * y2

                σ[:,:,1,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,1,j,i] 
                σ[:,:,2,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,2,j,i] 
                σ[:,:,3,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,3,j,i] 
                σ[:,:,4,j,i] .= 2μr * ϵ[:,:,4,j,i]
                σ[:,:,5,j,i] .= 2μr * ϵ[:,:,5,j,i]
                σ[:,:,6,j,i] .= 2μr * ϵ[:,:,6,j,i]

            end
        end

        return disp, ϵ, σ
    end

    function get_heating_profile(y, r, ρ, g, μ, κ, ω, ecc; res=20.0)
        dres = deg2rad(res)
        λ = κ .- 2μ/3
        R = r[end,end]

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))

        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)

        ϵs = zero(ϵ)
        σs = zero(ϵ)

        # Eccentricity tide forcing coefficients
        U22E =  7/8 * ω^2*R^2*ecc 
        U22W = -1/8 * ω^2*R^2*ecc
        U20  = -3/2 * ω^2*R^2*ecc

        # Better way to do this? (Analytical expression?)
        n = 2
        for m in [-2, 2, 0]
            Y = m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)
            S = m < 0 ? Snmc(n,abs(m),clats,lons) : Snm(n,abs(m),clats,lons)    

            if iszero(abs(m))
                dYdθ = -1.5sin.(2clats) * exp.(1im * m * lons)
                dYdϕ = Y * 1im * m

                Z = 0.0 * Y
                X = -6cos.(2clats)*exp.(1im *m * lons) .+ n*(n+1)*Y

            elseif  abs(m) == 2
                dYdθ = 3sin.(2clats) * exp.(1im * m * lons)
                dYdϕ = Y * 1im * m
                
                Z = 6 * 1im * m * cos.(clats) * exp.(1im * m * lons)
                X = 12cos.(2clats)* exp.(1im * m * lons) .+ n*(n+1)*Y 
            end

            for i in 2:size(r)[2] # Loop of layers
                ρr = ρ[i]
                κr = κ[i]
                μr = μ[i]
                λr = λ[i]

                for j in 1:size(r)[1]-1 # Loop over sublayers 
                    (y1, y2, y3, y4, y5, y6) = conj.(y[:,j,i])
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    A = get_A(rr, ρr, gr, μr, κr)
                    dy1dr = dot(A[1,:], y[:,j,i])
                    dy2dr = dot(A[2,:], y[:,j,i])

                    # Compute strain tensor
                    ϵ[:,:,1,j,i] = dy1dr * Y
                    ϵ[:,:,2,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
                    ϵ[:,:,3,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
                    ϵ[:,:,4,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdθ
                    ϵ[:,:,5,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdϕ .* 1.0 ./ sin.(clats) 
                    ϵ[:,:,6,j,i] = 0.5 * y2/rr * Z
                    ϵV = dy1dr .+ 2/rr * y1 .- n*(n+1)/rr * y2

                    # Compute stress tensor
                    σ[:,:,1,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,1,j,i] 
                    σ[:,:,2,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,2,j,i] 
                    σ[:,:,3,j,i] .= λr * ϵV * Y + 2μr*ϵ[:,:,3,j,i] 
                    σ[:,:,4,j,i] .= 2μr * ϵ[:,:,4,j,i]
                    σ[:,:,5,j,i] .= 2μr * ϵ[:,:,5,j,i]
                    σ[:,:,6,j,i] .= 2μr * ϵ[:,:,6,j,i]
                end
            end

            if m==-2
                ϵs .+= U22W*ϵ
                σs .+= U22W*σ
            elseif m==2
                ϵs .+= U22E*ϵ
                σs .+= U22E*σ
            else
                ϵs .+= U20*ϵ
                σs .+= U20*σ
            end
        end

        Eₛ_vol = zeros(  (size(σ)[1], size(σ)[2], size(σ)[4], size(σ)[5]) )
        Eₛ_vol_layer_sph_avg = zeros(  (size(r)[2]) )
        Eₛ_total = 0.0

        for j in 2:size(r)[2]   # loop from CMB to surface
            layer_volume = 4π/3 * (r[end,j]^3 - r[1,j]^3)

            for i in 1:size(r)[1]-1

                dr = (r[i+1, j] - r[i, j])
                dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

                # Dissipated energy per unit volume
                Eₛ_vol[:,:,i, j] =  ( sum(σs[:,:,1:3,i,j] .* conj.(ϵs[:,:,1:3,i,j]), dims=3) .- sum(conj.(σs[:,:,1:3,i,j]) .* ϵs[:,:,1:3,i,j], dims=3) ) * 1im 
                Eₛ_vol[:,:,i, j] += 2( sum(σs[:,:,4:6,i,j] .* conj.(ϵs[:,:,4:6,i,j]), dims=3) .- sum(conj.(σs[:,:,4:6,i,j]) .* ϵs[:,:,4:6,i,j], dims=3) ) * 1im 
                Eₛ_vol[:,:,i, j] .*= -0.25ω

                # # Integrate across r to find dissipated energy per unit area
                # Eₛ_area[:,:] .+= Eₛ_vol[:,:, i, j] * dr

                Eₛ_vol_layer_sph_avg[j] += sum(sin.(clats) .* (Eₛ_vol[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr
                # Eₛ_total += sum(sin.(clats) .* (Eₛ_vol[:,:,i,j] * dr)  * dres^2 * r[i,j]^2.0) 
            end

            Eₛ_vol_layer_sph_avg[j] /= layer_volume
        end

        return Eₛ_vol_layer_sph_avg
    end

    function get_Ic(r, ρ, g, μ, type, M=6, N=3)
        Ic = zeros(precc, M, N)

        if type=="liquid"
            Ic[1,1] = -r^n / g
            Ic[1,3] = 1.0
            Ic[2,2] = 1.0
            Ic[3,3] = g*ρ
            Ic[5,1] = r^n
            Ic[6,1] = 2(n-1)*r^(n-1)
            Ic[6,3] = 4π * G * ρ 
        else # incompressible solid core
            # First column
            Ic[1, 1] = n*r^( n+1 ) / ( 2*( 2n + 3) )
            Ic[2, 1] = ( n+3 )*r^( n+1 ) / ( 2*( 2n+3 ) * ( n+1 ) )
            Ic[3, 1] = ( n*ρ*g*r + 2*( n^2 - n - 3)*μ ) * r^n / ( 2*( 2n + 3) )
            Ic[4, 1] = n *( n+2 ) * μ * r^n / ( ( 2n + 3 )*( n+1 ) )
            Ic[6, 1] = 2π*G*ρ*n*r^( n+1 ) / ( 2n + 3 )

            # Second column
            Ic[1, 2] = r^( n-1 )
            Ic[2, 2] = r^( n-1 ) / n
            Ic[3, 2] = ( ρ*g*r + 2*( n-1 )*μ ) * r^( n-2 )
            Ic[4, 2] = 2*( n-1 ) * μ * r^( n-2 ) / n
            Ic[6, 2] = 4π*G*ρ*r^( n-1 )

            # Third column
            Ic[3, 3] = -ρ * r^n
            Ic[5, 3] = -r^n
            Ic[6, 3] = -( 2n + 1) * r^( n-1 )

        end

        return Ic
    end

    # inputs:
    #   r: Radii of main layers (core, mantle, crust, etc)
    #   nr: number of sublayers to discretize the main layers with (TODO: make nr an array)
    function expand_layers(r; nr::Int=80)
        set_nr(nr) # Update nr globally 

        rs = zeros(prec, (nr+1, length(r)-1))
        
        for i in 1:length(r)-1
            rfine = LinRange(r[i], r[i+1], nr+1)
            rs[:, i] .= rfine[1:end] 
        end
    
        return rs
    end

    
end