# Julia code to calculate tidal deformation, Love numbers, and heating
# This version is based on the method outlined in Kamata (2023).
# Author: H. Hay

# μ: solid shear modulus
# ρ: solid density 
# ρₗ: liquid density 
# K: solid bulk modulus 
# Kl: liquid bulk modulus 
# α: Biot's constant 
# λ: Lame's First Parameter
# η: solid viscosity 
# ηₗ: liquid viscosity 
# g: gravity 
# ϕ: porosity 
# k: permeability 
# ω: rotation rate


module TidalLoveNumbers

    using LinearAlgebra
    using DoubleFloats
    using AssociatedLegendrePolynomials
    include("SphericalHarmonics.jl")
    using .SphericalHarmonics
    using BenchmarkTools
    using StaticArrays

    export get_g, get_A!, get_A, get_B_product, get_Ic, get_B, get_B!
    export expand_layers, set_G, calculate_y
    export get_displacement, get_darcy_velocity, get_solution
    export get_total_heating, get_heating_profile
    export define_spherical_grid

    prec = Float64 #BigFloat
    precc = ComplexF64 #Complex{BigFloat}

    # prec = Double64 #BigFloat
    # precc = ComplexDF64 #Complex{BigFloat}

    # prec = BigFloat
    # precc = Complex{BigFloat}

    G = prec(6.6743e-11)
    n = 2

    porous = false

    M = 6 + 2porous         # Matrix size: 6x6 if only solid material, 8x8 for two-phases
    nr = 3000           # Number of sub-layers in each layer (TODO: change to an array)

    TM8 = MArray{Tuple{8, 8}, precc}
    TM6 = MArray{Tuple{6, 6}, precc}

    Abot_p = zeros(precc, 8, 8)
    Amid_p = zeros(precc, 8, 8)
    Atop_p = zeros(precc, 8, 8)

    k6 = zeros(TM6, 4)

    k18 = zeros(precc, 8, 8)
    k28 = zeros(precc, 8, 8)
    k38 = zeros(precc, 8, 8)
    k48 = zeros(precc, 8, 8)

    k16 = zeros(precc, 6, 6)
    k26 = zeros(precc, 6, 6)
    k36 = zeros(precc, 6, 6)
    k46 = zeros(precc, 6, 6)

    # I8 = Matrix{Float64}(I, 8, 8)
    I8 = SMatrix{8,8,precc}(I)
    I6 = SMatrix{6,6,precc}(I)

    Abot = zeros(precc, 6, 6)
    Amid = zeros(precc, 6, 6)
    Atop = zeros(precc, 6, 6)

    clats = 0.0
    lons = 0.0
    Y = 0.0
    dYdθ = 0.0
    dYdϕ = 0.0
    Z = 0.0
    X = 0.0
    res = 0.0

        
    

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

    function get_A(r, ρ, g, μ, K; ω=0.0)
        A = zeros(precc, 6, 6) 
        get_A!(A, r, ρ, g, μ, K, ω=ω)
        return A
    end

    function get_A!(A::Matrix, r, ρ, g, μ, K, λ=nothing; ω=0.0)
        if isnothing(λ)
            λ = K - 2μ/3
        end

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)
        rβ_inv = r_inv * β_inv

        A[1,1] = -2λ * rβ_inv
        A[2,1] = -r_inv
        A[3,1] = 4r_inv * (3K*μ*rβ_inv - ρ*g)       - ω^2 * ρ# 
        A[4,1] = -r_inv * (6K*μ*rβ_inv - ρ*g )
        A[5,1] = 4π * G * ρ
        A[6,1] = 4π*(n+1)*G*ρ*r_inv

        A[1,2] = n*(n+1) * λ * rβ_inv
        A[2,2] = r_inv
        A[3,2] = -n*(n+1)*r_inv * (6K*μ*rβ_inv - ρ*g ) 
        A[4,2] = 2μ*r_inv^2 * (2*n*(n+1)*(λ + μ)*β_inv - 1.0 )    - ω^2 * ρ
        A[6,2] = -4π*n*(n+1)*G*ρ*r_inv

        A[1,3] = β_inv
        A[3,3] = rβ_inv * (-4μ )
        A[4,3] = -λ * rβ_inv
        
        A[2,4] = 1.0 / μ
        A[3,4] = n*(n+1)*r_inv
        A[4,4] = -3r_inv

        A[3,5] = ρ * (n+1)*r_inv
        A[4,5] = -ρ*r_inv
        A[5,5] = -(n+1)r_inv

        A[3,6] = -ρ
        A[5,6] = 1.0
        A[6,6] = (n-1)r_inv

    end

    # Method 2 for matrix propagator: two-phase flow
    function get_A(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        A = zeros(precc, 8, 8)
        get_A!(A, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        return A
    end

    function get_A!(A::Matrix, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        if isinf(Kl) && isinf(K)
            α = 1.0
            comp = false

            # Kd can be finite? 
            λ = Kd .- 2μ/3

            M_inv = 0.0

        else
            λ = Kd .- 2μ/3
            Kₛ = K
            M_inv = 1 / ( K/(α - ϕ + ϕ*K/Kl) )
            comp = true
        end

        get_A!(A, r, ρ, g, μ, Kd, λ) 

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)

        # if ϕ == 0.0
        #     println(K, Kd)
        # end

        if !iszero(ϕ)
            A[1,7] = α * β_inv

            A[3,1] += 1im * k*ρₗ^2 *g^2 * n*(n+1) / (ω*ηₗ) * r_inv^2
            A[3,5] += -(n+1)r_inv * 1im *(k*ρₗ^2*g*n)/(ω*ηₗ) * r_inv                               
            A[3,7] = 1im * (k*ρₗ*g*n*(n+1))/(ω*ηₗ)*r_inv^2 - 4μ*α*β_inv*r_inv 
            A[3,8] =  1im * (k*ρₗ^2*g^2*n*(n+1))/(ω*ηₗ)*r_inv^2 - 4ϕ*ρₗ*g*r_inv 
        
            A[4,7] = 2α*μ*r_inv * β_inv
            A[4,8] = ϕ*ρₗ*g*r_inv 
            
            A[5,8] = 4π*G*ρₗ*ϕ

            A[6,1] += -1im * 4π*G*n*(n+1)*r_inv * (k*ρₗ^2*g/(ω*ηₗ)*r_inv)
            A[6,5] = 1im*4π*n*(n+1)G*(ρₗ)^2*k*r_inv^2 / (ω*ηₗ)  
            A[6,7] = -1im *4π*n*(n+1)G*ρₗ*k*r_inv^2 / ( ω*ηₗ) 
            A[6,8] = 4π*G*(n+1)*r_inv * (ϕ*ρₗ - 1im * n*k*ρₗ^2*g/(ω*ηₗ)*r_inv) 
            
            A[7,1] = ρₗ*g*r_inv * ( 4 - 1im *(k*ρₗ*g*n*(n+1)/(ω*ϕ*ηₗ))*r_inv)  
            A[7,2] = -ρₗ*n*(n+1)*r_inv*g
            A[7,5] = -ρₗ*(n+1)r_inv * (1 - 1im*(k*ρₗ*g*n)/(ω*ϕ*ηₗ)*r_inv)  
            A[7,6] = ρₗ 
            A[7,7] = - 1im*(k*ρₗ*g*n*(n+1))/(ω*ϕ*ηₗ)*r_inv^2
            A[7,8] = -1im*ω*ϕ*ηₗ/k - 4π*G*(ρ - ϕ*ρₗ)*ρₗ + ρₗ*g*r_inv*(4 - 1im*(k*ρₗ*g*n*(n+1))/(ω*ϕ*ηₗ)*r_inv) 
        
            A[8,1] = r_inv*( 1im * k*ρₗ*g*n*(n+1)/(ω*ϕ*ηₗ)*r_inv - α/ϕ * 4μ*β_inv) 
            A[8,2] = α/ϕ * 2n*(n+1)*μ *β_inv * r_inv
            A[8,3] = -α/ϕ * β_inv 
            A[8,5] = -1im * k *ρₗ *n*(n+1) / (ω*ϕ*ηₗ)*r_inv^2 
            A[8,7] = 1im*k*n*(n+1)/(ω*ϕ*ηₗ)*r_inv^2 - 1/ϕ * (M_inv + α^2 * β_inv) # If solid and liquid are compressible, keep the 1/M term
            A[8,8] = 1im * k *ρₗ*g *n*(n+1) / (ω*ϕ*ηₗ)*r_inv^2  - 2r_inv 
        end
        
    end

    function get_B(r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        B = zeros(precc, 8, 8)
        get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        return B
    end

    function get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        get_A!(Abot_p, r1, ρ, g1, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Amid_p, rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Atop_p, r2, ρ, g2, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        
        k18 .= dr * Abot_p 
        k28 .= dr *  (Amid_p .+ 0.5Amid_p *k18) 
        k38 .= dr *  (Amid_p .+ 0.5*Amid_p *k28)
        k48 .= dr *  (Atop_p .+ Atop_p*k38) 

        B .= (I8 + 1.0/6.0 .* (k18 .+ 2*(k28 .+ k38) .+ k48))

    end

    function get_B(r1, r2, g1, g2, ρ, μ, K; ω=0.0)
        B = zeros(precc, 6, 6)
        get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω=ω)
        return B
    end

    function get_B!(B, r1, r2, g1, g2, ρ, μ, K; ω=0.0)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        get_A!(Abot, r1, ρ, g1, μ, K, ω=ω)
        get_A!(Amid, rhalf, ρ, ghalf, μ, K, ω=ω)
        get_A!(Atop, r2, ρ, g2, μ, K, ω=ω)
        
        k16 .= dr * Abot 
        k26 .= dr * Amid * (I + 0.5k16)
        k36 .= dr * Amid * (I + 0.5k26)
        k46 .= dr * Atop * (I + k36) 

        # k6[1] .= dr * Abot 
        # k6[2] .= dr * Amid * (I + 0.5k6[1])
        # k6[3] .= dr * Amid * (I + 0.5k6[2])
        # k6[4] .= dr * Atop * (I + k6[3]) 

        # Abot[:] .= zero(Abot[1])
        # Atop[:] .= zero(Atop[1])
        # Amid[:] .= zero(Amid[1])

        # # display(Abot[1:6,1:6])
        # get_A!(Abot, r1, ρ, g1, μ, K)
        # get_A!(Amid, rhalf, ρ, ghalf, μ, K)
        # get_A!(Atop, r2, ρ, g2, μ, K)
        
        # # display(Abot[1:6,1:6] .- A1)

        # k16 = dr * Abot[1:6,1:6] 
        # k26 = dr * Amid[1:6,1:6] * (I + 0.5k16)
        # k36 = dr * Amid[1:6,1:6] * (I + 0.5k26)
        # k46 = dr * Atop[1:6,1:6] * (I + k36) 

        # println("here")

        # B[1:6,1:6] .= (I + 1.0/6.0 .* (k6[1] .+ 2*(k6[2] .+ k6[3]) .+ k6[4]))
        B[1:6,1:6] .= (I + 1.0/6.0 .* (k16 .+ 2*(k26 .+ k36) .+ k46))

        # return B
    end

    # second method: porous layer
    function get_B_product(r, ρ, g, μ, K, i1=2, iend=nothing)
        # Bprod = zeros(ComplexDF64, length(r), 8, 8)
    
        # B = zeros(ComplexDF64, 8, 8)
        B = zeros(precc, 6, 6)
        # B = I # Set B to the Identity matrix
        # B[7,7] = 0.0
        # Bprod[1,:,:] .= B[:,:]
        for i in 1:6
            B[i,i] = 1
        end

        layer_num = size(r)[2]
        nr = size(r)[1]


        # Bprod = zeros(ComplexDF64, (8, 8, nr-1, layer_num))
        Bprod = zeros(precc, (6, 6, nr-1, layer_num))

        r1 = r[1]
        if isnothing(iend)
            iend = layer_num
        end

        for i in i1:iend # start at the top of the innermost layer
            r1 = r[1,i]
            for j in 1:nr-1
                r2 = r[j+1,i]
                dr = r2 - r1
                rhalf = r1 + 0.5dr

                r2 = r[j+1, i]
                g1 = g[j, i]
                g2 = g[j+1, i]

                B = get_B(r1, r2, g1, g2, ρ[i], μ[i], K[i]) * B 
                
                Bprod[:,:,j,i] .= B[:,:]

                r1 = r2
            end
        end

        return Bprod
    end


    # second method: porous layer
    function get_B_product(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, i1=2, iend=nothing)
        # Bprod = zeros(ComplexDF64, length(r), 8, 8)
    
        # B = zeros(ComplexDF64, 8, 8)
        B = zeros(precc, 8, 8)
        # B = I # Set B to the Identity matrix
        # B[7,7] = 0.0
        # Bprod[1,:,:] .= B[:,:]
        for i in 1:6
            B[i,i] = 1
        end

        # if starting from a porous layer, 
        # don't filter out y7 and y8 components
        if ϕ[i1]>0
            B[7,7] = 1
            B[8,8] = 1
        end

        layer_num = size(r)[2]
        nr = size(r)[1]


        # Bprod = zeros(ComplexDF64, (8, 8, nr-1, layer_num))
        Bprod = zeros(precc, (8, 8, nr-1, layer_num))

        r1 = r[1]
        if isnothing(iend)
            iend = layer_num
        end

        # Pδ = zeros(Int64, 8, 8)
        Pδ = zeros(BigInt, 8, 8)
        Pδ[1,1] = 1
        Pδ[2,2] = 1
        Pδ[3,3] = 1
        Pδ[4,4] = 1
        Pδ[5,5] = 1
        Pδ[6,6] = 1

        for i in i1:iend # start at the top of the innermost layer
            r1 = r[1,i]
            for j in 1:nr-1
                r2 = r[j+1,i]
                dr = r2 - r1
                rhalf = r1 + 0.5dr

                r2 = r[j+1, i]
                g1 = g[j, i]
                g2 = g[j+1, i]

                if ϕ[i] > 0 && j>1
                    Pδ[7,7] = 1
                    Pδ[8,8] = 1
                else
                    Pδ[7,7] = 0
                    Pδ[8,8] = 0
                end

                # In the first integration, don't filter out 
                # y7 and y8. Better way to do this? 
                if i==i1 && j==1
                    Pδ[7,7] = 1
                    Pδ[8,8] = 1
                end
    
                B = get_B(r1, r2, g1, g2, ρ[i], μ[i], K[i], ω, ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i]) * B * Pδ
               
                Bprod[:,:,j,i] .= B[:,:]

                r1 = r2
            end
        end

        return Bprod
    end

    # second method: porous layer -- for a specific layer?
    function get_B_product2!(Bprod2, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        # Check dimensions of Bprod2

        nr = size(r)[1]

        Bstart = zeros(precc, 8, 8)
        B = zeros(precc, 8, 8)

        for i in 1:6
            Bstart[i,i,1] = 1
        end

        # if layer is porous, 
        # don't filter out y7 and y8 components
        if ϕ>0
            Bstart[7,7,1] = 1
            Bstart[8,8,1] = 1   # Should this be a 1 or zero?
        end

        r1 = r[1]
        g1 = g[1]
        for j in 1:nr-1
            r2 = r[j+1]
            g2 = g[j+1]

            if ϕ>0 
                get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
            else
                get_B!(B, r1, r2, g1, g2, ρ, μ, K)
            end

            Bprod2[:,:,j] .= B * (j==1 ? Bstart : @view(Bprod2[:,:,j-1])) 

            r1 = r2
            g1 = g2 
        end
    end

    # first method: solid layer -- for a specific layer?
    function get_B_product2!(Bprod2, r, ρ, g, μ, K; ω=0.0)
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

            get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω=ω)
            Bprod2[:,:,j] .= B * (j==1 ? Bstart : @view(Bprod2[:,:,j-1]))
            # @inline Bprod2[:,:,j] .= get_B(r1, r2, g1, g2, ρ, μ, K) * (j==1 ? Bstart : @view(Bprod2[:,:,j-1]))

            r1 = r2
        end
    end

    # Create R and S vectors?
    function set_sph_expansion(res=5.0)


    end

    function get_solution(y, n, m, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, ecc)
        R = r[end,end]

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)
        d_disp = zero(disp)
        
        disps = zero(disp)
        d_disps = zero(d_disp)
        σs = zero(σ)
        ϵs = zero(ϵ)
        ps = zero(p)
    

        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        
        for x in 1:length(ms)
            m = ms[x]
            for i in 2:size(r)[2] # Loop of layers
                ηlr = ηₗ[i]
                ρlr = ρₗ[i]
                ρr = ρ[i]
                kr  = k[i]
                Klr = Kl[i]
                Kr = K[i]
                μr = μ[i]
                ϕr = ϕ[i]
                Kdr = Kd[i]
                αr = α[i]

                for j in 1:size(r)[1]-1 # Loop over sublayers 
                    yrr = ComplexF64.(y[:,j,i])
                    (y1, y2, y3, y4, y5, y6, y7, y8) = yrr

                    rr = r[j,i]
                    gr = g[j,i]
                    
                    if ϕ[i] > 0 
                        compute_darcy_displacement!(@view(d_disp[:,:,:,j,i]), yrr, m, rr, ω, ϕr, ηlr, kr, gr, ρlr)
                        compute_pore_pressure!(@view(p[:,:,j,i]), yrr, m)
                    end
                    compute_displacement!(@view(disp[:,:,:,j,i]), yrr, m)
                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
                    compute_stress_ten!(@view(σ[:,:,:,j,i]), @view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
                end
            end

            disps .+= forcing[x]*disp
            ϵs .+= forcing[x]*ϵ
            σs .+= forcing[x]*σ
            d_disps .+= forcing[x]*d_disp
            ps .+= forcing[x]*p
        end

        return disps, ϵs, σs, ps, d_disps
    end

    function get_solution(y, n, m, r, ρ, g, μ, K, ω, ecc)
        R = r[end,end]

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)
        
        disps = zero(disp)
        σs = zero(σ)
        ϵs  = zero(ϵ)

        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        
        for x in 1:length(ms)
            m = ms[x]
            for i in 2:size(r)[2] # Loop of layers
                ρr = ρ[i]
                Kr = K[i]
                μr = μ[i]

                for j in 1:size(r)[1]-1 # Loop over sublayers 
                    yrr = ComplexF64.(y[:,j,i])
                    (y1, y2, y3, y4, y5, y6) = yrr
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_displacement!(@view(disp[:,:,:,j,i]), yrr, m)
                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr)
                    compute_stress_ten!(@view(σ[:,:,:,j,i]), @view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr)
                end
            end

            disps .+= forcing[x]*disp
            ϵs .+= forcing[x]*ϵ
            σs .+= forcing[x]*σ
        end

        return disp, ϵs, σs
    end

    


    # function get_darcy_velocity(y, mag, k, r, ηₗ, ρₗ, ωt=0.0, res=5.0,n=2, m=2)
    #     lons = deg2rad.(collect(0:res:360-0.001))'
    #     clats = deg2rad.(collect(0:res:180))
    #     cosTheta = cos.(clats)

    #     Y22 = Ynm(2,2,clats,lons)
    #     S22 = Snm(2,2,clats,lons)
        
    #     Y20 = Ynm(2,0,clats,lons) 
    #     S20 = Snm(2,0,clats,lons)
        
    #     # Need to take conjugate in y here depending on whether the Fourier
    #     # transform is taken with exp(-iωt) or exp(iωt)
    #     # Also need negative sign due to the sign convention of the tidal potential
    #     # U22 = -0.5*ω^2*R^2*e*(Y22 * (7/8 * exp(-1im *ωt) )) * conj.(y[1,end,end])
    #     U22 = -0.5*mag*(Y22 * (7/8 * exp(-1im *ωt) - 1/8 *exp(1im *ωt) ))
    #     U20 = -0.5*mag* -1.5(Y20 * exp(-1im * ωt) )

    #     U22_theta = -0.5*mag*(S22[1] * (7/8 * exp(-1im *ωt) - 1/8 *exp(1im *ωt) ))
    #     U22_phi = -0.5*mag*(S22[2] * (7/8 * exp(-1im *ωt) - 1/8 *exp(1im *ωt) ))
    #     U20_theta = -0.5*mag* -1.5(S20[1] * exp(-1im * ωt) )
    #     U20_phi = -0.5*mag* -1.5(S20[2] * exp(-1im * ωt) )

    #     q_R22 =  U22 * conj.(y[8,end-10,end-1]) * -k[end-1]/ηₗ[end-1]
    #     q_R22 .+= conj.(q_R22)

    #     q_R20 = U20 * conj.(y[8,end-10,end-1]) * -k[end-1]/ηₗ[end-1]
    #     q_R20 .+= conj.(q_R20) 

    #     q_S22_theta = U22_theta  * -k[end-1]/ηₗ[end-1] * 1/r * ( conj.(y[7,end,end-1]) .+ ρₗ[end-1]*conj.(y[5,end,end-1])) 
    #     q_S22_theta .+= conj.(q_S22_theta)

    #     q_S22_phi = U22_phi * -k[end-1]/ηₗ[end-1] * 1/r * ( conj.(y[7,end,end-1]) .+ ρₗ[end-1]*conj.(y[5,end,end-1]))
    #     q_S22_phi .+= conj.(q_S22_phi)

    #     q_S20_theta = U20_theta * -k[end-1]/ηₗ[end-1] * 1/r * ( conj.(y[7,end,end-1]) .+ ρₗ[end-1]*conj.(y[5,end,end-1]))
    #     q_S20_theta .+= conj.(q_S20_theta)

    #     q_S20_phi = U20_phi * -k[end-1]/ηₗ[end-1] * 1/r * ( conj.(y[7,end,end-1]) .+ ρₗ[end-1]*conj.(y[5,end,end-1]))
    #     q_S20_phi .+= conj.(q_S20_phi)

    #     # Return drops the imaginary component, which should be zero anyway. Add check?
    #     # Radial component, theta component, phi component

    #     q_vec = hcat(real.(q_R22 + q_R20), real(q_S22_theta + q_S20_theta), real(q_S22_phi + q_S20_phi))  

    #     return reshape(q_vec, (length(clats), length(lons), 3) )   
    # end

    

    function calculate_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, core="liquid")
        porous_layer = ϕ .> 0.0

        sum(porous_layer) > 1.0 && error("Can only handle one porous layer for now!")

        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core, 8, 4)

        y1_4 = zeros(precc, 8, 4, size(r)[1]-1, size(r)[2]) # Four linearly independent y solutions
        y = zeros(ComplexF64, 8, size(r)[1]-1, size(r)[2])  # Final y solutions to return
        # y_start = zeros(precc, 8, 4)                        # Starting y vector 
        
        for i in 2:nlayers
            Bprod = zeros(precc, 8, 8, nsublayers-1)
            get_B_product2!(Bprod, r[:,i], ρ[i], g[:,i], μ[i], K[i], ω, ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i])

            # Modify starting vector if the layer is porous
            # If a new porous layer (i.e., sitting on a non-porous layer)
            # reset the pore pressure and darcy flux
            if porous_layer[i] && !porous_layer[i-1]
                y_start[7,4] = 1.0          # Non-zero pore pressure
                y_start[8,4] = 0.0          # Zero radial Darcy flux
            elseif !porous_layer[i]
                y_start[7:8, :] .= 0.0      # Pore pressure and flux undefined
            end

            for j in 1:nsublayers-1
                y1_4[:,:,j,i] = Bprod[:,:,j] * y_start 
            end

            y_start[:,:] .= y1_4[:,:,end,i]

        end

        M = zeros(precc, 4,4)

        # Row 1 - Radial Stress
        M[1, :] .= y1_4[3,:,end,end]

        # # Row 2 - Tangential Stress
        M[2, :] .= y1_4[4,:,end,end]
    
        # # Row 3 - Potential Stress
        M[3, :] .= y1_4[6,:,end,end]
        
        #  Row 4 - Darcy flux (r = r_tp)
        for i in 2:nlayers
            if porous_layer[i]
                M[4, :] .= y1_4[8,:,end,i]
            end
        end

        b = zeros(precc, 4)
        b[3] = (2n+1)/r[end,end] 
        C = M \ b

        # Combine the linearly independent solutions
        # to get the solution vector in each sublayer
        for i in 2:nlayers
            for j in 1:nsublayers-1
                y[:,j,i] = y1_4[:,:,j,i]*C
            end
        end

        return y
    end

    function calculate_y(r, ρ, g, μ, K, core="liquid"; ω=0.0)

        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core, 6, 3)

        y1_4 = zeros(precc, 6, 3, size(r)[1]-1, size(r)[2]) # Three linearly independent y solutions
        y = zeros(ComplexF64, 6, size(r)[1]-1, size(r)[2])

        # y_start[:,:] .= Ic[:,:]
        
        for i in 2:nlayers
            Bprod = zeros(precc, 6, 6, nsublayers-1)
            get_B_product2!(Bprod, r[:, i], ρ[i], g[:, i], μ[i], K[i], ω=ω)

            for j in 1:nsublayers-1
                y1_4[:,:,j,i] = Bprod[:,:,j] * y_start #y1_4[:,:,1,i]
            end

            y_start[:,:] .= y1_4[:,:,end,i]   # Set starting vector for next layer
        end

        M = zeros(precc, 3,3)

        # Row 1 - Radial Stress
        M[1, :] .= y1_4[3,:,end,end]

        # Row 2 - Tangential Stress
        M[2, :] .= y1_4[4,:,end,end]
    
        # Row 3 - Potential Stress
        M[3, :] .= y1_4[6,:,end,end]
         
        b = zeros(precc, 3)
        b[3] = (2n+1)/r[end,end] 
        C = M \ b

        for i in 2:nlayers
            for j in 1:nsublayers-1
                y[:,j,i] = y1_4[:,:,j,i]*C
            end
        end

        return y
    end

    function get_Ic(r, ρ, g, μ, type, M=6, N=3)
        # Ic = zeros(Double64, M, N)
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

    # Get the total heating rate across the entire body
    function get_total_heating(y, ω, R, ecc)
        k2 = y[5, end,end] - 1.0    # Get k2 Love number at surface
        total_power = -21/2 * imag(k2) * (ω*R)^5/G * ecc^2

        return total_power
    end
    
    # Get a radial profile of the volumetric heating rate for 
    # solid-body tides
    function get_heating_profile(y, r, ρ, g, μ, κ, ω, ecc)
        dres = deg2rad(res)
        λ = κ .- 2μ/3
        R = r[end,end]

        @views clats = TidalLoveNumbers.clats[:]
        @views lons = TidalLoveNumbers.lons[:]
        cosTheta = cos.(clats)

        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        ϵs = zero(ϵ)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        for x in 1:length(ms)
            m = ms[x]

            @views Y    = TidalLoveNumbers.Y[x,:,:]
            @views dYdθ = TidalLoveNumbers.dYdθ[x,:,:]
            @views dYdϕ = TidalLoveNumbers.dYdϕ[x,:,:]
            @views Z    = TidalLoveNumbers.Z[x,:,:]
            @views X    = TidalLoveNumbers.X[x,:,:]

            for i in 2:size(r)[2] # Loop of layers
                ρr = ρ[i]
                κr = κ[i]
                μr = μ[i]
                λr = λ[i]

                for j in 1:size(r)[1]-1 # Loop over sublayers 
                    yrr = conj.(y[1:6,j,i])
                    (y1, y2, y3, y4, y5, y6) = yrr
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, κr)

                end
            end

            ϵs .+= forcing[x]*ϵ
        end

        Eμ = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        Eμ_layer_sph_avg = zeros(  (size(r)[2]) )
        Eμ_layer_sph_avg_rr = zeros(  size(r) )

        Eκ = zero(Eμ)
        Eκ_layer_sph_avg = zero( Eμ_layer_sph_avg )
        Eκ_layer_sph_avg_rr = zero( Eμ_layer_sph_avg_rr )

        for j in 2:size(r)[2]   # loop from CMB to surface
            layer_volume = 4π/3 * (r[end,j]^3 - r[1,j]^3)

            for i in 1:size(r)[1]-1

                dr = (r[i+1, j] - r[i, j])
                dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

                Eμ[:,:,i, j] = sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                Eμ[:,:,i, j] .*= ω * imag(μ[j])
    
                Eκ[:,:,i, j] = ω/2 *imag(κ[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
    
                # Integrate across r to find dissipated energy per unit area
                Eμ_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eμ_layer_sph_avg[j] += Eμ_layer_sph_avg_rr[i,j]*dvol
    
                Eκ_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (Eκ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eκ_layer_sph_avg[j] += Eκ_layer_sph_avg_rr[i,j]*dvol
            end

            Eμ_layer_sph_avg[j] /= layer_volume
            Eκ_layer_sph_avg[j] /= layer_volume
        end

        return (Eμ_layer_sph_avg, Eμ_layer_sph_avg_rr), 
                (Eκ_layer_sph_avg, Eκ_layer_sph_avg_rr) 
    end

    # Get a radial profile of the volumetric heating rate
    function get_heating_profile(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, ecc)
        dres = deg2rad(res)
        λ = Ks .- 2μ/3
        R = r[end,end]

        @views clats = TidalLoveNumbers.clats[:]
        @views lons = TidalLoveNumbers.lons[:]

        cosTheta = cos.(clats)

        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        d_disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])

        ϵs = zero(ϵ)
        d_disps = zero(d_disp)
        ps = zero(p)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        for x in 1:length(ms)
            m = ms[x]

            @views Y    = TidalLoveNumbers.Y[x,:,:]
            @views dYdθ = TidalLoveNumbers.dYdθ[x,:,:]
            @views dYdϕ = TidalLoveNumbers.dYdϕ[x,:,:]
            @views Z    = TidalLoveNumbers.Z[x,:,:]
            @views X    = TidalLoveNumbers.X[x,:,:]

            for i in 2:size(r)[2] # Loop of layers
                ρr = ρ[i]
                Ksr = Ks[i]
                μr = μ[i]
                λr = λ[i]
                ρlr = ρl[i]
                Klr = Kl[i]
                Kdr = Kd[i]
                αr = α[i]
                ηlr = ηl[i]
                ϕr = ϕ[i]
                kr = k[i]

                for j in 1:size(r)[1]-1 # Loop over sublayers 
                    yrr = conj.(y[:,j,i])
                    (y1, y2, y3, y4, y5, y6, y7, y8) = yrr
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)

                    if ϕ[i] > 0 
                        compute_darcy_displacement!(@view(d_disp[:,:,:,j,i]), yrr, m, rr, ω, ϕr, ηlr, kr, gr, ρlr)
                        compute_pore_pressure!(@view(p[:,:,j,i]), yrr, m)
                    end

                    p[:,:,j,i] .= y7 * Y    # pore pressure

                end
            end

            ϵs .+= forcing[x]*ϵ
            d_disps .+= forcing[x]*d_disp
            ps .+= forcing[x]*p
        end

        # Shear heating in the solid
        Eμ = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        Eμ_layer_sph_avg = zeros(  (size(r)[2]) )
        Eμ_layer_sph_avg_rr = zeros(  size(r) )

        # Darcy dissipation in the liquid
        El = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        El_layer_sph_avg = zeros(  (size(r)[2]) )
        El_layer_sph_avg_rr = zeros(  size(r) )

        # Bulk dissipation in the solid
        Eκ = zero(Eμ)
        Eκ_layer_sph_avg = zero( Eμ_layer_sph_avg )
        Eκ_layer_sph_avg_rr = zero( Eμ_layer_sph_avg_rr )

        for j in 2:size(r)[2]   # loop from CMB to surface
            layer_volume = 4π/3 * (r[end,j]^3 - r[1,j]^3)

            for i in 1:size(r)[1]-1
                dr = (r[i+1, j] - r[i, j])
                dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

                Eμ[:,:,i, j] = sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                Eμ[:,:,i, j] .*= ω * imag(μ[j])

                Eκ[:,:,i, j] = ω/2 *imag(Kd[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                if ϕ[j] > 0
                    Eκ[:,:,i, j] .+= ω/2 *imag(Kd[j]) * (abs.(ps[:,:,i,j]) ./ Ks[j]).^2
                end

                # Integrate across r to find dissipated energy per unit area
            
                Eμ_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eμ_layer_sph_avg[j] += Eμ_layer_sph_avg_rr[i,j]*dvol

                Eκ_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (Eκ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eκ_layer_sph_avg[j] += Eκ_layer_sph_avg_rr[i,j]*dvol
        
                if ϕ[j] > 0            
                    El[:,:,i, j] = 0.5 *  ϕ[j]^2 * ω^2 * ηl[j]/k[j] * (abs.(d_disps[:,:,1,i,j]).^2 + abs.(d_disps[:,:,2,i,j]).^2 + abs.(d_disps[:,:,3,i,j]).^2)
                    El_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (El[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                    El_layer_sph_avg[j] += El_layer_sph_avg_rr[i,j]*dvol

                end

            end

            Eμ_layer_sph_avg[j] /= layer_volume
            Eκ_layer_sph_avg[j] /= layer_volume
            El_layer_sph_avg[j] /= layer_volume
        end

        return (Eμ_layer_sph_avg, Eμ_layer_sph_avg_rr), 
                (Eκ_layer_sph_avg, Eκ_layer_sph_avg_rr), 
                (El_layer_sph_avg, El_layer_sph_avg_rr) 
    end

    function define_spherical_grid(res; n=2)
        TidalLoveNumbers.res = res

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))
        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        TidalLoveNumbers.Y    = zeros(ComplexF64, 3, length(clats), length(lons))
        TidalLoveNumbers.dYdθ = zero(TidalLoveNumbers.Y)
        TidalLoveNumbers.dYdϕ = zero(TidalLoveNumbers.Y)
        TidalLoveNumbers.Z    = zero(TidalLoveNumbers.Y)
        TidalLoveNumbers.X    = zero(TidalLoveNumbers.Y)

        ms = [-2, 0, 2]
        for i in 1:length(ms)
            m = ms[i]
            TidalLoveNumbers.Y[i,:,:] .= m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)

            if iszero(abs(m))
                TidalLoveNumbers.dYdθ[i,:,:] = -1.5sin.(2clats) * exp.(1im * m * lons)
                TidalLoveNumbers.dYdϕ[i,:,:] = TidalLoveNumbers.Y[i,:,:] * 1im * m

                TidalLoveNumbers.Z[i,:,:] = 0.0 * TidalLoveNumbers.Y[i,:,:]
                TidalLoveNumbers.X[i,:,:] = -6cos.(2clats)*exp.(1im *m * lons) .+ n*(n+1)*TidalLoveNumbers.Y[i]

            elseif  abs(m) == 2
                TidalLoveNumbers.dYdθ[i,:,:] = 3sin.(2clats) * exp.(1im * m * lons)
                TidalLoveNumbers.dYdϕ[i,:,:] = Y[i,:,:] * 1im * m
                
                TidalLoveNumbers.Z[i,:,:] = 6 * 1im * m * cos.(clats) * exp.(1im * m * lons)
                TidalLoveNumbers.X[i,:,:] = 12cos.(2clats)* exp.(1im * m * lons) .+ n*(n+1)*Y[i,:,:] 
            end

        end

        TidalLoveNumbers.clats = clats;
        TidalLoveNumbers.lons = lons;

    end

    function compute_strain_ten!(ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]
        @views dYdθ = TidalLoveNumbers.dYdθ[i,:,:]
        @views dYdϕ = TidalLoveNumbers.dYdϕ[i,:,:]
        @views Z    = TidalLoveNumbers.Z[i,:,:]
        @views X    = TidalLoveNumbers.X[i,:,:]

        A = get_A(rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
        dy1dr = dot(A[1,:], y[:])
        dy2dr = dot(A[2,:], y[:])

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]
        y7 = y[7]

        λr = Kdr - 2μr/3
        βr = λr + 2μr

        # Compute strain tensor
        # ϵ[:,:,1] = dy1dr * Y
        ϵ[:,:,1] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3 + rr*αr*y7)/(βr*rr)  * Y
        ϵ[:,:,2] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
        ϵ[:,:,3] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
        # ϵ[:,:,4] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdθ
        # ϵ[:,:,5] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,4] = 0.5/μr * y4 * dYdθ        
        ϵ[:,:,5] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,6] = 0.5 * y2/rr * Z
    end

    function compute_stress_ten!(σ, ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y7 = y[7]

        λr = Ksr .- 2μr/3

        ϵV = sum(ϵ[:,:,1:3], dims=3)   # trace of strain ten
        F = (λr * ϵV .- αr*y7) .* Y
        σ[:,:,1] .= F .+ 2μr*ϵ[:,:,1]  
        σ[:,:,2] .= F .+ 2μr*ϵ[:,:,2] 
        σ[:,:,3] .= F .+ 2μr*ϵ[:,:,3] 
        σ[:,:,4] .= 2μr * ϵ[:,:,4]
        σ[:,:,5] .= 2μr * ϵ[:,:,5]
        σ[:,:,6] .= 2μr * ϵ[:,:,6]
    end

    function compute_strain_ten!(ϵ, y, m, rr, ρr, gr, μr, Ksr)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]
        @views dYdθ = TidalLoveNumbers.dYdθ[i,:,:]
        @views dYdϕ = TidalLoveNumbers.dYdϕ[i,:,:]
        @views Z    = TidalLoveNumbers.Z[i,:,:]
        @views X    = TidalLoveNumbers.X[i,:,:]

        # A = get_A(rr, ρr, gr, μr, Ksr)
        # dy1dr = dot(A[1,:], y[:])
        # dy2dr = dot(A[2,:], y[:])

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
        # ϵ[:,:,4] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdθ
        # ϵ[:,:,5] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,4] = 0.5/μr * y4 * dYdθ        
        ϵ[:,:,5] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,6] = 0.5 * y2/rr * Z
    end

    function compute_stress_ten!(σ, ϵ, y, m, rr, ρr, gr, μr, κsr)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]

        y1 = y[1]
        y2 = y[2]

        λr = κsr .- 2μr/3

        ϵV = sum(ϵ[:,:,1:3], dims=3)   # trace of strain ten
        F = (λr * ϵV) .* Y
        σ[:,:,1] .= F .+ 2μr*ϵ[:,:,1]  
        σ[:,:,2] .= F .+ 2μr*ϵ[:,:,2] 
        σ[:,:,3] .= F .+ 2μr*ϵ[:,:,3] 
        σ[:,:,4] .= 2μr * ϵ[:,:,4]
        σ[:,:,5] .= 2μr * ϵ[:,:,5]
        σ[:,:,6] .= 2μr * ϵ[:,:,6]
    end

    function compute_displacement!(dis, y, m)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]
        @views dYdθ = TidalLoveNumbers.dYdθ[i,:,:]
        @views dYdϕ = TidalLoveNumbers.dYdϕ[i,:,:]

        y1 = y[1]
        y2 = y[2]

        dis[:,:,1] =  Y * y1
        dis[:,:,2] = dYdθ * y2
        dis[:,:,3] = dYdϕ * y2 ./ sin.(clats)
    end

    function compute_darcy_displacement!(dis_rel, y, m, r, ω, ϕ, ηl, k, g, ρl)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]
        @views dYdθ = TidalLoveNumbers.dYdθ[i,:,:]
        @views dYdϕ = TidalLoveNumbers.dYdϕ[i,:,:]
        
        y1 = y[1]
        y5 = y[5]
        y7 = y[7]
        y8 = y[8]
        y9 = 1im * k / (ω*ϕ*ηl*r) * (ρl*g*y1 - ρl * y5 + ρl*g*y8 + y7)
        
        dis_rel[:,:,1] = Y * y8 
        dis_rel[:,:,2] = dYdθ * y9
        dis_rel[:,:,3] = dYdϕ * y9 ./ sin.(clats)
    end

    function compute_pore_pressure!(p, y, m)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        @views Y    = TidalLoveNumbers.Y[i,:,:]
        @views dYdθ = TidalLoveNumbers.dYdθ[i,:,:]
        @views dYdϕ = TidalLoveNumbers.dYdϕ[i,:,:]
        
        y7 = y[7]
        
        p[:,:] = Y * y7 
    end


end