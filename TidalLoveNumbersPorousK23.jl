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

    # prec = Float64 #BigFloat
    # precc = ComplexF64 #Complex{BigFloat}

    prec = Double64 #BigFloat
    precc = ComplexDF64 #Complex{BigFloat}

    # prec = BigFloat
    # precc = Complex{BigFloat}

    G = prec(6.6743e-11)
    n = 2

    porous = false

    M = 6 + 2porous         # Matrix size: 6x6 if only solid material, 8x8 for two-phases
    nr = 3000           # Number of sub-layers in each layer (TODO: change to an array)

    # α = 0.95

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

    # Overwrite Gravitional constant for non-dimensional 
    # calculations
    function set_G(new_G)
        TidalLoveNumbers.G = new_G
    end

    function set_nr(new_nr)
        TidalLoveNumbers.nr = new_nr
    end

    function get_g(r, ρ)
        # g = zeros(Double64, size(r))
        # M = zeros(Double64, size(r))

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

            # if abs.(imag.(Kd)) > 0.0
            #     α = Kd/K
            # else
            #     α = 1 - Kd/K
            # end

            # Kd = (1-α)* K

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

        # A1 = get_A(r1, ρ, g1, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        # Ahalf = get_A(rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        # A2 = get_A(r2, ρ, g2, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        # k1 = dr * A1 
        # k2 = dr * Ahalf * (I + 0.5k1)
        # k3 = dr * Ahalf * (I + 0.5k2)
        # k4 = dr * A2 * (I + k3) 
        # mul

        # Abot_p[:] .= zero(Abot_p[1])
        # Atop_p[:] .= zero(Atop_p[1])
        # Amid_p[:] .= zero(Amid_p[1])

        get_A!(Abot_p, r1, ρ, g1, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Amid_p, rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Atop_p, r2, ρ, g2, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        
        # k18 = dr * Abot_p 
        # k28 = dr * Amid_p * (I8 + 0.5*k18) # dr*Amid_p + 0.5dr*Amid_p*k188
        # k38 = dr * Amid_p * (I8 + 0.5*k28)
        # k48 = dr * Atop_p * (I8 + k38) 

        k18 .= dr * Abot_p 
        k28 .= dr *  (Amid_p .+ 0.5Amid_p *k18) # dr*Amid_p + 0.5dr*Amid_p*k188
        k38 .= dr *  (Amid_p .+ 0.5*Amid_p *k28)
        k48 .= dr *  (Atop_p .+ Atop_p*k38) 

        # mul!(k18, Abot_p, I8, dr, 0.0); # k18 = dr * Abot_p * I + 0*k18
        # # mul!(k28, Amid_p, I8)
        # copy!(k28, Amid_p)
        # mul!(k28, Amid_p, k18, 0.5dr, 1.0)
        # # mul!(k38, Amid_p, I8)
        # copy!(k38, Amid_p)
        # mul!(k38, Amid_p, k28, 0.5dr, 1.0)
        # # mul!(k48, Atop_p, I8)
        # copy!(k48, Amid_p)
        # mul!(k48, Atop_p, k38, dr, 1.0)
        
        # mu

        # mu
        # get_A!(Abot, r1, ρ, g1, μ, K, ω, ρₗ, Kl, ηₗ, ϕ, k)
        # get_A!(Amid, rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, ηₗ, ϕ, k)
        # get_A!(Atop, r2, ρ, g2, μ, K, ω, ρₗ, Kl, ηₗ, ϕ, k)
        
        # k1 = dr * Abot 
        # k2 = dr * Amid * (I + 0.5k1)
        # k3 = dr * Amid * (I + 0.5k2)
        # k4 = dr * Atop * (I + k3) 

        # B .= (I + dr/6.0 .* (k18 .+ 2k28 .+ 2k38 .+ k48))
        # mul!(B, I8 + dr/6.0 .* (k18 .+ 2k28 .+ 2k38 .+ k48), I8)
        B .= (I8 + 1.0/6.0 .* (k18 .+ 2*(k28 .+ k38) .+ k48))

        # B .= (I + dr/6.0 .* (Abot_p .+ 2 .* Amid_p * (I .+ 0.5*(dr * Abot_p)) .+ 2 .* k3 .+ k4))

        # return B
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

        # A1 = get_A(r1, ρ, g1, μ, K)
        # Ahalf = get_A(rhalf, ρ, ghalf, μ, K)
        # A2 = get_A(r2, ρ, g2, μ, K)
        
        # k1 = dr * A1 
        # k2 = dr * Ahalf * (I + 0.5 * k1)
        # k3 = dr * Ahalf * (I + 0.5 * k2)
        # k4 = dr * A2 * (I + k3) 

        # Abot[:] .= zero(Abot[1])
        # Atop[:] .= zero(Atop[1])
        # Amid[:] .= zero(Amid[1])

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

    function get_solution(y, n, m, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, res=5.0)
        #K is the bulk modulus of the solid! The drained bulk modulus
        # is (1-α)*K

        λ = K .- 2μ/3
        Kₛ = K

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))

        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        Y = m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)
        S = m < 0 ? Snmc(n,abs(m),clats,lons) : Snm(n,abs(m),clats,lons)

        # Better way to do this? (Analytical expression?)
        if iszero(abs(m))
            d2Ydθ2 = -3cos.(2clats) * exp.(1im * m * lons)
            dYdθ = -1.5sin.(2clats) * exp.(1im * m * lons)
            Y = 0.5 *(3cos.(clats).^2 .- 1.0) * exp.(1im * m * lons)
            dYdϕ = Y .* 1im * m

        elseif  abs(m) == 2
            d2Ydθ2 = 6cos.(2clats) * exp.(1im * m * lons)
            dYdθ = 3sin.(2clats) * exp.(1im * m * lons)
            Y = 3 *(1 .- cos.(clats).^2) * exp.(1im * m * lons)
            dYdϕ = Y * 1im * m
            
            Z = 6 * 1im * m * cos.(clats) * exp.(1im * m * lons)
            X = 12cos.(2clats)* exp.(1im * m * lons) .+ n*(n+1)*Y 
        end
        
        
        d2Ydϕ2 = -Y * m^2
    
        X2 = cot.(clats) .* dYdθ .+ (1 ./ sin.(clats).^2) .* d2Ydϕ2
        
        X3 = 1 ./ sin.(clats) .* dYdθ * 1im * m .- cot.(clats) .* 1 ./ sin.(clats) .* dYdϕ
    

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        q_flux = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)
        p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])
        ζ = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])

        for i in 2:size(r)[2] # Loop of layers
            ηlr = ηₗ[i]
            ρlr = ρₗ[i]
            ρr = ρ[i]
            kr  = k[i]
            Klr = Kl[i]
            Kr = K[i]
            μr = μ[i]
            ϕr = ϕ[i]
            λr = λ[i]
            Kdr = Kd[i]
            βr = λr + 2μr


            if ϕr > 0
                Kₛ = K[i]        # 
                α = 1 - Kdr/Kₛ   # not used?
                # Ku = Kd + Klr .*Kₛ .*α^2 ./ (ϕ[i] .* Kₛ + (α-ϕ[i]) .* Klr)
                # Ku = Kdr + Klr .* (Kₛ - Kdr) ./ (ϕr*Kₛ*(Kₛ - Klr) + Klr*(Kₛ - Kdr)  )
                λr = Kdr .- 2μr/3
                βr = λr + 2μr
            end
            # λr = λ[i]

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                (y1, y2, y3, y4, y5, y6, y7, y8) = ComplexF64.(y[:,j,i])
                
                rr = r[j,i]
                gr = g[j,i]
                
                disp[:,:,:,j,i]   .= get_displacement(y1, y2, Y, S)
                if ϕ[i] > 0
                    y9 = 1im * kr / (ω*ϕr*ηlr*rr) * (ρlr*gr*y1 - ρlr * y5 + ρlr*gr*y8 + y7)

                    q_flux[:,:,1,j,i] .= y8 * Y
                    q_flux[:,:,2,j,i] .= y9 * dYdθ
                    q_flux[:,:,3,j,i] .= y9 * dYdϕ .* 1.0 ./ sin.(clats)

                    # q_flux[:,:,:,j,i] .= get_darcy_displacement(y1, y5, y7, y8, rr, ω, ϕr, ηlr, kr, gr, ρlr, Y, S)
                end

                # A = ComplexF64.(get_A(rr, ρr, gr, μr, Kr, ω, ρₗr, Klr, Kdr, ηₗr, ϕr, kr))

                ϵ[:,:,1,j,i] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3)/(βr*rr)  * Y                                                    
                ϵ[:,:,2,j,i] = (y1 * Y .+  y2 * d2Ydθ2)/rr
                ϵ[:,:,3,j,i] = (y1*Y .+ y2*X2)/rr
                # println(size(y4), " ", size(dYdθ))
                ϵ[:,:,4,j,i] = 0.5/μr * y4 * dYdθ
                
                ϵ[:,:,5,j,i] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
                
                ϵ[:,:,6,j,i] = y2/rr * X3
                
                ϵV = (4μr*y1 - 2n*(n+1)μr*y2 + rr*y3)/(βr*rr) 

                σ[:,:,1,j,i] .= λr * ϵV * Y .+ 2μr*ϵ[:,:,1,j,i] 
                σ[:,:,2,j,i] .= λr * ϵV * Y .+ 2μr*ϵ[:,:,2,j,i] 
                σ[:,:,3,j,i] .= λr * ϵV * Y .+ 2μr*ϵ[:,:,3,j,i] 
                σ[:,:,4,j,i] .= 2μr * ϵ[:,:,4,j,i]
                σ[:,:,5,j,i] .= 2μr * ϵ[:,:,5,j,i]
                σ[:,:,6,j,i] .= 2μr * ϵ[:,:,6,j,i]
            end
        end

        return disp, ϵ, σ, p, q_flux, ζ
    end

    function get_solution(y, n, m, r, ρ, g, μ, K, res=10.0)
        #K is the bulk modulus of the solid! The drained bulk modulus
        # is (1-α)*K

        λ = K .- 2μ/3
        # Kₛ = K

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))

        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        Y = m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)
        S = m < 0 ? Snmc(n,abs(m),clats,lons) : Snm(n,abs(m),clats,lons)

        # Better way to do this? (Analytical expression?)
        if iszero(abs(m))
            d2Ydθ2 = -3cos.(2clats) * exp.(1im * m * lons)
            dYdθ = -1.5sin.(2clats) * exp.(1im * m * lons)
            Y = 0.5 *(3cos.(clats).^2 .- 1.0) * exp.(1im * m * lons)
            dYdϕ = Y .* 1im * m

        elseif  abs(m) == 2
            d2Ydθ2 = 6cos.(2clats) * exp.(1im * m * lons)
            dYdθ = 3sin.(2clats) * exp.(1im * m * lons)
            Y = 3 *(1 .- cos.(clats).^2) * exp.(1im * m * lons)
            dYdϕ = Y * 1im * m
        end
        
        
        d2Ydϕ2 = -Y * m^2
    
        X2 = cot.(clats) .* dYdθ .+ (1 ./ sin.(clats).^2) .* d2Ydϕ2
        
        X3 = 1 ./ sin.(clats) .* dYdθ * 1im * m .- cot.(clats) .* 1 ./ sin.(clats) .* dYdϕ
    

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
            # Klr = Kl[i]
            Kr = K[i]
            μr = μ[i]
            # ϕr = ϕ[i]
            λr = λ[i]
            βr = λr + 2μr

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                (y1, y2, y3, y4, y5, y6) = y[:,j,i]
                
                rr = r[j,i]
                gr = g[j,i]
                
                disp[:,:,:,j,i]   .= get_displacement(y1, y2, Y, S)

                # A = get_A(rr, ρr, gr, μr, Kr)
                # dy1dr = dot(A[1,:], y[:,j,i])
                # dy2dr = dot(A[2,:], y[:,j,i])
                
                ϵ[:,:,1,j,i] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3)/(βr*rr)  * Y                                                    
                ϵ[:,:,2,j,i] = (y1 * Y .+  y2 * d2Ydθ2)/rr
                ϵ[:,:,3,j,i] = (y1*Y .+ y2*X2)/rr
                # println(size(y4), " ", size(dYdθ))
                ϵ[:,:,4,j,i] = 0.5/μr * y4 * dYdθ
                
                ϵ[:,:,5,j,i] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
                
                ϵ[:,:,6,j,i] = y2/rr * X3
                
                ϵV = (4μr*y1 - 2n*(n+1)μr*y2 + rr*y3)/(βr*rr) 

                

                # ϵ[:,:,1,j,i] = dy1dr * Y
                # ϵ[:,:,2,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
                # ϵ[:,:,3,j,i] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
                # ϵ[:,:,4,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdθ
                # ϵ[:,:,5,j,i] = 0.5 * (dy2dr + (y1 - y2)/rr) .* dYdϕ .* 1.0 ./ sin.(clats) 
                # ϵ[:,:,6,j,i] = 0.5 * y2/rr * Z
                # ϵV = dy1dr .+ 2/rr * y1 .- n*(n+1)/rr * y2

                

                σ[:,:,1,j,i] .= λr * ϵV * Y .+ 2μr*ϵ[:,:,1,j,i] 
                σ[:,:,2,j,i] .= λr * ϵV * Y .+ 2μr*ϵ[:,:,2,j,i] 
                σ[:,:,3,j,i] .= λr * ϵV * Y .+ 2μr*ϵ[:,:,3,j,i] 
                σ[:,:,4,j,i] .= 2μr * ϵ[:,:,4,j,i]
                σ[:,:,5,j,i] .= 2μr * ϵ[:,:,5,j,i]
                σ[:,:,6,j,i] .= 2μr * ϵ[:,:,6,j,i]

            end
        end
        
        return disp, ϵ, σ
    end

    

    function get_displacement(y1, y2, Y, S)
        # displ_R =  mag_r * -conj.(y1)
        # displ_R .+= conj.(displ_R)

        # Conjugate y for eastward forcing, not westward
        displ_R =  Y * -y1
        # displ_R .+= conj.(displ_R)


        displ_theta = S[1] * -y2
        # displ_theta .+= conj.(displ_theta)

        displ_phi = S[2] * -y2
        # displ_phi .+= conj.(displ_phi)

        # Return drops the imaginary component, which should be zero anyway. Add check?
        # Radial component, theta component, phi component
        displ_vec = hcat(displ_R, displ_theta, displ_phi)  

        return reshape(displ_vec, (size(displ_R)[1], size(displ_R)[2], 3) )  
    end

    function get_darcy_displacement(y1, y5, y7, y8, r, ω, ϕ, ηₗ, k, g,  ρₗ, Y, S)
        q_R =  Y * y8 
        # q_R .+= conj.(q_R)

        f1 = 1im * k*ρₗ*g/(ω*ϕ*ηₗ*r)
        f2 = 1im * k/(ω*ϕ*ηₗ*r)
        f3 = -1im * k*ρₗ/(ω*ϕ*ηₗ*r)

        q_theta = S[1]  * ( f1*y1 .+ f3*y5 .+ f2*y7 .+ f1*y8) 
        # q_theta .+= conj.(q_theta)

        q_phi = S[2] * ( f1*y1 .+ f3*y5 .+ f2*y7 .+ f1*y8) 

        # q_R =  Y * y8 * -k/ηₗ
        # # q_R .+= conj.(q_R)

        # q_theta = S[1]  * -k/ηₗ * 1/r * ( y7 .+ ρₗ*y5) 
        # # q_theta .+= conj.(q_theta)

        # q_phi = S[2] * -k/ηₗ * 1/r * ( y7 .+ ρₗ*y5)
        # q_phi .+= conj.(q_phi)

        q_vec = hcat(q_R, q_theta, q_phi)  

        return reshape(q_vec, (size(q_R)[1], size(q_R)[2], 3) )   
    end


    # function get_displacement(y1, y2)
    #     # lons = deg2rad.(collect(0:res:360-0.001))'
    #     # clats = deg2rad.(collect(0:res:180))
    #     # cosTheta = cos.(clats)

    #     # Y22 = Ynm(2,2,clats,lons)
    #     # S22 = Snm(2,2,clats,lons)
        
    #     # Y20 = Ynm(2,0,clats,lons) 
    #     # S20 = Snm(2,0,clats,lons)
        
    #     # # Need to take conjugate in y here depending on whether the Fourier
    #     # # transform is taken with exp(-iωt) or exp(iωt)
    #     # # Also need negative sign due to the sign convention of the tidal potential
    #     # # U22 = -0.5*ω^2*R^2*e*(Y22 * (7/8 * exp(-1im *ωt) )) * conj.(y[1,end,end])
    #     # U22 = 0.5*mag*(Y22 * (7/8 * exp(-1im *ωt) - 1/8 *exp(1im *ωt) ))
    #     # U20 = 0.5*mag* -1.5(Y20 * exp(-1im * ωt) )

    #     # U22_theta = 0.5*mag*(S22[1] * (7/8 * exp(-1im *ωt) - 1/8 *exp(1im *ωt) ))
    #     # U22_phi = 0.5*mag*(S22[2] * (7/8 * exp(-1im *ωt) - 1/8 *exp(1im *ωt) ))
    #     # U20_theta = 0.5*mag* -1.5(S20[1] * exp(-1im * ωt) )
    #     # U20_phi = 0.5*mag* -1.5(S20[2] * exp(-1im * ωt) )

    #     displ_R22 =  U22 * -conj.(y[1])
    #     displ_R22 .+= conj.(displ_R22)

    #     displ_R20 = U20 * -conj.(y[1])
    #     displ_R20 .+= conj.(displ_R20) 

    #     displ_S22_theta = U22_theta * -conj.(y[2])
    #     displ_S22_theta .+= conj.(displ_S22_theta)

    #     displ_S22_phi = U22_phi * -conj.(y[2])
    #     displ_S22_phi .+= conj.(displ_S22_phi)

    #     displ_S20_theta = U20_theta * -conj.(y[2])
    #     displ_S20_theta .+= conj.(displ_S20_theta)

    #     displ_S20_phi = U20_phi * -conj.(y[2])
    #     displ_S20_phi .+= conj.(displ_S20_phi)

    #     # Return drops the imaginary component, which should be zero anyway. Add check?
    #     # Radial component, theta component, phi component
    #     displ_vec = hcat(real(displ_R22 + displ_R20), real(displ_S22_theta + displ_S20_theta), real(displ_S22_phi + displ_S20_phi))  
        
    #     return reshape(displ_vec, (length(clats), length(lons), 3) )  
    # end

    function get_strain(y)
    end

    function get_stress(y)
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

        # Eccentricity tide forcing coefficients (These should be called from a function)
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
                    (y1, y2, y3, y4, y5, y6) = conj.(y[1:6,j,i])
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    A = get_A(rr, ρr, gr, μr, κr)
                    dy1dr = dot(A[1,:], y[1:6,j,i])
                    dy2dr = dot(A[2,:], y[1:6,j,i])

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

        Eμ = zeros(  (size(σ)[1], size(σ)[2], size(σ)[4], size(σ)[5]) )
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
    
                # # Integrate across r to find dissipated energy per unit area
                # Eμ_area[:,:] .+= Eμ[:,:, i, j] * dr
    
                # Eμ_layer_sph_avg[j] += sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr
                
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
function get_heating_profile(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, ecc; res=20.0)
    dres = deg2rad(res)
    λ = Ks .- 2μ/3
    R = r[end,end]

    lons = deg2rad.(collect(0:res:360-0.001))'
    clats = deg2rad.(collect(0:res:180))

    clats[1] += 1e-6
    clats[end] -= 1e-6
    cosTheta = cos.(clats)

    ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
    σ = zero(ϵ)
    q_flux = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])

    p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])

    ϵs = zero(ϵ)
    σs = zero(ϵ)
    qs = zero(q_flux)
    ps = zero(p)

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
                (y1, y2, y3, y4, y5, y6, y7, y8) = conj.(y[:,j,i])
                
                rr = r[j,i]
                gr = g[j,i]

                A = get_A(rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
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

                if ϕ[i] > 0
                    y9 = 1im * kr / (ω*ϕr*ηlr*rr) * (ρlr*gr*y1 - ρlr * y5 + ρlr*gr*y8 + y7)

                    q_flux[:,:,1,j,i] .= y8 * Y
                    q_flux[:,:,2,j,i] .= y9 * dYdθ
                    q_flux[:,:,3,j,i] .= y9 * dYdϕ .* 1.0 ./ sin.(clats)

                    # q_flux[:,:,:,j,i] .= get_darcy_displacement(y1, y5, y7, y8, rr, ω, ϕr, ηlr, kr, gr, ρlr, Y, S)
                end

                p[:,:,j,i] .= y7 * Y    # pore pressure

            end
        end

        if m==-2
            ϵs .+= U22W*ϵ
            σs .+= U22W*σ
            qs .+= U22W*q_flux
            ps .+= U22W*p
        elseif m==2
            ϵs .+= U22E*ϵ
            σs .+= U22E*σ
            qs .+= U22E*q_flux
            ps .+= U22E*p
        else
            ϵs .+= U20*ϵ
            σs .+= U20*σ
            qs .+= U20*q_flux
            ps .+= U20*p
        end
    end

    # Shear heating in the solid
    Eμ = zeros(  (size(σ)[1], size(σ)[2], size(σ)[4], size(σ)[5]) )
    Eμ_layer_sph_avg = zeros(  (size(r)[2]) )
    Eμ_layer_sph_avg_rr = zeros(  size(r) )

    # Darcy dissipation in the liquid
    El = zeros(  (size(σ)[1], size(σ)[2], size(σ)[4], size(σ)[5]) )
    El_layer_sph_avg = zeros(  (size(r)[2]) )
    El_layer_sph_avg_rr = zeros(  size(r) )

    # Bulk dissipation in the solid
    Eκ = zero(Eμ)
    Eκ_layer_sph_avg = zero( Eμ_layer_sph_avg )
    Eκ_layer_sph_avg_rr = zero( Eμ_layer_sph_avg_rr )

    # Bulk dissipation in the liquid
    ES = zero(Eμ)
    ES_layer_sph_avg = zero( Eμ_layer_sph_avg )
    ES_layer_sph_avg_rr = zero( Eμ_layer_sph_avg_rr )

    for j in 2:size(r)[2]   # loop from CMB to surface
        layer_volume = 4π/3 * (r[end,j]^3 - r[1,j]^3)

        for i in 1:size(r)[1]-1

            dr = (r[i+1, j] - r[i, j])
            dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

            # Dissipated energy per unit volume
            # Eμ_vol[:,:,i, j] =  ( sum(σs[:,:,1:3,i,j] .* conj.(ϵs[:,:,1:3,i,j]), dims=3) .- sum(conj.(σs[:,:,1:3,i,j]) .* ϵs[:,:,1:3,i,j], dims=3) ) * 1im 
            # Eμ_vol[:,:,i, j] += 2( sum(σs[:,:,4:6,i,j] .* conj.(ϵs[:,:,4:6,i,j]), dims=3) .- sum(conj.(σs[:,:,4:6,i,j]) .* ϵs[:,:,4:6,i,j], dims=3) ) * 1im 
            # Eμ_vol[:,:,i, j] .*= -0.25ω

            Eμ[:,:,i, j] = sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
            Eμ[:,:,i, j] .*= ω * imag(μ[j])

            Eκ[:,:,i, j] = ω/2 *imag(Kd[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
            if ϕ[j] > 0
                Eκ[:,:,i, j] .+= ω/2 *imag(Kd[j]) * (abs.(ps[:,:,i,j]) ./ Ks[j]).^2
            end

            # # Integrate across r to find dissipated energy per unit area
            # Eμ_area[:,:] .+= Eμ[:,:, i, j] * dr

            # Eμ_layer_sph_avg[j] += sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr
            
            Eμ_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
            Eμ_layer_sph_avg[j] += Eμ_layer_sph_avg_rr[i,j]*dvol

            Eκ_layer_sph_avg_rr[i,j] = sum(sin.(clats) .* (Eκ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
            Eκ_layer_sph_avg[j] += Eκ_layer_sph_avg_rr[i,j]*dvol
       
            if ϕ[j] > 0            
                El[:,:,i, j] = 0.5 *  ϕ[j]^2 * ω^2 * ηl[j]/k[j] * (abs.(qs[:,:,1,i,j]).^2 + abs.(qs[:,:,2,i,j]).^2 + abs.(qs[:,:,3,i,j]).^2)
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
end