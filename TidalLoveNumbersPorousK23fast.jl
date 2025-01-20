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
    # using LoopVectorization
    # using Octavian

    export get_g, get_A!, get_A, get_B_product, get_Ic, get_B, get_B!
    export expand_layers, set_G, calculate_y
    export get_displacement, get_darcy_velocity, get_solution

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
    nr = 300          # Number of sub-layers in each layer (TODO: change to an array)

    TM8 = MMatrix{8, 8, precc}
    TM6 = MMatrix{6, 6, precc}
    # TM366 = MMatrix{3, 6, 6, precc}
    TM63 = MMatrix{6,3, precc}
    TM84 = MMatrix{8,4, precc}
    TM3 = MMatrix{3,3, precc}
    TV6 = MVector{6, precc}
    TV3 = MVector{3, precc}

    idx6 = SVector(1,2,3,4,5,6)
    idx3 = SVector(1,2,3)
    idx4 = SVector(1,2,3,4)

    # I8 = Matrix{Float64}(I, 8, 8)
    I8 = SMatrix{8,8,precc}(I)
    I6 = SMatrix{6,6,precc}(I)

    # Overwrite Gravitional constant for non-dimensional 
    # calculations
    function set_G(new_G)
        TidalLoveNumbers.G = new_G
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

    function get_A(r, ρ, g, μ, K)
        A = zeros(TM6) 
        get_A!(A, r, ρ, g, μ, K)
        return A
    end

    function get_Av(r, ρ, g, μ, K)



        # println(size(rβ_inv_μ))
        # println(size(A[:]))

        A = zeros(precc, 6, 6, 2size(r)[1]-1)
        rr = zeros(prec, 2size(r)[1]-1)
        gr = zeros(prec, 2size(r)[1]-1)

        dr = r[2] - r[1]
        for i in 1:size(r)[1]-1
            rr[2i]   = r[i]
            rr[2i+1] = rr[2i]+0.5dr 
            gr[2i]   = g[i]
            gr[2i+1] = g[i] + (g[i+1]-g[i])*0.5
        end

        λ = K - 2μ/3.
        r_inv = 1.0 ./ rr

        rβ_inv_λ = r_inv .* λ/(2μ + λ)
        rβ_inv_μ = r_inv .* μ/(2μ + λ)

        # println(size(r))
        # println(size(A))

        # for i in 1:3
        @. A[1,1,:] = -2rβ_inv_λ
        @. A[2,1,:] = -r_inv
        @. A[3,1,:] = 4r_inv * (3K*rβ_inv_μ - ρ*gr)
        @. A[4,1,:] = -r_inv * (6K*rβ_inv_μ - ρ*gr )
        @. A[5,1,:] = 4π * G * ρ
        @. A[6,1,:] = 4π*(n+1)*G*ρ*r_inv

        @. A[1,2,:] = n*(n+1) * rβ_inv_λ
        @. A[2,2,:] = r_inv
        @. A[3,2,:] = -n*(n+1)*r_inv * (6K*rβ_inv_μ - ρ*gr ) 
        @. A[4,2,:] = 2μ*r_inv * (2*n*(n+1)*(rβ_inv_λ + rβ_inv_μ) - r_inv )
        @. A[6,2,:] = -4π*n*(n+1)*G*ρ*r_inv

        @. A[1,3,:] = 1.0/(2μ + λ)
        @. A[3,3,:] = -4rβ_inv_μ
        @. A[4,3,:] = -rβ_inv_λ
        
        @. A[2,4,:] = 1.0 / μ
        @. A[3,4,:] = n*(n+1)*r_inv
        @. A[4,4,:] = -3r_inv

        @. A[3,5,:] = ρ * (n+1)*r_inv
        @. A[4,5,:] = -ρ*r_inv
        @. A[5,5,:] = -(n+1)r_inv

        @. A[3,6,:] = -ρ
        @. A[5,6,:] = 1.0
        @. A[6,6,:] = (n-1)r_inv
        # end

        return A
    end

    function get_A!(A, r, ρ, g, μ, K, λ=nothing)
        if isnothing(λ)
            λ = K - 2μ/3
        end

        r_inv = 1.0/r::prec
        # β_inv = 1.0/(2μ + λ)
        # rβ_inv = r_inv * β_inv
        rβ_inv_λ = r_inv * λ/(2μ + λ)
        rβ_inv_μ = r_inv * μ/(2μ + λ)

        A[1,1] = -2rβ_inv_λ
        A[2,1] = -r_inv
        A[3,1] = 4r_inv * (3K*rβ_inv_μ - ρ*g)
        A[4,1] = -r_inv * (6K*rβ_inv_μ - ρ*g )
        A[5,1] = 4π * G * ρ
        A[6,1] = 4π*(n+1)*G*ρ*r_inv

        A[1,2] = n*(n+1) * rβ_inv_λ
        A[2,2] = r_inv
        A[3,2] = -n*(n+1)*r_inv * (6K*rβ_inv_μ - ρ*g ) 
        A[4,2] = 2μ*r_inv * (2*n*(n+1)*(rβ_inv_λ + rβ_inv_μ) - r_inv )
        A[6,2] = -4π*n*(n+1)*G*ρ*r_inv

        A[1,3] = 1.0/(2μ + λ)
        A[3,3] = -4rβ_inv_μ
        A[4,3] = -rβ_inv_λ
        
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


    function get_A!(A, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

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

        get_A!( @view(A[idx6,idx6]), r, ρ, g, μ, Kd, λ) 

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
        B = zeros(TM8)
        get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        return B
    end

    function get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        c1 = 1/6. 
        c2 = 2/6.

        get_A!(Abot_p, r1, ρ, g1, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Amid_p, rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Atop_p, r2, ρ, g2, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        
        k18 .= dr * Abot_p 
        k28 .= dr *  (Amid_p .+ 0.5Amid_p *k18) # dr*Amid_p + 0.5dr*Amid_p*k188
        k38 .= dr *  (Amid_p .+ 0.5*Amid_p *k28)
        k48 .= dr *  (Atop_p .+ Atop_p*k38) 

     
        B .= (I8 + 1.0/6.0 .* (k18 .+ 2*(k28 .+ k38) .+ k48))
    end


    function get_B(r1, r2, g1, g2, ρ, μ, K)
        B = zeros(precc, 6, 6)
        get_B!(B, r1, r2, g1, g2, ρ, μ, K)
        return B
    end

    
    @inline function get_B2!(B, A, k, dr)
        c1 = 1/6. * dr
        c2 = 2/6. * dr
        
        k1 =  A[1]
        
        k[2] .= A[2]
        mul!(k[2], A[2], k1, 0.5dr, 1.0)

        k[3] .= A[2]
        mul!(k[3], A[2], k[2], 0.5dr, 1.0)
        
        k[4] .= A[3]
        mul!(k[4], A[3], k[3], dr, 1.0)
        
        B .= I + c1 .* (k1 .+ k[4]) .+  c2 .* (k[2] .+ k[3]) 
    end

    function get_B!(B, r1, r2, g1, g2, ρ, μ, K)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        ghalf = g1 + 0.5*(g2 - g1)

        c1 = 1/6. * dr
        c2 = 2/6. * dr

        get_A!(A6[1], r1, ρ, g1, μ, K)
        get_A!(A6[2], rhalf, ρ, ghalf, μ, K)
        get_A!(A6[3], r2, ρ, g2, μ, K)
        
        
        k6[1] .=  A6[1] 
        k6[2] .=  A6[2] * (I + 0.5dr .* k6[1])
        k6[3] .=  A6[2] * (I + 0.5dr .* k6[2])
        k6[4] .=  A6[3] * (I + dr .* k6[3]) 

        @view(B[idx6,idx6]) .= I + c1 .* (k6[1] .+ k6[4]) .+  c2 .* (k6[2] .+ k6[3]) 
    end

    function Amul_k6!(k1, k2, A, α, β, i)
        r = β/α
        # println(r)
        k1[1,i] = α*(A[1,1]*k2[1,i]     + A[1,2]*k2[2,i] + A[1,3]*k2[3,i] + A[1,4]*k2[4,i] + A[1,5]*k2[5,i] + A[1,6]*k2[6,i])
        k1[2,i] = α*(A[2,1]*k2[1,i]     + A[2,2]*k2[2,i]              + A[2,4]*k2[4,i]                                      )
        k1[3,i] = α*(A[3,1]*k2[1,i]     + A[3,2]*k2[2,i] + A[3,3]*k2[3,i] + A[3,4]*k2[4,i] + A[3,5]*k2[5,i] + A[3,6]*k2[6,i])
        k1[4,i] = α*(A[4,1]*k2[1,i]     + A[4,2]*k2[2,i] + A[4,3]*k2[3,i] + A[4,4]*k2[4,i] + A[4,5]*k2[5,i]                 )
        k1[5,i] = α*(A[5,1]*k2[1,i]                                       + A[5,4]*k2[4,i]     + A[5,5]*k2[5,i] + A[5,6]*k2[6,i])
        k1[6,i] = α*(A[6,1]*k2[1,i]     + A[6,2]*k2[2,i] +                + A[6,4]*k2[4,i]                      + A[6,6]*k2[6,i])
        # k1[1,1] +=  

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
        # nr = size(r)[1]


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
        # # Check dimensions of Bprod2

        # nr = size(r)[1]

        # Bstart = zeros(TM8)
        # B = zeros(TM8)

        # for i in 1:6
        #     Bstart[i,i,1] = 1
        # end

        # # if layer is porous, 
        # # don't filter out y7 and y8 components
        # if ϕ>0
        #     Bstart[7,7,1] = 1
        #     Bstart[8,8,1] = 1   # Should this be a 1 or zero?
        # end

        # r1 = r[1]
        # g1 = g[1]
        # for j in 1:nr-1
        #     r2 = r[j+1]
        #     g2 = g[j+1]

        #     if ϕ>0 
        #         get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        #     else
        #         get_B!(B, r1, r2, g1, g2, ρ, μ, K)
        #     end

        #     Bprod2[:,:,j] .= B * (j==1 ? Bstart : @view(Bprod2[:,:,j-1])) 

        #     r1 = r2
        #     g1 = g2 
        # end


        #################################################


        B = zeros(TM8)

        k8 = rand(TM8, 4) .* 0.0
        A8 = rand(TM8, 3) .* 0.0
        
        r1 = r[1]
        g1 = g[1]
        dr = r[2]-r[1]
        
        # get_A!(A8[1], r1, ρ, g1, μ, K)
        get_A!(A8[1], r1, ρ, g1, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        for j in 1:nr
            r2 = r[j+1]
            g2 = g[j+1]
            rhalf = r1 + 0.5dr
            ghalf = g1 + 0.5(g2-g1)

            # get_A!(A8[2], rhalf, ρ, ghalf, μ, K)
            get_A!(A8[2], rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
            # get_A!(A8[3], r2, ρ, g2, μ, K)
            get_A!(A8[3], r2, ρ, g2, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

            get_B2!(B, A8, k8, dr)

            mul!(Bprod2[j+1], B, Bprod2[j])

            r1 = r2
            g1 = g2

            A8[1] .= A8[3]
        end
    end

    # first method: solid layer -- for a specific layer?
    function get_B_product2!(Bprod2, r, ρ, g, μ, K)
        B = zeros(TM6)

        k6 = rand(TM6, 4) .* 0.0
        A6 = rand(TM6, 3) .* 0.0
        
        r1 = r[1]
        g1 = g[1]
        dr = r[2]-r[1]

        get_A!(A6[1], r1, ρ, g1, μ, K)
        for j in 1:nr
            r2 = r[j+1]
            g2 = g[j+1]
            rhalf = r1 + 0.5dr
            ghalf = g1 + 0.5(g2-g1)

            get_A!(A6[2], rhalf, ρ, ghalf, μ, K)
            get_A!(A6[3], r2, ρ, g2, μ, K)

            get_B2!(B, A6, k6, dr)

            Btemp1 = Bprod2[j]          # Create temp object to access 6x6 portion of Bprod
            Btemp2 = Bprod2[j+1]

            @views mul!(Btemp2[1:6,1:3], B, Btemp1[1:6,1:3])

            r1 = r2
            g1 = g2

            A6[1] .= A6[3]
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


    function calculate_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, core="liquid")

        porous_layer = ϕ .> 0.0

        sum(porous_layer) > 1.0 && error("Can only handle one porous layer for now!")

        nlayers = size(r)[2]
        # nsublayers = size(r)[1]

        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core, 8, 4)

        y = rand(MVector{8, ComplexF64}, nr+1, size(r)[2] - 1)

        y1_4 = rand(TM84, nr+1, size(r)[2] - 1) .* 0.0
                           # Starting y vector 
        
        Bprod = rand(TM84, nr+1)*0.0
        for i in 2:nlayers
            # Modify starting vector if the layer is porous
            # If a new porous layer (i.e., sitting on a non-porous layer)
            # reset the pore pressure and darcy flux
            if porous_layer[i] && !porous_layer[i-1]
                y_start[7,4] = 1.0          # Non-zero pore pressure
                y_start[8,4] = 0.0          # Zero radial Darcy flux
            elseif !porous_layer[i]
                y_start[7:8, :] .= 0.0      # Pore pressure and flux undefined
            end

            Bprod[1] .= copy(y_start)
            
            if porous_layer[i] 
                @views get_B_product2!(Bprod, r[:,i], ρ[i], g[:,i], μ[i], K[i], ω, ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i])
            else
                @views get_B_product2!(Bprod, r[:,i], ρ[i], g[:,i], μ[i], K[i])
            end

            for j in 1:nr+1
                # y1_4[:,:,j,i] .= Bprod[j] #* y_start #y1_4[:,:,1,i]
                y1_4[j, i-1] .= Bprod[j]
            end

            # y_start .= y1_4[end,i]   # Set starting vector for next layer
            y_start .= Bprod[end]

        end

         # M = zeros(TM3)
        M = zeros(MMatrix{4,4, ComplexDF64})

        # Row 1 - Radial Stress
        M[1, :] .= y1_4[end,end][3,:]

        # Row 2 - Tangential Stress
        M[2, :] .= y1_4[end,end][4,:]
    
        # Row 3 - Potential Stress
        M[3, :] .= y1_4[end,end][6,:]

        #  Row 4 - Darcy flux (r = r_tp)
        for i in 2:nlayers
            if porous_layer[i]
                M[4, :] .= y1_4[end,end][8,:]
            end
        end
         
        # b = zeros(TV3)
        b = zeros(MVector{4, ComplexDF64})
        # b = zeros(Complex{BigFloat}, 3)
        b[3] = (2n+1)/r[end,end] 
        C = ComplexDF64.(M \ b)

        for i in 1:nlayers-1
            for j in 1:nr+1
                y[j,i] .= y1_4[j,i]*C
            end
        end
        


##################################################################
        # porous_layer = ϕ .> 0.0

        # sum(porous_layer) > 1.0 && error("Can only handle one porous layer for now!")

        # nlayers = size(r)[2]
        # nsublayers = size(r)[1]

        # y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core, 8, 4)

        # y1_4 = zeros(precc, 8, 4, size(r)[1]-1, size(r)[2]) # Four linearly independent y solutions
        # y = zeros(ComplexF64, 8, size(r)[1]-1, size(r)[2])  # Final y solutions to return
        # # y_start = zeros(precc, 8, 4)                        # Starting y vector 
        
        # for i in 2:nlayers
        #     Bprod = zeros(precc, 8, 8, nsublayers-1)



        #     get_B_product2!(Bprod, r[:,i], ρ[i], g[:,i], μ[i], K[i], ω, ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i])

        #     # Modify starting vector if the layer is porous
        #     # If a new porous layer (i.e., sitting on a non-porous layer)
        #     # reset the pore pressure and darcy flux
        #     if porous_layer[i] && !porous_layer[i-1]
        #         y_start[7,4] = 1.0          # Non-zero pore pressure
        #         y_start[8,4] = 0.0          # Zero radial Darcy flux
        #     elseif !porous_layer[i]
        #         y_start[7:8, :] .= 0.0      # Pore pressure and flux undefined
        #     end

        #     for j in 1:nsublayers-1
        #         y1_4[:,:,j,i] = Bprod[:,:,j] * y_start 
        #     end

        #     y_start[:,:] .= y1_4[:,:,end,i]

        # end

        # # M = zeros(precc, 4,4)
        # M = zeros(MMatrix{3,3, ComplexDF64})

        # # Row 1 - Radial Stress
        # M[1, :] .= y1_4[3,:,end,end]

        # # # Row 2 - Tangential Stress
        # M[2, :] .= y1_4[4,:,end,end]
    
        # # # Row 3 - Potential Stress
        # M[3, :] .= y1_4[6,:,end,end]
        
        # #  Row 4 - Darcy flux (r = r_tp)
        # for i in 2:nlayers
        #     if porous_layer[i]
        #         M[4, :] .= y1_4[8,:,end,i]
        #     end
        # end

        # # b = zeros(precc, 4)
        # b = zeros(MVector{3, ComplexDF64})
        # b[3] = (2n+1)/r[end,end] 
        # C = M \ b

        # # Combine the linearly independent solutions
        # # to get the solution vector in each sublayer
        # for i in 2:nlayers
        #     for j in 1:nsublayers-1
        #         y[:,j,i] = y1_4[:,:,j,i]*C
        #     end
        # end

        return y
    end

    function calculate_y(r, ρ, g, μ, K, core="liquid")
        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core, 6, 3)

        y = rand(MVector{6, ComplexF64}, nr+1, size(r)[2] - 1)

        y1_4 = rand(TM63, nr+1, size(r)[2] - 1) .* 0.0
        
        Bprod = rand(TM63, nr+1)*0.0        # Bproduct needed at every interface, so nr+1
        for i in 2:nlayers
            
            Bprod[1] .= copy(y_start)
            @views get_B_product2!(Bprod, r[:, i], ρ[i], g[:, i], μ[i], K[i])

            for j in 1:nr+1
                # y1_4[:,:,j,i] .= Bprod[j] #* y_start #y1_4[:,:,1,i]
                y1_4[j, i-1] .= Bprod[j]
            end

            # y_start .= y1_4[end,i]   # Set starting vector for next layer
            y_start .= Bprod[end]
        end

        # M = zeros(TM3)
        M = zeros(MMatrix{3,3, ComplexDF64})

        # M = zeros(Complex{BigFloat}, 3, 3)
        
        # Row 1 - Radial Stress
        M[1, :] .= y1_4[end,end][3,:]

        # Row 2 - Tangential Stress
        M[2, :] .= y1_4[end,end][4,:]
    
        # Row 3 - Potential Stress
        M[3, :] .= y1_4[end,end][6,:]
         
        # b = zeros(TV3)
        b = zeros(MVector{3, ComplexDF64})
        # b = zeros(Complex{BigFloat}, 3)
        b[3] = (2n+1)/r[end,end] 
        C = ComplexDF64.(M \ b)

        for i in 1:nlayers-1
            for j in 1:nr+1
                y[j,i] .= y1_4[j,i]*C
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

    function expand_layers(r)#, ρ, μ, K, η)
        # rs = zeros(Float64, (length(r)-1)*nr - length(r) + 2)
        # rs = zeros(Double64, (nr+1, length(r)-1))
        rs = zeros(prec, (nr+1, length(r)-1))
        
        for i in 1:length(r)-1
            rfine = LinRange(r[i], r[i+1], nr+1)
            rs[:, i] .= rfine[1:end] 
        end
    
        return rs
    end

end