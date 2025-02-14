using LinearAlgebra
using DoubleFloats
using AssociatedLegendrePolynomials
include("SphericalHarmonics.jl")
using .SphericalHarmonics
using BenchmarkTools
using StaticArrays

# export get_g, get_A!, get_A, get_B_product, get_Ic, get_B, get_B!
# export expand_layers, set_G, calculate_y
# export get_displacement, get_darcy_velocity, get_solution

# prec = Float64 #BigFloat
# precc = ComplexF64 #Complex{BigFloat}

prec = Double64 #BigFloat
precc = ComplexDF64 #Complex{BigFloat}

TM8 = MArray{Tuple{8, 8}, precc}
TM6 = MArray{Tuple{6, 6}, precc}




function get_A!(A)
    A .= rand(6,6)
end

function get_B!(B, A1, A2, A3, k1, k2, k3, k4)
    dr = 1.0 + 1e-11
    k1 .= dr * A1 
    k2 .= dr *  (A2 .+ 0.5A2 *k1) # dr*A2 + 0.5dr*A2*k1
    k3 .= dr *  (A2 .+ 0.5*A2 *k2)
    k4 .= dr *  (A3 .+ A3*k3) 
    

    return I + dr/6.0 * (k1 + 2k2 + 2k3 + k4)
end

function get_B_product2!(Bprod, B)
    # Check dimensions of Bprod2

    Bstart = zeros(precc, 6, 6)

    for i in 1:6
        Bstart[i,i,1] = 1
    end

    for i in 1:4
        for j in 1:300-1
            if j ==1
                mul!(Bprod[j,i], B[j,i], Bstart)
            else
                mul!(Bprod[j,i], B[j,i], Bprod[j-1,i])
            end
        end
    end
end

function test()
    B = zeros(TM6, 300, 4)
    Bprod = zero(B)
    A = zeros(TM6, 300, 4)

    k1 = zeros(TM6)#
    k2 = zeros(TM6)
    k3 = zeros(TM6)
    k4 = zeros(TM6)
    # A2 = zeros(TM6)
    # A3 = zeros(TM6)

    for j in 1:4
        for i in 1:300
            get_A!(A[1])
            get_A!(A[2])
            get_A!(A[3])
            get_B!(B[i, j], A[1], A[2], A[3], k1, k2, k3, k4) 
        end
    end

    get_B_product2!(Bprod, B)
end

@benchmark test() setup=()