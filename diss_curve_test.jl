include("TidalLoveNumbersPorousK23fast.jl")
using .TidalLoveNumbers
using DoubleFloats
using BenchmarkTools
using LinearAlgebra
using StaticArrays
using PyPlot

prec = TidalLoveNumbers.prec
precc = TidalLoveNumbers.precc

# non_dim = true

G = prec(6.6743e-11)
e = 0.0041

h_core = 1e1
R = 1500e3

# h_crust = 20.
ω0 = 2*2.047e-5
# ω = ω0

# #enceladus test model:
n = 2
ρₛ = [3300, 3300]
r = [0, 
     h_core, 
     R] 
μ = [60, prec(60)] .* 1e9
K = [100e9, 100e9] .* 1000000.0
η = [1e25, 1e25]

# # ρₛ = [3300, prec(3300), 3300]
# # r = [0, 
# #      h_core, R*0.5, 
# #      R] 
# # μ = [60, 60., prec(60)] .* 1e9
# # K = [100e9, 100e9, 100e9] .* 1000000.0

# ρₛ = [3300, prec(3300), 3300, 3300]
# r = [0, 
#      h_core, R/3, 2R/3, 
#      R] 
# μ = [60, 60., prec(60), 60] .* 1e9
# K = [100e9, 100e9, 100e9, 100e9] .* 1000000.0

# r = expand_layers(r)
# g = get_g(r, ρₛ)

non_dim = false

# G = prec(6.6743e-11)
# e = 0.0041

# h_core = 700.0 - 690.0
# h_mantle_low = 800. + 690
# h_mantle_up = 300. - 0.

# h_crust = 20.
# ω0 = 2*2.047e-5
# ω = ω0

#enceladus test model:
n = 2
# ρₛ = [3300, 3300, 3300, prec(3300)]
# r = [0, 
#      h_core, 
#      h_core+h_mantle_low, 
#      h_core+h_mantle_low+h_mantle_up, 
#      h_core+h_mantle_low+h_mantle_up+h_crust] .* 1e3
# μ = [60+0im, 60, 60, prec(60)] .* 1e9
# K = [100e9, 100e9, 100e9, 100e9] .* 100000.0
# η = [1e25, 1e25, 1e25, 1e25]

R = r[end]
μ0 = μ[end]
bp = 3
tp = 3


# ρₗ = [0, 0, 3300, 0]
# # α  = [0, 0, 0.95, 0., 0]
# Kl = [0, 0, 100e9, 0] .* 100000.0
# Kd = K

# α = zero(Kd)
# α[3] = Kd[3]/K[3]


# k = [0, 0, 1e-7, 0]

# ηₗ = [0, 0, 1.0, 0]
# ϕ =  [0, 0, prec(0.1), 0]

# ρ = (1 .- ϕ) .* ρₛ + ϕ .* ρₗ # bulk density
# # ρ = ρₛ
r = expand_layers(r)
g = get_g(r, ρₛ)

# Set Biot's coefficient


if non_dim
    R = r[end]
    μ0 = μ[end]

    ρ0 = ρₛ[end]
    μ0 = ρ0*g[end,end]*R

    Gnd = G*ρ0^2*R^2/μ0;

    r = r./R
    ρₛ = ρₛ ./ ρ0
    # ρ = ρ ./ ρ0
    μ = μ ./ μ0
    K = K ./ μ0
    # η = η ./ (μ0 * 2π/ω0 )

    set_G(Gnd)
    g = get_g(r, ρₛ)

end
    



function test_y(r, ρ, g, μ, K)
    Threads.@threads for i in 1:100
        y = calculate_y(r, ρ, g, μ, K)
        # println(y[end,end][5]-1 )
    end
end

function test_yp(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
    Threads.@threads for i in 1:100
        y = calculate_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        # println(y[end,end][5]-1 )
    end
end

# @btime test_y(r, ρₛ, g,, K) 
# @btime test_yp(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
# # @time test_y(r, ρₛ, g, μ, K) 
# # @time test_y(r, ρₛ, g, μ, K) 
# # @time test_y(r, ρₛ, g, μ, K) 

# # y = calculate_y(r, ρₛ, g, μ, K)
# y = calculate_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)



# if non_dim
#     r = r .* R
#     ρₛ = ρₛ  .*  ρ0
#     μ = μ  .*  μ0
#     K = K  .*  μ0

#     set_G(G)
#     g = get_g(r, ρₛ)
# end

ηs = 1:0.5:22


Edot = zero(ηs)
Edot_an = zero(ηs)

for i in eachindex(ηs)

    μc = copy(μ)
    μc =  1im*ω0*μ / (1im*ω0 .+ μ / (10.0 .^ ηs[i]))

    y = calculate_y(r, ρₛ, g, μc, K)

    k2 = y[end,end][5]-1 # 0.02646256#
    k2_an = ComplexF64(3/2 / (1 + 19/2 *μc[end,end]/(3300 * g[end,end] * r[end,end]) ))

    Edot[i] = 21/2 * -imag(k2) * (ω0*R)^5/G * e^2
    Edot_an[i] = 21/2 * -imag(k2_an) * (ω0*R)^5/G * e^2

end

fig, ax = plt.subplots()
# println(y)

ax.loglog(10 .^ ηs, Edot)
ax.loglog(10 .^ ηs, Edot_an, "k--")

plt.show()



k2_an = ComplexF64(3/2 / (1 + 19/2 *μ[end,end]/(3300 * g[end,end] * r[end,end]) ))

# println(k2_num)
# println(k2_an)
# println((k2_an-k2_num)/k2_an * 100.0)