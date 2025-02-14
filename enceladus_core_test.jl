include("TidalLoveNumbersPorous.jl")
using .TidalLoveNumbers
using PyPlot
using DoubleFloats
using DelimitedFiles

#enceladus core model:




n = 2
G = 6.6743e-11
e = 0.0047
ρₛ = Double64[2422, 2422] 
r = Double64[0, 0.1, 191.1] .* 1e3
μ = Double64[1, 1] .* 1e9
κ = Double64[10e9, 10e9]
η = Double64[1.0, 1.0] 

ρₛ = Double64[2422, 2422, 2422] 
r = Double64[0, 0.1, 150.0, 191.1] .* 1e3
μ = Double64[1, 1, 1] .* 1e9
κ = Double64[10e9, 10e9, 10e9]
η = Double64[1.0, 1.0, 1.0] 






bp = 2
tp = 2


# To recover solid-body behaviour
# set ρₗ=0, ϕ=0, α=0

# ρₗ = Double64[0, 1000.0] 
# # α  = Double64[0, 0, 0.95, 0., 0]
# κₗ = Double64[0, 2.2e9]
# ω = 2π / (33*60*60)
# ηₗ = Double64[0, 1.9e-3]
# ϕ =  Double64[0, 0.2]
# k = [0, 1e-6]

ρₗ = Double64[0, 1000.0, 1000.0] 
# α  = Double64[0, 0, 0.95, 0., 0]
κₗ = Double64[0, 2.2e9, 2.2e9]
ω = 2π / (33*60*60)
ηₗ = Double64[0, 1.9e-3, 1.9e-3]
ϕ =  Double64[0, 0.2, 0.2]
k = [0, 1e-6, 1e-6]



# μc = 1im*ω*μ ./ (1im*ω .+ μ./η)

# ρₛ = (ρₛ[1]-ϕ[2]*ρₗ[2])/(1-ϕ[2])
ρ = (1 .- ϕ) .* ρₛ + ϕ .* ρₗ # bulk density

# ρ = ρₛ # Matches better with Marc's results. Perhaps he turned off self-gravity 
       # in his Figure 2.

r = expand_layers(r)

g = get_g(r, ρ)


# Add extra column to Ic 
Ic = get_Ic(r[end,1], ρₛ[1], g[end,1], μ[1], "solid", 8, 4)

ηs = 10 .^ collect(9:0.1:24)

Edot = zeros(length(ηs))
Edot2 = zeros(length(ηs))



for i in eachindex(ηs) 
    # Calculate the integration matrix from 
    # bottom to top
    # Complex shear modulus for a Maxwell rheology
    μc =  1im*ω*μ ./ (1im*ω .+ μ ./ (η .* ηs[i]))
    Bprod1 = get_B_product(r, ρ, g, μc, κ, ω, ρₗ, κₗ, ηₗ, ϕ, k)[:, :, :, :]

    # Projection matrix for the third, fourth, sixth, and eigth 
    # components of the solution
    P4 = zeros(4, 8)
    P4[1,3] = 1
    P4[2,4] = 1
    P4[3,6] = 1
    P4[4,8] = 1

    Pl = zeros(Double64, 8,4)
    Pl[7,4] = 1

    yR = Bprod1[:,:,end,end]*(Ic + Pl)


    # Get boundary condtion matrix
    M = P4*yR

    b = zeros(ComplexDF64, 4)
    b[3] = (2n+1)/r[end,end] 
    C2 = M \ b

    y = yR*C2

    # C2 = M[1:3,1:3] \ b[1:3]

    # y = yR[1:6,1:3]*C2


    k2 = y[5] - 1
    println("k2 = ", k2)  

    k2_homog = 3/2 / (1 + 19/2 * μc[end]/(ρ[end]*g[end,end]*r[end,end]))

    # μc = 1im*ω*1e9 / (1im*ω + 1e9 / (η[end]*ηs[i]) )
    println("k2 = ", 3/2 / (1 + 19/2 * μc[end]/(ρ[end]*g[end,end]*r[end,end])))
    # println("h2 = ", -g[end]*y[1] )

    Ediss = 21/2 * -imag(k2) * (ω*r[end,end])^5/G * e^2

    Edot[i] = Ediss
    Edot2[i] = 21/2 * -imag(k2_homog) * (ω*r[end,end])^5/G * e^2
    # println(Ediss/1e9 )

end

μs = 10 .^ collect(4:0.1:10)
k2_all = zeros(ComplexF64, length(μs))
k2_an = zeros(ComplexF64, length(μs))
for i in eachindex(μs) 
    # Calculate the integration matrix from 
    # bottom to top
    # Complex shear modulus for a Maxwell rheology
    μc =  1im*ω*μs[i] ./ (1im*ω .+ μs[i] ./ (η .* 1e21))
    Bprod1 = get_B_product(r, ρ, g, μc, κ, ω, ρₗ, κₗ, ηₗ, ϕ, k)[:, :, :, :]

    # Projection matrix for the third, fourth, sixth, and eigth 
    # components of the solution
    P4 = zeros(4, 8)
    P4[1,3] = 1
    P4[2,4] = 1
    P4[3,6] = 1
    P4[4,8] = 1

    Pl = zeros(Double64, 8,4)
    Pl[7,4] = 1

    yR = Bprod1[:,:,end,end]*(Ic + Pl)


    # Get boundary condtion matrix
    M = P4*yR

    b = zeros(ComplexDF64, 4)
    b[3] = (2n+1)/r[end,end] 
    C2 = M \ b

    y = yR*C2

    # C2 = M[1:3,1:3] \ b[1:3]

    # y = yR[1:6,1:3]*C2


    k2 = y[5] - 1
    k2_all[i] = k2
    k2_an[i] = 3/2 / (1 + 19/2 * μc[end]/(ρ[end]*g[end,end]*r[end,end]))
    # println("k2 = ", k2)  

    # k2_homog = 3/2 / (1 + 19/2 * μc[end]/(ρ[end]*g[end,end]*r[end,end]))

    # # μc = 1im*ω*1e9 / (1im*ω + 1e9 / (η[end]*ηs[i]) )
    # # println("k2 = ", 3/2 / (1 + 19/2 * μc[end]/(ρ[end]*g[end,end]*r[end,end])))
    # # println("h2 = ", -g[end]*y[1] )

    # Ediss = 21/2 * -imag(k2) * (ω*r[end,end])^5/G * e^2

    # Edot[i] = Ediss
    # Edot2[i] = 21/2 * -imag(k2_homog) * (ω*r[end,end])^5/G * e^2
    # println(Ediss/1e9 )

end

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,3.5))

ax2.loglog(ηs, Edot/1e9, color="C1", label="Numerical")
ax2.loglog(ηs, Edot2/1e9, "k:", label="Analytical (solid only)")

ax2.set_xlabel("Solid Viscosity [Pa s]")
ax2.set_ylabel("Tidal Heating Rate [GW]")

ax1.loglog(μs , k2_all, color="C1", label="Numerical")
ax1.loglog(μs, k2_an, "k:", label="Analytical (solid only)")
ax1.legend(frameon=false)
ax1.set_xlabel("Shear Modulus [Pa]")
ax1.set_ylabel("k2 Tidal Love Number")

ax1.axhline(1.5, color="k", linestyle="--", alpha=0.5)
mat = readdlm("benchmarking.csv", ',')

ax2.loglog(mat[:,1], mat[:,2], "g:", alpha=0.8, label="Rovira-Navarro et al., (2022)")

ax2.legend(frameon=false)

ax2.set_xlim([ηs[1], ηs[end]])
fig.subplots_adjust(wspace=.3)

fig.savefig("homog_enc_core_porous2.png", dpi=600, bbox_inches="tight")

show()



