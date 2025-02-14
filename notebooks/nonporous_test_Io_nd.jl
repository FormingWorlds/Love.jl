include("TidalLoveNumbersPorous.jl")
using .TidalLoveNumbers
using DoubleFloats
using PyPlot
using Statistics

prec = TidalLoveNumbers.prec
precc = TidalLoveNumbers.precc

non_dim = false

G = prec(6.6743e-11)
e = 0.0041

h_core = 700 
h_mantle_low = 800. 
h_mantle_up = 300. - 0.

h_crust = 20.
ω0 = 2*2.047e-5
ω = ω0

#enceladus test model:
n = 2
ρₛ = [3300, 3300, 3300, prec(3300)]
r = [0, 
     h_core, 
     h_core+h_mantle_low, 
     h_core+h_mantle_low+h_mantle_up, 
     h_core+h_mantle_low+h_mantle_up+h_crust] .* 1e3
μ = [60+0im, 60+0im, 60+0im, prec(60)+0im] .* 1e9
κ = [10000e9, 10000e9, 10000e9, 10000e9]
η = [1e23, 1e23, 1e23, 1e23]

ρₛ = [3300, 3300, 3300, 3300]
r = [0, 
     0.1,
     800+0.1-400, 
     800+0.1+400+300] .* 1e3
μ = [60+0im, prec(60.0)+0im, prec(60)+0im] .* 1e9
κ = [10000e9, 10000e9, 10000e9]
η = [1e23, 1e23, 1e23]

R = r[end]
μ0 = μ[end]
bp = 3
tp = 3


ρₗ = [0, 0, 3300, 0]
# α  = [0, 0, 0.95, 0., 0]
κₗ = [0, 0, 10e10, 0]
k = [0, 0, 1e-6, 0]

ηₗ = [0, 0, 1e1, 0]
ϕ =  [0, 0, prec(0.1), 0]

# ρ = (1 .- ϕ) .* ρₛ + ϕ .* ρₗ # bulk density
ρ = ρₛ
r = expand_layers(r)
g = get_g(r, ρ)

 
ρ0 = ρ[end]
μ0 = ρ0*g[end,end]*R

Gnd = G*ρ0^2*R^2/μ0;



# Now non-dimensionalise?
T = 2π/ω0

if non_dim
    r = r./R
    ρₛ = ρₛ ./ ρ0
    ρ = ρ ./ ρ0
    μ = μ ./ μ0
    κ = κ ./ μ0
    η = η ./ (μ0 * 2π/ω0 )

    ρₗ = ρₗ ./ ρ0
    κₗ = κₗ ./ μ0
    k = k ./ R^2

    ηₗ = ηₗ ./ (μ0 * 2π/ω0 )


    ω = 2π

    set_G(Gnd)
    g = get_g(r, ρ)

end







ηs = 10 .^ collect(8:0.25:18)

Edot1 = zeros(length(ηs))
Edot2 = zeros(length(ηs))
k2_1 = zeros(ComplexF64, length(ηs))
k2_2 = zeros(ComplexF64, length(ηs))

yall = zeros(precc, 8, size(r)[1]-1,size(r)[2])

ytest = 0
disp = 0
vel = 0
ϵ = 0
σ = 0
p = 0
ζ = 0

for j in -6:1:-6
    for i in eachindex(ηs)
        #######################################################################
        lay = 3
        η[lay] = ηs[i] * (non_dim ? 1.0/(μ0*T) : 1.0)
        # k[lay] = 10.0^j * (non_dim ? 1.0/R^2 : 1.0)

        # μc =  1im*ω*μ ./ (1im*ω .+ μ ./ η)
        μc = copy(μ)
        μc[lay] = 1im*ω*μ[lay] / (1im*ω .+ μ[lay] / η[lay])
        # μc[:] .= 1im*ω*μ[:] ./ (1im*ω .+ μ[:] ./ η[lay])
        # μc[1] = 1im*ω*μ[1] / (1im*ω .+ μ[1] / η[1])
        # μc[2] = 1im*ω*μ[2] / (1im*ω .+ μ[2] / η[2])
        # μc[4] = 1im*ω*μ[4] / (1im*ω .+ μ[4] / η[4])

        # μc[3] = μ[3]

        # global y = calculate_y(r, ρ, g, μc, κ, "solid")

        Ic = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], "solid")
        Bprod = get_B_product(r, ρ, g, μc, κ)[:,:,end, end]
        println(μc)

        yR3 = Bprod*Ic

        M = zeros(ComplexF64, 3,3)

        # Row 1 - Radial Stress
        M[1, :] .= yR3[3,:,end,end]

        # Row 2 - Tangential Stress
        M[2, :] .= yR3[4,:,end,end]
    
        # Row 3 - Potential Stress
        M[3, :] .= yR3[6,:,end,end]
        
        #  Row 4 - Darcy flux (r = r_tp)
        
        b = zeros(ComplexF64, 3)
        b[3] = (2n+1)/r[end,end] 
        C = M \ b

        yR = yR3 * C

        # yR = y[:,end,end]

        # U22E =  7/8 * ω0^2*R^2*e * (non_dim ? ρ0/μ0 : 1) 
        # U22W = -1/8 * ω0^2*R^2*e * (non_dim ? ρ0/μ0 : 1)
        # U20  = -3/2 * ω0^2*R^2*e * (non_dim ? ρ0/μ0 : 1) 

        # # global disp = get_displacement.(y[:,end,end-1], mag, 0.25π)
        # sol_22  = get_solution(conj.(y), 2,  2, r, ρ, g, μc, κ)
        # sol_22c = get_solution(conj.(y), 2, -2, r, ρ, g, μc, κ)
        # sol_20  = get_solution(conj.(y), 2,  0, r, ρ, g, μc, κ)

        # global disp = U22E*sol_22[1] + U22W*sol_22c[1] + U20*sol_20[1] 
        # # global vel = U22E*sol_22[2] + U22W*sol_22c[2] + U20*sol_20[2] 
        
        # global ϵ = U22E*sol_22[2] + U22W*sol_22c[2] + U20*sol_20[2] 
        # global σ = U22E*sol_22[3] + U22W*sol_22c[3] + U20*sol_20[3] 

        # global p = U22E*sol_22[5] + U22W*sol_22c[5] + U20*sol_20[5] 
        # global ζ = U22E*sol_22[6] + U22W*sol_22c[6] + U20*sol_20[6] 


        k2 = yR[5] - 1

        k2_2[i] = k2    # println("Porous body k2 = ", k2)  
        println("k2 = ", yR[5] - 1)  
        println("h2 = ", -g[end]*yR[1] )
        Ediss2 = 21/2 * -imag(k2) * (ω0*R)^5/G * e^2
        Edot2[i] = Ediss2
        println("Dissipated Energy Total: ", Ediss2/1e9)

        k2_analy = 3/2 / (1 + 19/2 * μc[end] / (ρ[end]*R*g[end,end]))
        Ediss2 = 21/2 * -imag(k2_analy) * (ω0*R)^5/G * e^2
        println("Dissipated Energy Total: ",    Ediss2/1e9)
        # println(g[end,end])
        # println(R^2 * ω0^2 * e / g[end,end] )

    end

    # ax2.loglog(ηs, Edot2/1e12, label="Melt, k=\$10^{$(j)}\$ m\$^2\$")
    # ax1.semilogx(ηs, real(k2_2))
end

fig, axes = plt.subplots()
axes.loglog(ηs, Edot2)
show()



# Now dimensionalise the solutions, if doing non-dimensional calculations
# if non_dim
#     disp[:] .*= R
#     # vel[:] .*= R / T
#     σ[:] .*= μ0
#     p[:] .*= μ0
# end

# # Scale the solutions by (r/R)^2
# for i in 1:size(r)[2]
#     for j in 1:size(r)[1]-1
#         scale = non_dim ? 1.0 : 1.0/R
#         disp[:,:,:,j,i] .*= (r[j,i] * scale).^2
#         # vel[:,:,:,j,i] .*= (r[j,i] * scale ).^2
#         # ϵ[:,:,:,j,i] .*= (r[j,i] * scale ).^2
#         # σ[:,:,:,j,i] .*= (r[j,i] * scale ).^2
#         # p[:,:,j,i]   .*= (r[j,i] * scale ).^2
#         # ζ[:,:,j,i]   .*= (r[j,i] * scale ).^2
#     end 
# end

# Eₗ_vol = zeros(  (size(disp)[1], size(disp)[2], size(disp)[4], size(disp)[5]) )
# Eₛ_vol = zeros(  (size(disp)[1], size(disp)[2], size(disp)[4], size(disp)[5]) )
# Eₛ_area = zeros( (size(disp)[1], size(disp)[2]) )
# Eₗ_area = zeros( (size(disp)[1], size(disp)[2]) )
# Eₗ_total = 0.0
# Eₛ_total = 0.0

# res = 10.0
# lons = deg2rad.(collect(0:res:360-0.001))'
# clats = deg2rad.(collect(0:res:180))
# dres = deg2rad(res)



# for j in 2:size(r)[2]
#     for i in 1:size(r)[1]-1

#         dr = (r[i+1, j] - r[i, j]) * (non_dim ? R : 1.0)
#         dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

#         # Dissipated energy per unit volume
#         Eₛ_vol[:,:,i, j] +=  ( sum(σ[:,:,1:3,i,j] .* conj.(ϵ[:,:,1:3,i,j]), dims=3) .- sum(conj.(σ[:,:,1:3,i,j]) .* ϵ[:,:,1:3,i,j], dims=3) ) * 1im 
#         Eₛ_vol[:,:,i, j] += 2( sum(σ[:,:,4:6,i,j] .* conj.(ϵ[:,:,4:6,i,j]), dims=3) .- sum(conj.(σ[:,:,4:6,i,j]) .* ϵ[:,:,4:6,i,j], dims=3) ) * 1im 

#         # if ϕ[j] > 0
#         #     # Eₛ_vol[:,:,i, j] += ( p[:,:,i,j] .* conj.(ζ[:,:,i,j]) .- conj.(p[:,:,i,j]) .* ζ[:,:,i,j] ) * 1im 
#         #     Eₗ_vol[:,:,i, j] = 0.5 * ηₗ[j]/k[j] * (abs.(vel[:,:,1,i,j]).^2 + abs.(vel[:,:,2,i,j]).^2 + abs.(vel[:,:,3,i,j]).^2)
#         #     Eₗ_vol[:,:,i, j] *= (non_dim ? μ0 * T / R^2 : 1.0) # dimensionalise ηₗ and k
#         # end

#         Eₛ_vol[:,:,i, j] .*= -0.25ω0

#         # Integrate across r to find dissipated energy per unit area
#         # Eₗ_area[:,:] += Eₗ_vol[:, :, i, j] * dr
#         Eₛ_area[:,:] .+= Eₛ_vol[:,:,i, j] * dr

#         # global Eₗ_total += sum(sin.(clats) .* (Eₗ_vol[:,:,i,j] * dr)  * dres^2 * r[i,j]^2.0) 
#         global Eₛ_total += sum(sin.(clats) .* (Eₛ_vol[:,:,i,j] * dr)  * dres^2 * r[i,j]^2.0) 

        
#     end
# end

# # Eₗ_total += sum(sin.(clats) .* Eₗ_area  * dres^2 * R^2.0) 
# # Eₛ_total += sum(sin.(clats) .* Eₛ_area  * dres^2 * R^2.0) 
# # println(Eₗ_total / 1e9)
# println(Eₛ_total / 1e9)

# println(2*(Eₗ_total + Eₛ_total) / 1e9)

# To scale y functions for a given tida potential, use Eq. D6 in Rovira-Navarro et al., (2022)

# for i in 1:8
#     axes[i].plot(Double64.(real.(y[i,:,2:end][:])), R/1e3*Double64.(r[1:end-1, 2:end][:]), "-")
# end
# ax2.loglog(ηs, Edot1/1e12,"k--", label="No melt")
# println(size(disp))
# c = axes.contourf(disp[:,:,1,end,end])

# Eₗ_area = sum(Eₗ_vol*r[j,end-1], dims=3)[:,:,1]
# println(size(Eₗ_area))
# fig, axes = plt.subplots(ncols=2,figsize=(12,3.5))

# # for i in 1:6
# #     c = axes[i].contourf( (σ[:,:,i,end,end-1].*conj.(ϵ[:,:,i,end,end-1]) .- conj.(σ[:,:,i,end,end-1]).*ϵ[:,:,i,end,end-1]) * 1im
# #                         )

# #     PyPlot.colorbar(c)
# # end

# c = axes[1].contourf(Eₛ_area)
# PyPlot.colorbar(c)
# c = axes[2].contourf(Eₗ_area)
# PyPlot.colorbar(c)
# ax2.set_xlabel("Asthenosphere Solid Viscosity [Pa s]")
# ax2.set_ylabel("Tidal Heating Rate [TW]")

# ax2.axhspan((9.33-1.87)*1e13/1e12, (9.33+1.87)*1e13/1e12, alpha=0.5)

# ax1.semilogx(ηs, real(k2_1), "k--")

# # ax1.loglog(μs , k2_all, color="C1", label="Numerical")
# # ax1.loglog(μs, k2_an, "k:", label="Analytical (solid only)")
# # ax1.legend(frameon=false)
# ax1.set_xlabel("Asthenosphere Solid Viscosity [Pa s]")
# ax1.set_ylabel("k2 Tidal Love Number")

# ax2.legend(frameon=false, bbox_to_anchor=(1.0,0.5))

# ax2.set_xlim([ηs[1], ηs[end]])
# fig.subplots_adjust(wspace=.3)

# fig.savefig("io_porous.png", dpi=600, bbox_inches="tight")

# show()


# # P3 = zeros(3, 8)
# # P3[1,3] = 1
# # P3[2,4] = 1
# # P3[3,6] = 1

# P4 = zeros(4, 8)
# P4[1,3] = 1
# P4[2,4] = 1
# P4[3,6] = 1
# P4[4,7] = 1

# Pl = zeros(Int64, 8,4)
# Pl[7,4] = 1

# # display(P4*Bprod4*Ic)
# # display(P4*Bprod2*Pl)
# # # display(Bprod4)

# # Ps = zeros(3, 4)
# # Ps[1,1] = 1
# # Ps[2,2] = 1
# # Ps[3,4] = 1

# # Bp = zeros(8)
# # Bp[7] = 1.0

# # y1_T1 = Bprod1*Ic

# # # display(y1_T1)

# # y_tp = Bprod4*Ic #+ Bprod2*Bp
# # # display(P4*Bprod4*Ic)
# # # display(Bprod2)
# # # display(T1)

# # # display(P3*Bprod1*Ic)
# # # display((Bprod3*Bprod2)*Bp)
# # # display(Bprod3)

# # # Bbp = get_B(r[bp], r[bp+1], g[bp], g[bp+1], ρ[bp], μ[bp], κ[bp], η[bp], ω, ρₗ[bp], κₗ[bp], ηₗ[bp], ϕ[bp])
# # # display(P3*Bprod1*Ic)
# # # y1 = Bprod*Ic
# # # # display(y1)

# # P1 = zeros(Float64, 3, 6)
# # P1[1,3] = 1
# # P1[2,4] = 1
# # P1[3,6] = 1

# # b = zeros(Float64, 3)
# # b[3] = (2n+1)/r[end,end] 

# # C = (P3* y1_T1) \ b

# # y_tp = Bprod4*Ic #+ Bprod2*Bp
# # # display(Bprod1*Ic*C)
# # # display(Bprod2)
# # # # display(T1)


# # # C = (P1 * y1) \ b

# # # Seems to be equivalent to using P1 and P2 
# # # y = C[1]*y1[:,1] + C[2]*y1[:,2] + C[3]*y1[:,3]
# # y = C[1]*y1_T1[:,1] + C[2]*y1_T1[:,2] + C[3]*y1_T1[:,3]
# # # display(y)
# # # P2 = zeros(Float64, 3, 6)
# # # P2[1,1] = 1
# # # P2[2,2] = 1
# # # P2[3,5] = 1

# # # y1 = P1 * Bprod * Ic * C 
# # # y2 = P2 * Bprod * Ic * C 

# # # println("k2 = ", y[5] - 1)  
# # # println("h2 = ", -g[end]*y[1] )



# ############################################ Marc's method ###############################################

# # println( -g[end]*y[2]  )

# # Bprod = get_B_product(r, ρ, g, μ, κ, η, ω)[:,:,end,end]

# # display(Bprod)

# yR = Bprod1[:,:,end,end]*Ic
# ytp = Bprod4[:,:,end,tp]*Ic + Bprod2[:,:,end,tp]*Pl
# # ybp = Bprod4[:,:,end,tp]*Ic + Bprod2[:,:,end,tp]*Pl

# # display(yR)
# # display(ytp)


# C1v = zeros(4)
# C2v = zeros(4)
# C3v = zeros(4)
# C4v = zeros(4)

# C1v[1] = 1
# C2v[2] = 1
# C3v[3] = 1
# C4v[4] = 1

# display(yR)
# display(yR *C1v)
# display(yR *C2v)
# display(yR *C3v)
# display(yR *C4v)

# y1R = yR * C1v
# y2R = yR * C2v
# y3R = yR * C3v
# y4R = yR * C4v 

# y1tp = ytp * C1v
# y2tp = ytp * C2v
# y3tp = ytp * C3v
# y4tp = ytp * C4v 



# # Construct boundary condition matrix
# M = zeros(ComplexF64, 4,4)

# # Row 1 - Radial Stress
# M[1, 1] = y1R[3]
# M[1, 2] = y2R[3]
# M[1, 3] = y3R[3]
# M[1, 4] = y4R[3]

# # Row 2 - Tangential Stress
# M[2, 1] = y1R[4]
# M[2, 2] = y2R[4]
# M[2, 3] = y3R[4]
# M[2, 4] = y4R[4]

# # Row 3 - Potential Stress
# M[3, 1] = y1R[6]
# M[3, 2] = y2R[6]
# M[3, 3] = y3R[6]
# M[3, 4] = y4R[6]

# #  Row 4 - Darcy flux (r = r_tp)
# M[4, 1] = y1tp[8]
# M[4, 2] = y2tp[8]
# M[4, 3] = y3tp[8]
# M[4, 4] = y4tp[8]


# # M[:,]


# # display(ytp*C1v)

# # display(P4*yR*C1v)
# # M[1,1] = (P4*yR*C1v)[1]
# # M[1:end-1, 1] .= (P4*yR*C1v)[1:end-1]
# # M[1:end-1, 2] .= (P4*yR*C2v)[1:end-1]
# # M[1:end-1, 3] .= (P4*yR*C3v)[1:end-1]
# # M[1:end-1, 4] .= (P4*yR*C4v)[1:end-1]
# # M[end, 1] = (ytp*C1v)[8]
# # M[end, 2] = (ytp*C2v)[8]
# # M[end, 3] = (ytp*C3v)[8]
# # M[end, 4] = (ytp*C4v)[8]

# # M[1:end-1,:] = P4*yR
# # M[end, :] = ytp

# # display(M)

# # M[:, 3] = P4*y1*C3v

# # display(M)

# # # M[1,1] = (y1*C1v)[3]
# # # M[1,2] = (y1*C2v)[3]

# # # display(M)P4*
# # P4*
# # P4*
# # P4*
# b = zeros(ComplexF64, 4)
# b[3] = (2n+1)/r[end,end] 
# C2 = M \ b

# # C3 = M[1:3,1:3] \ b[1:3]

# y = yR*C2

# # display(ytp[:,1:3]*C3)
# # b[1] = (ytp[:,1:3]*C3)[3]
# # b[2] = (ytp[:,1:3]*C3)[4]
# # b[3] = (ytp[:,1:3]*C3)[6]


# # C4 = M \ b

# # # display(C2)
# # # display(C3)
# # # display(C4)

# # y1 = ytp*C2
# # y2 = ytp*C4

# # # y = C2[1]*yR[:,1] + C2[2]*yR[:,2] + C2[3]*yR[:,3]+ C2[4]*yR[:,4]

# # # y = C2[1]*ytp[:,1] + C2[2]*ytp[:,2] + C2[3]*ytp[:,3]+ C2[4]*ytp[:,4]
# # # display(C2)
# # display(y1)
# # display(y2)

# # # display(C)
# # display(C2)

# k2 = y[5] - 1
# println("k2 = ", y[5] - 1)  
# println("h2 = ", -g[end]*y[1] )
# Ediss = 21/2 * -imag(k2) * (ω*r[end,end])^5/G * e^2
# println(Ediss/1e9)



