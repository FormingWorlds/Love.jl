include("./rheology_params.jl")

"Plot Figure 4"
function plot_darcy_dissipation(filename="./../data/data_ED.jld")
    data = load(filename)

    El = data["E_Darcy_total"]
    Mx = data["M"]
    El_prof = data["E_Darcy_profile"]
    r_prof = data["r_profile"]
    M_prof = data["M_profile"]
    κdx = data["κdx"]
    κlx = data["κlx"]

    fig, (ax1, ax2) = subplots(ncols=2, width_ratios=[1, 0.4], figsize=(10, 5))

    cmap = PyPlot.cm.cool;
    ims = []
    ls = ["--", "-"]
    for i in 1:length(κdx)
        for j in 1:length(κlx)
            lineplot, = ax1.loglog(Mx[j], El[i][j][:], ls=ls[i], color=cmap(j/length(κlx)))  
            push!(ims, lineplot)
        end
    end

    ax1.set_ylabel("Darcy heating rate, \$\\dot{E}_{D}\$ [W]")
    ax1.set_xlabel("Mobility, \$M_\\phi = k_\\phi / \\eta_l \$ [m\$^\\mathregular{2}\$ Pa\$^{\\mathregular{-1}}\$ s\$^{\\mathregular{-1}}\$ ] ")

    ax1.axhspan(IO_OBVS[1], IO_OBVS[2], color="red", alpha=0.4, edgecolor=nothing)
    ax1.text(2e-9, 3e13, "Io's heating rate", fontsize=12, color="red", alpha=0.6)
    ax1.set_xlim(1e-9, 1e-2)

    line_dash = mpl.lines.Line2D([0], [0], label="\$\\alpha = \\mathregular{0.01}\$", color="k", linestyle="--")
    line_solid = mpl.lines.Line2D([0], [0], label="\$\\alpha = \\mathregular{0.99}\$", color="k", linestyle="-")
    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles; [line_dash, line_solid]]

    ax1.legend(handles=handles, handlelength=2.5,
            columnspacing=2, borderpad=0.5, framealpha=0.75,prop=Dict("size" => 11),
                loc=(0.1,0.03))

    ax1.grid(which="major", alpha=0.5, linewidth=0.5)


    cmap2 = PyPlot.cm.winter;
    for i in eachindex(M_prof)
        ax2.plot(El_prof[1:end-1,3,i], r_prof, color=cmap2(i/length(M_prof)))

        # Get index of corresponding total heating rate from panel a)
        ind = findfirst(x -> x == M_prof[i], Mx[end])
        ax1.plot(M_prof[i], El[2][end][ind], "o", color=cmap2(i/length(M_prof)), alpha=0.8, ms=8,zorder=20, markerfacecolor="None", markeredgewidth=2)

    end

    ax2.set_xlabel("Normalised Volumetric\nDissipation Rate")
    ax2.set_ylabel("Radius [km]")

    ax2.grid(which="both", alpha=0.5, linewidth=0.5)
    ax2.set_title("\$H\$ = 50 km, \$\\alpha = \\mathregular{0.99}\$")

    make_log_ticks_pretty(ax1, 5:1:14, "y")
    make_log_ticks_pretty(ax1, -9:1:-2, "x")

    ax1.text(9e-4, 1.2e14, "\$\\kappa_l = \\mathregular{1}\$ GPa", rotation=0, color=cmap(1/4), fontsize=11)
    ax1.text(4e-4, 9e11, "\$\\kappa_l = \\mathregular{10}\$ GPa", rotation=-34, color=cmap(2/4), fontsize=11)
    ax1.text(2e-4, 4e7, "\$\\kappa_l = \\kappa_s = \\mathregular{200}\$ GPa", rotation=-35, color=cmap(3/4), fontsize=11)
    ax1.text(4e-4, 2e9, "\$\\kappa_l \\rightarrow \\infty\$ GPa", rotation=-34, color=cmap(4/4), fontsize=11)

    norm = mpl.colors.Normalize(vmin=log10(M_prof[1]), vmax=log10(M_prof[end]))
    sm = PyPlot.cm.ScalarMappable(cmap=cmap2, norm=norm)
    colorbar(sm, ax=ax2, shrink=0.8, label="log\$_{\\mathregular{10}} (\$Mobility, \$M_\\phi\$ [m\$^\\mathregular{2}\$ Pa\$^{\\mathregular{-1}}\$ s\$^{\\mathregular{-1}}\$ ]) ")

    subplots_adjust(wspace=0.25)

    ax1.set_ylim(5e6, 2.2e14)

    fig.text(0.055, 0.85, "a)", fontsize=14, fontweight="bold")
    fig.text(0.63, 0.85, "b)", fontsize=14, fontweight="bold")

    fig.savefig("./figures/fig4_Edarcy_vs_M.pdf", bbox_inches="tight")#

    return fig, (ax1, ax2)
end

"Plot figure 3"
function plot_compaction_dissipation(filename="./../data/data_EC.jld")
    data = load(filename)
    
    Ec = data["E_Comp_total"]
    E_nomelt = data["E_no_melt_total"]
    ζx = data["ζ"]
    Ec_prof = data["E_Comp_profile"]
    r_prof = data["r_profile"]
    ζ_prof = data["ζ_profile"]
    κdx = data["κdx"]
    κlx = data["κlx"]
    
    fig, (ax1, ax2) = subplots(ncols=2, width_ratios=[1, 0.4], figsize=(10, 5))

    #, label="\$D\$ = $(Ds[j]) km"

    cmap = PyPlot.cm.cool;
    ims1 = []
    ls = ["--", "-"]
    for x in 1:length(κdx)
        for j in 1:length(κlx)
            # ζ[3] = 10.0^ζs
            # τ = 10.0 .^ ζs * ω/ (κdx[x][3] )
            # lineplot, = ax1.loglog( τ, El[x][j][:], color=cmap(j/length(Ds)))  
            lineplot, = ax1.loglog( ζx, Ec[x][j][:], ls=ls[x], color=cmap(j/length(κlx)))
            ax1.plot(ζx, E_nomelt, "k:", linewidth=0.75, zorder=-5)
            # lineplot, = ax2.loglog( 10.0.^ζs, Es2[x][j][:], ls=ls[x], color=cmap(j/length(κlx)))
            push!(ims1, lineplot)
        end
    end

    ax1.set_ylabel("Compaction heating rate, \$\\dot{E}_{C}\$ [W]")
    ax1.set_xlabel("Compaction viscosity, \$\\zeta\$ [Pa s]")

    ax1.axhspan(IO_OBVS[1], IO_OBVS[2], color="red", alpha=0.4, edgecolor=nothing)
    ax1.text(2e13, 2.2e13, "Io's heating rate", fontsize=12, color="red", alpha=0.6)
    # ax1.set_xlim(1e-11, 1e-3)
    # ax1.legend(ncols=2, columnspacing=0.0)

    line_dash = mpl.lines.Line2D([0], [0], label="\$\\alpha = \\mathregular{0.01}\$", color="k", linestyle="--")
    line_solid = mpl.lines.Line2D([0], [0], label="\$\\alpha = \\mathregular{0.99}\$", color="k", linestyle="-")
    line_dotted = mpl.lines.Line2D([0], [0], label="no melt", color="k", linestyle=":", linewidth=0.75)
    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles; [line_dash, line_solid, line_dotted]]
    ax1.legend(handles=handles)

    ax1.legend(handles=handles, handlelength=2.5,
            columnspacing=2, borderpad=0.5, framealpha=0.75,prop=Dict("size" => 11),
                loc=(0.73,0.75))


    ax1.grid(which="both", alpha=0.5, linewidth=0.5)

    cmap2 = PyPlot.cm.winter;
    for i in eachindex(ζ_prof)
        ax2.plot(Ec_prof[1:end-1,3,i], r_prof, color=cmap2(i/length(ζ_prof)))

        ind = findfirst(x -> x == ζ_prof[i], ζx)
        ax1.plot(ζx[ind], Ec[1][1][ind], "o", color=cmap2(i/length(ζ_prof)), alpha=0.8, ms=8,zorder=20, markerfacecolor="None", markeredgewidth=2)

        
    end

    ax2.set_xlabel("Normalised Volumetric\nDissipation Rate")
    ax2.set_ylabel("Radius [km]")

    ax2.grid(which="both", alpha=0.5, linewidth=0.5)
    ax2.set_title("\$\\kappa_l\$ = 1 GPa, \$\\alpha = \\mathregular{0.01}\$", fontsize=12)

    make_log_ticks_pretty(ax1, 3:1:14, "y")
    make_log_ticks_pretty(ax1, 13:1:22, "x")

    ax1.set_ylim(1e3, 3e14)
    # ax2.set_ylim(1e3, 3e14)
    ax1.set_xlim(1e13, 1e22)

    norm = mpl.colors.Normalize(vmin=log10(ζx[1]), vmax=log10(ζx[end]))
    sm = PyPlot.cm.ScalarMappable(cmap=cmap2, norm=norm)
    colorbar(sm, ax=ax2, shrink=0.8, label="log\$_{\\mathregular{10}} (\$Compaction viscosity, \$\\zeta\$ [Pa s]) ")

    subplots_adjust(wspace=0.25)

    fig.text(0.055, 0.85, "a)", fontsize=14, fontweight="bold")
    fig.text(0.63, 0.85, "b)", fontsize=14, fontweight="bold")

    ax1.annotate("\$\\kappa_l = \\mathregular{1}\$ GPa", fontsize=11, xy=(0.8e19, 5e5), xytext=(1e18, 8e6), color=cmap(1/3),
                arrowprops=Dict("arrowstyle"=>"->", "connectionstyle"=>"arc3,rad=0.05", "color"=>cmap(1/3)))

    ax1.annotate("\$\\kappa_l = \\mathregular{10}\$ GPa", fontsize=11, xy=(1e19, 1.2e5), xytext=(0.7e20, 1.5e5), color=cmap(2/3),
                arrowprops=Dict("arrowstyle"=>"->", "connectionstyle"=>"arc3,rad=0.05", "color"=>cmap(2/3)))

    ax1.annotate("\$\\kappa_l \\rightarrow \\infty\$ GPa", fontsize=11, xy=(5e18, 1e5), xytext=(5e16, 1e4), color=cmap(3/3),
                arrowprops=Dict("arrowstyle"=>"->", "connectionstyle"=>"arc3,rad=0.05", "color"=>cmap(3/3)))


    fig.savefig("./figures/fig3_Ecomp_vs_zeta.pdf", bbox_inches="tight")
    # fig.savefig("/home/hamish/Research/Presentations/FoalabJuly25/fig3_bulk_visc_v_heat_all_khigh.png", bbox_inches="tight", dpi=600)#

    return fig, (ax1, ax2)

end

"Plot figure 2"
function plot_shear_dissipation(filename="./../data/data_ES.jld")
    data = load(filename)
    
    Es = data["E_Shear_total"]
    E_nomelt = data["E_no_melt_total"]
    ηx = data["η"]
    Es_prof = data["E_Shear_profile"]
    r_prof = data["r_profile"]
    η_prof = data["η_profile"]
    κdx = data["κdx"]
    κlx = data["κlx"]
    

    fig, (ax1, ax2) = subplots(ncols=2, width_ratios=[1, 0.4], figsize=(10, 5))

    #, label="\$D\$ = $(Ds[j]) km"

    cmap = PyPlot.cm.cool;
    ims1 = []
    ls = ["--", "-"]
    for x in eachindex(κdx)
        for j in eachindex(κlx)
            # ζ[3] = 10.0^ζs
            # τ = 10.0 .^ ζs * ω/ (κdx[x][3] )
            # lineplot, = ax1.loglog( τ, El[x][j][:], color=cmap(j/length(Ds)))  
            lineplot, = ax1.loglog( ηx, Es[x][j][:], ls=ls[x], color=cmap(j/length(κlx)))
            ax1.plot(ηx, E_nomelt, "k:", linewidth=0.75)
            # lineplot, = ax2.loglog( 10.0.^ζs, Es2[x][j][:], ls=ls[x], color=cmap(j/length(κlx)))
            push!(ims1, lineplot)
        end
    end

    ax1.set_ylabel("Shear heating rate, \$\\dot{E}_{S}\$ [W]")
    ax1.set_xlabel("Shear viscosity, \$\\eta\$ [Pa s]")

    ax1.axhspan(IO_OBVS[1], IO_OBVS[2], color="red", alpha=0.4, edgecolor=nothing)
    ax1.text(1e18, 2.2e13, "Io's heating rate", fontsize=12, color="red", alpha=0.6)
    # ax1.legend(ncols=2, columnspacing=0.0)

    line_dash = mpl.lines.Line2D([0], [0], label="\$\\alpha = \\mathregular{0.01}\$", color="k", linestyle="--")
    line_solid = mpl.lines.Line2D([0], [0], label="\$\\alpha = \\mathregular{0.99}\$", color="k", linestyle="-")
    line_dotted = mpl.lines.Line2D([0], [0], label="no melt", color="k", linestyle=":", linewidth=0.75)
    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles; [line_dash, line_solid, line_dotted]]
    ax1.legend(handles=handles)

    ax1.legend(handles=handles, handlelength=2.5,
            columnspacing=2, borderpad=0.5, framealpha=0.75,prop=Dict("size" => 11))

    ax1.grid(which="both", alpha=0.5, linewidth=0.5)

    cmap2 = PyPlot.cm.winter;
    for i in eachindex(η_prof)
        ax2.plot(Es_prof[1:end-1,3,i], r_prof, color=cmap2(i/length(η_prof)))

        ind = findfirst(x -> x == η_prof[i], ηx)
        ax1.plot(ηx[ind], Es[2][1][ind], "o", color=cmap2(i/length(η_prof)), alpha=0.8, ms=8,zorder=20, markerfacecolor="None", markeredgewidth=2)

    end

    ax2.set_xlabel("Normalised Volumetric\nDissipation Rate")
    ax2.set_ylabel("Radius [km]")

    ax2.grid(which="both", alpha=0.5, linewidth=0.5)
    ax2.set_title("\$\\kappa_l\$ = 1 GPa, \$\\alpha = \\mathregular{0.99}\$", fontsize=12)

    tick_loc = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    ax1.yaxis.set_major_locator(mpl.ticker.FixedLocator(tick_loc)) 
    ax1.set_yticklabels(["10\$^{\\mathregular{$(i)}}\$" for i in 3:1:16])
    ax1.set_ylim(1e6, 3e16)
    # ax2.set_ylim(1e3, 3e14)
    ax1.set_xlim(1e9, 1e21)

    norm = mpl.colors.Normalize(vmin=log10(η_prof[1]), vmax=log10(η_prof[end]))
    sm = PyPlot.cm.ScalarMappable(cmap=cmap2, norm=norm)
    colorbar(sm, ax=ax2, shrink=0.8, label="log\$_{\\mathregular{10}} (\$ Shear viscosity, \$\\eta\$ [Pa s]) ")

    subplots_adjust(wspace=0.25)

    fig.text(0.055, 0.85, "a)", fontsize=14, fontweight="bold")
    fig.text(0.63, 0.85, "b)", fontsize=14, fontweight="bold")

    fig.savefig("./figures/fig2_Eshear_vs_eta.pdf", bbox_inches="tight")

    return fig, (ax1, ax2)

end

"Plot figure 6"
function plot_E_vs_phi(filename="./../data/data_E_vs_phi.csv")
    fig, (ax1, ax2) = subplots(ncols=2, figsize=(8, 3.5), sharey=true)

    df1 = DataFrame(CSV.File(filename))

    Ds = unique(sort(df1[!, :D]))
    # Ds = [50.0, 300.0]
    as = unique(sort(df1[!, :a]))

    axes = [ax1, ax2]
    lws = [0.5, 1.0, 2.0]

    for i in eachindex(as)
        for j in eachindex(Ds)
            mask = (df1.a .== as[i]) .* (df1.D .== Ds[j])

            if as[i] == 0.01 && Ds[j] == 50.0
                mask .*= df1.φ .> 0.04 
            end
            if as[i] == 0.1 && Ds[j] == 300.0
                mask .*= df1.φ .> 0.03 
            end
            if as[i] == 0.01 && Ds[j] == 300.0
                mask .*= df1.φ .> 0.16
            end
            
            mask .*= df1.φ .<= 0.3

            axes[j].loglog(df1.φ[mask], df1.Edarcy[mask], "r-" , lw=lws[i])#, color=cmap(0.5))
            axes[j].loglog(df1.φ[mask], df1.Eshear[mask], "b-", lw=lws[i])#, color=cmap(0.25))
            axes[j].loglog(df1.φ[mask], df1.Ecomp[mask], "g-", lw=lws[i])
            if i==3
                axes[j].loglog(df1.φ[mask], df1.Eshear_solid[mask], "b--", lw=lws[i])  
                axes[j].loglog(df1.φ[mask], df1.Ecomp_solid[mask], "g--", lw=lws[i]) 
                axes[j].loglog(df1.φ[mask], df1.Etotal_solid[mask], "k--", lw=lws[i]) 
                axes[j].loglog(df1[mask,:φ], df1[mask,:Etotal], "k-", lw=lws[i])#, color=cmap(0.0)
            end

            xs = 10 .^ collect(-2:0.001:0.4)
            if i==1 && j==1 
                ax1.fill_between(xs, 1e15, where=xs .> 0.3, facecolor="grey", alpha=.5, zorder=100)
                ax2.fill_between(xs, 1e15, where=xs .> 0.3, facecolor="grey", alpha=.5, zorder=100)
            end

            if j ==2
                ax2.annotate("a = $(Int64(as[i]*100)) cm", fontsize=12, xy=(0.31, df1.Edarcy[mask][end]), xytext=(0.6, df1.Edarcy[mask][end]), color="r",
                    arrowprops=Dict("arrowstyle"=>"->", "connectionstyle"=>"arc3,rad=0.05", "color"=>"r"))
            end 
        end
    end

    ax1.set_xlabel("Melt fraction, \$\\phi\$ [-]")
    ax1.set_ylabel("Tidal dissipation [GW]")

    for ax in axes
        ax.set_ylim([1e6, 1e15]) 
        ax.grid(which="both", alpha=0.2)
        ax.set_xlim([1e-2, 0.4])

        
        ax.set_xlabel("Melt fraction, \$\\phi \$ [-]")
    end

    ax1.set_ylabel("Tidal dissipation [W]")

    ax1.set_title("H = 50 km, \$b = \\mathregular{0.1}\$", fontsize=12)
    ax2.set_title("H = 300 km, \$b = \\mathregular{0.1}\$", fontsize=12)

    line_dash = mpl.lines.Line2D([0], [0], label="no melt", color="k", linestyle="--", linewidth=lws[3])
    line_bulk = mpl.lines.Line2D([0], [0], label="compaction", color="g", linestyle="-")
    line_shear = mpl.lines.Line2D([0], [0], label="shear", color="b", linestyle="-")
    line_darcy = mpl.lines.Line2D([0], [0], label="darcy", color="r", linestyle="-")
    line_total = mpl.lines.Line2D([0], [0], label="total", color="k", linestyle="-")
    # line_dotted = mpl.lines.Line2D([0], [0], label="no melt", color="k", linestyle=":", linewidth=0.75)
    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles; [line_dash, line_shear, line_darcy, line_bulk, line_total]]
    ax2.legend(handles=handles, bbox_to_anchor=(1, 0.48))

    make_log_ticks_pretty(ax1, [-2,-1], "x")
    make_log_ticks_pretty(ax2, [-2,-1], "x")
    make_log_ticks_pretty(ax1, collect(6:15), "y")

    ax1.axhspan(IO_OBVS[1], IO_OBVS[2], color="red", alpha=0.4, edgecolor=nothing)
    ax2.axhspan(IO_OBVS[1], IO_OBVS[2], color="red", alpha=0.4, edgecolor=nothing)
    ax1.text(0.012, 1.3e14, "Io's heating rate", fontsize=12, color="red", alpha=0.6)
        # ax1.legend(ncols=2, columnspacing=0.0)

    subplots_adjust(wspace=0.1)
    fig.savefig("./figures/fig6_diss_vs_phi.pdf", bbox_inches="tight")

    return fig, (ax1, ax2)
end

"Plot figure 7"
function plot_b_sensitivity(filename="./../data/data_E_vs_b.csv")

    df2 = DataFrame(CSV.File(filename))

    cmap = ColorMap(reverse(ColorSchemes.hawaii.colors[:]))


    tcoords = reverse([(5e-3, 0.86), (0.01, 0.735), (0.02, 0.57), (0.035, 0.33), (0.05, 0.06)])
    rot = reverse([35, 35, 32, 22, 3])
    fig, (ax1, ax2, ax3) = subplots(ncols=3, figsize=(14,5))

    bs = unique(sort(df2.b))
    φs = 10 .^ collect(-3:0.01:log10(0.4))
    
    as = collect(1e-3:0.0001:1.0)
     
    φpos = [2e-2, 1e-1, 2e-1, 3e-1]
    φlab = ["2%", "10%", "20%", "30%"]
    φscal = φpos .* 150.0

    ax3.axhline(1.0, linestyle="--", color="k", linewidth=0.8, alpha=0.8, zorder=1)

    for i in eachindex(bs)
        αs = df2.α[df2.b .== bs[i]]

        ζs = df2.ζ[df2.b .== bs[i]]
        φs = df2.φ[df2.b .== bs[i]]
        κds = df2.κd[df2.b .== bs[i]]
        Ebulk = df2.Ebulk[df2.b .== bs[i]]

        c = cmap((i)/length(bs))
        ax1.semilogx(φs, αs, color=c, label="\$ b = \\mathregular{$(bs[i])} \$")
        ax2.loglog(φs, Ebulk, color=c)    
        
        ax1.text(tcoords[i][1], tcoords[i][2], "\$ b = \\mathregular{$(bs[i])} \$", fontsize=11, rotation=rot[i], color=c)
        

        φs_a = φ_alpha.(as, bs[i])
        κds_a = κ_phi.(φs_a, 200e9, bs[i])
        ηs_a = ηs_phi.(φs_a, 1.0)
        ζs_a = ζ_phi.(φs_a, ηs_a)
        ωtc = ω .* ζs_a ./ κds_a
        ax3.semilogy(as, ωtc, color=c)

        κdpos = κ_phi.(φpos, 200e9, bs[i])
        apos = α_phi.(φpos, bs[i])
        ηpos = ηs_phi.(φpos, 1.0)
        ζpos = ζ_phi.(φpos, ηpos)
        ωtcpos = ω .* ζpos ./ κdpos
        ax3.scatter(apos, ωtcpos, c=repeat([c], length(φpos)), s=φscal)

        if i == 2
            for j in eachindex(φpos)
                if j > 1
                    ax3.text(apos[j]-0.28, ωtcpos[j]*(0.7) , "\$ \\phi =\$"*φlab[j], fontsize=10, rotation=7, alpha=0.5)
                else
                    ax3.text(apos[j]-0.2, ωtcpos[j]*(1 -0.05) , "\$ \\phi =\$"*φlab[j], fontsize=10, rotation=7, alpha=0.5)
                end
            end
        end
    end

    a2 = as[:]
    for i in eachindex(φpos)
        b = b_from_a_phi.(φpos[i], a2)
        κds_a = κ_phi.(φpos[i], 200e9, b)
        ηs_a = ηs_phi.(φpos[i], 1.0)
        ζs_a = ζ_phi.(φpos[i], ηs_a)

        ωtc = ω .* ζs_a ./ κds_a

        ax3.semilogy(a2, ωtc, "k--", linewidth=0.5, alpha=0.5)
    end

    ax1.set_ylabel("Biot's coefficient, \$\\alpha\$ [-]")
    ax3.set_xlabel("Biot's coefficient, \$\\alpha\$ [-]")
    ax2.set_xlabel("Melt fraction, \$\\phi\$ [-]")

    ax1.set_xlabel("Melt fraction, \$\\phi\$ [-]")
    ax3.set_ylabel("Compaction Maxwell time, \$\\omega\\tau_C\$ [-]")
    ax2.set_ylabel("Compaction dissipation rate, \$\\dot{E}_C\$ [W]")

    ax3.set_ylim([1e-1,1e8])
    ax1.set_xlim([1e-3, 0.3])
    ax2.set_xlim([1e-3, 0.3])
    ax3.set_xlim([0.0, 1.0])
    fig.legend(ncols=5,loc=(0.25, 0.01), frameon=false)

    make_log_ticks_pretty(ax3, -1:1:8, "y")

    subplots_adjust(wspace=0.3)
    # tight_layout()

    ax1.text(-0.22, 0.97, "a)", fontsize=14, fontweight="bold", transform=ax1.transAxes)
    ax2.text(-0.22, 0.97, "b)", fontsize=14, fontweight="bold", transform=ax2.transAxes)
    ax3.text(-0.22, 0.97, "c)", fontsize=14, fontweight="bold", transform=ax3.transAxes)

    ax3.text(0.83, 1.2, "\$\\omega\\tau_C =\$1", fontsize=11)

    subplots_adjust(bottom=0.2)

    fig.savefig("./figures/fig7_b_impact.pdf", bbox_inches="tight")

    return fig, (ax1, ax2, ax3)
end

"Plot figure S1"
function plot_RN22(filename="./../data/data_RN22")
    df3 = DataFrame(CSV.File(filename))

    fig, (ax1, ax2) = subplots(ncols=2, figsize=(9, 4), sharey=true)

    Ds = unique(sort(df3[!, :D]))
    as = unique(sort(df3[!, :a]))

    axes = [ax1, ax2]

    lws = [1.5]
    for i in eachindex(as)
        for j in eachindex(Ds)
            mask = (df3.a .== as[i]) .* (df3.D .== Ds[j])

            if as[i] == 0.01 && Ds[j] == 50.0
                mask .*= df3.φ .> 0.04 
            end
            if as[i] == 0.1 && Ds[j] == 300.0
                mask .*= df3.φ .> 0.03 
            end
            
            ax1.loglog(df3.φ[mask], df3.Edarcy1[mask], "r-" , lw=lws[i],label="Darcy" )
            ax1.loglog(df3.φ[mask], df3.Eshear1[mask], "b-", lw=lws[i],label="Shear")
            ax1.loglog(df3.φ[mask], df3.Etotal1[mask], "k-", lw=lws[i],label="Total from Im(\$k_\\mathregular{2}\$)")
            ax1.loglog(df3.φ[mask], df3.Edarcy1[mask] .+ df3.Eshear1[mask], "k--", lw=lws[i], label="Total from Darcy + Shear")

            ax2.loglog(df3.φ[mask], df3.Edarcy2[mask], "r-" , lw=lws[i])
            ax2.loglog(df3.φ[mask], df3.Eshear2[mask], "b-", lw=lws[i])
            ax2.loglog(df3.φ[mask], df3.Edarcy2[mask] .+ df3.Eshear2[mask], "k--" , lw=lws[i])
            ax2.loglog(df3.φ[mask], df3.Etotal2[mask], "k-", lw=lws[i])
        
        end
    end

    ax1.set_ylabel("Tidal dissipation [GW]")

    for ax in axes
        ax.set_ylim([1e7, 2e13]) 
        ax.grid(which="both", alpha=0.2)
        ax.set_xlim([1e-2, 0.4])

        
        ax.set_xlabel("Melt fraction, \$\\phi \$ [-]")
    end

    ax1.legend(prop=Dict("size" => 11), frameon=false)

    ax1.set_title("this work/K23")
    ax2.set_title("RN22")

    subplots_adjust(wspace=0.05)

    make_log_ticks_pretty(ax1, 7:1:13, "y")
    make_log_ticks_pretty(ax1, [-2, -1], "x")
    make_log_ticks_pretty(ax2, [-2, -1], "x")

    fig.savefig("./figures/K23_vs_R22.pdf", bbox_inches="tight")

    return fig, (ax1, ax2)
end

"Plot figure 5"
function plot_param_vs_phi()
    fig = plt.figure(layout="constrained", figsize=(12, 3.5))
    subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 2])
    axs0 = subfigs[1].subplots(1,2, width_ratios=[0.05, 1.0])
    axs1 = subfigs[2].subplots(1, 2)

    axc1 = axs0[1]
    ax1  = axs0[2]
    ax2  = axs1[1]
    ax3  = axs1[2]

    φs = 10 .^ collect(-3:0.01:log10(0.3))
    a = 10 .^ collect(-4:0.1:-1)

    kφ = k_phi.(φs, a')'

    c = ax1.contourf(φs, a .* 100, log10.(kφ), levels=-19:0.1:-4, cmap=cm.batlow)
    ax1.contour(φs, a .* 100, log10.(kφ), colors="k", levels=-18:1:-5, linestyles="-", linewidths=0.75)
    ax1.set_xlabel("Melt fraction, \$\\phi \$ [-]")
    ax1.set_ylabel("Grain size, \$a\$ [cm]")

    ax1.set_xscale("log")
    ax1.set_yscale("log")

    cbar = colorbar(c,cax=axc1, orientation="vertical")
    cbar.set_ticks(-18:2:-4)
    cbar.set_label("log\$_{\\mathregular{10}}\$(Permeability, \$ k \$ [m\$^2\$])")

    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    ax1.set_title("\$K_0 = \\mathregular{50}\$")
    ax2.set_title(" ")
    ax3.set_title(" ")

    b = 1.0
    ηφ = ηs_phi.(φs, 1.0)
    ζφ = ζ_phi.(φs, ηφ)

    τη = (ηφ / 60e9) * ω

    ax2.loglog(φs, ηφ, label="κ", color="k", linewidth=2.0)
    ax2.loglog(φs, ζφ, label="κ", color="k", linewidth=2.0)

    ax5 = ax2.twinx()
    ax4 = ax3.twinx() 

    ax5.loglog(φs, τη, color="blue")
    
    ax3.set_ylabel("Compaction modulus, \$\\kappa_d\$ [Pa]", color = "k")
    ax4.set_ylabel("Biot's coefficient, \$\\alpha\$", color = "blue") 
    ax5.set_ylabel("Maxwell time, \$\\omega\\tau_S \$, \$\\omega\\tau_C\$ [-]", color = "blue") 
    ax4.tick_params(axis ="y", labelcolor = "blue") 
    ax5.tick_params(axis ="y", labelcolor = "blue") 

    bs = [0.5, 1.0, 1.5]
    dashes = [(1, 0, 1.0, 0), (3, 3, 3), (2, 2, 2)]
    for i in eachindex(bs)
        b = bs[i]

        κφ = κ_phi.(φs, 200e9, b)
        αφ = α_phi.(φs, b)

        l1, = ax3.loglog(φs, κφ, label="κ", color="k", dashes=dashes[i])
        l2, = ax4.plot(φs, αφ, color = "blue", dashes=dashes[i]) 

        τζ = (ζφ ./ ((1. .- αφ ) .* 200e9)) * ω

        l3, = ax5.loglog(φs, τζ, color="blue", dashes=dashes[i])
    end

    ax1.fill_between(φs, 10, where=φs .> 0.3, facecolor="grey", alpha=.5, zorder=100)
    ax2.fill_between(φs, 1e30, where=φs .> 0.3, facecolor="grey", alpha=.5, zorder=100)
    ax3.fill_between(φs, 1e30, where=φs .> 0.3, facecolor="grey", alpha=.5, zorder=100)

    ax2.set_ylim([1e13, 1e23])
    ax2.set_xlim([minimum(φs), maximum(φs)])

    ax3.set_ylim([1e5,1e12])
    ax3.set_xlim([minimum(φs), maximum(φs)])

    ax2.set_ylabel("Shear and compaction\nviscosity, \$\\eta\$, \$\\zeta\$  [Pa s]", color = "k")
    ax2.set_xlabel("Melt fraction, \$\\phi \$ [-]")
    ax3.set_xlabel("Melt fraction, \$\\phi \$ [-]")

    axc1.text(-4, 1.0, "a)", fontsize=14, fontweight="bold", transform=axc1.transAxes)
    ax2.text(-0.25, 1.0, "b)", fontsize=14, fontweight="bold", transform=ax2.transAxes)
    ax3.text(-0.25, 1.0, "c)", fontsize=14, fontweight="bold", transform=ax3.transAxes)

    ax3.text(0.04, 2.5e9, "\$ b = \\mathregular{0.5} \$", fontsize=11, rotation=-20)
    ax3.text(0.04, 5e7, "\$ b = \\mathregular{1} \$", fontsize=11, rotation=-38)
    ax3.text(0.04, 5e5, "\$ b = \\mathregular{1.5} \$", fontsize=11, rotation=-50)

    ax5.text(0.05, 2e6, "\$ b = \\mathregular{1.5} \$", fontsize=10, rotation=-20, color="blue")
    ax5.text(0.05, 3e5, "\$ b = \\mathregular{1} \$", fontsize=10, rotation=-25, color="blue")
    ax5.text(0.035, 4.5e4, "\$ b = \\mathregular{0.5} \$", fontsize=10, rotation=-25, color="blue")

    ax2.annotate("\$ \\omega\\tau_S \$", fontsize=13, xy=(φs[120], 1.8e19), xytext=(0.003, 1e20), color="blue",
                arrowprops=Dict("arrowstyle"=>"-", "connectionstyle"=>"arc3,rad=0.05", "color"=>"blue"))


    ax2.annotate("\$ \\omega\\tau_C \$", fontsize=13, xy=(0.01, 3e21), xytext=(0.0016, 2e23), color="white",
                arrowprops=Dict("arrowstyle"=>"-", "connectionstyle"=>"arc3,rad=0.05", "color"=>"blue"))

    ax2.annotate("\$ \\omega\\tau_C \$", fontsize=13, xy=(0.01, 1e22), xytext=(0.0016, 2e23), color="white",
                arrowprops=Dict("arrowstyle"=>"-", "connectionstyle"=>"arc3,rad=0.05", "color"=>"blue"))

    ax2.annotate("\$ \\omega\\tau_C \$", fontsize=13, xy=(0.01, 3.5e22), xytext=(0.0016, 2e23), color="blue",
                arrowprops=Dict("arrowstyle"=>"-", "connectionstyle"=>"arc3,rad=0.05", "color"=>"blue"))

    ax2.annotate("\$ \\eta \$", fontsize=13, xy=(φs[100], ηφ[100]), xytext=(0.003, 3e17),
                arrowprops=Dict("arrowstyle"=>"-", "connectionstyle"=>"arc3,rad=0.05"))

    ax2.annotate("\$ \\zeta \$", fontsize=13, xy=(φs[60], ζφ[60]), xytext=(0.0016, 5e20),
                arrowprops=Dict("arrowstyle"=>"-", "connectionstyle"=>"arc3,rad=0.05"))

    ax2.set_ylim([1e13, 1e24])
    ax5.set_ylim([0.01, 1e8])
    ax1.set_ylim([1e-2, 1e1])

    make_log_ticks_pretty(ax1, [-3,-2,-1], "x")
    make_log_ticks_pretty(ax2, [-3,-2,-1], "x")
    make_log_ticks_pretty(ax3, [-3,-2,-1], "x")
    make_log_ticks_pretty(ax1, [-2,-1, 0, 1], "y")
    make_log_ticks_pretty(ax2, collect(14:2:24), "y")
    make_log_ticks_pretty(ax5, collect(-2:2:8), "y")
    make_log_ticks_pretty(ax3, collect(5:1:12), "y")

    fig.savefig("./figures/fig5_params_vs_porosity.pdf", bbox_inches="tight")  
    
    return fig, (ax1, ax2, ax3)
end

function make_log_ticks_pretty(ax, tick_loc, axis)
    if axis=="x"
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(10.0 .^ tick_loc)) 
        ax.set_xticklabels(["10\$^{\\mathregular{$(i)}}\$" for i in tick_loc])
    else
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(10.0 .^ tick_loc)) 
        ax.set_yticklabels(["10\$^{\\mathregular{$(i)}}\$" for i in tick_loc])
    end
end