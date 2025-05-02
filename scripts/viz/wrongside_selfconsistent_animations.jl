using HDF5
using CairoMakie

all_files = readdir("data/wrong_side_results")

function h5_lookup(teff, grav_ms2, fsed, nc, cloudmode)
    if cloudmode == "selfconsistent"
        fname = last(filter(x -> startswith(x, "wrongside_selfconsistent_teff$(teff)_gravms2$(grav_ms2)_fsed$(fsed)_nc$(nc)"), all_files))
    else
        fname = last(filter(x -> startswith(x, "wrongside_teff$(teff)_gravms2$(grav_ms2)_fsed$(fsed)_nc$(nc)_cloudmodefixed"), all_files))
    end
    return h5open("data/wrong_side_results/$fname")
end

for fsed in [1, 8]
    teff, grav_ms2 = 1100, 316
    nc_vals = fsed == 1 ?  [46, 51, 81] : [66, 72, 77] 
    for nc in nc_vals
        r = h5_lookup(teff, grav_ms2, fsed, nc, "selfconsistent")
        p = Array(r["pressure"])
        p_virga = [sqrt(p[i] * p[i+1]) for i in 1:90]
        t = Array(r["temperature"])
        nstrs = Array(r["nstrs"])
        ptop = p[1]
        pbottom = p[91]
        total_cloud_mmr = Array(r["condensate_mmr"]) .+ 1e-15
        pcloud = p[nc]
        cloud_species = HDF5.attrs(r["pressure"])["cloud_species"]

        i = Observable(1)
        title = @lift("Iteration number = $($i)")
        pvals = @lift(p[nstrs[[2,3,5,6],$i].+1])
        fig = Figure()
        ax = Axis(fig[1,1], yscale=log10, xlabel="Temperature (K)", ylabel="Pressure (bar)", title="teff = $teff K, grav = $grav_ms2, fsed = $fsed")
        xlims!(ax, 0, 5199)
        ylims!(ax, ptop[], pbottom[])
        ax.yreversed = true
        lines!(ax, @lift(t[:,$i+1]), p_virga)
        band!(ax, [0,5199], @lift($pvals[1]), pbottom, label="Convective zone", alpha=0.5, color=:darkgreen)
        ax2 = Axis(fig[1,2], xscale=log10, yscale=log10, xlabel="Cloud mass mixing ratio", ylabel="Pressure (bar)", title=title)
        ylims!(ax2, ptop[], pbottom[])
        xlims!(ax2, 1e-8, 1e-2)
        ax2.yreversed = true
        for j in 1:4
            lines!(ax2, @lift(total_cloud_mmr[j,:,$i]), pvirga, label=cloud_species[j])
        end
        axislegend(ax2)
        fig

        record(fig, "figures/wrong_side_selfconsistent_temp$(teff)_gravms2$(grav_ms2)_fsed$(fsed)_nc$nc.mp4", 1:(size(t,2)-1); framerate=20) do n
            i[] = n
        end
    end
end