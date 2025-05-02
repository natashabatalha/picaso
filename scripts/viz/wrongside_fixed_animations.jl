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
    nc_range = fsed == 1 ? (46:80) : (66:76)
    nc = Observable(first(nc_range))
    r = @lift(h5_lookup(teff, grav_ms2, fsed, $nc, "fixed"))
    t, p, pvirga = @lift(Array($r["temperature_picaso"])[end-90:end]), @lift(Array($r["pressure_picaso"])), @lift(Array($r["pressure_virga"]))
    nstrs = @lift(Array($r["nstrs"]))
    ptop = @lift($p[1])
    pvals = @lift($p[$nstrs[[2,3,5,6],end].+1])
    pbottom = @lift($p[91])
    total_cloud_mmr = @lift(
        Array($r["condensate_mmr"])[:,:,1] .+ 1e-15
    )
    pcloud = @lift($p[$nc])
    title = @lift("cloud location = $($pcloud) bar")
    cloud_species = @lift(HDF5.attrs($r["pressure_virga"])["cloud_species"])

    begin
        fig = Figure()
        ax = Axis(fig[1,1], yscale=log10, xlabel="Temperature (K)", ylabel="Pressure (bar)", title="teff = $teff K, grav = $grav_ms2, fsed = $fsed")
        xlims!(ax, 0, 5199)
        ylims!(ax, ptop[], pbottom[])
        ax.yreversed = true
        lines!(ax, t, p)
        band!(ax, [0,5199], @lift($pvals[1]), pbottom, label="Convective zone", alpha=0.5, color=:darkgreen)
        # band!(ax, [0,5199], @lift($pvals[3]), pbottom, label="Convective zone", alpha=0.5, color=:darkgreen)
        ax2 = Axis(fig[1,2], xscale=log10, yscale=log10, xlabel="Total cloud mass mixing ratio", ylabel="Pressure (bar)", title=title)
        ylims!(ax2, ptop[], pbottom[])
        xlims!(ax2, 1e-8, 1e-2)
        ax2.yreversed = true
        for i in 1:4
            lines!(ax2, @lift($total_cloud_mmr[i,:]), pvirga, label=@lift($cloud_species[i]))
        end
        axislegend(ax2)
        fig
    end

    record(fig, "figures/wrong_side_fixed_temp$(teff)_gravms2$(grav_ms2)_fsed$fsed.mp4", nc_range; framerate=3) do n
        nc[] = n
    end
end