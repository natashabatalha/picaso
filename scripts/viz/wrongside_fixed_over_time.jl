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

fsed = 1
nc = 46
teff, grav_ms2 = 1100, 316
r = h5_lookup(teff, grav_ms2, fsed, nc, "fixed")
p = Array(r["pressure_picaso"])
p_virga = [sqrt(p[i] * p[i+1]) for i in 1:90]
t = reshape(Array(r["temperature_picaso"]), (91, 75))
nstrs = Array(r["nstrs"])
ptop = p[1]
pbottom = p[91]
total_cloud_mmr = Array(r["condensate_mmr"]) .+ 1e-15
pcloud = p[nc]
cloud_species = HDF5.attrs(r["pressure_virga"])["cloud_species"]

begin
    i = Observable(1)
    title = @lift("Iteration number = $($i)")
    #pvals = @lift(p[nstrs[[2,3,5,6],$i].+1])
    fig = Figure()
    ax = Axis(fig[1,1], yscale=log10, xlabel="Temperature (K)", ylabel="Pressure (bar)", title="teff = $teff K, grav = $grav_ms2, fsed = $fsed")
    xlims!(ax, 0, 5199)
    ylims!(ax, ptop[], pbottom[])
    ax.yreversed = true
    lines!(ax, @lift(t[:,$i+1]), p)
    #band!(ax, [0,5199], @lift($pvals[1]), pbottom, label="Convective zone", alpha=0.5, color=:darkgreen)
    ax2 = Axis(fig[1,2], xscale=log10, yscale=log10, xlabel="Cloud mass mixing ratio", ylabel="Pressure (bar)", title=title)
    ylims!(ax2, ptop[], pbottom[])
    xlims!(ax2, 1e-8, 1e-2)
    ax2.yreversed = true
    for j in 1:4
        lines!(ax2, total_cloud_mmr[j,:,1], pvirga, label=cloud_species[j])
    end
    axislegend(ax2)
    fig
end

begin
    record(fig, "figures/wrong_side_fixedovertime_temp$(teff)_gravms2$(grav_ms2)_fsed$(fsed)_nc$nc.mp4", 1:(size(t,2)-1); framerate=20) do n
        i[] = n
    end
end
