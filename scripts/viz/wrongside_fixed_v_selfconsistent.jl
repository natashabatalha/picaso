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

selfcon = h5_lookup(1100, 316, 1, 46, "selfconsistent")
fixeds = [h5_lookup(1100, 316, 1, nc, "fixed") for nc in 46:80]

begin
    fig = Figure()
    ax = Axis(fig[1,1], yscale=log10, xlabel="Temperature (K)", ylabel="Pressure (bar)", title="teff = $teff K, grav = $grav_ms2, fsed = $fsed")
    lines!(ax, Array(selfcon["temperature_picaso"]), Array(selfcon["pressure"]), label="self-consistent")
    for f in fixeds
        lines!(ax, Array(f["temperature_picaso"])[end-90:end], Array(f["pressure_picaso"]), label="fixed", alpha=0.2)
    end
    ax.yreversed = true
    fig
end