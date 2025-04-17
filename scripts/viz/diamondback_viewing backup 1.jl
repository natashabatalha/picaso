using Plots
using NPZ
using PythonCall

animations = Dict()

for fsed in 1:4
    t_all = npzread("data/convergence_checking/fsed$(fsed)_teff1100_all_profiles.npy")
    opd_all = npzread("data/convergence_checking/fsed$(fsed)_teff1100_all_opd.npy")
    p = npzread("data/convergence_checking/pressure_grid.npy")

    anim = @animate for i in 1:(length(opd_all)÷91)
        plot(t_all[i*91+1:(i+1)*91], p, yscale=:log10, yflip=true, label=nothing, xlabel="Temperature (K)", ylabel="Pressure (bar)", yticks=10 .^ (-6.0:2.0:2.0), xlim=(minimum(t_all), maximum(t_all)), color=:red, xguidefont=font(:red))
        plot!(twiny(), opd_all[i*90+1:(i+1)*90] .+ 1e-5, p[1:90], xscale=:log10, yscale=:log10, yflip=true, label=nothing, title="fsed = $fsed, iteration $i", xlabel="Optical depth", ylabel="Pressure (bar)", yticks=10 .^ (-6.0:2.0:2.0), xticks=10 .^ (-4.0:2.0:2.0), xlim=(1e-5, 1e2), color=:teal, xguidefont=font(:teal))
    end
    animations[fsed] = anim
end

gif(animations[4], "convergence_checking_t_opd.gif", fps=10)

all_cloud = npzread("data/convergence_checking/fsed2_teff1100_all_cloud.npy")

for fsed in 1:4
    all_cloud = npzread("data/convergence_checking/fsed$(fsed)_teff1100_all_cloud.npy")
    all_profiles = npzread("data/convergence_checking/fsed$(fsed)_teff1100_all_profiles.npy")
    p = npzread("data/convergence_checking/pressure_grid.npy")

    anim = @animate for i in eachindex(axes(all_cloud)[1])
        plot(all_profiles[i,:], p[1:90], yscale=:log10, yflip=true, xlabel="Temperature (K)", ylabel="Pressure (bar)", yticks=10 .^ (-6.0:2.0:2.0), xlim=(minimum(all_profiles), maximum(all_profiles)), color=:red, xguidefont=font(:red), label=nothing, dpi=500)
        plot!(twiny(), all_cloud[i,:,:] .+ 1e-10, p[1:90], xscale=:log10, yscale=:log10, yflip=true, label=["MgSiO₃" "Mg₂SiO₄" "Fe" "Al₂O₃"], title="fsed = $fsed, iteration $i", xlabel="Condensate mass mixing ratio", ylabel="Pressure (bar)", yticks=10 .^ (-6.0:2.0:2.0), xlim=(1e-10, 1))
    end

    gif(anim, "condmmr_t_fsed$fsed.gif", fps=10)
end
