### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f96cf408-ff74-11ef-305a-576de76a1f13
begin
	using Pkg
	Pkg.activate("../..")
	using HDF5
	using PlutoUI
	using Plots
	using DataInterpolations
end

# ╔═╡ 5a7cd874-7272-4b66-8503-4880be423051
@bind nstr_upper Select([77, 88])

# ╔═╡ e168698a-56e4-45c6-8b38-af9a622c0adf
@bind semimajor Select([0.03, 0.05, 0.07])

# ╔═╡ 0b540bae-12f7-4d52-9a2a-d3c4686742ee
@bind fsed Select([1, 2, 3, 4])

# ╔═╡ ad69062c-5cea-48bd-a6ce-1ae6cab8f9ad
all_files = readdir("../../data/convh5_hotjupiter");

# ╔═╡ 928e3931-4358-4131-b945-ca4b98218ff6
fname = last(filter(x -> startswith(x, "convergence_fsed$(fsed)_semimajor_$(semimajor)_nstrupper$(nstr_upper)"), all_files));

# ╔═╡ ef9f4c8f-0896-46d9-99b8-4cc84dc3183d
begin
	f = h5open("../../data/convh5_hotjupiter/$fname", "r")
	pressure = Array(f["pressure"])
	temperatures = Array(f["temperature"])
	nstrs = Array(f["nstrs"])
	cloud_decks = Array(f["cloud_deck"])
	condensate_mmr = Array(f["condensate_mmr"])
	total_condmmr = sum(condensate_mmr, dims=1)[1,:,:]
end;

# ╔═╡ 2b326e22-2e67-4365-95d9-9d3b60c0a3b6
N = (size(temperatures,2)-1);

# ╔═╡ c5a8fd6a-7b49-457c-b699-bdbba1478dba
@bind k Slider(N:-1:1)

# ╔═╡ e5741278-44f1-4dbc-8545-0625dab51e2c
function plot_temp_cond(i)
	plot(temperatures[:,i], pressure[1:90], yscale=:log10, yflip=true, xlabel="Temperature (K)", ylabel="Pressure (bar)", label="Temperature", xlim=(minimum(temperatures), maximum(temperatures) * 1.1), yticks=10 .^ (-6.0:2.0:2.0), color=:red, xguidefont=font(:red))
	plot!([], [], color=:teal, label="Total condensate MMR")
	plot!(twiny(), total_condmmr[:,i] .+ 1e-10, pressure[1:90], xscale=:log10, yscale=:log10, yflip=true, xlabel="Total condensate MMR", xlim=(1e-10, 1), title="fsed=$fsed, sep=$semimajor au, nstr_upper=$nstr_upper, iteration $i", legend=nothing, color=:teal, xguidefont=font(:teal))
	hspan!([pressure[nstrs[2,i]+1], pressure[nstrs[3,i]+1]], color=RGBA(0.2, 1, 0.2, 1), alpha=0.3, label="Convective zone")
	hspan!([pressure[nstrs[5,i]+1], pressure[nstrs[6,i]+1]], color=RGBA(0.2, 1, 0.2, 1), alpha=0.3, label=nothing)
end;

# ╔═╡ ed9c34ab-9aef-4304-8624-7d75f18871c0
plot_temp_cond(k)

# ╔═╡ Cell order:
# ╠═f96cf408-ff74-11ef-305a-576de76a1f13
# ╠═5a7cd874-7272-4b66-8503-4880be423051
# ╠═e168698a-56e4-45c6-8b38-af9a622c0adf
# ╠═0b540bae-12f7-4d52-9a2a-d3c4686742ee
# ╟─c5a8fd6a-7b49-457c-b699-bdbba1478dba
# ╟─ed9c34ab-9aef-4304-8624-7d75f18871c0
# ╟─2b326e22-2e67-4365-95d9-9d3b60c0a3b6
# ╠═e5741278-44f1-4dbc-8545-0625dab51e2c
# ╠═ad69062c-5cea-48bd-a6ce-1ae6cab8f9ad
# ╠═928e3931-4358-4131-b945-ca4b98218ff6
# ╠═ef9f4c8f-0896-46d9-99b8-4cc84dc3183d
