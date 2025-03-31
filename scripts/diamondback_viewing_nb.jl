### A Pluto.jl notebook ###
# v0.20.5

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

# ╔═╡ e5731840-0b38-11f0-1a19-0b1a8bbee0b5
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using PlutoUI
	using Plots
	using CSV
	using DataFrames
end

# ╔═╡ 1c5427ce-51d1-425a-9de7-3ec584dd8089
Pkg.add(path="../../ExoClouds.jl")

# ╔═╡ c7671786-63e7-40a7-860c-208f089501be
pwd()

# ╔═╡ f7089ea2-4014-443b-b832-aff903cbcfdd
@bind temp Slider(900:100:2400)

# ╔═╡ 45932636-5cec-4452-9357-6e9d50aca201
@bind g Select([31, 100, 316, 1000, 3160])

# ╔═╡ 977c188a-82a7-4a7e-8fc0-79022c7b4c5c
@bind fsed Slider(1:4)

# ╔═╡ c148a0f8-dd15-4e3c-8388-a15fd674f6d6
# ╠═╡ show_logs = false
diamondback_df = CSV.read("../data/sonora_diamondback/t$(temp)g$(g)f$(fsed)_m0.0_co1.0.pt", DataFrame, delim=" ", ignorerepeated=true, header=1, skipto=3);

# ╔═╡ a5adf2c6-d865-436e-88f7-ecd520f2be56
plot(diamondback_df[!, :T], diamondback_df[!, :P], yscale=:log10, yflip=true, yticks=exp10.(-4:3), xlabel="Temperature (K)", ylabel="Pressure (bar)", label=nothing, xlims=(0, 4000), ylims=(1e-3, 1e3))

# ╔═╡ Cell order:
# ╠═e5731840-0b38-11f0-1a19-0b1a8bbee0b5
# ╠═1c5427ce-51d1-425a-9de7-3ec584dd8089
# ╠═c7671786-63e7-40a7-860c-208f089501be
# ╠═f7089ea2-4014-443b-b832-aff903cbcfdd
# ╠═45932636-5cec-4452-9357-6e9d50aca201
# ╠═977c188a-82a7-4a7e-8fc0-79022c7b4c5c
# ╠═c148a0f8-dd15-4e3c-8388-a15fd674f6d6
# ╠═a5adf2c6-d865-436e-88f7-ecd520f2be56
