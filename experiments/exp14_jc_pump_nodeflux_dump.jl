# experiments/exp14_jc_pump_nodeflux_dump.jl
# Dump JC nonlinear pump nodeflux (modes x nodes) + node names for one case,
# to warm-start the Python pump solve onto JC's (post-fold) branch.
#
# Usage:
#   julia --project=<JC.jl> exp14_jc_pump_nodeflux_dump.jl <case> <out_npz_dir>

using JosephsonCircuits
using Symbolics
using Statistics
using DelimitedFiles

cases_dir = raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs"
case    = ARGS[1]
outdir  = ARGS[2]
mkpath(outdir)

if !(@isdefined BenchmarkCase)
    struct BenchmarkCase
        name::String; kind::Symbol; parameters::Any; metadata::Any
    end
end

include(joinpath(cases_dir, "build_jc_common.jl"))
include(joinpath(cases_dir, "build_jc_$(case)_case.jl"))
builder = getfield(Main, Symbol("build_jc_$(case)_case"))
_c, art = builder()

# Single-frequency HB just to get the nonlinear pump solution.
ws = art["ws"][1:1]
hbkw = get(art, "hbsolve_kwargs", Dict{Symbol,Any}())
hbkw_pairs = [Symbol(k) => v for (k, v) in hbkw]

sol = JosephsonCircuits.hbsolve(ws, art["wp"], art["sources"],
    art["Nmodulationharmonics"], art["Npumpharmonics"],
    art["circuit"], art["circuitdefs"]; hbkw_pairs...)

nl = sol.nonlinear
modes = nl.modes            # Vector of tuples
nodes = nl.nodes            # Vector{String}
nf = Matrix{ComplexF64}(nl.nodeflux)   # (Nmodes, Nnodes)

println("case=", case)
println("modes=", modes)
println("n_nodes=", length(nodes))
println("nodeflux_size=", size(nf))
println("wp=", art["wp"])
println("nodeflux_max_abs=", maximum(abs.(nf)))

writedlm(joinpath(outdir, "jc_nodeflux_real.csv"), real.(nf), ',')
writedlm(joinpath(outdir, "jc_nodeflux_imag.csv"), imag.(nf), ',')
writedlm(joinpath(outdir, "jc_nodeflux_nodes.csv"), nodes, ',')
writedlm(joinpath(outdir, "jc_nodeflux_modes.csv"),
         reduce(vcat, [reshape(collect(m), 1, :) for m in modes]), ',')
println("wrote ", outdir)
