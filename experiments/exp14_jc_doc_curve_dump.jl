# experiments/exp14_jc_doc_curve_dump.jl
# Generic JosephsonCircuits.jl reference-curve dumper for a JC doc design.
#
# Usage:
#   julia --project=<JC.jl> exp14_jc_doc_curve_dump.jl <case> <fstart_ghz> <fstop_ghz> <npoints> <out_csv>
#
# <case> is the short name, e.g. jtwpa, fqjtwpa, fxjpa, fxjtwpa (maps to
# build_jc_<case>_case.jl / build_jc_<case>_case()).
#
# S-parameter (output/input mode+port) and pump sources / harmonics come straight
# from the case builder's `artifacts`, so the same script serves every design.

using JosephsonCircuits
using Symbolics
using Statistics

cases_dir = get(ENV, "JC_DOCS_CASES_DIR", raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs")

case      = ARGS[1]
fstart    = parse(Float64, ARGS[2])
fstop     = parse(Float64, ARGS[3])
npoints   = parse(Int, ARGS[4])
out_csv   = ARGS[5]

mkpath(dirname(out_csv))

# Builder returns BenchmarkCase; we only need artifacts, so shim the type.
if !(@isdefined BenchmarkCase)
    struct BenchmarkCase
        name::String
        kind::Symbol
        parameters::Any
        metadata::Any
    end
end

include(joinpath(cases_dir, "build_jc_common.jl"))
include(joinpath(cases_dir, "build_jc_$(case)_case.jl"))

builder = getfield(Main, Symbol("build_jc_$(case)_case"))
_case_obj, artifacts = builder()

ws = 2π .* collect(range(fstart, fstop; length=npoints)) .* 1.0e9

println("=== JC $(case) curve: $(fstart)-$(fstop) GHz, $(npoints) pts ===")

hbkw = get(artifacts, "hbsolve_kwargs", Dict{Symbol,Any}())
hbkw_pairs = [Symbol(k) => v for (k, v) in hbkw]

timed = @timed JosephsonCircuits.hbsolve(
    ws,
    artifacts["wp"],
    artifacts["sources"],
    artifacts["Nmodulationharmonics"],
    artifacts["Npumpharmonics"],
    artifacts["circuit"],
    artifacts["circuitdefs"];
    hbkw_pairs...,
)

sol = timed.value
sp = artifacts["sparam"]

svec = ComplexF64.(collect(sol.linearized.S(
    sp["outputmode"],
    sp["outputport"],
    sp["inputmode"],
    sp["inputport"],
    :,
)))

freq_ghz = ws ./ (2π * 1.0e9)
gain_db = real.(10.0 .* log10.(abs2.(svec)))

open(out_csv, "w") do io
    println(io, "signal_ghz,gain_db,sparam_real,sparam_imag,sparam_abs")
    for i in eachindex(freq_ghz)
        println(io, string(freq_ghz[i], ",", gain_db[i], ",",
                           real(svec[i]), ",", imag(svec[i]), ",", abs(svec[i])))
    end
end

best_i = argmax(gain_db)
println("wrote_csv=", out_csv)
println("points=", length(freq_ghz))
println("gain_db_max=", maximum(gain_db))
println("peak_frequency_ghz=", freq_ghz[best_i])
println("runtime_s=", timed.time)
