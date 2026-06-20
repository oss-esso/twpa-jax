using JosephsonCircuits
using Symbolics
using Statistics

cases_dir = raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs"
outdir = raw"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_fast_scale2"
mkpath(outdir)

# Minimal shim because the builder returns BenchmarkCase, but we only need artifacts.
if !(@isdefined BenchmarkCase)
    struct BenchmarkCase
        name::String
        kind::Symbol
        parameters::Any
        metadata::Any
    end
end

include(joinpath(cases_dir, "build_jc_common.jl"))
include(joinpath(cases_dir, "build_jc_jtwpa_case.jl"))

_case_obj, artifacts = build_jc_jtwpa_case()

# Match the Python fast diagnostic grid.
ws = 2π .* collect(range(4.0, 8.0; length=21)) .* 1.0e9

println("=== running JC JTWPA 21-point curve ===")

timed = @timed JosephsonCircuits.hbsolve(
    ws,
    artifacts["wp"],
    artifacts["sources"],
    artifacts["Nmodulationharmonics"],
    artifacts["Npumpharmonics"],
    artifacts["circuit"],
    artifacts["circuitdefs"],
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

out_csv = joinpath(outdir, "jc_jtwpa_curve_21pt.csv")
open(out_csv, "w") do io
    println(io, "signal_ghz,gain_db,sparam_real,sparam_imag,sparam_abs")
    for i in eachindex(freq_ghz)
        println(io, string(freq_ghz[i], ",", gain_db[i], ",", real(svec[i]), ",", imag(svec[i]), ",", abs(svec[i])))
    end
end

best_i = argmax(gain_db)
println("wrote_csv=", out_csv)
println("points=", length(freq_ghz))
println("gain_db_max=", maximum(gain_db))
println("gain_db_mean=", mean(gain_db))
println("gain_db_min=", minimum(gain_db))
println("peak_frequency_ghz=", freq_ghz[best_i])
println("runtime_s=", timed.time)
