using JosephsonCircuits
using Statistics

cases_dir = raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs"
outdir = raw"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_harmonic_math"
mkpath(outdir)

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

ws = 2π .* collect(range(4.0, 8.0; length=21)) .* 1.0e9
sp = artifacts["sparam"]

# JC convention:
# Npump=6  -> odd pump modes 1,3,5
# Npump=8  -> odd pump modes 1,3,5,7
# Npump=10 -> odd pump modes 1,3,5,7,9  (the reference builder value)
npump_values = [6, 8, 10]

summary_csv = joinpath(outdir, "jc_jtwpa_npump_ladder_summary.csv")

open(summary_csv, "w") do sio
    println(sio, "npump,points,gain_db_max,gain_db_mean,gain_db_min,peak_frequency_ghz,runtime_s,csv")

    for npump in npump_values
        println("\n=== JC JTWPA npump=$(npump) 21-point curve ===")

        timed = @timed JosephsonCircuits.hbsolve(
            ws,
            artifacts["wp"],
            artifacts["sources"],
            artifacts["Nmodulationharmonics"],
            (npump,),
            artifacts["circuit"],
            artifacts["circuitdefs"],
        )

        sol = timed.value

        svec = ComplexF64.(collect(sol.linearized.S(
            sp["outputmode"],
            sp["outputport"],
            sp["inputmode"],
            sp["inputport"],
            :,
        )))

        freq_ghz = ws ./ (2π * 1.0e9)
        gain_db = real.(10.0 .* log10.(abs2.(svec)))

        out_csv = joinpath(outdir, "jc_jtwpa_npump$(npump)_curve_21pt.csv")

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

        println(
            sio,
            string(
                npump, ",",
                length(freq_ghz), ",",
                maximum(gain_db), ",",
                mean(gain_db), ",",
                minimum(gain_db), ",",
                freq_ghz[best_i], ",",
                timed.time, ",",
                out_csv,
            )
        )
    end
end

println("\nwrote_summary=", summary_csv)
