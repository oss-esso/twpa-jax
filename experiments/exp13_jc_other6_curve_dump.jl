using JosephsonCircuits
using Statistics

cases_dir = raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs"
outdir = raw"D:\Projects\Thesis\twpa_jax\outputs\exp13_other6_compare"
mkpath(outdir)

cases = [
    "jc_dpjpa",
    "jc_fxjpa",
    "jc_jtwpa",
    "jc_fqjtwpa",
    "jc_fqjtwpa_diss",
    "jc_fxjtwpa",
]

include(joinpath(cases_dir, "build_jc_common.jl"))

function maybe_get_sparam(artifacts)
    if haskey(artifacts, "sparam")
        return artifacts["sparam"]
    end

    return Dict{String,Any}(
        "outputmode" => artifacts["outputmode"],
        "outputport" => artifacts["outputport"],
        "inputmode" => artifacts["inputmode"],
        "inputport" => artifacts["inputport"],
    )
end

summary_rows = Vector{Dict{String,Any}}()

for case in cases
    println("\n=== JC curve: $(case) ===")

    build_file = joinpath(cases_dir, "build_$(case)_case.jl")
    if !isfile(build_file)
        println("SKIP missing builder: ", build_file)
        push!(summary_rows, Dict(
            "case" => case,
            "status" => "MISSING_BUILDER",
            "points" => 0,
            "gain_db_max" => NaN,
            "gain_db_mean" => NaN,
            "gain_db_min" => NaN,
            "peak_frequency_ghz" => NaN,
            "runtime_s" => NaN,
            "csv" => "",
            "error" => build_file,
        ))
        continue
    end

    include(build_file)

    builder_name = Symbol("build_$(case)_case")
    if !isdefined(Main, builder_name)
        println("SKIP missing function: ", builder_name)
        push!(summary_rows, Dict(
            "case" => case,
            "status" => "MISSING_FUNCTION",
            "points" => 0,
            "gain_db_max" => NaN,
            "gain_db_mean" => NaN,
            "gain_db_min" => NaN,
            "peak_frequency_ghz" => NaN,
            "runtime_s" => NaN,
            "csv" => "",
            "error" => String(builder_name),
        ))
        continue
    end

    builder = getfield(Main, builder_name)

    try
        _case_obj, artifacts = builder()

        timed = @timed JosephsonCircuits.hbsolve(
            artifacts["ws"],
            artifacts["wp"],
            artifacts["sources"],
            artifacts["Nmodulationharmonics"],
            artifacts["Npumpharmonics"],
            artifacts["circuit"],
            artifacts["circuitdefs"],
        )

        sol = timed.value
        sp = maybe_get_sparam(artifacts)

        svec = ComplexF64.(collect(sol.linearized.S(
            sp["outputmode"],
            sp["outputport"],
            sp["inputmode"],
            sp["inputport"],
            :,
        )))

        freq_ghz = artifacts["ws"] ./ (2π * 1.0e9)
        gain_db = real.(10.0 .* log10.(abs2.(svec)))

        out_csv = joinpath(outdir, "$(case)_jc_curve.csv")

        open(out_csv, "w") do io
            println(io, "signal_ghz,gain_db,sparam_real,sparam_imag,sparam_abs")
            for i in eachindex(freq_ghz)
                println(
                    io,
                    string(
                        freq_ghz[i], ",",
                        gain_db[i], ",",
                        real(svec[i]), ",",
                        imag(svec[i]), ",",
                        abs(svec[i]),
                    )
                )
            end
        end

        best_i = argmax(gain_db)

        row = Dict(
            "case" => case,
            "status" => "VALID_CONVERGED",
            "points" => length(freq_ghz),
            "gain_db_max" => maximum(gain_db),
            "gain_db_mean" => mean(gain_db),
            "gain_db_min" => minimum(gain_db),
            "peak_frequency_ghz" => freq_ghz[best_i],
            "runtime_s" => timed.time,
            "csv" => out_csv,
            "error" => "",
        )

        push!(summary_rows, row)

        println("wrote_csv=", out_csv)
        println("points=", row["points"])
        println("gain_db_max=", row["gain_db_max"])
        println("gain_db_mean=", row["gain_db_mean"])
        println("gain_db_min=", row["gain_db_min"])
        println("peak_frequency_ghz=", row["peak_frequency_ghz"])
        println("runtime_s=", row["runtime_s"])

    catch err
        msg = sprint(showerror, err)
        println("FAILED $(case): ", msg)
        push!(summary_rows, Dict(
            "case" => case,
            "status" => "FAILED",
            "points" => 0,
            "gain_db_max" => NaN,
            "gain_db_mean" => NaN,
            "gain_db_min" => NaN,
            "peak_frequency_ghz" => NaN,
            "runtime_s" => NaN,
            "csv" => "",
            "error" => msg,
        ))
    end
end

summary_csv = joinpath(outdir, "jc_other6_summary.csv")

open(summary_csv, "w") do io
    println(io, "case,status,points,gain_db_max,gain_db_mean,gain_db_min,peak_frequency_ghz,runtime_s,csv,error")
    for r in summary_rows
        println(
            io,
            string(
                r["case"], ",",
                r["status"], ",",
                r["points"], ",",
                r["gain_db_max"], ",",
                r["gain_db_mean"], ",",
                r["gain_db_min"], ",",
                r["peak_frequency_ghz"], ",",
                r["runtime_s"], ",",
                replace(string(r["csv"]), "," => ";"), ",",
                replace(string(r["error"]), "," => ";"),
            )
        )
    end
end

println("\nwrote_summary=", summary_csv)
