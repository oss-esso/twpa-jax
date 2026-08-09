# JosephsonCircuits.jl point probe for the source IPM topology.
#
# This is intentionally a pump-only reference run. It rebuilds the same
# source-IPM netlist used by the IndependentTWPA diagnostics, then calls JC's
# public hbsolve at one (frequency, current) operating point. Run one process
# per point so a failed/heavy point cannot corrupt the other probes.
#
# Usage:
#   julia --project=D:/Projects/Thesis/Harmonia.jl \
#     experiments/exp15_jc_ipm_point_probe.jl \
#     --freq-ghz 7.32894736842 --current-a 4.7e-6 \
#     --outdir outputs/track_a_discriminator/jc_fp7p329_low

using Dates
using JSON
using JosephsonCircuits

const ROOT = normpath(joinpath(@__DIR__, ".."))
const HARMONIA_ROOT = normpath(joinpath(ROOT, "..", "Harmonia.jl"))
const SOURCE_BUILDER = joinpath(
    HARMONIA_ROOT,
    "experiments",
    "independent_twpa",
    "source_ipm_jtwpa_netlist_audit.jl",
)

include(SOURCE_BUILDER)

function parse_args(argv)
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if startswith(arg, "--")
            key = replace(arg[3:end], "_" => "-")
            if i == length(argv) || startswith(argv[i + 1], "--")
                opts[key] = "true"
                i += 1
            else
                opts[key] = argv[i + 1]
                i += 2
            end
        else
            i += 1
        end
    end
    return opts
end

function float_opt(opts, key, default)
    return parse(Float64, get(opts, key, string(default)))
end

function int_opt(opts, key, default)
    return parse(Int, get(opts, key, string(default)))
end

function write_summary(path, values)
    open(path, "w") do io
        println(io, "key,value")
        for (key, value) in values
            text = replace(string(value), "," => ";")
            println(io, "$(key),$(text)")
        end
    end
end

function main(argv=ARGS)
    opts = parse_args(argv)
    freq_ghz = float_opt(opts, "freq-ghz", 7.9)
    current_a = float_opt(opts, "current-a", 4.7e-6)
    source_port = int_opt(opts, "source-port", 4)
    npump = int_opt(opts, "npump", 1)
    nmod = int_opt(opts, "nmod", 1)
    iterations = int_opt(opts, "iterations", 50)
    outdir = normpath(get(
        opts,
        "outdir",
        joinpath(ROOT, "outputs", "track_a_discriminator", "jc_point_probe"),
    ))
    mkpath(outdir)

    build_t0 = time()
    circuit, circuitdefs, metadata = build_source_ipm_jtwpa_netlist()
    build_runtime_s = time() - build_t0

    wp = (2π * freq_ghz * 1e9,)
    sources = [(mode=(1,), port=source_port, current=current_a)]

    solve_t0 = time()
    status = "FAIL"
    failure_reason = ""
    result_type = ""
    nodeflux_path = ""
    try
        sol = hbsolve(
            wp,
            wp,
            sources,
            (nmod,),
            (npump,),
            circuit,
            circuitdefs;
            dc=false,
            iterations=iterations,
            returnnodeflux=true,
            keyedarrays=Val(false),
        )
        result_type = string(typeof(sol))
        if hasproperty(sol, :nonlinear) && hasproperty(sol.nonlinear, :nodeflux)
            nodeflux = vec(Array(sol.nonlinear.nodeflux))
            nodeflux_path = joinpath(outdir, "jc_nodeflux.json")
            open(nodeflux_path, "w") do io
                JSON.print(io, Dict(
                    "real" => real.(nodeflux),
                    "imag" => imag.(nodeflux),
                    "unit" => "dimensionless_reduced_flux",
                ))
            end
        end
        status = "VALID_RAN"
    catch err
        failure_reason = sprint(showerror, err)
    end
    runtime_s = time() - solve_t0

    write_summary(joinpath(outdir, "summary.csv"), [
        "status" => status,
        "failure_reason" => failure_reason,
        "result_type" => result_type,
        "build_runtime_s" => build_runtime_s,
        "runtime_s" => runtime_s,
        "pump_frequency_ghz" => freq_ghz,
        "pump_current_a" => current_a,
        "source_port" => source_port,
        "npump" => npump,
        "nmod" => nmod,
        "iterations" => iterations,
        "nodeflux_path" => nodeflux_path,
        "topology_source" => "Harmonia source_ipm_jtwpa_netlist_audit.jl",
        "solver" => "JosephsonCircuits.hbsolve",
        "created_at" => string(now()),
    ])
    println("status=$(status)")
    println("summary=$(joinpath(outdir, "summary.csv"))")
    return status == "VALID_RAN" ? 0 : 1
end

exit(main())
