# JosephsonCircuits.jl continuation probe for the source IPM topology.
#
# Calls JC's public hbnlsolve directly, carrying the previous nonlinear
# node-flux vector through a current ladder. The stdout emitted by JC is
# captured so the per-point convergence diagnostics are preserved in CSV.

using Dates
using JosephsonCircuits

const ROOT = normpath(joinpath(@__DIR__, ".."))
const HARMONIA_ROOT = normpath(joinpath(ROOT, "..", "Harmonia.jl"))
include(joinpath(
    HARMONIA_ROOT,
    "experiments",
    "independent_twpa",
    "source_ipm_jtwpa_netlist_audit.jl",
))

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

function fopt(opts, key, default)
    parse(Float64, get(opts, key, string(default)))
end

function iopt(opts, key, default)
    parse(Int, get(opts, key, string(default)))
end

function parse_diag(text)
    rel = match(r"norm\(F\)/norm\(x\): ([^\r\n]+)", text)
    inf = match(r"Infinity norm: ([^\r\n]+)", text)
    return (
        nonconverged_marker = occursin("Solver did not converge", text),
        rel_norm = rel === nothing ? "" : rel.captures[1],
        inf_norm = inf === nothing ? "" : inf.captures[1],
    )
end

function write_rows(path, rows)
    headers = [
        "step", "pump_frequency_ghz", "pump_current_a", "current_fraction",
        "status", "jc_nonconverged_marker", "rel_norm", "inf_norm",
        "runtime_s", "stdout_tail", "error",
    ]
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            vals = [replace(string(get(row, h, "")), "," => ";") for h in headers]
            println(io, join(vals, ","))
        end
    end
end

function main(argv=ARGS)
    opts = parse_args(argv)
    freq_ghz = fopt(opts, "freq-ghz", 7.9)
    current_max = fopt(opts, "current-max-a", 1.0e-5)
    current_min_fraction = fopt(opts, "current-min-fraction", 0.05)
    points = iopt(opts, "points", 9)
    source_port = iopt(opts, "source-port", 4)
    npump = iopt(opts, "npump", 1)
    iterations = iopt(opts, "iterations", 60)
    outdir = normpath(get(
        opts,
        "outdir",
        joinpath(ROOT, "outputs", "track_a_discriminator", "jc_continuation"),
    ))
    mkpath(outdir)

    build_t0 = time()
    circuit, circuitdefs, _metadata = build_source_ipm_jtwpa_netlist()
    build_runtime_s = time() - build_t0
    wp = (2π * freq_ghz * 1e9,)
    currents = collect(range(current_min_fraction * current_max, current_max; length=points))
    x0 = nothing
    rows = Vector{Dict{String,Any}}()

    for (step, current_a) in enumerate(currents)
        pipe = Pipe()
        t0 = time()
        err_text = ""
        status = "FAIL"
        diag = (nonconverged_marker=false, rel_norm="", inf_norm="")
        text = ""
        try
            sol = redirect_stdout(pipe) do
                hbnlsolve(
                    wp,
                    (npump,),
                    [(mode=(1,), port=source_port, current=current_a)],
                    circuit,
                    circuitdefs;
                    iterations=iterations,
                    keyedarrays=Val(false),
                    x0=x0,
                )
            end
            close(pipe.in)
            text = read(pipe, String)
            x0 = vec(Array(sol.nodeflux))
            diag = parse_diag(text)
            status = diag.nonconverged_marker ? "NONCONVERGED_RETURNED" : "CONVERGED_OR_UNMARKED"
        catch err
            try
                close(pipe.in)
            catch
            end
            text = try
                read(pipe, String)
            catch
                ""
            end
            err_text = sprint(showerror, err)
            diag = parse_diag(text)
        end
        push!(rows, Dict(
            "step" => step,
            "pump_frequency_ghz" => freq_ghz,
            "pump_current_a" => current_a,
            "current_fraction" => current_a / current_max,
            "status" => status,
            "jc_nonconverged_marker" => diag.nonconverged_marker,
            "rel_norm" => diag.rel_norm,
            "inf_norm" => diag.inf_norm,
            "runtime_s" => time() - t0,
            "stdout_tail" => replace(text[max(1, end - 500):end], '\n' => ' '),
            "error" => err_text,
        ))
        println("[JC] step=$(step)/$(points) current=$(current_a) status=$(status) rel=$(diag.rel_norm) inf=$(diag.inf_norm)")
    end

    write_rows(joinpath(outdir, "jc_continuation_rows.csv"), rows)
    open(joinpath(outdir, "summary.csv"), "w") do io
        println(io, "key,value")
        println(io, "topology,source_ipm_jtwpa_netlist_audit")
        println(io, "solver,JosephsonCircuits.hbnlsolve")
        println(io, "pump_frequency_ghz,$freq_ghz")
        println(io, "current_max_a,$current_max")
        println(io, "points,$points")
        println(io, "npump,$npump")
        println(io, "iterations,$iterations")
        println(io, "build_runtime_s,$build_runtime_s")
        println(io, "created_at,$(now())")
    end
    println("wrote=$(joinpath(outdir, "jc_continuation_rows.csv"))")
end

main()
