"""One-off native JosephsonCircuits.jl map for the exported ipm_2c_fixed design.

The netlist is loaded directly from designs/ipm_2c_fixed/ipm_elements.csv so
that the JC calculation uses the same design as the Python campaign. The
current is intentionally scaled by 0.5 for the JosephsonCircuits convention.
"""

using Dates
using Printf
using JosephsonCircuits

const NPOWER = 20
const NFREQ = 20
const POWER_MIN_DBM = -26.0
const POWER_MAX_DBM = -16.0
const FREQ_MIN_GHZ = 7.6
const FREQ_MAX_GHZ = 7.85
const SIGNAL_DETUNING_MHZ = 500.0
const Z0_OHM = 50.0
const JC_CURRENT_SCALE = 0.5
const MAX_ITER = 100

# Same pump-line loss model used by the Python 2c_base campaign.
const LOSS_A10_C_DB = 27.3882157727
const LOSS_A10_A_DB = 0.4579029666
const LOSS_A10_B_DB = 0.8354288817

# These are the ports in designs/ipm_2c_fixed/ipm_ports.csv.
const SIGNAL_INPUT_PORT = 1
const SIGNAL_OUTPUT_PORT = 2
const PUMP_SOURCE_PORT = 4

# Match the harmonic budget used by the existing native-JC gain diagnostics.
const N_PUMP_HARMONICS = (10,)
const N_MOD_HARMONICS = (6,)

function csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, '"' => "\"\"") * "\""
    end
    return s
end

function write_rows(path, rows, header)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join([csv_escape(get(row, h, "")) for h in header], ','))
        end
    end
end

function write_grid(path, row_label, row_axis, col_axis, values)
    open(path, "w") do io
        println(io, join(vcat([row_label], [@sprintf("fp_%.9g_ghz", f) for f in col_axis]), ','))
        for i in eachindex(row_axis)
            println(io, join(vcat([row_axis[i]], [values[i, j] for j in eachindex(col_axis)]), ','))
        end
    end
end

loss_a10_db(f_ghz) = LOSS_A10_C_DB + LOSS_A10_A_DB * sqrt(f_ghz) + LOSS_A10_B_DB * f_ghz

function python_peak_current(power_dbm, f_ghz)
    source_dbm = power_dbm - loss_a10_db(f_ghz)
    source_w = 1.0e-3 * 10.0^(source_dbm / 10.0)
    return sqrt(2.0 * source_w / Z0_OHM)
end

function parse_outdir()
    for i in eachindex(ARGS)
        if ARGS[i] == "--outdir" && i < length(ARGS)
            return abspath(ARGS[i + 1])
        end
    end
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    return normpath(joinpath(@__DIR__, "..", "outputs", "jc_ipm_2c_map_20x20_" * stamp))
end

function load_ipm_circuit(path)
    # The exported CSV has no quoted commas, so split is sufficient here.
    circuit = Tuple{String,String,String,Any}[]
    for (line_number, line) in enumerate(eachline(path))
        line_number == 1 && continue
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == 6 || error("Malformed ipm_elements.csv row $(line_number): $(line)")
        _, name, node1, node2, value, _kind = fields
        push!(circuit, (name, node1, node2, parse(Float64, value)))
    end
    return circuit
end

function main()
    outdir = parse_outdir()
    mkpath(outdir)

    design_dir = normpath(joinpath(@__DIR__, "..", "designs", "ipm_2c_fixed"))
    elements_path = joinpath(design_dir, "ipm_elements.csv")
    circuit = load_ipm_circuit(elements_path)
    circuitdefs = Dict{Symbol,Any}()

    powers = collect(range(POWER_MIN_DBM, POWER_MAX_DBM; length=NPOWER))
    freqs = collect(range(FREQ_MIN_GHZ, FREQ_MAX_GHZ; length=NFREQ))
    gain_grid = fill(NaN, NPOWER, NFREQ)
    iter_grid = fill(-1, NPOWER, NFREQ)
    status_grid = fill("NOT_RUN", NPOWER, NFREQ)
    rows = Dict{String,Any}[]

    println("JC ipm_2c_fixed map: $(NPOWER)x$(NFREQ), output=$(outdir)")
    println("netlist=$(elements_path)")
    println("JC current scale=$(JC_CURRENT_SCALE), axes P=$(POWER_MIN_DBM)..$(POWER_MAX_DBM) dBm, fp=$(FREQ_MIN_GHZ)..$(FREQ_MAX_GHZ) GHz")

    for (ip, power_dbm) in enumerate(powers), (ifp, fp_ghz) in enumerate(freqs)
        signal_ghz = fp_ghz - SIGNAL_DETUNING_MHZ / 1000.0
        jc_current = JC_CURRENT_SCALE * python_peak_current(power_dbm, fp_ghz)
        timing = Dict{Symbol,Any}()
        t0 = time()
        gain_db = NaN
        status = "ERROR"
        iters = -1
        err = ""

        try
            sources = [(mode=(1,), port=PUMP_SOURCE_PORT, current=jc_current)]
            sol = hbsolve(
                [2π * signal_ghz * 1.0e9],
                (2π * fp_ghz * 1.0e9,),
                sources,
                N_MOD_HARMONICS,
                N_PUMP_HARMONICS,
                circuit,
                circuitdefs;
                iterations=MAX_ITER,
                ftol=1e-8,
                nbatches=1,
                returnS=true,
                returnSnoise=false,
                returnQE=false,
                returnCM=false,
                keyedarrays=Val(true),
                internaltiming=timing,
            )
            s21 = sol.linearized.S(outputmode=(0,), outputport=SIGNAL_OUTPUT_PORT,
                                    inputmode=(0,), inputport=SIGNAL_INPUT_PORT, freqindex=:)
            s21v = ComplexF64(first(collect(s21)))
            gain_db = 10.0 * log10(abs2(s21v))
            iters = Int(get(timing, :nlsolve_iteration_count_actual,
                            get(timing, :nlsolve_iterations, -1)))
            status = isfinite(gain_db) ?
                (iters >= MAX_ITER ? "MAXITER_FINITE" : "CONVERGED_FINITE") :
                "NONFINITE"
        catch ex
            err = sprint(showerror, ex)
        end

        elapsed_s = time() - t0
        gain_grid[ip, ifp] = gain_db
        iter_grid[ip, ifp] = iters
        status_grid[ip, ifp] = status
        push!(rows, Dict(
            "power_dbm" => power_dbm,
            "pump_frequency_ghz" => fp_ghz,
            "signal_frequency_ghz" => signal_ghz,
            "loss_a10_db" => loss_a10_db(fp_ghz),
            "python_peak_current_a" => python_peak_current(power_dbm, fp_ghz),
            "jc_current_a" => jc_current,
            "jc_current_scale" => JC_CURRENT_SCALE,
            "gain_db" => gain_db,
            "iterations" => iters,
            "status" => status,
            "elapsed_s" => elapsed_s,
            "error" => err,
        ))
        @printf("[%3d/%3d] P=%7.3f dBm fp=%7.4f GHz status=%-17s gain=%9.4f iters=%3d t=%7.2fs\n",
                (ip - 1) * NFREQ + ifp, NPOWER * NFREQ, power_dbm, fp_ghz,
                status, gain_db, iters, elapsed_s)
    end

    header = ["power_dbm", "pump_frequency_ghz", "signal_frequency_ghz", "loss_a10_db",
              "python_peak_current_a", "jc_current_a", "jc_current_scale", "gain_db",
              "iterations", "status", "elapsed_s", "error"]
    write_rows(joinpath(outdir, "rows.csv"), rows, header)
    write_grid(joinpath(outdir, "gain_db_grid.csv"), "power_dbm", powers, freqs, gain_grid)
    write_grid(joinpath(outdir, "iterations_grid.csv"), "power_dbm", powers, freqs, iter_grid)
    write_grid(joinpath(outdir, "status_grid.csv"), "power_dbm", powers, freqs, status_grid)

    n_converged = count(==("CONVERGED_FINITE"), status_grid)
    n_maxiter = count(==("MAXITER_FINITE"), status_grid)
    n_error = count(==("ERROR"), status_grid)
    open(joinpath(outdir, "report.md"), "w") do io
        println(io, "# Native JosephsonCircuits.jl ipm_2c_fixed 20x20 quick map")
        println(io, "")
        println(io, "- design: `designs/ipm_2c_fixed/ipm_elements.csv`")
        println(io, "- axes: $(NPOWER) power points $(POWER_MIN_DBM)..$(POWER_MAX_DBM) dBm; $(NFREQ) pump-frequency points $(FREQ_MIN_GHZ)..$(FREQ_MAX_GHZ) GHz")
        println(io, "- signal: `ws = wp - $(SIGNAL_DETUNING_MHZ) MHz`")
        println(io, "- ports: signal input $(SIGNAL_INPUT_PORT), signal output $(SIGNAL_OUTPUT_PORT), pump source $(PUMP_SOURCE_PORT)")
        println(io, "- current: `I_JC = $(JC_CURRENT_SCALE) * sqrt(2 P_source / 50 ohm)`, with `P_source = P_axis - loss_A10(fp)`")
        println(io, "- statuses: `CONVERGED_FINITE=$(n_converged)`, `MAXITER_FINITE=$(n_maxiter)`, `ERROR=$(n_error)`")
        println(io, "")
        println(io, "`MAXITER_FINITE` and `ERROR` cells are retained as native-JC convergence diagnostics.")
    end
    println("COMPLETE outdir=$(outdir) converged=$(n_converged) maxiter=$(n_maxiter) errors=$(n_error)")
end

main()
