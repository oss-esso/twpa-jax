# JosephsonCircuits.jl reference: IPM JTWPA S21 gain vs signal frequency.
#
# Mirrors ../Harmonia/IPM_JTWPA.jl (same make_IPM topology + circuitdefs) but
# sweeps the signal frequency at a fixed pump and dumps a CSV (signal_ghz,
# gain_db) for overlaying on the Python exp08/exp09 gain profile.
#
# Run from the Harmonia.jl package env (has Harmonia + JosephsonCircuits):
#   julia --project=<...>/Harmonia.jl experiments/jc_ipm_signal_sweep.jl \
#       --pump-current-a 8.934e-6 --pump-freq-ghz 7.9 \
#       --signal-min-ghz 5 --signal-max-ghz 10 --points 51 --out <csv>

using JosephsonCircuits
using Harmonia

function getarg(args, key, default)
    i = findfirst(==("--" * key), args)
    return i === nothing ? default : args[i + 1]
end

args = ARGS
pump_current_a = parse(Float64, getarg(args, "pump-current-a", "8.934e-6"))
pump_freq_ghz = parse(Float64, getarg(args, "pump-freq-ghz", "7.9"))
signal_min = parse(Float64, getarg(args, "signal-min-ghz", "5.0"))
signal_max = parse(Float64, getarg(args, "signal-max-ghz", "10.0"))
npoints = parse(Int, getarg(args, "points", "51"))
outcsv = getarg(args, "out", "jc_ipm_signal_sweep.csv")

@variables Rleft Rright Cg Lj Cj Cl Ll
circuit = Tuple{String,String,String,Any}[]

start_node_top = 1
start_node_bot = 10000
ground = 0
array_length = 418
num_rows = 6
arrays_per_dc = 3
coupler_freq = 8.0e9
coupler_factor = -14.0
couplers_params = (coupling_dB = coupler_factor, Z0 = 50.0, freq = coupler_freq)
length_of_long_TL = 250
len1 = 100
len2 = 50
len3 = 100
len4 = 300
inter_twpa_section_length = 30
coupler_section_length = 1500
total_jj_count = num_rows * array_length
test_mod_array = ones(Float64, total_jj_count)

n_t_end, n_b_end = make_IPM(
    circuit, start_node_top, start_node_bot,
    array_length, num_rows, arrays_per_dc, couplers_params,
    length_of_long_TL, inter_twpa_section_length, coupler_section_length,
    len1, len2, len3, len4, ground,
    Cg, Lj, Cj, Ll, Cl, Rleft, Rleft, Rright; mod_array = test_mod_array,
)

circuitdefs = Dict(
    Lj => 79e-12, Cj => 145.0e-15, Cg => 33e-15,
    Cl => 10 * 1.73e-15, Ll => 10 * 4.13e-12,
    Rleft => 50.0, Rright => 50.0,
)

ws = 2 * pi * range(signal_min, signal_max; length = npoints) * 1e9
wp = (2 * pi * pump_freq_ghz * 1e9,)
sources = [(mode = (1,), port = 4, current = pump_current_a)]
Npumpharmonics = (20,)
Nmodulationharmonics = (10,)

println("Ic = ", LjtoIc(79e-12))
println("pump source current = ", pump_current_a, " A (",
        pump_current_a / LjtoIc(79e-12), " Ic), pump ", pump_freq_ghz, " GHz")

@time rpm = hbsolve(ws, wp, sources, Nmodulationharmonics, Npumpharmonics,
    circuit, circuitdefs)

S21 = rpm.linearized.S(outputmode=(0,), outputport=2, inputmode=(0,),
    inputport=1, freqindex=:)
gain_db = 10 .* log10.(abs2.(S21))
freqs_ghz = ws ./ (2 * pi * 1e9)

open(outcsv, "w") do io
    println(io, "signal_ghz,gain_db")
    for (f, g) in zip(freqs_ghz, gain_db)
        println(io, "$(f),$(g)")
    end
end

imax = argmax(gain_db)
println("PEAK_GAIN_DB = ", gain_db[imax], " at signal ", freqs_ghz[imax], " GHz")
println("wrote ", outcsv)
