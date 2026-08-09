using JosephsonCircuits
using Symbolics
using Statistics

@variables R Cc Lj Cj

outdir = raw"D:\Projects\Thesis\twpa_jax\outputs\exp13_compare"
mkpath(outdir)

out_csv = joinpath(outdir, "jc_jpa_curve.csv")

circuit = [
    ("P1","1","0",1),
    ("R1","1","0",R),
    ("C1","1","2",Cc),
    ("Lj1","2","0",Lj),
    ("C2","2","0",Cj),
]

circuitdefs = Dict(
    Lj => 1000.0e-12,
    Cc => 100.0e-15,
    Cj => 1000.0e-15,
    R => 50.0,
)

ws = 2π .* collect(4.5:0.001:5.0) .* 1.0e9
wp = (2π * 4.75001e9,)
Ip = 0.00565e-6

sources = [(mode=(1,), port=1, current=Ip)]
Npumpharmonics = (16,)
Nmodulationharmonics = (8,)

println("=== running JC JPA 501-point gain curve ===")

timed = @timed JosephsonCircuits.hbsolve(
    ws,
    wp,
    sources,
    Nmodulationharmonics,
    Npumpharmonics,
    circuit,
    circuitdefs,
)

sol = timed.value

svec = ComplexF64.(collect(sol.linearized.S((0,), 1, (0,), 1, :)))
gain_db = real.(10.0 .* log10.(abs2.(svec)))
freq_ghz = ws ./ (2π * 1.0e9)

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

println("wrote_csv=", out_csv)
println("points=", length(freq_ghz))
println("gain_db_max=", maximum(gain_db))
println("gain_db_mean=", mean(gain_db))
println("gain_db_min=", minimum(gain_db))
println("peak_frequency_ghz=", freq_ghz[best_i])
println("runtime_s=", timed.time)
