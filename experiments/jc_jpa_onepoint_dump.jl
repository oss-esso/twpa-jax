using JosephsonCircuits
using Symbolics
using LinearAlgebra
using SparseArrays

@variables R Cc Lj Cj

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

ws = 2π .* [4.75] .* 1.0e9
wp = (2π * 4.75001e9,)
Ip = 0.00565e-6
sources = [(mode=(1,), port=1, current=Ip)]
Npumpharmonics = (16,)
Nmodulationharmonics = (8,)

function safe_show(obj, name::Symbol)
    if hasproperty(obj, name)
        try
            println(string(name), " = ", getproperty(obj, name))
        catch err
            println(string(name), " inspect failed: ", err)
        end
    else
        println(string(name), " = <missing>")
    end
end

println("=== keyed JC observable solve ===")

sol_keyed = JosephsonCircuits.hbsolve(
    ws,
    wp,
    sources,
    Nmodulationharmonics,
    Npumpharmonics,
    circuit,
    circuitdefs;
    returnnodeflux=true,
    returnvoltage=true,
)

svec = collect(sol_keyed.linearized.S((0,), 1, (0,), 1, :))
s11 = ComplexF64(svec[1])
gain_db = real(10.0 * log10(abs2(s11)))

println("\n=== exact adapter-style observable ===")
println("s11 = ", s11)
println("s11_abs = ", abs(s11))
println("gain_db = ", gain_db)

println("\n=== raw-array solve for diagnostics ===")

sol = JosephsonCircuits.hbsolve(
    ws,
    wp,
    sources,
    Nmodulationharmonics,
    Npumpharmonics,
    circuit,
    circuitdefs;
    returnnodeflux=true,
    returnvoltage=true,
    keyedarrays=Val(false),
)

lin = sol.linearized
nl = sol.nonlinear

println("\n=== nonlinear ===")
println("typeof(nonlinear) = ", typeof(nl))
println("nonlinear fields = ", fieldnames(typeof(nl)))
safe_show(nl, :modes)
safe_show(nl, :ports)
safe_show(nl, :nodes)
safe_show(nl, :Nmodes)
safe_show(nl, :Nbranches)

try
    println("nonlinear nodeflux size = ", size(nl.nodeflux))
    println("nonlinear nodeflux first 20 = ", vec(nl.nodeflux)[1:min(end,20)])
catch err
    println("nonlinear nodeflux inspect failed: ", err)
end

println("\n=== linearized ===")
println("typeof(linearized) = ", typeof(lin))
println("linearized fields = ", fieldnames(typeof(lin)))
safe_show(lin, :modes)
safe_show(lin, :signalindex)
safe_show(lin, :portnumbers)
safe_show(lin, :portindices)
safe_show(lin, :portimpedanceindices)
safe_show(lin, :Nsignalmodes)
safe_show(lin, :Nports)
safe_show(lin, :Nnodes)

S = lin.S
println("S typeof = ", typeof(S))
println("S size = ", size(S))

sig = hasproperty(lin, :signalindex) ? lin.signalindex : 1
s11_raw = S[sig, sig, 1]
gain_db_raw = real(10.0 * log10(abs2(s11_raw)))

println("\n=== raw S central entry ===")
println("sig = ", sig)
println("s11_raw = ", s11_raw)
println("s11_raw_abs = ", abs(s11_raw))
println("gain_db_raw = ", gain_db_raw)

try
    println("\n=== linearized nodeflux/voltage raw arrays ===")
    println("nodeflux typeof = ", typeof(lin.nodeflux))
    println("nodeflux size = ", size(lin.nodeflux))
    println("voltage typeof = ", typeof(lin.voltage))
    println("voltage size = ", size(lin.voltage))
    println("nodeflux first 40 = ", vec(lin.nodeflux)[1:min(end,40)])
    println("voltage first 40 = ", vec(lin.voltage)[1:min(end,40)])
catch err
    println("linearized nodeflux/voltage inspect failed: ", err)
end
