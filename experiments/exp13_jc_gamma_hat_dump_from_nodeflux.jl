using JosephsonCircuits
using LinearAlgebra
using SparseArrays
using Statistics

cases_dir = raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs"
outdir = raw"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_gamma_hat_compare"
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

ws = 2*pi .* [6.6e9]

println("=== running JC JTWPA reference solve for gamma_hat dump ===")

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

println("runtime_s=", timed.time)
println("typeof(sol)=", typeof(sol))

modes = [m[1] for m in collect(sol.nonlinear.modes)]
phin = Array(sol.nonlinear.nodeflux)
Rbnm = sol.nonlinear.Rbnm
Ljb = Vector(sol.nonlinear.Ljb)

Nm = length(modes)
Nnodes = size(phin, 2)
Nbranches = length(Ljb)

println("modes=", modes)
println("phin_size=", size(phin))
println("Rbnm_size=", size(Rbnm))
println("Ljb_len=", length(Ljb))
println("Ljb_nonzero=", count(!=(0.0), Ljb))

phi0 = 2.067833848e-15 / (2*pi)

# Candidate ways to flatten nodeflux and reshape branch-mode output.
# We pick the one whose reconstructed branch flux is most real in time.
candidates = []

push!(candidates, ("vec_phin__reshape_Nm_Nb",
    vec(phin),
    (v -> reshape(v, Nm, Nbranches))
))

push!(candidates, ("vec_permuted_phin__reshape_Nm_Nb",
    vec(permutedims(phin)),
    (v -> reshape(v, Nm, Nbranches))
))

push!(candidates, ("vec_phin__reshape_Nb_Nm_transpose",
    vec(phin),
    (v -> permutedims(reshape(v, Nbranches, Nm)))
))

push!(candidates, ("vec_permuted_phin__reshape_Nb_Nm_transpose",
    vec(permutedims(phin)),
    (v -> permutedims(reshape(v, Nbranches, Nm)))
))

junction_idx = findall(x -> x != 0.0, Ljb)

function reality_error(psi_modes)
    nt = 96
    worst = 0.0
    for ti in 0:(nt-1)
        th = 2*pi*ti/nt
        vals = zeros(ComplexF64, length(junction_idx))
        for (ki, k) in enumerate(modes)
            vals .+= psi_modes[ki, junction_idx] .* exp(im*k*th)
        end
        denom = maximum(abs.(vals))
        if denom > 0
            err = maximum(abs.(imag.(vals))) / denom
            worst = max(worst, err)
        end
    end
    return worst
end

best_name = ""
best_err = Inf
best_psi_modes = nothing

for (name, nodevec, reshape_fn) in candidates
    if length(nodevec) != size(Rbnm, 2)
        println("candidate=", name, " skipped length mismatch nodevec=", length(nodevec))
        continue
    end

    branch_vec = Rbnm * nodevec
    psi_modes = reshape_fn(branch_vec)

    err = reality_error(psi_modes)
    println("candidate=", name, " reality_err=", err)

    if err < best_err
        global best_err = err
        global best_name = name
        global best_psi_modes = psi_modes
    end
end

println("BEST_LAYOUT=", best_name)
println("BEST_REALITY_ERR=", best_err)

psi_modes = best_psi_modes

max_ell = 20
nt = 96
ells = collect(-max_ell:max_ell)

# gamma_hat[ell] over only physical Josephson nonlinear branches.
gamma_hat = Dict{Int, Vector{ComplexF64}}()

for ell in ells
    gamma_hat[ell] = zeros(ComplexF64, length(junction_idx))
end

for ti in 0:(nt-1)
    th = 2*pi*ti/nt

    psi_t = zeros(ComplexF64, length(junction_idx))
    for (ki, k) in enumerate(modes)
        psi_t .+= psi_modes[ki, junction_idx] .* exp(im*k*th)
    end

    # The physical branch flux should be real. Drop tiny numerical imaginary part.
    psi_real = real.(psi_t)

    gamma_t = similar(psi_t)
    for (jj, bidx) in enumerate(junction_idx)
        gamma_t[jj] = (1.0 / Ljb[bidx]) * cos(psi_real[jj] / phi0)
    end

    for ell in ells
        # Same convention as Khat_ell = (1/T) int K(t) exp(-i ell wp t) dt
        gamma_hat[ell] .+= gamma_t .* exp(-im*ell*th) / nt
    end
end

zero_gamma = [1.0 / Ljb[bidx] for bidx in junction_idx]
zero_norm = max(norm(zero_gamma), 1e-300)

summary_path = joinpath(outdir, "jc_gamma_hat_summary.csv")
coeff_path = joinpath(outdir, "jc_gamma_hat_branch_coeffs_selected.csv")

open(summary_path, "w") do io
    println(io, "ell,nbranches,l2_abs,l2_abs_over_zero_l2,max_abs,mean_abs,mean_real,mean_imag,conj_symmetry_rel_err")
    for ell in ells
        gh = gamma_hat[ell]
        ghneg = gamma_hat[-ell]
        denom = max(norm(gh), norm(ghneg), 1e-300)
        conjerr = norm(ghneg .- conj.(gh)) / denom
        println(io, join([
            ell,
            length(gh),
            norm(gh),
            norm(gh) / zero_norm,
            maximum(abs.(gh)),
            mean(abs.(gh)),
            mean(real.(gh)),
            mean(imag.(gh)),
            conjerr,
        ], ","))
    end
end

selected_ells = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]

open(coeff_path, "w") do io
    println(io, "branch_rank,jc_branch_index,ell,real,imag,abs,rel_to_zero_branch")
    for ell in selected_ells
        gh = gamma_hat[ell]
        for (rank, bidx) in enumerate(junction_idx)
            z = gh[rank]
            rel = abs(z) / max(abs(zero_gamma[rank]), 1e-300)
            println(io, join([rank, bidx, ell, real(z), imag(z), abs(z), rel], ","))
        end
    end
end

println("JC_GAMMA_DUMP_OK")
println("layout=", best_name)
println("reality_err=", best_err)
println("nbranches=", length(junction_idx))
println("wrote_summary=", summary_path)
println("wrote_coeffs=", coeff_path)
println()
println("Most relevant summary rows:")
for ell in [-10,-8,-6,-4,-2,0,2,4,6,8,10]
    gh = gamma_hat[ell]
    ghneg = gamma_hat[-ell]
    denom = max(norm(gh), norm(ghneg), 1e-300)
    conjerr = norm(ghneg .- conj.(gh)) / denom
    println(
        "ell=", lpad(string(ell), 3),
        " rel_l2=", norm(gh)/zero_norm,
        " max_abs=", maximum(abs.(gh)),
        " mean_abs=", mean(abs.(gh)),
        " conj_err=", conjerr
    )
end
