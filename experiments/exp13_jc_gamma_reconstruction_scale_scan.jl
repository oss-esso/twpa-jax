using JosephsonCircuits
using LinearAlgebra
using SparseArrays
using Statistics
using Printf

cases_dir = raw"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs"
outdir = raw"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_gamma_hat_compare"
mkpath(outdir)

py_summary_path = joinpath(outdir, "python_gamma_hat_summary.csv")

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

function read_python_rel_l2(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    ell_i = findfirst(==("ell"), header)
    rel_i = findfirst(==("l2_abs_over_zero_l2"), header)
    d = Dict{Int, Float64}()
    for line in lines[2:end]
        s = split(line, ",")
        d[Int(round(parse(Float64, s[ell_i])))] = parse(Float64, s[rel_i])
    end
    return d
end

py_rel = read_python_rel_l2(py_summary_path)

_case_obj, artifacts = build_jc_jtwpa_case()
ws = 2*pi .* [6.6e9]

println("=== running JC JTWPA reference solve for gamma reconstruction scan ===")

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

modes = [m[1] for m in collect(sol.nonlinear.modes)]
phin = Array(sol.nonlinear.nodeflux)
Rbnm = sol.nonlinear.Rbnm
Ljb = Vector(sol.nonlinear.Ljb)

Nm = length(modes)
Nnodes = size(phin, 2)
Nbranches = length(Ljb)
junction_idx = findall(x -> x != 0.0, Ljb)

phi0 = 2.067833848e-15 / (2*pi)
Lmean = mean(Ljb[junction_idx])

println("runtime_s=", timed.time)
println("modes=", modes)
println("phin_size=", size(phin))
println("Rbnm_size=", size(Rbnm))
println("Nbranches=", Nbranches)
println("junctions=", length(junction_idx))
println("Lmean=", Lmean)
println("phi0=", phi0)

layout_candidates = [
    ("vec_phin__reshape_Nm_Nb",
        vec(phin),
        v -> reshape(v, Nm, Nbranches)),

    ("vec_permuted_phin__reshape_Nm_Nb",
        vec(permutedims(phin)),
        v -> reshape(v, Nm, Nbranches)),

    ("vec_phin__reshape_Nb_Nm_transpose",
        vec(phin),
        v -> permutedims(reshape(v, Nbranches, Nm))),

    ("vec_permuted_phin__reshape_Nb_Nm_transpose",
        vec(permutedims(phin)),
        v -> permutedims(reshape(v, Nbranches, Nm))),
]

# phase = phase_scale * reconstructed_branch_value
# We do not know yet whether JC's nodeflux is raw SI flux, phase, or scaled.
scale_candidates = [
    ("phase_already", 1.0),
    ("half_phase", 0.5),
    ("double_phase", 2.0),
    ("SI_flux_to_phase_1_over_phi0", 1.0 / phi0),
    ("scaled_by_Lmean_over_phi0", Lmean / phi0),
    ("scaled_by_1_over_Lmean", 1.0 / Lmean),
    ("scaled_by_phi0_over_Lmean", phi0 / Lmean),
    ("scaled_by_Lmean", Lmean),
    ("scaled_by_phi0", phi0),
]

# If JC stores positive-frequency phasors, physical real signal is 2*Re(sum).
# If it stores cosine-like amplitudes, factor 1 may be right.
real_factors = [1.0, 2.0]

max_ell = 20
nt = 96
ells = collect(-max_ell:max_ell)
score_ells = [-10,-8,-6,-4,-2,0,2,4,6,8,10]

function reconstruct_branch_time(psi_modes, real_factor, ti, nt)
    th = 2*pi*ti/nt
    vals = zeros(ComplexF64, length(junction_idx))
    for (ki, k) in enumerate(modes)
        vals .+= psi_modes[ki, junction_idx] .* exp(im*k*th)
    end
    return real_factor .* real.(vals)
end

function gamma_summary_for(psi_modes, phase_scale, real_factor)
    gamma_hat = Dict{Int, Vector{ComplexF64}}()
    for ell in ells
        gamma_hat[ell] = zeros(ComplexF64, length(junction_idx))
    end

    for ti in 0:(nt-1)
        th = 2*pi*ti/nt
        psi_real = reconstruct_branch_time(psi_modes, real_factor, ti, nt)
        phase = phase_scale .* psi_real

        gamma_t = Vector{ComplexF64}(undef, length(junction_idx))
        for (jj, bidx) in enumerate(junction_idx)
            gamma_t[jj] = (1.0 / Ljb[bidx]) * cos(phase[jj])
        end

        for ell in ells
            gamma_hat[ell] .+= gamma_t .* exp(-im*ell*th) / nt
        end
    end

    zero_gamma = [1.0 / Ljb[bidx] for bidx in junction_idx]
    zero_norm = max(norm(zero_gamma), 1e-300)

    rel = Dict{Int, Float64}()
    maxabs = Dict{Int, Float64}()
    meanabs = Dict{Int, Float64}()

    for ell in ells
        gh = gamma_hat[ell]
        rel[ell] = norm(gh) / zero_norm
        maxabs[ell] = maximum(abs.(gh))
        meanabs[ell] = mean(abs.(gh))
    end

    return rel, maxabs, meanabs
end

function score_against_python(rel)
    vals = Float64[]
    for ell in score_ells
        a = max(rel[ell], 1e-300)
        b = max(py_rel[ell], 1e-300)
        push!(vals, log10(a / b)^2)
    end
    return sqrt(mean(vals))
end

rows = []

best = nothing
best_rel = nothing
best_maxabs = nothing
best_meanabs = nothing

for (layout_name, nodevec, reshape_fn) in layout_candidates
    if length(nodevec) != size(Rbnm, 2)
        println("skip layout=", layout_name, " nodevec_len=", length(nodevec))
        continue
    end

    branch_vec = Rbnm * nodevec
    psi_modes = reshape_fn(branch_vec)

    for real_factor in real_factors
        for (scale_name, phase_scale) in scale_candidates
            rel, maxabs, meanabs = gamma_summary_for(psi_modes, phase_scale, real_factor)
            score = score_against_python(rel)

            row = (
                score=score,
                layout_name=layout_name,
                real_factor=real_factor,
                scale_name=scale_name,
                phase_scale=phase_scale,
                rel0=rel[0],
                rel2=rel[2],
                rel4=rel[4],
                rel6=rel[6],
                rel8=rel[8],
                rel10=rel[10],
            )
            push!(rows, row)

            global best, best_rel, best_maxabs, best_meanabs
            if best === nothing || score < best.score
                best = row
                best_rel = rel
                best_maxabs = maxabs
                best_meanabs = meanabs
            end
        end
    end
end

sort!(rows, by = r -> r.score)

println()
println("TOP_CANDIDATES")
for r in rows[1:min(12, length(rows))]
    @printf(
        "score=%.6f layout=%s real_factor=%.1f scale=%s phase_scale=%.6e rel0=%.9e rel2=%.9e rel4=%.9e rel6=%.9e rel8=%.9e rel10=%.9e\n",
        r.score, r.layout_name, r.real_factor, r.scale_name, r.phase_scale,
        r.rel0, r.rel2, r.rel4, r.rel6, r.rel8, r.rel10
    )
end

println()
println("BEST_DETAIL")
println("score=", best.score)
println("layout=", best.layout_name)
println("real_factor=", best.real_factor)
println("scale_name=", best.scale_name)
println("phase_scale=", best.phase_scale)

println()
println("ell py_rel jc_rel ratio jc_max_abs jc_mean_abs")
for ell in [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
    pyr = py_rel[ell]
    jcr = best_rel[ell]
    ratio = jcr / max(pyr, 1e-300)
    @printf(
        "%+4d %.9e %.9e %.6f %.9e %.9e\n",
        ell, pyr, jcr, ratio, best_maxabs[ell], best_meanabs[ell]
    )
end

summary_path = joinpath(outdir, "jc_gamma_reconstruction_scale_scan.csv")
open(summary_path, "w") do io
    println(io, "score,layout,real_factor,scale_name,phase_scale,rel0,rel2,rel4,rel6,rel8,rel10")
    for r in rows
        println(io, join([
            r.score,
            r.layout_name,
            r.real_factor,
            r.scale_name,
            r.phase_scale,
            r.rel0,
            r.rel2,
            r.rel4,
            r.rel6,
            r.rel8,
            r.rel10,
        ], ","))
    end
end

println()
println("wrote_scan=", summary_path)
