using JosephsonCircuits
using Statistics
using LinearAlgebra

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

# One signal point is enough to force the same pump solve and construct the linearized object.
ws = 2π .* [6.6e9]

println("=== running JC JTWPA reference solve for introspection ===")

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
println("artifact_keys=", sort(collect(keys(artifacts))))

report_path = joinpath(outdir, "jc_solution_introspection.txt")
array_path = joinpath(outdir, "jc_solution_candidate_arrays.csv")

function short_type(x)
    return string(typeof(x))
end

function shape_string(x)
    if x isa AbstractArray
        return string(size(x))
    else
        return ""
    end
end

function scalar_summary(x)
    try
        if x isa Number
            return string(x)
        elseif x isa AbstractArray
            return "array eltype=$(eltype(x)) size=$(size(x)) length=$(length(x))"
        else
            return ""
        end
    catch
        return ""
    end
end

function safe_props(x)
    try
        return collect(propertynames(x))
    catch
        return Symbol[]
    end
end

function is_leaf(x)
    return x isa Number || x isa AbstractString || x isa Symbol || x isa Function || x isa Module || x isa AbstractArray || x === nothing
end

function print_tree(io, label, x; depth=0, maxdepth=4, seen=IdDict())
    indent = repeat("  ", depth)

    println(io, indent, label, " :: ", typeof(x), " ", scalar_summary(x))

    if depth >= maxdepth || is_leaf(x)
        return
    end

    if haskey(seen, x)
        println(io, indent, "  <already seen>")
        return
    end
    seen[x] = true

    props = safe_props(x)
    if isempty(props)
        return
    end

    for p in props
        child_label = string(label, ".", p)
        try
            child = getproperty(x, p)
            print_tree(io, child_label, child; depth=depth+1, maxdepth=maxdepth, seen=seen)
        catch err
            println(io, indent, "  ", child_label, " :: <error reading: ", err, ">")
        end
    end
end

candidate_rows = Vector{Tuple{String,String,String,Int,String}}()

function collect_arrays!(rows, label, x; depth=0, maxdepth=5, seen=IdDict())
    if x isa AbstractArray
        push!(rows, (label, string(typeof(x)), string(size(x)), length(x), string(eltype(x))))
        return
    end

    if depth >= maxdepth || is_leaf(x)
        return
    end

    if haskey(seen, x)
        return
    end
    seen[x] = true

    for p in safe_props(x)
        try
            child = getproperty(x, p)
            collect_arrays!(rows, string(label, ".", p), child; depth=depth+1, maxdepth=maxdepth, seen=seen)
        catch
        end
    end
end

open(report_path, "w") do io
    println(io, "JC JTWPA solution introspection")
    println(io, "runtime_s=", timed.time)
    println(io, "typeof(sol)=", typeof(sol))
    println(io, "artifact keys:")
    for k in sort(collect(keys(artifacts)))
        v = artifacts[k]
        println(io, "  ", k, " :: ", typeof(v), " ", scalar_summary(v))
    end
    println(io)
    println(io, "PROPERTY_TREE")
    print_tree(io, "sol", sol; maxdepth=5)
end

collect_arrays!(candidate_rows, "sol", sol; maxdepth=6)

open(array_path, "w") do io
    println(io, "path,type,size,length,eltype")
    for (path, typ, siz, len, elt) in candidate_rows
        println(io, "\"", path, "\",\"", typ, "\",\"", siz, "\",", len, ",\"", elt, "\"")
    end
end

println("INTROSPECTION_OK")
println("wrote_report=", report_path)
println("wrote_arrays=", array_path)
println()
println("CANDIDATE_ARRAYS")
for (path, typ, siz, len, elt) in candidate_rows
    low = lowercase(path)
    if occursin("phi", low) || occursin("pump", low) || occursin("harm", low) || occursin("freq", low) || occursin("mode", low) || occursin("sol", low) || occursin("linear", low)
        println(path, " :: ", typ, " size=", siz, " length=", len, " eltype=", elt)
    end
end
