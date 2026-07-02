# experiments/jc_raw_timing.jl
# Time a raw JosephsonCircuits.jl doc standalone (…/jc_docs/raw/<DESIGN>.jl),
# EXCLUDING plotting and JIT. Run under an env that has JosephsonCircuits
# (e.g. --project=<Harmonia.jl>); Plots need NOT be installed -- its calls are
# stubbed to no-ops.
#
# Usage:
#   julia --project=<Harmonia.jl> jc_raw_timing.jl <path-to-raw/DESIGN.jl>
#
# The raw file is evaluated twice: pass 1 pays JIT (warmup), pass 2 is timed.
# The design's own `@time hbsolve(...)` lines print the runtime; we echo the
# timed-pass total as `runtime_s=`.

using JosephsonCircuits
using Printf

# Plotting is out of scope: stub the Plots entry points to no-ops so `using
# Plots` (removed below) and any plot(...) calls do nothing.
plot(args...; kwargs...) = nothing
plot!(args...; kwargs...) = nothing
scatter(args...; kwargs...) = nothing
scatter!(args...; kwargs...) = nothing
savefig(args...; kwargs...) = nothing
gui(args...; kwargs...) = nothing

# One or more raw files; concatenated in order (e.g. a base design + its
# continuation variant that reuses the base circuit).
bodies = [replace(read(p, String), "using Plots" => "") for p in ARGS]
# Drop the plotting dependency; plot(...) resolves to the stubs above.
# Wrap in a `let` block so the doc files' top-level `for` loops that mutate a
# preceding global (e.g. `j += 1`) bind to block-locals -- this is soft scope,
# matching how the standalones behave when run interactively.
src = "let\n" * join(bodies, "\n") * "\nend\n"

# Time every hbsolve call in a pass by wrapping the printed @time; simplest is to
# capture wall time around the whole evaluation of the solve section. The doc
# files already annotate the solve with @time, whose seconds we parse from the
# captured stdout.
function run_pass(tag::String)
    println("=== $(tag) ===")
    # Capture the design's own `@time hbsolve(...)` prints; sum their seconds so
    # runtime_s is the pure solve time (excludes circuit setup and plotting).
    # include() a real file (soft top-level scope, unlike include_string) so the
    # doc files' top-level `for` loops that mutate globals work as when run
    # directly.
    srcfile = tempname() * ".jl"
    write(srcfile, src)
    tmp = tempname()
    open(tmp, "w") do io
        redirect_stdout(io) do
            include(srcfile)
        end
    end
    rm(srcfile; force = true)
    out = read(tmp, String)
    rm(tmp; force = true)
    print(out)
    return [parse(Float64, m.captures[1])
            for m in eachmatch(r"([0-9]+\.[0-9]+) seconds", out)]
end

run_pass("WARMUP (JIT)")
timed = run_pass("TIMED (warm)")

@printf("hbsolve_time_calls=%d\n", length(timed))
println("hbsolve_times_s=", join(timed, ","))
@printf("runtime_s=%.6f\n", isempty(timed) ? NaN : sum(timed))
