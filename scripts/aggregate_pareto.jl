# Pareto analysis of 3L tune configurations.
# For each results/tune_3L/<cfg>/ subfolder (and the legacy 8-seed baselines
# in results/), compute 12-mo ACC, RMSE, std_ratio, and spatial pc; then
# identify the Pareto frontier on (RMSE↓, pc↑) — i.e. configs not dominated
# by any other.

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CrossScaleRC, JLD2, Statistics, LinearAlgebra, Glob, Plots, Measures, Printf

const RESULTS = joinpath(@__DIR__, "..", "results")
const TUNE = joinpath(RESULTS, "tune_3L")

function spatial_pc(A::AbstractMatrix, B::AbstractMatrix)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end

"""Compute 12-mo metrics for an ensemble of 3L runs in `dir`."""
function metrics(dir; seed_filter=nothing, lead=12)
    paths = sort(glob("enso_monthly_preds_three_layer_seed*.jld2", dir))
    isempty(paths) && return nothing
    n34_true = nothing
    P = Vector{Vector{Float64}}()
    fields = Array{Float64,3}[]
    test_f = nothing
    seeds = Int[]
    for p in paths
        f = jldopen(p)
        s = Int(f["seed"])
        if seed_filter !== nothing && !(s in seed_filter); close(f); continue; end
        n34_true = Float64.(f["n34_true"])
        push!(P, Float64.(f["n34_pred"]))
        push!(fields, Float64.(f["preds_f"]))
        push!(seeds, s)
        if test_f === nothing && "test_f" in keys(f)
            test_f = Float64.(f["test_f"])
        end
        close(f)
    end
    if test_f === nothing
        ref = joinpath(dir, "enso_monthly_reference_three_layer.jld2")
        if isfile(ref); f = jldopen(ref); test_f = Float64.(f["test_f"]); close(f); end
    end
    test_f === nothing && return nothing
    ensn = vec(mean(reduce(vcat, [reshape(p, 1, length(p)) for p in P]), dims=1))
    truth_w = n34_true[1:lead]; pred_w = ensn[1:lead]
    acc = skill_score(truth_w, pred_w).acc
    rmse = sqrt(mean((pred_w .- truth_w) .^ 2))
    std_ratio = std(pred_w) / std(truth_w)
    field_ens = cat(fields...; dims=4)
    field_mean = dropdims(mean(field_ens; dims=4); dims=4)
    pc = spatial_pc(test_f[:, :, lead], field_mean[:, :, lead])
    return (acc=acc, rmse=rmse, std_ratio=std_ratio, pc=pc, nseeds=length(seeds))
end

# Collect all configs.
rows = NamedTuple[]
match_seeds = Set([1, 42, 99])
match8 = Set([1, 2, 3, 7, 11, 23, 42, 99])

# Reference points
m = metrics(RESULTS; seed_filter=match8)
m !== nothing && push!(rows, (name="3L baseline (match8)", values=m))
m = metrics(RESULTS; seed_filter=match_seeds)
m !== nothing && push!(rows, (name="3L baseline (match3)", values=m))

# All tuning subfolders
for d in sort(readdir(TUNE; join=true))
    isdir(d) || continue
    name = basename(d)
    m = metrics(d)
    m !== nothing && push!(rows, (name=name, values=m))
end

# ===== Print sorted table =====
sort!(rows; by=r -> r.values.rmse)  # sort by RMSE ascending
println("="^110)
println("All 3L configurations — 12-month metrics (sorted by RMSE)")
println("="^110)
@printf("%-30s | %-2s | %-7s | %-7s | %-8s | %-7s | %s\n",
        "config", "N", "ACC", "RMSE", "std_rat", "pc", "Pareto")
println("-"^110)

# Compute Pareto frontier on (RMSE↓, pc↑)
function dominated(rs, ix)
    # ix is dominated if any other has lower-or-equal RMSE AND higher-or-equal pc,
    # with at least one strict.
    here = rs[ix].values
    for jx in eachindex(rs)
        jx == ix && continue
        other = rs[jx].values
        if other.rmse ≤ here.rmse && other.pc ≥ here.pc &&
           (other.rmse < here.rmse || other.pc > here.pc)
            return true
        end
    end
    return false
end
pareto_idx = [ix for ix in eachindex(rows) if !dominated(rows, ix)]
pareto_set = Set(pareto_idx)

for (ix, r) in enumerate(rows)
    v = r.values
    pflag = ix in pareto_set ? "★" : " "
    @printf("%-30s | %2d | %+.3f  | %.3f  | %5.2f    | %+.3f  | %s\n",
            r.name, v.nseeds, v.acc, v.rmse, v.std_ratio, v.pc, pflag)
end
println("="^110)
println("★ = Pareto-optimal on (RMSE↓, pc↑)")

# ===== Plot =====
xs = [r.values.rmse for r in rows]
ys = [r.values.pc for r in rows]
labels = [r.name for r in rows]

# Color by family
function color_of(name)
    if startswith(name, "3L baseline"); return :gray
    elseif startswith(name, "tauC");    return :orange
    elseif startswith(name, "joint");   return :purple
    elseif startswith(name, "tauF");    return :green
    elseif startswith(name, "rF");      return :brown
    elseif startswith(name, "gB") && !startswith(name, "gB-"); return :red
    elseif startswith(name, "gA");      return :magenta
    elseif startswith(name, "tauM");    return :cyan
    elseif startswith(name, "grid");    return :teal
    elseif startswith(name, "Ncoarse") || startswith(name, "Nfine"); return :pink
    elseif startswith(name, "radCoarse") || startswith(name, "radFine"); return :navy
    else; return :black
    end
end
markers = [ix in pareto_set ? :star5 : :circle for ix in eachindex(rows)]
sizes   = [ix in pareto_set ? 10 : 5 for ix in eachindex(rows)]
colors  = [color_of(n) for n in labels]

p = scatter(xs, ys; markershape=markers, markersize=sizes,
            color=colors, alpha=0.85,
            xlabel="12-mo Niño 3.4 RMSE  (lower = better)",
            ylabel="12-mo spatial pattern correlation  (higher = better)",
            title="3L Pareto frontier — RMSE vs spatial pc  (★ = Pareto-optimal)",
            legend=false, grid=true, size=(1300, 850))

# Connect Pareto frontier
pareto_sorted = sort(collect(pareto_idx); by=ix -> rows[ix].values.rmse)
pxs = [rows[ix].values.rmse for ix in pareto_sorted]
pys = [rows[ix].values.pc for ix in pareto_sorted]
plot!(p, pxs, pys; lw=2, color=:gold, ls=:dash, label="Pareto frontier")

# Annotate Pareto points and a few notable ones
for ix in pareto_idx
    annotate!(p, rows[ix].values.rmse, rows[ix].values.pc + 0.015,
              text(rows[ix].name, 7, :black))
end

savefig(p, joinpath(RESULTS, "tune_3L_pareto.png"))
println("\nSaved: results/tune_3L_pareto.png")
