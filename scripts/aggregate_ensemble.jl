# Aggregate ENSO ensemble runs saved as results/enso_preds_*_seed*.jld2
# by main_enso.jl. Produces an ensemble-mean Niño 3.4 forecast and
# reports skill metrics, plus a comparison plot.
#
# Usage:
#   julia --project=. scripts/aggregate_ensemble.jl [mode_tag]
# Default mode_tag = "three_layer".

ENV["GKSwstype"] = "100"
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CrossScaleRC
using JLD2, Statistics, Plots, Measures, Glob

mode_tag = length(ARGS) > 0 ? ARGS[1] : "three_layer"
results_dir = joinpath(@__DIR__, "..", "results")
paths = sort(glob("enso_preds_$(mode_tag)_seed*.jld2", results_dir))
isempty(paths) && error("No ensemble files found for mode=$(mode_tag) in $(results_dir)")
println("Found $(length(paths)) ensemble members:")
foreach(p -> println("  ", basename(p)), paths)

# Load n34_pred from each seed; n34_true is the same across seeds (sanity check).
function load_ensemble(paths)
    preds = Matrix{Float64}[]
    n34_true_ref = nothing
    seeds = Int[]
    per_seed = Dict{Int,Float64}()
    for p in paths
        f = jldopen(p)
        n34_true = Float64.(f["n34_true"])
        n34_pred = Float64.(f["n34_pred"])
        seed     = Int(f["seed"])
        close(f)
        if n34_true_ref === nothing
            n34_true_ref = n34_true
        else
            @assert n34_true_ref ≈ n34_true "n34_true differs across seeds"
        end
        push!(preds, reshape(n34_pred, 1, length(n34_pred)))
        push!(seeds, seed)
        per_seed[seed] = skill_score(n34_true, n34_pred).acc
    end
    return preds, n34_true_ref, seeds, per_seed
end
preds, n34_true_ref, seeds, per_seed = load_ensemble(paths)

P = vcat(preds...)   # (n_seeds × T)
ensemble_mean   = vec(mean(P, dims=1))
ensemble_median = vec(median(P, dims=1))

sc_mean   = skill_score(n34_true_ref, ensemble_mean)
sc_median = skill_score(n34_true_ref, ensemble_median)

println()
println("Per-seed full-window Niño 3.4 ACC:")
for s in seeds
    println("  seed=$(lpad(s, 3)):  ACC=$(round(per_seed[s]; digits=3))")
end
println()
println("Ensemble mean   (n=$(length(seeds))):  ACC=$(round(sc_mean.acc;   digits=3))   RMSE=$(round(sc_mean.rmse;   digits=4))   skill_vs_pers=$(round(sc_mean.rmse_skill;   digits=3))")
println("Ensemble median (n=$(length(seeds))):  ACC=$(round(sc_median.acc; digits=3))   RMSE=$(round(sc_median.rmse; digits=4))   skill_vs_pers=$(round(sc_median.rmse_skill; digits=3))")

# Lead-time ACC for the ensemble mean
month_steps = 120
horizons = [1, 2, 3, 4, 6, 8] .* month_steps
println("\nEnsemble-mean cumulative lead-time ACC:")
for H in horizons
    H > length(ensemble_mean) && continue
    s = skill_score(n34_true_ref[1:H], ensemble_mean[1:H])
    println("  ~$(round(H / month_steps; digits=1)) mo  (H=$H): ACC=$(round(s.acc; digits=3))   RMSE=$(round(s.rmse; digits=4))")
end

# Plot: each seed's prediction (thin grey), ensemble mean (bold red), ground truth (black)
p1 = plot(n34_true_ref, lw=2.5, color=:black, label="Observed",
          xlabel="step (post-warmup)", ylabel="Niño 3.4 (norm.)",
          title="ENSO ensemble  (n=$(length(seeds))) — mode=$(mode_tag)",
          legend=:topright)
for (i, s) in enumerate(seeds)
    plot!(p1, P[i, :], lw=0.8, color=:gray, alpha=0.45,
          label=i == 1 ? "individual seeds" : false)
end
plot!(p1, ensemble_mean, lw=2.2, color=:red, linestyle=:dash,
      label="ensemble mean  (ACC=$(round(sc_mean.acc; digits=2)))")

# Lead-time curve: per-seed + ensemble
p2 = plot(xlabel="Lead (months)", ylabel="Niño 3.4 ACC",
          title="Lead-time skill  (cumulative ACC)", ylim=(-1.0, 1.0), legend=:bottomleft)
hline!(p2, [0.0], color=:black, ls=:dot, lw=0.8, label=false)
hline!(p2, [0.5], color=:gray,  ls=:dash, lw=1.2, label="ACC = 0.5 (useful)")
for (i, s) in enumerate(seeds)
    accs = Float64[]
    months = Float64[]
    for H in horizons
        H > size(P, 2) && continue
        push!(accs, skill_score(n34_true_ref[1:H], P[i, 1:H]).acc)
        push!(months, H / month_steps)
    end
    plot!(p2, months, accs, lw=0.8, color=:gray, alpha=0.5,
          label=i == 1 ? "individual seeds" : false)
end
mean_accs = Float64[]
mean_months = Float64[]
for H in horizons
    H > length(ensemble_mean) && continue
    push!(mean_accs, skill_score(n34_true_ref[1:H], ensemble_mean[1:H]).acc)
    push!(mean_months, H / month_steps)
end
plot!(p2, mean_months, mean_accs, lw=2.5, color=:red, marker=:circle,
      label="ensemble mean")

p = plot(p1, p2, layout=(2, 1), size=(1200, 800), left_margin=4mm)
out = joinpath(results_dir, "enso_ensemble_$(mode_tag).png")
savefig(p, out)
println("\nSaved: $(out)")
