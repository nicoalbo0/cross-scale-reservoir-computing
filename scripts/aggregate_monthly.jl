# Aggregate monthly-resolution ENSO ensemble runs.
# Mirrors scripts/aggregate_ensemble.jl but reads results/enso_monthly_preds_*.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CrossScaleRC, JLD2, Statistics, Plots, Measures, Glob

mode_tag = length(ARGS) > 0 ? ARGS[1] : "three_layer"
results_dir = joinpath(@__DIR__, "..", "results")
paths = sort(glob("enso_monthly_preds_$(mode_tag)_seed*.jld2", results_dir))
isempty(paths) && error("no monthly ensemble files for mode=$(mode_tag)")
println("Aggregating $(length(paths)) monthly runs")

function load_monthly_ensemble(paths)
    preds = Matrix{Float64}[]
    n34_true_ref = nothing
    seeds = Int[]
    per_seed_acc = Dict{Int,Float64}()
    for p in paths
        f = jldopen(p)
        n34_true = Float64.(f["n34_true"])
        n34_pred = Float64.(f["n34_pred"])
        s = Int(f["seed"])
        close(f)
        if n34_true_ref === nothing; n34_true_ref = n34_true; end
        push!(preds, reshape(n34_pred, 1, length(n34_pred)))
        push!(seeds, s)
        per_seed_acc[s] = skill_score(n34_true, n34_pred).acc
    end
    return preds, n34_true_ref, seeds, per_seed_acc
end
preds, n34_true_ref, seeds, per_seed_acc = load_monthly_ensemble(paths)

P = vcat(preds...)
ensemble_mean   = vec(mean(P, dims=1))
ensemble_median = vec(median(P, dims=1))

sc_mean   = skill_score(n34_true_ref, ensemble_mean)
sc_median = skill_score(n34_true_ref, ensemble_median)

println("\nPer-seed full-window ACC:")
for s in seeds
    println("  seed=$(lpad(s,3)):  ACC=$(round(per_seed_acc[s]; digits=3))")
end
println("\nEnsemble mean   (n=$(length(seeds))):  ACC=$(round(sc_mean.acc;  digits=3))  RMSE=$(round(sc_mean.rmse;  digits=4))")
println("Ensemble median (n=$(length(seeds))):  ACC=$(round(sc_median.acc; digits=3))  RMSE=$(round(sc_median.rmse; digits=4))")

leads = [3, 6, 9, 12, 18, 24, 36, 48]
println("\nLead-time ACC — ensemble mean and range over seeds:")
lead_mean_accs = Float64[]; lead_min = Float64[]; lead_max = Float64[]
for L in leads
    L > length(ensemble_mean) && continue
    sc_L = skill_score(n34_true_ref[1:L], ensemble_mean[1:L])
    per = [skill_score(n34_true_ref[1:L], P[i, 1:L]).acc for i in 1:size(P,1)]
    push!(lead_mean_accs, sc_L.acc); push!(lead_min, minimum(per)); push!(lead_max, maximum(per))
    println("  $(L) mo : ens_mean ACC=$(round(sc_L.acc; digits=3))   seed range=[$(round(minimum(per); digits=3)), $(round(maximum(per); digits=3))]")
end

# Plot: ground truth + each seed (thin gray) + ensemble mean (red)
p1 = plot(n34_true_ref; lw=2.5, color=:black, label="Observed",
          xlabel="month since forecast start", ylabel="Niño 3.4 (norm.)",
          title="Monthly ENSO ensemble  n=$(length(seeds))  ens-ACC=$(round(sc_mean.acc; digits=2))")
for i in 1:size(P,1)
    plot!(p1, P[i, :]; lw=0.8, color=:gray, alpha=0.45,
          label = i == 1 ? "individual seeds" : false)
end
plot!(p1, ensemble_mean; lw=2.2, color=:red, linestyle=:dash,
      label="ensemble mean")

# Lead-time with min/max band
p2 = plot(leads[1:length(lead_mean_accs)], lead_mean_accs; marker=:circle, lw=2,
          color=:red, label="ens mean", xlabel="Lead (months)",
          ylabel="Niño 3.4 ACC", title="Lead-time skill",
          ylim=(-1, 1), legend=:bottomleft)
plot!(p2, leads[1:length(lead_mean_accs)], lead_min; fillrange=lead_max, alpha=0.2,
      color=:gray, label="seed range")
hline!(p2, [0.5]; ls=:dash, color=:gray, label="ACC=0.5")
hline!(p2, [0.0]; ls=:dot,  color=:black, label=false)

p = plot(p1, p2; layout=(2, 1), size=(1200, 800), left_margin=4mm)
out = joinpath(results_dir, "enso_monthly_ensemble_$(mode_tag).png")
savefig(p, out)
println("\nSaved: $(out)")
