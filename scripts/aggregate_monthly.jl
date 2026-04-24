# Aggregate monthly-resolution ENSO ensemble runs.
# Produces three figures:
#   (1) Niño 3.4 trajectory (ensemble mean + percentile bands) with lead-time skill
#   (2) Pattern correlation + spatial RMSE of the ensemble-mean forecast vs truth,
#       as a function of forecast time — this tests whether the SPATIAL field is
#       right, not just the Niño 3.4 scalar. Cumulative ACC can hide phase-off
#       forecasts that match in aggregate.
#   (3) Side-by-side SST-anomaly maps (observed / ens-mean forecast / difference)
#       at several lead times.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CrossScaleRC, JLD2, LinearAlgebra, Statistics, Plots, Measures, Glob

mode_tag = length(ARGS) > 0 ? ARGS[1] : "three_layer"
results_dir = joinpath(@__DIR__, "..", "results")
paths = sort(glob("enso_monthly_preds_$(mode_tag)_seed*.jld2", results_dir))
isempty(paths) && error("no monthly ensemble files for mode=$(mode_tag)")
println("Aggregating $(length(paths)) monthly runs")

function load_ensemble(paths)
    n34_preds = Matrix{Float64}[]
    field_preds = Array{Float64,3}[]
    n34_true_ref = nothing
    test_f_ref = nothing
    lons, lats = nothing, nothing
    seeds = Int[]
    per_seed_acc = Dict{Int,Float64}()
    for p in paths
        f = jldopen(p)
        n34_true = Float64.(f["n34_true"])
        n34_pred = Float64.(f["n34_pred"])
        test_f   = Float64.(f["test_f"])     # (nlon × nlat × T)
        preds_f  = Float64.(f["preds_f"])
        if n34_true_ref === nothing
            n34_true_ref = n34_true
            test_f_ref   = test_f
            lons = Float64.(f["lons_f"])
            lats = Float64.(f["lats_f"])
        end
        s = Int(f["seed"])
        close(f)
        push!(n34_preds, reshape(n34_pred, 1, length(n34_pred)))
        push!(field_preds, preds_f)
        push!(seeds, s)
        per_seed_acc[s] = skill_score(n34_true, n34_pred).acc
    end
    return n34_preds, field_preds, n34_true_ref, test_f_ref, lons, lats, seeds, per_seed_acc
end

n34_preds, field_preds, n34_true_ref, test_f_ref, lons, lats,
    seeds, per_seed_acc = load_ensemble(paths)

# ----- Niño 3.4 ensemble stats -----
P = vcat(n34_preds...)
ensemble_mean = vec(mean(P, dims=1))
sc_mean = skill_score(n34_true_ref, ensemble_mean)

println("\nPer-seed Niño 3.4 ACC:")
for s in sort(collect(keys(per_seed_acc)))
    println("  seed=$(lpad(s,3)): ACC=$(round(per_seed_acc[s]; digits=3))")
end
println("\nEnsemble mean ACC: $(round(sc_mean.acc; digits=3))   RMSE: $(round(sc_mean.rmse; digits=4))")

leads = [3, 6, 9, 12, 18, 24, 36, 48]
lead_mean_accs = Float64[]; lead_p10 = Float64[]; lead_p90 = Float64[]
lead_p25 = Float64[]; lead_p75 = Float64[]
leads_used = Int[]
println("\nLead-time cumulative ACC (ens-mean / seed range):")
for L in leads
    L > length(ensemble_mean) && continue
    per = [skill_score(n34_true_ref[1:L], P[i, 1:L]).acc for i in 1:size(P,1)]
    mean_acc = skill_score(n34_true_ref[1:L], ensemble_mean[1:L]).acc
    push!(lead_mean_accs, mean_acc)
    push!(lead_p10, quantile(per, 0.1)); push!(lead_p90, quantile(per, 0.9))
    push!(lead_p25, quantile(per, 0.25)); push!(lead_p75, quantile(per, 0.75))
    push!(leads_used, L)
    println("  $(lpad(L,2)) mo: ens-mean=$(round(mean_acc; digits=3))   range=[$(round(minimum(per); digits=3)), $(round(maximum(per); digits=3))]")
end

# ----- Spatial pattern stats -----
nlon, nlat, T = size(test_f_ref)
field_ens = cat(field_preds...; dims=4)   # (nlon × nlat × T × seeds)
field_mean = dropdims(mean(field_ens; dims=4); dims=4)

function spatial_pattern_corr(A::AbstractMatrix, B::AbstractMatrix)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end

pattern_corr = Float64[spatial_pattern_corr(test_f_ref[:, :, t], field_mean[:, :, t]) for t in 1:T]
spatial_rmse = Float64[sqrt(mean((test_f_ref[:, :, t] .- field_mean[:, :, t]) .^ 2)) for t in 1:T]

println("\nSpatial pattern correlation (ens-mean forecast vs truth):")
for L in leads
    L ≤ T && println("  $(lpad(L,2)) mo: pattern-corr=$(round(pattern_corr[L]; digits=3))   spatial-RMSE=$(round(spatial_rmse[L]; digits=3))")
end

# ===== FIGURE 1: Niño 3.4 trajectory + lead-time =====
p10v = [quantile(P[:, t], 0.10) for t in 1:size(P,2)]
p25v = [quantile(P[:, t], 0.25) for t in 1:size(P,2)]
p75v = [quantile(P[:, t], 0.75) for t in 1:size(P,2)]
p90v = [quantile(P[:, t], 0.90) for t in 1:size(P,2)]

pA = plot(ensemble_mean; fillrange=p10v, alpha=0.15, color=:red, linewidth=0, label="10-90%")
plot!(pA, ensemble_mean; fillrange=p90v, alpha=0.15, color=:red, linewidth=0, label=false)
plot!(pA, ensemble_mean; fillrange=p25v, alpha=0.25, color=:red, linewidth=0, label="25-75%")
plot!(pA, ensemble_mean; fillrange=p75v, alpha=0.25, color=:red, linewidth=0, label=false)
plot!(pA, ensemble_mean; lw=2.5, color=:red, label="ensemble mean")
plot!(pA, n34_true_ref;  lw=2.5, color=:black, label="Observed",
      xlabel="month since forecast start", ylabel="Niño 3.4 (norm.)",
      title="Monthly ENSO — $(length(seeds))-seed ensemble   ACC=$(round(sc_mean.acc; digits=2))",
      legend=:topright, grid=true)

pB = plot(leads_used, lead_mean_accs; fillrange=lead_p10, alpha=0.15, color=:red, linewidth=0, label="10-90%")
plot!(pB, leads_used, lead_mean_accs; fillrange=lead_p90, alpha=0.15, color=:red, linewidth=0, label=false)
plot!(pB, leads_used, lead_mean_accs; fillrange=lead_p25, alpha=0.25, color=:red, linewidth=0, label="25-75%")
plot!(pB, leads_used, lead_mean_accs; fillrange=lead_p75, alpha=0.25, color=:red, linewidth=0, label=false)
plot!(pB, leads_used, lead_mean_accs; marker=:circle, lw=2.5, color=:red, label="ensemble mean",
      xlabel="Lead (months)", ylabel="Niño 3.4 ACC (cumulative)",
      title="Lead-time skill", ylim=(-0.3, 1.0), grid=true, legend=:bottomleft)
hline!(pB, [0.5]; ls=:dash, color=:gray, lw=1.5, label="ACC = 0.5")
hline!(pB, [0.0]; ls=:dot,  color=:black, lw=1.0, label=false)

p1 = plot(pA, pB; layout=(2, 1), size=(1200, 800), left_margin=5mm)
out1 = joinpath(results_dir, "enso_monthly_nino34_$(mode_tag).png")
savefig(p1, out1)
println("\nSaved: $(out1)")

# ===== FIGURE 2: Spatial pattern skill over time =====
pC = plot(1:T, pattern_corr; lw=2, color=:red, label=false,
          xlabel="month", ylabel="spatial pattern correlation",
          title="Spatial pattern ACC (ens-mean forecast vs truth)",
          ylim=(-0.5, 1.0), grid=true)
hline!(pC, [0.5]; ls=:dash, color=:gray, lw=1.2, label="0.5")
hline!(pC, [0.0]; ls=:dot,  color=:black, lw=1.0, label=false)

pD = plot(1:T, spatial_rmse; lw=2, color=:orange, label=false,
          xlabel="month", ylabel="spatial RMSE (σ units)",
          title="Point-wise RMSE (ens-mean forecast vs truth)",
          grid=true)

p2 = plot(pC, pD; layout=(2, 1), size=(1200, 700), left_margin=5mm)
out2 = joinpath(results_dir, "enso_monthly_pattern_$(mode_tag).png")
savefig(p2, out2)
println("Saved: $(out2)")

# ===== FIGURE 3: Side-by-side SST maps at multiple leads =====
lons_plot = mod.(lons, 360.0)
perm = sortperm(lons_plot); lons_plot = lons_plot[perm]

# Fixed color scale ±1.5 σ so patterns aren't washed out.
scale = 1.5

function map_panel(field2d, ttl; clim_=(-scale, scale), cmap=:RdBu)
    heatmap(lons_plot, lats, permutedims(field2d[perm, :]);
            clim=clim_, c=cmap,
            xlabel="lon (°E)", ylabel="lat (°N)",
            aspect_ratio=:equal,
            xlims=(minimum(lons_plot), maximum(lons_plot)),
            ylims=(minimum(lats), maximum(lats)),
            title=ttl, colorbar=true)
end

snap_leads = [3, 6, 9, 12, 18, 24]
snap_leads = filter(L -> L ≤ T, snap_leads)

panels = Plots.Plot[]
for L in snap_leads
    obs  = test_f_ref[:, :, L]
    fore = field_mean[:, :, L]
    diff = obs .- fore
    pc   = round(spatial_pattern_corr(obs, fore); digits=2)
    push!(panels, map_panel(obs,  "Observed  ($(L) mo)"))
    push!(panels, map_panel(fore, "Ens-mean forecast  ($(L) mo)  pc=$(pc)"))
    push!(panels, map_panel(diff, "Obs − Forecast  ($(L) mo)"))
end

p3 = plot(panels...; layout=(length(snap_leads), 3),
          size=(1800, 320 * length(snap_leads)),
          plot_title="2° SST anomaly — $(length(seeds))-seed ensemble mean vs observed",
          left_margin=4mm, bottom_margin=3mm)
out3 = joinpath(results_dir, "enso_monthly_maps_ens_$(mode_tag).png")
savefig(p3, out3)
println("Saved: $(out3)")
