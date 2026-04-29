# Aggregate ensemble runs of main_enso_temporal_multiscale_field.jl over seeds.
# Reads results/<outdir>/enso_temporal_field_preds_<mode>_seed*.jld2 and emits
# per-seed and ensemble summary statistics: 12/18/24-mo ACC + RMSE + std_ratio,
# spatial pc at 3/12/24-mo, plus an ensemble-mean Niño 3.4 / spatial-pc plot.
#
# Usage:
#   julia --project=. scripts/aggregate_temporal_field.jl <outdir> <mode_tag>
#
# Examples:
#   julia --project=. scripts/aggregate_temporal_field.jl results/temporal_multiscale/multi_tau_3_field_best multi_tau_3_field
#   julia --project=. scripts/aggregate_temporal_field.jl results/temporal_multiscale/multi_tau_2_field      multi_tau_2_field

ENV["GKSwstype"] = "100"
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CrossScaleRC
using JLD2, Statistics, LinearAlgebra, Plots, Measures, Glob

length(ARGS) ≥ 2 ||
    error("Usage: aggregate_temporal_field.jl <outdir> <mode_tag>")
outdir   = ARGS[1]
mode_tag = ARGS[2]

paths = sort(glob("enso_temporal_field_preds_$(mode_tag)_seed*.jld2", outdir))
isempty(paths) && error("No per-seed files matching enso_temporal_field_preds_$(mode_tag)_seed*.jld2 in $(outdir)")

ref_path = joinpath(outdir, "enso_temporal_field_reference_$(mode_tag).jld2")
isfile(ref_path) || error("Missing reference file $(ref_path)")

ref = jldopen(ref_path, "r") do f
    (n34_true   = Float64.(f["n34_true"]),
     field_true = Float64.(f["field_true"]))
end
n34_true   = ref.n34_true
field_true = ref.field_true
predict_len = length(n34_true)

# Per-seed extraction
n_seeds = length(paths)
println("Found $(n_seeds) ensemble members in $(outdir):")
foreach(p -> println("  ", basename(p)), paths)

n34_preds  = Vector{Vector{Float64}}(undef, n_seeds)
field_preds = Vector{Array{Float64,3}}(undef, n_seeds)
seeds      = Vector{Int}(undef, n_seeds)

for (i, p) in enumerate(paths)
    jldopen(p, "r") do f
        n34_preds[i]   = Float64.(f["n34_pred"])
        field_preds[i] = Float64.(f["field_pred"])
        seeds[i]       = Int(f["seed"])
    end
end

function spatial_pc(A::AbstractMatrix, B::AbstractMatrix)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end

function lead_metrics(n34_pred::Vector{Float64})
    leads = [3, 6, 9, 12, 18, 24, 36, 48]
    out = Dict{Int, NamedTuple{(:acc, :rmse, :std_ratio), NTuple{3, Float64}}}()
    for L in leads
        L > predict_len && continue
        s = skill_score(n34_true[1:L], n34_pred[1:L])
        sr = std(n34_pred[1:L]) / std(n34_true[1:L])
        out[L] = (acc = s.acc, rmse = s.rmse, std_ratio = sr)
    end
    return out
end

function pc_at(field_pred_3d::AbstractArray{<:Real, 3}, leads = [3, 6, 9, 12, 18, 24])
    out = Dict{Int, Float64}()
    for L in leads
        L > predict_len && continue
        out[L] = spatial_pc(field_true[:, :, L], field_pred_3d[:, :, L])
    end
    return out
end

# Per-seed table
println("\n=== Per-seed Niño 3.4 cumulative ACC + std_ratio ===")
println("seed |  3 mo |  6 mo |  9 mo | 12 mo | 18 mo | 24 mo | 36 mo")
for i in 1:n_seeds
    m = lead_metrics(n34_preds[i])
    fmt(L) = haskey(m, L) ? "$(round(m[L].acc; digits=3))" : "    -"
    println("$(lpad(seeds[i], 4)) | $(fmt(3)) | $(fmt(6)) | $(fmt(9)) | $(fmt(12)) | $(fmt(18)) | $(fmt(24)) | $(fmt(36))")
end

println("\n=== Per-seed spatial pattern correlation ===")
println("seed |  3 mo |  6 mo |  9 mo | 12 mo | 18 mo | 24 mo")
for i in 1:n_seeds
    pc = pc_at(field_preds[i])
    fmt(L) = haskey(pc, L) ? "$(round(pc[L]; digits=3))" : "    -"
    println("$(lpad(seeds[i], 4)) | $(fmt(3)) | $(fmt(6)) | $(fmt(9)) | $(fmt(12)) | $(fmt(18)) | $(fmt(24))")
end

# Ensemble mean / std across seeds
println("\n=== Ensemble means (n=$(n_seeds) seeds, mean ± std) ===")
for L in [3, 6, 9, 12, 18, 24, 36]
    accs = [haskey(lead_metrics(n34_preds[i]), L) ? lead_metrics(n34_preds[i])[L].acc : NaN for i in 1:n_seeds]
    srs  = [haskey(lead_metrics(n34_preds[i]), L) ? lead_metrics(n34_preds[i])[L].std_ratio : NaN for i in 1:n_seeds]
    pcs  = [haskey(pc_at(field_preds[i]), L) ? pc_at(field_preds[i])[L] : NaN for i in 1:n_seeds]
    accs = filter(!isnan, accs); srs = filter(!isnan, srs); pcs = filter(!isnan, pcs)
    isempty(accs) && continue
    println("  $(lpad(L, 2)) mo : ACC=$(round(mean(accs); digits=3))±$(round(std(accs); digits=3))   std_ratio=$(round(mean(srs); digits=3))±$(round(std(srs); digits=3))   pc=$(round(mean(pcs); digits=3))±$(round(std(pcs); digits=3))")
end

# Ensemble mean prediction
P = vcat([reshape(p, 1, :) for p in n34_preds]...)
ens_mean_n34 = vec(mean(P; dims = 1))
sc_ens = skill_score(n34_true, ens_mean_n34)
println("\n=== Ensemble-mean N3.4 over full $(predict_len)-mo window ===")
println("  ACC=$(round(sc_ens.acc; digits=3))   RMSE=$(round(sc_ens.rmse; digits=4))   std_ratio=$(round(std(ens_mean_n34)/std(n34_true); digits=3))")

# Plot: per-seed N3.4 + ensemble mean
p1 = plot(n34_true; lw=2.5, color=:black, label="Observed",
          xlabel="month since forecast start", ylabel="Niño 3.4",
          title="Ensemble  $(mode_tag)  (n=$(n_seeds))",
          legend=:topright)
for i in 1:n_seeds
    plot!(p1, n34_preds[i]; lw=0.7, color=:gray, alpha=0.5,
          label = i == 1 ? "individual seeds" : false)
end
plot!(p1, ens_mean_n34; lw=2.0, color=:red, ls=:dash,
      label="ensemble mean (ACC=$(round(sc_ens.acc; digits=2)))")

# Lead-time curves: per seed + ensemble
p2 = plot(xlabel="Lead (months)", ylabel="cumulative ACC",
          title="Lead-time skill", ylim=(-1, 1), legend=:bottomleft)
hline!(p2, [0.0]; ls=:dot, color=:black, label=false)
hline!(p2, [0.5]; ls=:dash, color=:gray, label="0.5")

leads_x = [3, 6, 9, 12, 18, 24, 36, 48]
for i in 1:n_seeds
    accs = Float64[]; xs = Int[]
    for L in leads_x
        L > predict_len && continue
        push!(accs, skill_score(n34_true[1:L], n34_preds[i][1:L]).acc)
        push!(xs, L)
    end
    plot!(p2, xs, accs; lw=0.7, color=:gray, alpha=0.5,
          label = i == 1 ? "individual seeds" : false)
end
ens_accs = Float64[]; xs = Int[]
for L in leads_x
    L > predict_len && continue
    push!(ens_accs, skill_score(n34_true[1:L], ens_mean_n34[1:L]).acc)
    push!(xs, L)
end
plot!(p2, xs, ens_accs; lw=2.5, color=:red, marker=:circle,
      label="ensemble mean")

# Spatial pc per month, ensemble
ens_pc_per_t = [
    spatial_pc(field_true[:, :, t],
               dropdims(mean(cat([field_preds[i][:, :, t] for i in 1:n_seeds]...; dims=3); dims=3); dims=3))
    for t in 1:predict_len
]
p3 = plot(1:predict_len, ens_pc_per_t; lw=2, color=:purple,
          label="ensemble", xlabel="month", ylabel="spatial pc",
          title="Spatial pc — ensemble mean field forecast",
          ylim=(-0.5, 1.0), legend=:topright)
hline!(p3, [0.5]; ls=:dash, color=:gray, label="0.5"); hline!(p3, [0.0]; ls=:dot, color=:black, label=false)

p_ens = plot(p1, p2, p3; layout=(3, 1), size=(1200, 1100), left_margin=4mm)
out_png = joinpath(outdir, "ensemble_$(mode_tag).png")
savefig(p_ens, out_png)
println("\nSaved: $(out_png)")
