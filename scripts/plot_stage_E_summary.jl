# Stage E publication-grade comparison figure.
# Two PNGs:
#   1) summary.png      — N3.4 trajectory (8-seed ensemble mean, A/B/D vs truth),
#                         lead-time ACC curve (A/B/D), spatial pc per month
#   2) A_field_maps.png — A champion: truth vs ensemble-mean forecast vs error,
#                         at leads 3, 6, 9, 12, 18, 24 mo

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Glob, Plots, Measures, LinearAlgebra, Printf

ROOT = "results/temporal_multiscale_compare/tier4_head_to_head"
ARCHS = [
    ("A — frequency-band",   joinpath(ROOT, "no_xscale_field_W1"),   "no_xscale_field",   :steelblue),
    ("B — multi-τ",          joinpath(ROOT, "multi_tau_3_field_W1"), "multi_tau_3_field", :firebrick),
    ("D — bandpass + xscale",joinpath(ROOT, "full_cascade_field_W1"), "full_cascade_field", :darkorange),
]

# --- Load truth (use any reference) ---
ref_path = joinpath(ARCHS[1][2], "enso_temporal_field_reference_no_xscale_field.jld2")
n34_true, field_true, lons, lats = jldopen(ref_path, "r") do f
    Float64.(f["n34_true"]), Float64.(f["field_true"]),
    Float64.(f["lons"]), Float64.(f["lats"])
end
nt = length(n34_true)
println("truth window: $(nt) months")

# --- Load ensemble-mean predictions per arch ---
function ensemble_mean(dir, mode)
    paths = sort(glob("enso_temporal_field_preds_$(mode)_seed*.jld2", dir))
    n34s = Vector{Vector{Float64}}()
    fields = Vector{Array{Float64,3}}()
    seeds = Int[]
    for p in paths
        jldopen(p, "r") do f
            push!(seeds, Int(f["seed"]))
            push!(n34s,   Float64.(f["n34_pred"]))
            push!(fields, Float64.(f["field_pred"]))
        end
    end
    n34_stk = hcat(n34s...)              # (nt, n_seeds)
    field_stk = cat(fields...; dims=4)   # (nlon, nlat, nt, n_seeds)
    return (seeds = seeds,
            n34_mean = vec(mean(n34_stk; dims=2)),
            n34_std  = vec(std(n34_stk;  dims=2)),
            field_mean = dropdims(mean(field_stk; dims=4); dims=4))
end

ensembles = [(label, ensemble_mean(dir, mode), color) for (label, dir, mode, color) in ARCHS]

# --- Skill metrics for the captions ---
function pearson(a, b)
    am = a .- mean(a); bm = b .- mean(b)
    σa = std(a); σb = std(b)
    (σa > 0 && σb > 0) ? mean(am .* bm) / (σa * σb) : NaN
end
function lead_acc_curve(n34_pred)
    leads = [1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72, 84, 96]
    leads = filter(L -> L ≤ nt, leads)
    [pearson(n34_true[1:L], n34_pred[1:L]) for L in leads], leads
end
function spatial_pc_curve(field_pred)
    [let a = vec(field_true[:, :, t]), b = vec(field_pred[:, :, t]),
         am = a .- mean(a), bm = b .- mean(b);
         dot(am, bm) / (norm(am) * norm(bm) + eps())
     end for t in 1:nt]
end

# ============================================================================
# Figure 1 — head-to-head summary (3 panels)
# ============================================================================

p1 = plot(1:nt, n34_true; lw = 3, color = :black, label = "Observed",
    xlabel = "month since forecast start", ylabel = "Niño 3.4 (z-scored)",
    title = "Niño 3.4 forecast — 8-seed ensemble means at W1",
    legend = :topright, ylim = (-3, 3))
for (label, e, color) in ensembles
    acc12 = pearson(n34_true[1:12], e.n34_mean[1:12])
    plot!(p1, 1:nt, e.n34_mean;
        ribbon = e.n34_std, fillalpha = 0.15,
        lw = 2, color = color,
        label = "$(label)  (12-mo ACC=$(round(acc12; digits=3)))")
end
vline!(p1, [12]; ls = :dot, color = :gray, label = false)
annotate!(p1, 12, 2.7, text("12-mo lead", 9, :gray, :left))

p2 = plot(xlabel = "Lead (months)", ylabel = "cumulative ACC",
    title = "Lead-time skill", ylim = (-0.5, 1), legend = :bottomleft)
hline!(p2, [0.5]; ls = :dash, color = :gray, label = "0.5")
hline!(p2, [0.0]; ls = :dot,  color = :black, label = false)
for (label, e, color) in ensembles
    accs, leads = lead_acc_curve(e.n34_mean)
    plot!(p2, leads, accs; lw = 2.5, marker = :circle,
        color = color, label = label)
end
vline!(p2, [12]; ls = :dot, color = :gray, label = false)

p3 = plot(xlabel = "month", ylabel = "spatial pattern correlation",
    title = "Spatial pattern correlation (8-seed mean field)",
    ylim = (-0.3, 1), legend = :topright)
hline!(p3, [0.5]; ls = :dash, color = :gray, label = "0.5")
hline!(p3, [0.0]; ls = :dot,  color = :black, label = false)
for (label, e, color) in ensembles
    pcs = spatial_pc_curve(e.field_mean)
    plot!(p3, 1:nt, pcs; lw = 2, color = color, label = label)
end

p_summary = plot(p1, p2, p3; layout = (3, 1), size = (1300, 1200),
    left_margin = 6mm, bottom_margin = 4mm,
    plot_title = "Stage E head-to-head — A (winner) vs B vs D, W1, 8 seeds")
out1 = joinpath(ROOT, "stage_E_summary.png")
savefig(p_summary, out1)
println("Saved $(out1)")

# ============================================================================
# Figure 2 — A champion spatial maps at multiple leads
# ============================================================================

A_label, A_e, _ = ensembles[1]   # A = no_xscale_field

lons360 = mod.(lons, 360.0); perm = sortperm(lons360); lons_sorted = lons360[perm]

function map_panel(field2d, ttl; clim_=(-2, 2), cmap=:RdBu)
    heatmap(lons_sorted, lats, permutedims(field2d[perm, :]);
        clim = clim_, c = cmap, xlabel = "lon (°E)", ylabel = "lat (°N)",
        aspect_ratio = :equal,
        xlims = (minimum(lons_sorted), maximum(lons_sorted)),
        ylims = (minimum(lats), maximum(lats)),
        title = ttl, colorbar = true)
end

map_leads = [3, 6, 9, 12, 18, 24]
# Panel sizing: data aspect = 160°lon × 70°lat = 2.29:1. With per-panel colorbar
# consuming ~25% of horizontal space, panel width must be 2.29 × 1.25 ≈ 2.86×
# the heatmap's vertical extent for cells to render visually square.
# Use 720 × 280 per panel (2.57:1) — close to the target with comfortable margins.
panel_w, panel_h = 720, 280
panels = Plots.Plot[]
for L in map_leads
    L > nt && continue
    truth_L = field_true[:, :, L]
    pred_L  = A_e.field_mean[:, :, L]
    pc_L = let a = vec(truth_L), b = vec(pred_L)
        am = a .- mean(a); bm = b .- mean(b)
        dot(am, bm) / (norm(am) * norm(bm) + eps())
    end
    push!(panels, map_panel(truth_L, "Observed  L=$(L) mo"))
    push!(panels, map_panel(pred_L,  "A forecast  L=$(L) mo  pc=$(round(pc_L; digits=2))"))
    push!(panels, map_panel(truth_L .- pred_L, "Obs − Forecast  L=$(L) mo"))
end
p_maps = plot(panels...; layout = (length(map_leads), 3),
    size = (3 * panel_w, panel_h * length(map_leads)),
    plot_title = "A champion (frequency-band, no_xscale_field) — 8-seed ensemble mean",
    left_margin = 4mm, bottom_margin = 3mm)
out2 = joinpath(ROOT, "stage_E_A_champion_maps.png")
savefig(p_maps, out2)
println("Saved $(out2)")

# ============================================================================
# Print the headline numbers
# ============================================================================
println("\n" * "="^72)
println("Headline numbers (8-seed ensemble means at W1)")
println("="^72)
@printf("%-22s  %-10s  %-10s  %-10s  %-10s\n", "arch", "ACC@12mo", "ACC@24mo", "ACC@36mo", "pc@12mo")
for (label, e, _) in ensembles
    acc12 = pearson(n34_true[1:12], e.n34_mean[1:12])
    acc24 = pearson(n34_true[1:min(24,nt)], e.n34_mean[1:min(24,nt)])
    acc36 = pearson(n34_true[1:min(36,nt)], e.n34_mean[1:min(36,nt)])
    pc12  = let a = vec(field_true[:,:,12]), b = vec(e.field_mean[:,:,12])
        am = a .- mean(a); bm = b .- mean(b)
        dot(am, bm) / (norm(am) * norm(bm) + eps())
    end
    @printf("%-22s  %.3f      %.3f      %.3f      %.3f\n", label, acc12, acc24, acc36, pc12)
end
