# Event-centric Stage-E comparison: identify major ENSO events in the test
# window (peaks/troughs of Niño 3.4) and for each event show the spatial
# pattern at THAT month under (truth, A forecast, B forecast, D forecast).
# Forecast lead = (event month) since the forecast starts at month 1.
#
# Output:
#   stage_E_event_aligned.png  — trajectory with markers + per-event maps
#   stage_E_phase_diag_event25.png — for the strongest cold event (mo 25),
#       show the A forecast pattern at lead 25, 22, ..., 4 — does the
#       model "see" the event coming?

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Glob, Plots, Measures, LinearAlgebra, Printf

ROOT = "results/temporal_multiscale_compare/tier4_head_to_head"
ARCHS = [
    ("A — frequency-band",   joinpath(ROOT, "no_xscale_field_W1"),   "no_xscale_field",   :steelblue),
    ("B — multi-τ",          joinpath(ROOT, "multi_tau_3_field_W1"), "multi_tau_3_field", :firebrick),
    ("D — bandpass+xscale",  joinpath(ROOT, "full_cascade_field_W1"), "full_cascade_field", :darkorange),
]

ref = jldopen(joinpath(ARCHS[1][2], "enso_temporal_field_reference_no_xscale_field.jld2"), "r")
n34_true   = Float64.(ref["n34_true"])
field_true = Float64.(ref["field_true"])
lons       = Float64.(ref["lons"])
lats       = Float64.(ref["lats"])
close(ref)
nt = length(n34_true)

function ensemble_mean_field(dir, mode)
    paths = sort(glob("enso_temporal_field_preds_$(mode)_seed*.jld2", dir))
    fs = Vector{Array{Float64,3}}()
    for p in paths
        jldopen(p, "r") do f
            push!(fs, Float64.(f["field_pred"]))
        end
    end
    return dropdims(mean(cat(fs...; dims=4); dims=4); dims=4)
end
function ensemble_mean_n34(dir, mode)
    paths = sort(glob("enso_temporal_field_preds_$(mode)_seed*.jld2", dir))
    ns = Vector{Vector{Float64}}()
    for p in paths
        jldopen(p, "r") do f
            push!(ns, Float64.(f["n34_pred"]))
        end
    end
    return vec(mean(hcat(ns...); dims=2))
end

A_field = ensemble_mean_field(ARCHS[1][2], ARCHS[1][3])
B_field = ensemble_mean_field(ARCHS[2][2], ARCHS[2][3])
D_field = ensemble_mean_field(ARCHS[3][2], ARCHS[3][3])
A_n34 = ensemble_mean_n34(ARCHS[1][2], ARCHS[1][3])
B_n34 = ensemble_mean_n34(ARCHS[2][2], ARCHS[2][3])
D_n34 = ensemble_mean_n34(ARCHS[3][2], ARCHS[3][3])

# --- pick events: greedily take strongest |N3.4| with min separation 6 mo ---
order = sortperm(abs.(n34_true); rev=true)
events = Int[]
for i in order
    all(abs(i - e) > 6 for e in events) || continue
    push!(events, i)
    length(events) ≥ 6 && break
end
sort!(events)

start_year = 2006
function event_label(t)
    yr = start_year + (t-1) ÷ 12
    mo = ((t-1) % 12) + 1
    sgn = n34_true[t] > 0 ? "warm" : "cold"
    return "$(sgn)  $(yr)-$(lpad(mo,2,'0'))  L=$(t)mo  N3.4=$(round(n34_true[t];digits=2))"
end

# --- shared colour limits / lon order ---
lons360 = mod.(lons, 360.0); perm = sortperm(lons360); lons_sorted = lons360[perm]
function map_panel(field2d, ttl; clim_=(-2, 2), cmap=:RdBu)
    heatmap(lons_sorted, lats, permutedims(field2d[perm, :]);
        clim = clim_, c = cmap, xlabel = "lon (°E)", ylabel = "lat (°N)",
        aspect_ratio = :equal,
        xlims = (minimum(lons_sorted), maximum(lons_sorted)),
        ylims = (minimum(lats), maximum(lats)),
        title = ttl, colorbar = true, titlefontsize = 9)
end
function spatial_pc(A, B)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end

# --- Figure 1: trajectory with markers + per-event 4-column rows ---
ptraj = plot(1:nt, n34_true; lw = 3, color = :black, label = "Observed",
    xlabel = "month since forecast start (Jan 2006)", ylabel = "Niño 3.4 (z)",
    title = "Niño 3.4 trajectory — major events marked", legend = :topright,
    ylim = (-3, 3), size = (1700, 320))
plot!(ptraj, 1:nt, A_n34; lw = 1.6, color = :steelblue, alpha=0.8, label = "A")
plot!(ptraj, 1:nt, B_n34; lw = 1.6, color = :firebrick, alpha=0.8, label = "B")
plot!(ptraj, 1:nt, D_n34; lw = 1.6, color = :darkorange, alpha=0.8, label = "D")
for e in events
    color = n34_true[e] > 0 ? :navy : :darkred
    vline!(ptraj, [e]; color=color, ls=:dash, alpha=0.5, label=false)
    annotate!(ptraj, e, n34_true[e] + (n34_true[e]>0 ? 0.4 : -0.4),
        text("L=$e", 8, color))
end

panels = Plots.Plot[]
push!(panels, ptraj)
# placeholder columns to make the trajectory span 4 cols
push!(panels, plot(framestyle=:none, grid=false))
push!(panels, plot(framestyle=:none, grid=false))
push!(panels, plot(framestyle=:none, grid=false))

for e in events
    truth_e = field_true[:, :, e]
    A_e = A_field[:, :, e]
    B_e = B_field[:, :, e]
    D_e = D_field[:, :, e]
    push!(panels, map_panel(truth_e, "Observed  $(event_label(e))"))
    push!(panels, map_panel(A_e,   "A  pc=$(round(spatial_pc(truth_e, A_e); digits=2))"))
    push!(panels, map_panel(B_e,   "B  pc=$(round(spatial_pc(truth_e, B_e); digits=2))"))
    push!(panels, map_panel(D_e,   "D  pc=$(round(spatial_pc(truth_e, D_e); digits=2))"))
end

# Panel sizing: data 160°×70° with per-panel colorbar → 720×280 per panel renders
# cells visually square. 4 columns of map panels.
panel_w, panel_h = 720, 280
p_evt = plot(panels...; layout = (1 + length(events), 4),
    size = (4 * panel_w, panel_h * (1 + length(events))),
    plot_title = "Stage-E event-aligned comparison (W1, 8-seed ensemble means)",
    left_margin = 4mm, bottom_margin = 3mm)
out1 = joinpath(ROOT, "stage_E_event_aligned.png")
savefig(p_evt, out1)
println("Saved $(out1)")

# ----------------------------------------------------------------------------
# Figure 2 — phase diagnostic for the strongest cold event (month 25)
# Show A's forecast pattern at L=25, 22, 19, 16, 13, 10, 7, 4, 1 vs truth at
# L=25, plus B for comparison. Question: does the prediction's spatial pattern
# evolve toward the truth, or does it appear/disappear at random times?
# ----------------------------------------------------------------------------
target = events[argmax(n34_true[events] .* -1)]   # strongest cold event
target_truth = field_true[:, :, target]
println("Phase diag target event: L=$(target)  N3.4=$(round(n34_true[target]; digits=2))")

leads_phase = filter(L -> 1 ≤ L ≤ nt, [target, target-3, target-6, target-9, target-12, target-15, target-18, target-21])
sort!(leads_phase)

panels_phase = Plots.Plot[]
push!(panels_phase, map_panel(target_truth, "TARGET  L=$(target)  N3.4=$(round(n34_true[target];digits=2))"))
push!(panels_phase, plot(framestyle=:none, grid=false))
push!(panels_phase, plot(framestyle=:none, grid=false))
for L in leads_phase
    A_L = A_field[:, :, L]
    B_L = B_field[:, :, L]
    D_L = D_field[:, :, L]
    push!(panels_phase, map_panel(A_L,
        "A @ L=$L  pc(vs target)=$(round(spatial_pc(target_truth, A_L); digits=2))  N3.4=$(round(A_n34[L]; digits=2))"))
    push!(panels_phase, map_panel(B_L,
        "B @ L=$L  pc(vs target)=$(round(spatial_pc(target_truth, B_L); digits=2))  N3.4=$(round(B_n34[L]; digits=2))"))
    push!(panels_phase, map_panel(D_L,
        "D @ L=$L  pc(vs target)=$(round(spatial_pc(target_truth, D_L); digits=2))  N3.4=$(round(D_n34[L]; digits=2))"))
end

p_phase = plot(panels_phase...; layout = (1 + length(leads_phase), 3),
    size = (3 * panel_w, panel_h * (1 + length(leads_phase))),
    plot_title = "Phase diagnostic — strongest La Niña (target=L$(target))  pc evaluated against TARGET pattern",
    left_margin = 4mm, bottom_margin = 3mm)
out2 = joinpath(ROOT, "stage_E_phase_diag.png")
savefig(p_phase, out2)
println("Saved $(out2)")

# ----------------------------------------------------------------------------
# Print event-skill table
# ----------------------------------------------------------------------------
println("\n" * "="^72)
println("Event-aligned spatial pattern correlation (truth vs ensemble-mean forecast)")
println("="^72)
@printf("%-30s  %-10s  %-10s  %-10s\n", "event", "A (pc)", "B (pc)", "D (pc)")
for e in events
    truth_e = field_true[:, :, e]
    pcA = spatial_pc(truth_e, A_field[:, :, e])
    pcB = spatial_pc(truth_e, B_field[:, :, e])
    pcD = spatial_pc(truth_e, D_field[:, :, e])
    @printf("%-30s  %+0.3f      %+0.3f      %+0.3f\n", event_label(e), pcA, pcB, pcD)
end

# Also: how often does each model predict the right SIGN of the N3.4 anomaly at events?
println("\nSign-correctness at events:")
@printf("%-30s  %-8s  %-8s  %-8s\n", "event", "A_sgn", "B_sgn", "D_sgn")
for e in events
    s = sign(n34_true[e])
    sA = sign(A_n34[e]); sB = sign(B_n34[e]); sD = sign(D_n34[e])
    @printf("%-30s  %s        %s        %s\n", event_label(e),
            sA == s ? "✓" : "✗", sB == s ? "✓" : "✗", sD == s ? "✓" : "✗")
end
