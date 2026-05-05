# Per-pixel ACC field maps for the Stage-E champions at W1.
# Produces a 3-row × 2-column figure: each row = one architecture,
# columns = ensemble-mean per-pixel ACC, ensemble-std.
#
# Usage:
#   julia --project=. scripts/plot_per_pixel_maps.jl

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Glob, Plots, Measures

ROOT = "results/temporal_multiscale_compare/tier4_head_to_head"
ARCHS = [("A — no_xscale_field",   joinpath(ROOT, "no_xscale_field_W1")),
         ("B — multi_tau_3_field", joinpath(ROOT, "multi_tau_3_field_W1")),
         ("D — full_cascade_field", joinpath(ROOT, "full_cascade_field_W1"))]

function load_ppacc(dir)
    paths = sort(glob("enso_temporal_field_preds_*_seed*.jld2", dir))
    isempty(paths) && error("No per-seed in $dir")
    maps = Vector{Matrix{Float64}}()
    for p in paths
        jldopen(p, "r") do f
            push!(maps, Float64.(f["ppacc_map"]))
        end
    end
    stk = cat(maps...; dims = 3)
    return dropdims(mean(stk; dims = 3); dims = 3),
           dropdims(std(stk;  dims = 3); dims = 3)
end

ref_path = joinpath(ARCHS[1][2], "enso_temporal_field_reference_no_xscale_field.jld2")
lons_raw = jldopen(ref_path, "r") do f; Float64.(f["lons"]); end
lats     = jldopen(ref_path, "r") do f; Float64.(f["lats"]); end
lons360  = mod.(lons_raw, 360.0)
perm     = sortperm(lons360)
lons     = lons360[perm]

panels = Plots.Plot[]
for (label, dir) in ARCHS
    m, s = load_ppacc(dir)
    push!(panels, heatmap(lons, lats, permutedims(m[perm, :]);
        clim = (-1, 1), c = :RdBu, aspect_ratio = :equal,
        title = "$(label) — mean per-pixel ACC",
        xlabel = "lon (°E)", ylabel = "lat (°N)"))
    push!(panels, heatmap(lons, lats, permutedims(s[perm, :]);
        clim = (0, 0.5), c = :Reds, aspect_ratio = :equal,
        title = "$(label) — std across 8 seeds",
        xlabel = "lon (°E)", ylabel = "lat (°N)"))
end

panel_w, panel_h = 720, 280
p = plot(panels...; layout = (length(ARCHS), 2), size = (2 * panel_w, panel_h * length(ARCHS)),
         left_margin = 4mm, bottom_margin = 4mm, plot_title = "Stage E Tier 4 — per-pixel ACC at W1")
out = joinpath(ROOT, "per_pixel_acc_maps.png")
savefig(p, out)
println("Saved $(out)")

# Also print mean/median ACC inside Niño 3.4 box per arch
println("\nNiño 3.4 box mean per-pixel ACC (8-seed ensemble):")
lon_mask = 190.0 .<= lons360 .<= 240.0
lat_mask = -5.0 .<= lats .<= 5.0
for (label, dir) in ARCHS
    m, _ = load_ppacc(dir)
    box = m[lon_mask, lat_mask]
    valid = filter(!isnan, vec(box))
    println("  $(rpad(label, 30)) mean=$(round(mean(valid); digits=3))   median=$(round(median(valid); digits=3))   n=$(length(valid))")
end
