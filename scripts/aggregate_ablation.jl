# Cross-mode comparison of single_layer / two_layer / three_layer monthly runs.
#
# For each mode it loads the per-seed JLD2 ensemble, then produces a single
# figure that lets the multi-layer claim be read at a glance:
#   (1) Per-mode distribution of Niño 3.4 ACC (full window) across seeds.
#   (2) Per-mode lead-time cumulative ACC (mean across seeds, with seed range).
#   (3) Per-mode spatial pattern correlation of the ensemble-mean forecast,
#       as a function of forecast month.
#
# Side panel: side-by-side ensemble-mean SST-anomaly maps at lead 6, 12, 18 mo
# for each mode against the truth.
#
# Read-only: writes only to results/.

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CrossScaleRC, JLD2, LinearAlgebra, Statistics, Plots, Measures, Glob

const MODES = ["single_layer", "two_layer", "three_layer"]
const MODE_COLORS = Dict("single_layer" => :gray, "two_layer" => :orange, "three_layer" => :red)
const MODE_LABEL  = Dict("single_layer" => "1L", "two_layer" => "2L", "three_layer" => "3L")
const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

# ----- helpers -----
function spatial_pattern_corr(A::AbstractMatrix, B::AbstractMatrix)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end

function load_mode(mode_tag::String)
    paths = sort(glob("enso_monthly_preds_$(mode_tag)_seed*.jld2", RESULTS_DIR))
    isempty(paths) && return nothing
    n34_preds = Vector{Vector{Float64}}()
    field_preds = Array{Float64,3}[]
    seeds = Int[]
    n34_true_ref = nothing
    test_f_ref = nothing
    lons = nothing; lats = nothing
    for p in paths
        f = jldopen(p)
        ks = keys(f)
        push!(n34_preds, Float64.(f["n34_pred"]))
        push!(field_preds, Float64.(f["preds_f"]))
        push!(seeds, Int(f["seed"]))
        if n34_true_ref === nothing
            n34_true_ref = Float64.(f["n34_true"])
        end
        if test_f_ref === nothing && "test_f" in ks
            test_f_ref = Float64.(f["test_f"])
            lons = Float64.(f["lons_f"])
            lats = Float64.(f["lats_f"])
        end
        close(f)
    end
    # If per-seed files didn't carry test_f (slim format), pull it from the
    # mode's reference file.
    if test_f_ref === nothing
        ref = joinpath(RESULTS_DIR, "enso_monthly_reference_$(mode_tag).jld2")
        if isfile(ref)
            f = jldopen(ref)
            test_f_ref = Float64.(f["test_f"])
            lons = Float64.(f["lons_f"]); lats = Float64.(f["lats_f"])
            close(f)
        end
    end
    test_f_ref === nothing && error("missing test_f for mode=$(mode_tag)")
    return (
        n34_preds = reduce(hcat, [reshape(x, length(x), 1) for x in n34_preds])',  # seeds × T
        field_preds = field_preds,
        seeds = seeds, n34_true = n34_true_ref,
        test_f = test_f_ref, lons = lons, lats = lats,
    )
end

# ----- load all modes -----
loaded = Dict{String,Any}()
for m in MODES
    d = load_mode(m)
    if d === nothing
        @warn "no JLD2s for mode=$(m); skipping"
        continue
    end
    println("Loaded mode=$(m): $(length(d.seeds)) seeds, T=$(length(d.n34_true))")
    loaded[m] = d
end
isempty(loaded) && error("no modes loaded")

# Pick a reference truth & coords (all modes share the same observation).
ref_mode = first(keys(loaded))
n34_true = loaded[ref_mode].n34_true
test_f = loaded[ref_mode].test_f
lons = loaded[ref_mode].lons; lats = loaded[ref_mode].lats
T = length(n34_true)

# ----- per-mode metrics -----
const LEADS = [3, 6, 9, 12, 18, 24, 36, 48]
mode_full_accs = Dict{String,Vector{Float64}}()
mode_lead_mean = Dict{String,Vector{Float64}}()
mode_lead_min = Dict{String,Vector{Float64}}()
mode_lead_max = Dict{String,Vector{Float64}}()
mode_pattern_corr = Dict{String,Vector{Float64}}()
mode_ens_mean_n34 = Dict{String,Vector{Float64}}()

println("\n=== Per-mode summary ===")
for m in MODES
    haskey(loaded, m) || continue
    d = loaded[m]
    P = d.n34_preds   # seeds × T
    seeds = d.seeds
    # Full-window ACC per seed
    accs = [skill_score(n34_true, P[i, :]).acc for i in 1:size(P,1)]
    mode_full_accs[m] = accs
    # Lead-time cumulative ACC: per seed, then summarize
    L_used = [L for L in LEADS if L ≤ size(P,2)]
    means = Float64[]; mins = Float64[]; maxs = Float64[]
    for L in L_used
        per = [skill_score(n34_true[1:L], P[i, 1:L]).acc for i in 1:size(P,1)]
        push!(means, mean(per)); push!(mins, minimum(per)); push!(maxs, maximum(per))
    end
    mode_lead_mean[m] = means; mode_lead_min[m] = mins; mode_lead_max[m] = maxs
    # Ensemble-mean Niño 3.4
    ensn34 = vec(mean(P, dims=1))
    mode_ens_mean_n34[m] = ensn34
    # Ensemble-mean spatial field, per-month pattern correlation vs truth
    field_ens = cat(d.field_preds...; dims=4)
    field_mean = dropdims(mean(field_ens; dims=4); dims=4)
    pcorr = [spatial_pattern_corr(test_f[:, :, t], field_mean[:, :, t]) for t in 1:size(test_f,3)]
    mode_pattern_corr[m] = pcorr

    println("\n--- $(m) ($(length(seeds)) seeds) ---")
    println("  Full-window N3.4 ACC:")
    for (s, a) in zip(seeds, accs)
        println("    seed=$(lpad(s,3)): ACC=$(round(a; digits=3))")
    end
    println("  median=$(round(median(accs); digits=3))   mean=$(round(mean(accs); digits=3))")
    println("  N3.4 ens-mean ACC: $(round(skill_score(n34_true, ensn34).acc; digits=3))")
    println("  Spatial pc (ens-mean vs truth):")
    for L in [3, 6, 9, 12, 18, 24]
        L ≤ length(pcorr) && println("    $(lpad(L,2)) mo: pc=$(round(pcorr[L]; digits=3))")
    end
end

# ----- FIGURE: 2×2 panel comparing modes -----
# Panel A: per-mode ACC distribution (jittered scatter + median bar)
pA = plot(xlabel="architecture", ylabel="full-window Niño 3.4 ACC",
          title="Per-seed ACC distribution",
          ylim=(-1.0, 1.0), grid=true, legend=false)
mode_keys = [m for m in MODES if haskey(mode_full_accs, m)]
for (i, m) in enumerate(mode_keys)
    accs = mode_full_accs[m]
    xs = fill(i, length(accs)) .+ 0.05 .* randn(length(accs))
    scatter!(pA, xs, accs; color=MODE_COLORS[m], ms=6, alpha=0.75)
    plot!(pA, [i - 0.3, i + 0.3], [median(accs), median(accs)];
          color=MODE_COLORS[m], lw=3, alpha=0.9)
end
xticks!(pA, 1:length(mode_keys), [MODE_LABEL[m] for m in mode_keys])
hline!(pA, [0.0]; ls=:dot, color=:black, lw=0.7)
hline!(pA, [0.5]; ls=:dash, color=:gray, lw=0.7)

# Panel B: lead-time cumulative ACC, three architectures overlaid
pB = plot(xlabel="lead (months)", ylabel="cumulative Niño 3.4 ACC",
          title="Lead-time skill (across seeds)",
          ylim=(-0.5, 1.0), grid=true, legend=:bottomleft)
for m in mode_keys
    L_used = [L for L in LEADS if L ≤ T]
    plot!(pB, L_used, mode_lead_mean[m]; ribbon=(mode_lead_mean[m] .- mode_lead_min[m],
                                                 mode_lead_max[m] .- mode_lead_mean[m]),
          color=MODE_COLORS[m], lw=2.5, marker=:circle, fillalpha=0.18,
          label=MODE_LABEL[m])
end
hline!(pB, [0.5]; ls=:dash, color=:gray, label="0.5"); hline!(pB, [0.0]; ls=:dot, color=:black, label=false)

# Panel C: spatial pattern correlation over time (ensemble mean), three modes
pC = plot(xlabel="month since forecast start", ylabel="spatial pattern correlation",
          title="Spatial pc — ensemble-mean field vs truth",
          ylim=(-0.5, 1.0), grid=true, legend=:topright)
for m in mode_keys
    plot!(pC, 1:length(mode_pattern_corr[m]), mode_pattern_corr[m];
          color=MODE_COLORS[m], lw=2, label=MODE_LABEL[m])
end
hline!(pC, [0.5]; ls=:dash, color=:gray, label=false)
hline!(pC, [0.0]; ls=:dot,  color=:black, label=false)

# Panel D: Niño 3.4 ensemble-mean trajectory overlay
pD = plot(xlabel="month since forecast start", ylabel="Niño 3.4 (norm.)",
          title="Niño 3.4 ensemble-mean trajectories",
          grid=true, legend=:topright)
plot!(pD, n34_true; lw=2.5, color=:black, label="Observed")
for m in mode_keys
    plot!(pD, mode_ens_mean_n34[m]; lw=2, ls=:dash, color=MODE_COLORS[m], label=MODE_LABEL[m])
end

p_grid = plot(pA, pB, pC, pD; layout=(2, 2), size=(1500, 950),
              plot_title="Multi-layer ablation — monthly ENSO ($(join([MODE_LABEL[m] for m in mode_keys], "/")))",
              left_margin=4mm, bottom_margin=3mm)
out_main = joinpath(RESULTS_DIR, "enso_ablation_compare.png")
savefig(p_grid, out_main)
println("\nSaved: $(out_main)")

# ----- maps figure: each mode's ensemble-mean SST forecast at selected leads -----
const MAP_LEADS = filter(L -> L ≤ T, [6, 12, 18])
sc_map = 1.5
lons_plot = mod.(lons, 360.0); perm = sortperm(lons_plot); lons_plot = lons_plot[perm]
function map_panel(field2d, ttl; clim_=(-sc_map, sc_map), cmap=:RdBu)
    heatmap(lons_plot, lats, permutedims(field2d[perm, :]);
            clim=clim_, c=cmap, xlabel="lon (°E)", ylabel="lat (°N)",
            aspect_ratio=:equal,
            xlims=(minimum(lons_plot), maximum(lons_plot)),
            ylims=(minimum(lats), maximum(lats)),
            title=ttl, colorbar=true)
end

map_panels = Plots.Plot[]
for L in MAP_LEADS
    push!(map_panels, map_panel(test_f[:, :, L], "Observed  $(L) mo"))
    for m in mode_keys
        d = loaded[m]
        field_ens = cat(d.field_preds...; dims=4)
        field_mean = dropdims(mean(field_ens; dims=4); dims=4)
        pc = round(spatial_pattern_corr(test_f[:, :, L], field_mean[:, :, L]); digits=2)
        push!(map_panels, map_panel(field_mean[:, :, L],
                                    "$(MODE_LABEL[m])  $(L) mo  pc=$(pc)"))
    end
end

ncol = 1 + length(mode_keys)
p_maps = plot(map_panels...; layout=(length(MAP_LEADS), ncol),
              size=(450 * ncol, 320 * length(MAP_LEADS)),
              plot_title="Ensemble-mean SST anomaly — observed vs each mode",
              left_margin=4mm, bottom_margin=3mm)
out_maps = joinpath(RESULTS_DIR, "enso_ablation_maps.png")
savefig(p_maps, out_maps)
println("Saved: $(out_maps)")
