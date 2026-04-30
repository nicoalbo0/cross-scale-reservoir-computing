# Aggregate Tier 3 rolling-window results into per-window×architecture summaries
# and a side-by-side robustness plot.
#
# Usage:
#   julia --project=. scripts/aggregate_rolling.jl <ROOT>
# where ROOT defaults to results/temporal_multiscale_compare/tier3_rolling.

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Glob, Plots, Measures, Printf

ROOT = length(ARGS) ≥ 1 ? ARGS[1] : "results/temporal_multiscale_compare/tier3_rolling"
isdir(ROOT) || error("Missing $(ROOT)")

# discover (mode, champ) directories under ROOT/<mode>_<champ>/<window>/
champ_dirs = filter(isdir, glob("*_*", ROOT))

function load_window(dir)
    paths = sort(glob("enso_temporal_field_preds_*_seed*.jld2", dir))
    isempty(paths) && return nothing
    seeds = Int[]; acc12 = Float64[]; pc3 = Float64[]; sr12 = Float64[]
    for p in paths
        jldopen(p, "r") do f
            push!(seeds, Int(f["seed"]))
            push!(acc12, Float64(f["acc12"]))
            push!(pc3,   Float64(f["pc3"]))
            push!(sr12,  Float64(f["std_ratio12"]))
        end
    end
    return (seeds = seeds, acc12 = acc12, pc3 = pc3, sr12 = sr12)
end

println("="^72)
println("Stage E Tier 3 — rolling-window robustness")
println("ROOT = $(ROOT)")
println("="^72)

results = Dict{String, Dict{String, NamedTuple}}()
for cd in champ_dirs
    rel = relpath(cd, ROOT)
    occursin('/', rel) && continue
    win_dirs = sort(filter(isdir, glob("W*", cd)))
    isempty(win_dirs) && continue
    println("\n## $(rel)")
    println(@sprintf("  %-4s  %-15s  %-10s  %-15s  %-15s  %-3s",
                     "win", "acc12 (mean±std)", "worst", "pc3 (mean±std)", "sr12 (mean±std)", "n"))
    results[rel] = Dict{String, NamedTuple}()
    for wd in win_dirs
        wlabel = basename(wd)
        m = load_window(wd)
        m === nothing && continue
        results[rel][wlabel] = m
        println(@sprintf("  %-4s  %0.3f±%0.3f      %0.3f       %0.3f±%0.3f      %0.3f±%0.3f      %d",
                         wlabel, mean(m.acc12), std(m.acc12), minimum(m.acc12),
                         mean(m.pc3), std(m.pc3), mean(m.sr12), std(m.sr12), length(m.seeds)))
    end
end

# Robust composite per arch
println("\n--- Robust composite: 0.5·mean(ACC) + 0.3·worst-window ACC + 0.2·mean(pc3) ---")
for (arch, win_dict) in results
    accs = vcat([w.acc12 for w in values(win_dict)]...)
    pc3s = vcat([w.pc3   for w in values(win_dict)]...)
    worst = minimum([mean(w.acc12) for w in values(win_dict)])
    rc = 0.5 * mean(accs) + 0.3 * worst + 0.2 * mean(pc3s)
    println(@sprintf("  %-30s robust_composite = %0.3f   (mean_acc=%0.3f  worst=%0.3f  mean_pc3=%0.3f)",
                     arch, rc, mean(accs), worst, mean(pc3s)))
end

# side-by-side bar plot
plt = plot(layout = (1, 3), size = (1500, 400), left_margin = 4mm, bottom_margin = 4mm,
           legendfontsize = 8)
arch_keys = sort(collect(keys(results)))
colors = [:steelblue, :firebrick, :darkgoldenrod, :purple, :seagreen]
for (i, metric) in enumerate([:acc12, :pc3, :sr12])
    title_metric = string(metric)
    title!(plt[i], title_metric)
    for (j, arch) in enumerate(arch_keys)
        win_dict = results[arch]
        wlabels = sort(collect(keys(win_dict)))
        means = [mean(getfield(win_dict[w], metric)) for w in wlabels]
        stds  = [std(getfield(win_dict[w], metric)) for w in wlabels]
        plot!(plt[i], wlabels, means; yerr = stds, label = arch,
              marker = :circle, lw = 2, color = colors[mod1(j, length(colors))],
              xlabel = "window", ylabel = title_metric)
    end
end
out_png = joinpath(ROOT, "rolling_summary.png")
savefig(plt, out_png)
println("\nSaved $(out_png)")
