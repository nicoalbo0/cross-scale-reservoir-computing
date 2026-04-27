# Aggregate 3L hyperparameter sweep.
#
# Walks results/tune_3L/<cfg>/ subfolders, computes per-config N3.4 ACC and
# spatial pattern correlation at standard leads, and produces a summary table
# + per-parameter 1D sensitivity plot.
#
# The "baseline 3L center point" comes from results/enso_monthly_preds_three_layer_seed*.jld2
# (8 seeds, the existing E4 ensemble) — matched on the same 3 seeds used in
# the sweep when we want apples-to-apples per-seed comparison; full 8-seed
# baseline shown alongside for context.

ENV["GKSwstype"] = "100"
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CrossScaleRC, JLD2, Statistics, LinearAlgebra, Glob, Plots, Measures, Printf

const RESULTS = joinpath(@__DIR__, "..", "results")
const TUNE = joinpath(RESULTS, "tune_3L")
const SWEEP_SEEDS = [1, 42, 99]

function spatial_pc(A::AbstractMatrix, B::AbstractMatrix)
    a = vec(A); b = vec(B)
    am = a .- mean(a); bm = b .- mean(b)
    return dot(am, bm) / (norm(am) * norm(bm) + eps())
end

function load_runs(dir::AbstractString; seed_filter=nothing)
    paths = sort(glob("enso_monthly_preds_three_layer_seed*.jld2", dir))
    isempty(paths) && return nothing
    n34_true = nothing
    test_f = nothing
    seeds = Int[]
    n34_preds = Vector{Vector{Float64}}()
    fields = Array{Float64,3}[]
    for p in paths
        f = jldopen(p)
        s = Int(f["seed"])
        if seed_filter !== nothing && !(s in seed_filter); close(f); continue; end
        n34_true = Float64.(f["n34_true"])
        push!(n34_preds, Float64.(f["n34_pred"]))
        push!(fields, Float64.(f["preds_f"]))
        push!(seeds, s)
        if test_f === nothing && "test_f" in keys(f)
            test_f = Float64.(f["test_f"])
        end
        close(f)
    end
    if test_f === nothing
        ref = joinpath(dir, "enso_monthly_reference_three_layer.jld2")
        if isfile(ref)
            f = jldopen(ref); test_f = Float64.(f["test_f"]); close(f)
        end
    end
    return (seeds=seeds, n34_true=n34_true, n34_preds=n34_preds,
            fields=fields, test_f=test_f)
end

# Compute (12-mo cum N3.4 ACC, 12-mo spatial pc) for an ensemble.
function metrics(d; lead=12)
    isnothing(d) && return nothing
    P = reduce(vcat, [reshape(p, 1, length(p)) for p in d.n34_preds])
    accs = [skill_score(d.n34_true[1:lead], P[i, 1:lead]).acc for i in 1:size(P, 1)]
    field_ens = cat(d.fields...; dims=4)
    field_mean = dropdims(mean(field_ens; dims=4); dims=4)
    pc = spatial_pc(d.test_f[:, :, lead], field_mean[:, :, lead])
    return (accs=accs, pc=pc)
end

# Load baseline 3L (full 8 seeds) and seed-matched subset for fairness.
println("Loading baseline 3L from $(RESULTS)")
base_full = load_runs(RESULTS)
base_match = load_runs(RESULTS; seed_filter=Set(SWEEP_SEEDS))
println("  full baseline: $(length(base_full.seeds)) seeds")
println("  seed-matched : $(length(base_match.seeds)) seeds  ($(base_match.seeds))")

m_base_full  = metrics(base_full)
m_base_match = metrics(base_match)

# Load each tuning config.
cfg_dirs = sort(filter(isdir, [joinpath(TUNE, d) for d in readdir(TUNE)]))
cfg_results = Dict{String, NamedTuple}()
for d in cfg_dirs
    name = basename(d)
    runs = load_runs(d)
    if runs === nothing
        @warn "no runs in $(name)"; continue
    end
    cfg_results[name] = runs
    println("Loaded cfg=$(name): $(length(runs.seeds)) seeds")
end

# ===== Print summary table =====
println("\n" * "="^78)
println("3L HYPERPARAMETER SWEEP — 12-month metrics")
println("="^78)
@printf("%-14s | %-2s | %-30s | %-10s | %s\n",
        "config", "N", "12-mo N3.4 ACC (per-seed)", "12-mo pc", "median ACC")
println("-"^78)
@printf("%-14s | %2d | %-30s | %+10.3f | %+10.3f\n",
        "BASELINE-8", length(base_full.seeds),
        "[" * join([@sprintf("%+.2f", a) for a in m_base_full.accs], ",") * "]",
        m_base_full.pc, median(m_base_full.accs))
@printf("%-14s | %2d | %-30s | %+10.3f | %+10.3f\n",
        "BASELINE-3", length(base_match.seeds),
        "[" * join([@sprintf("%+.2f", a) for a in m_base_match.accs], ",") * "]",
        m_base_match.pc, median(m_base_match.accs))
println("-"^78)

# Sort configs by config family then numeric value
function parse_cfg(name)
    # tauF10 / tauC60 / rF0p1 / gBpm2p0
    if startswith(name, "tauF"); return (1, parse(Float64, replace(name[5:end], "p"=>".")))
    elseif startswith(name, "tauC"); return (2, parse(Float64, replace(name[5:end], "p"=>".")))
    elseif startswith(name, "rF"); return (3, parse(Float64, replace(name[3:end], "p"=>".")))
    elseif startswith(name, "gB"); return (4, parse(Float64, replace(replace(name[3:end], "p"=>"."), "m"=>"-")))
    else; return (99, 0.0)
    end
end
ordered = sort(collect(keys(cfg_results)); by=parse_cfg)

for name in ordered
    d = cfg_results[name]
    m = metrics(d)
    @printf("%-14s | %2d | %-30s | %+10.3f | %+10.3f\n",
            name, length(d.seeds),
            "[" * join([@sprintf("%+.2f", a) for a in m.accs], ",") * "]",
            m.pc, median(m.accs))
end
println("="^78)

# ===== Per-parameter sensitivity plot =====
function family_params(prefix)
    if prefix == "tauF"; return ("τ_fine", 3.0, [10.0, 20.0])
    elseif prefix == "tauC"; return ("τ_coarse", 30.0, [15.0, 60.0])
    elseif prefix == "rF"; return ("ridge_fine", 1.0, [0.1, 10.0])
    elseif prefix == "gB"; return ("glayer_B_exp", -1.0, [-2.0, 0.0])
    end
end

panels = Plots.Plot[]
for prefix in ["tauF", "tauC", "rF", "gB"]
    label, default_x, _ = family_params(prefix)
    cfgs = filter(n -> startswith(n, prefix), ordered)
    isempty(cfgs) && continue
    xs = Float64[default_x]
    acc_med = Float64[median(m_base_match.accs)]
    acc_lo  = Float64[minimum(m_base_match.accs)]
    acc_hi  = Float64[maximum(m_base_match.accs)]
    pcs = Float64[m_base_match.pc]
    for c in cfgs
        d = cfg_results[c]
        m = metrics(d)
        x = parse_cfg(c)[2]
        push!(xs, x); push!(acc_med, median(m.accs)); push!(acc_lo, minimum(m.accs))
        push!(acc_hi, maximum(m.accs)); push!(pcs, m.pc)
    end
    perm = sortperm(xs)
    xs = xs[perm]; acc_med = acc_med[perm]; acc_lo = acc_lo[perm]
    acc_hi = acc_hi[perm]; pcs = pcs[perm]

    p = plot(xs, acc_med; lw=2, marker=:circle, color=:red,
             ribbon=(acc_med .- acc_lo, acc_hi .- acc_med),
             fillalpha=0.25, label="N3.4 ACC (12mo)",
             xlabel=label, ylabel="12-mo metric",
             title="Sensitivity: $(label)",
             ylim=(-0.2, 1.0), grid=true, legend=:bottomright)
    plot!(p, xs, pcs; lw=2, marker=:diamond, color=:blue, ls=:dash,
          label="spatial pc (12mo)")
    vline!(p, [default_x]; ls=:dot, color=:gray, lw=1, label="default")
    push!(panels, p)
end

p_grid = plot(panels...; layout=(2, 2), size=(1500, 950),
              plot_title="3L hyperparameter sensitivity — 12-mo metrics  (3 seeds/config; baseline = 8 seeds)",
              left_margin=4mm, bottom_margin=3mm)
out = joinpath(RESULTS, "tune_3L_sensitivity.png")
savefig(p_grid, out)
println("\nSaved: $(out)")
