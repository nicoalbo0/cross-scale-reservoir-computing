# Paired statistical test for the Stage-E head-to-head Tier 4.
#
# Compares architecture A (e.g. no_xscale_field) vs B (e.g. multi_tau_3_field)
# on the same set of seeds at the same window. Reports paired
# Wilcoxon signed-rank p-values, Cohen's d, and a recommendation.
#
# Usage:
#   julia --project=. scripts/stat_test_head_to_head.jl <dirA> <dirB> [<dirD>]
#
# Each <dir> contains enso_temporal_field_preds_<mode>_seed*.jld2 for the same set
# of seeds.

using JLD2, Statistics, Glob, HypothesisTests, Printf

length(ARGS) ≥ 2 || error("Usage: stat_test_head_to_head.jl <dirA> <dirB> [<dirD>]")

function load_metrics(dir)
    paths = sort(glob("enso_temporal_field_preds_*_seed*.jld2", dir))
    isempty(paths) && error("No per-seed files in $(dir)")
    seeds = Int[]; acc12 = Float64[]; rmse12 = Float64[]; sr12 = Float64[]
    pc3 = Float64[]; pc12 = Float64[]
    ppn = Float64[]; ppg = Float64[]
    for p in paths
        jldopen(p, "r") do f
            push!(seeds,  Int(f["seed"]))
            push!(acc12,  Float64(f["acc12"]))
            push!(rmse12, Float64(f["rmse12"]))
            push!(sr12,   Float64(f["std_ratio12"]))
            push!(pc3,    Float64(f["pc3"]))
            push!(pc12,   Float64(f["pc12"]))
            push!(ppn,    Float64(f["ppacc_n34_mean"]))
            push!(ppg,    Float64(f["ppacc_global_mean"]))
        end
    end
    perm = sortperm(seeds)
    return (seeds = seeds[perm], acc12 = acc12[perm], rmse12 = rmse12[perm],
            sr12 = sr12[perm], pc3 = pc3[perm], pc12 = pc12[perm],
            ppn = ppn[perm], ppg = ppg[perm])
end

function pair_test(label, xa, xb, seeds_a, seeds_b)
    common = intersect(seeds_a, seeds_b)
    isempty(common) && error("No common seeds")
    ia = [findfirst(==(s), seeds_a) for s in common]
    ib = [findfirst(==(s), seeds_b) for s in common]
    a = xa[ia]; b = xb[ib]
    d = a .- b
    w = SignedRankTest(d)
    p = pvalue(w)
    cohen = mean(d) / (std(d) + eps())
    @printf("  %-22s  ΔA−B=%+0.4f ± %0.4f   p=%.4f   d=%+0.3f   (n=%d)\n",
            label, mean(d), std(d), p, cohen, length(d))
    return (Δmean = mean(d), p = p, cohen = cohen, n = length(d))
end

dirA = ARGS[1]; dirB = ARGS[2]
A = load_metrics(dirA); B = load_metrics(dirB)

println("="^72)
println("Stage-E Tier 4 paired test")
println("  A = $(dirA)  (n=$(length(A.seeds)))")
println("  B = $(dirB)  (n=$(length(B.seeds)))")
println("="^72)
println("Seeds A: $(A.seeds)")
println("Seeds B: $(B.seeds)")

println("\n--- A vs B (signed = A − B) ---")
res_acc  = pair_test("acc12 (A−B)",   A.acc12,  B.acc12,  A.seeds, B.seeds)
res_rmse = pair_test("rmse12 (A−B)",  A.rmse12, B.rmse12, A.seeds, B.seeds)
res_sr   = pair_test("std_ratio12",   A.sr12,   B.sr12,   A.seeds, B.seeds)
res_pc3  = pair_test("pc3 (A−B)",     A.pc3,    B.pc3,    A.seeds, B.seeds)
res_pc12 = pair_test("pc12 (A−B)",    A.pc12,   B.pc12,   A.seeds, B.seeds)
res_ppn  = pair_test("ppacc_n34",     A.ppn,    B.ppn,    A.seeds, B.seeds)
res_ppg  = pair_test("ppacc_global",  A.ppg,    B.ppg,    A.seeds, B.seeds)

if length(ARGS) ≥ 3
    dirD = ARGS[3]; D = load_metrics(dirD)
    println("\n--- A vs D (signed = A − D) ---")
    pair_test("acc12 (A−D)",  A.acc12,  D.acc12,  A.seeds, D.seeds)
    pair_test("pc3 (A−D)",    A.pc3,    D.pc3,    A.seeds, D.seeds)
    pair_test("std_ratio12",  A.sr12,   D.sr12,   A.seeds, D.seeds)

    println("\n--- B vs D (signed = B − D) ---")
    pair_test("acc12 (B−D)",  B.acc12,  D.acc12,  B.seeds, D.seeds)
    pair_test("pc3 (B−D)",    B.pc3,    D.pc3,    B.seeds, D.seeds)
    pair_test("std_ratio12",  B.sr12,   D.sr12,   B.seeds, D.seeds)
end

# ----------------------------------------------------------------------------
# Pre-registered decision tree (from plan)
# ----------------------------------------------------------------------------
ΔACC = res_acc.Δmean; pACC = res_acc.p

println("\n" * "="^72)
println("Pre-registered decision (Stage-E §Tier 4)")
println("="^72)
verdict = if pACC < 0.05
    if ΔACC > 0
        "DECLARE A WINS (A − B = $(round(ΔACC; digits=3)), Wilcoxon p=$(round(pACC;digits=4)))"
    else
        "DECLARE B WINS (B − A = $(round(-ΔACC; digits=3)), Wilcoxon p=$(round(pACC;digits=4)))"
    end
elseif abs(ΔACC) < 0.01
    "DECLARE TIE (|ΔACC|=$(round(abs(ΔACC);digits=4))<0.01, p=$(round(pACC;digits=4))). Recommend simpler arch (A)."
else
    "INCONCLUSIVE (p=$(round(pACC;digits=4)) ≥ 0.05 but |ΔACC|=$(round(abs(ΔACC);digits=4)) ≥ 0.01). Extend to 16 seeds."
end
println("  $(verdict)")

# Pareto check: ACC and amplitude metric agreement
acc_winner = ΔACC > 0 ? "A" : "B"
sr_winner_a_better = abs(1 - mean(A.sr12)) < abs(1 - mean(B.sr12))
sr_winner = sr_winner_a_better ? "A" : "B"
if acc_winner != sr_winner
    println("  ⚠ Pareto disagreement: ACC favours $(acc_winner) but std_ratio favours $(sr_winner) (A=$(round(mean(A.sr12);digits=2)), B=$(round(mean(B.sr12);digits=2)))")
    println("    Report both. Decide by Stage-F preference.")
end
