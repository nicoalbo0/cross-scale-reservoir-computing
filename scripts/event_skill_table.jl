# Compute event-skill metric for the Stage-E Tier 4 archs (no recompute).
# Reads per-seed JLD2 files in tier4_head_to_head/<arch>_W1/ and the truth
# reference, runs event_skill on each (arch, seed), reports a table.
#
# Usage:
#   julia --project=. scripts/event_skill_table.jl
#
# Optional: pass `--lead-window=10,20` `--phase-tol=3` etc. as args, or use the
# defaults below.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Glob, Printf, CrossScaleRC

ROOT = "results/temporal_multiscale_compare/tier4_head_to_head"
ARCHS = [
    ("A_no_xscale_field",     joinpath(ROOT, "no_xscale_field_W1"),    "no_xscale_field"),
    ("B_multi_tau_3_field",   joinpath(ROOT, "multi_tau_3_field_W1"),  "multi_tau_3_field"),
    ("D_full_cascade_field",  joinpath(ROOT, "full_cascade_field_W1"), "full_cascade_field"),
    # Baselines (seed=0 only; deterministic). Comment out if not yet run.
    ("baseline_climatology",          joinpath(ROOT, "climatology_W1"),          "climatology"),
    ("baseline_persistence",          joinpath(ROOT, "persistence_W1"),          "persistence"),
    ("baseline_damped_pers_τ=3",      joinpath(ROOT, "damped_persistence_3_W1"), "damped_persistence_3"),
    ("baseline_damped_pers_τ=6",      joinpath(ROOT, "damped_persistence_6_W1"), "damped_persistence_6"),
    ("baseline_damped_pers_τ=12",     joinpath(ROOT, "damped_persistence_12_W1"), "damped_persistence_12"),
]

# Knobs. lead_window is the range of lead times (months since forecast start)
# at which we evaluate event-prediction. The user's stated goal is "predict
# events 10-20+ months out"; in W1 the forecast starts Jan 2006 and the actual
# events are at L=12, 25, 37, 48, 60, 71. Use (3, 36) to cover the first 3
# events comfortably; the long-lead ones (L=48+) are beyond the practical
# horizon and should be reported separately if at all.
lead_window     = (3, 36)
phase_tol       = 3
event_threshold = 1.0
min_separation  = 6

# --- truth (any reference works — they all share the same truth in W1) ---
ref_path = joinpath(ARCHS[1][2], "enso_temporal_field_reference_no_xscale_field.jld2")
n34_true, field_true = jldopen(ref_path, "r") do f
    Float64.(f["n34_true"]), Float64.(f["field_true"])
end

# --- Print truth-detected events restricted to lead_window ---
all_events = find_enso_events(n34_true; threshold = event_threshold,
                              min_separation = min_separation,
                              n_events = length(n34_true))
events_in_window = filter(t -> lead_window[1] ≤ t ≤ lead_window[2], all_events)
println("Lead window: $(lead_window[1])–$(lead_window[2]) months  (phase_tol=±$(phase_tol))")
println("Truth events DETECTED in window: $(events_in_window)")
println("    n34_true at events: $(round.(n34_true[events_in_window]; digits=2))")
println()

# --- Compute event_skill per (arch, seed) ---
println("="^96)
@printf("%-26s  %-5s  %-7s  %-9s  %-9s  %-7s  %-7s  %-7s\n",
    "arch", "seed", "n_evts", "sign_acc", "mean_pc", "wt_pc", "FA", "phase_b")
println("-"^96)

results = Dict{String, NamedTuple}()
for (label, dir, mode) in ARCHS
    paths = sort(glob("enso_temporal_field_preds_$(mode)_seed*.jld2", dir))
    accs = Float64[]; pcs = Float64[]; wpcs = Float64[]
    fas  = Int[]; pbias = Float64[]; nevs = Int[]
    seeds = Int[]
    for p in paths
        jldopen(p, "r") do f
            n34_pred = Float64.(f["n34_pred"])
            fld_pred = Float64.(f["field_pred"])
            seed     = Int(f["seed"])
            s = event_skill(n34_true, field_true, n34_pred, fld_pred;
                            lead_window = lead_window, phase_tol = phase_tol,
                            event_threshold = event_threshold,
                            min_separation = min_separation)
            push!(seeds, seed)
            push!(accs,  s.sign_accuracy)
            push!(pcs,   s.mean_event_pc)
            push!(wpcs,  s.weighted_event_pc)
            push!(fas,   s.false_alarms)
            push!(pbias, s.phase_bias_mean)
            push!(nevs,  s.n_events)
            @printf("%-26s  %-5d  %-7d  %-9.3f  %-9.3f  %-7.3f  %-7d  %-+7.2f\n",
                    label, seed, s.n_events, s.sign_accuracy,
                    s.mean_event_pc, s.weighted_event_pc, s.false_alarms,
                    s.phase_bias_mean)
        end
    end
    results[label] = (sign_acc = accs, mean_pc = pcs, w_pc = wpcs,
                       fa = fas, pbias = pbias, n_ev = nevs, seeds = seeds)
end

# --- Ensemble summary table (mean ± std across 8 seeds) ---
println("\n" * "="^96)
println("ENSEMBLE SUMMARY (8 seeds, mean ± std)")
println("="^96)
@printf("%-26s  %-15s  %-15s  %-15s  %-7s\n",
    "arch", "sign_acc", "mean_event_pc", "weighted_pc", "FA mean")
println("-"^96)
for (label, _, _) in ARCHS
    r = results[label]
    @printf("%-26s  %.3f ± %.3f  %.3f ± %.3f  %.3f ± %.3f  %.2f\n",
        label,
        mean(r.sign_acc), std(r.sign_acc),
        mean(r.mean_pc),  std(r.mean_pc),
        mean(r.w_pc),     std(r.w_pc),
        mean(r.fa))
end

# --- Persist as CSV for later joining with baselines ---
mkpath("results/stage_G")
out_csv = "results/stage_G/event_skill_one_step.csv"
open(out_csv, "w") do io
    write(io, "arch,seed,n_events,sign_accuracy,mean_event_pc,weighted_event_pc,false_alarms,phase_bias_mean\n")
    for (label, _, _) in ARCHS
        r = results[label]
        for i in 1:length(r.seeds)
            write(io, "$(label),$(r.seeds[i]),$(r.n_ev[i]),$(r.sign_acc[i]),$(r.mean_pc[i]),$(r.w_pc[i]),$(r.fa[i]),$(r.pbias[i])\n")
        end
    end
end
println("\nSaved $(out_csv)")
