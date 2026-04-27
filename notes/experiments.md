# ENSO Experiments Log

A running record of every ENSO experiment we've tried. Goal: never repeat a
mistake, never re-run a settled question. Newest experiments at the top.

Conventions:
- One row per discrete experiment.
- "Outcome" is what we learned, not just numbers.
- Output dirs are namespaced under `results/<tag>/`.

---

## E5 — Tau-matched single-layer (2026-04-27)
**Question:** is single-layer's competitive 12-mo ACC (E4) just because we
gave it the wrong τ? If we match the deepest τ to 3L's coarse layer (τ=30),
does 1L beat 3L (multi-layer claim dead) or does it now underperform 3L
(multi-layer claim alive, E4 was a tuning artifact)?
**Setup:** main_enso_monthly.jl, mode=single_layer, ENSO_TAU_FINE=30,
ridge=1.0 unchanged, 8 seeds {1,2,3,7,11,23,42,99}.
**Output:** `results/ablation_taumatch/single_layer/`
**Outcome:**
- **1L τ=30 is now the best 12-mo Niño 3.4 forecaster: cum ACC = 0.891**
  (vs 0.855 for 1L τ=3, 0.828 for 3L). Per-seed range only 0.888–0.895 —
  near-zero seed variance, a strong sign of a converged closed-loop attractor.
- 1L τ=30 collapses harder at long leads: 24mo=−0.36, 36mo=−0.46. With τ=30
  the reservoir has a strong attractor but its phase drifts past one ENSO
  cycle.
- 3L still wins at long leads (>18 mo) AND wins decisively on 12-mo spatial
  pattern correlation (3L=0.42 vs 1L_τ30=0.11).
- **Net interpretation:** the architectures are complementary, not competing.
  - Best Niño 3.4 12-mo skill → single-layer with τ matched to ENSO timescale.
  - Best spatial-pattern fidelity → 3-layer cross-scale.
  - Best long-lead skill → 3-layer cross-scale (1L τ=30 is too aggressive).
- The E4 conclusion ("multi-layer doesn't help") was partly a tuning artifact
  — but multi-layer's true contribution is at SPATIAL FIDELITY and LONG LEAD,
  not at headline 12-mo ACC.

## E4 — Multi-layer ablation 1L / 2L / 3L (2026-04-27)
**Question:** does cross-scale wiring improve ENSO forecast over single-layer?
**Setup:** monthly pixel, 8 common seeds, identical hyperparams except mode.
**Outcome:**
- 12-mo cumulative N3.4 ACC essentially tied: 1L=0.855, 2L=0.820, 3L=0.828.
- 12-mo spatial pattern correlation: 1L=0.31, 2L=0.39, 3L=0.42 (3L marginally best).
- Long-lead (24+ mo): 2L collapses to ~0; 1L and 3L hold at ~0.4-0.5.
- **Major caveat (E5):** 1L was using τ=3 while 3L's coarse layer uses τ=30 —
  asymmetric tuning. E4 conclusions need confirmation from E5.
- Working interpretation: ENSO is spatially coherent / temporally nonlinear,
  so spatial cross-scale architecture isn't the right prior. See memory note
  `project_temporal_multiscale_idea.md` for proposed alternative.
**Outputs:** `results/enso_ablation_compare.png`, `results/enso_ablation_maps.png`,
per-seed JLD2s in `results/`.

## E3 — EOF scale-up (2026-04-26)
**Question:** does scaling K (modes) and N (reservoir) on a single-reservoir
EOF model improve spatial pattern correlation beyond the K=20/N=500 ceiling?
**Setup:** main_enso_eof.jl, single seed (42), three configs:
- K=50, N=1500, ridge=1e-2
- K=50, N=500, ridge=1e-2
- K=20, N=1500, ridge=1.0
**Outcome:** all three regressed vs K=20/N=500 baseline at long leads.
- K=20/N=1500/ridge=1.0 had best 3-mo spatial pc (0.512) but collapsed at
  12+ mo (0.18 at 12mo Niño 3.4). Classic capacity-vs-data overfit.
- Top-50 EOFs explain only 93.8% variance — modes 21-50 ≈ noise at this
  dataset size (288-month training window).
- **Key flaw:** all single-seed; for high-N reservoirs seed variance dominates
  (we'd need 4+ seeds per config to compare distributions).
- **Bigger flaw (in retrospect):** EOF single-reservoir is architecturally
  flat — not the multi-layer cross-scale claim. Wasn't the right scale-up
  to test.
**Outputs:** per-config JLD2/PNG at `results/enso_eof_*_K{K}_N{N}_seed42.{jld2,png}`.

## E2 — EOF baseline 8-seed ensemble (2026-04-25)
**Question:** does EOF projection (K=20, N=500) match the multi-layer pixel
model on Niño 3.4 and beat it on spatial pattern correlation?
**Setup:** main_enso_eof.jl, K=20, N=500, 8 seeds.
**Outcome:**
- Best seed (42): 12-mo cum ACC=0.87, 18-mo=0.88. Competitive with 3L pixel.
- Wide variance across seeds (12-mo cum ACC: 0.10 to 0.88). Not robust.
- Spatial pattern correlation peak ~0.4-0.5 at 3-6 mo, similar to 3L pixel.
- **Conclusion:** EOF doesn't break the spatial-pc ceiling. Same data limit.
**Outputs:** `results/enso_eof_*_K20_N500_seed{1,2,3,7,11,23,42,99}.{jld2,png}`.

## E1 — Monthly resolution 3-layer pixel (2026-04-24)
**Question:** does binning to monthly fix the daily-resolution failure?
**Setup:** main_enso_monthly.jl, mode=three_layer, 8 seeds initially, then 12.
**Outcome:**
- **Yes.** 12-mo cumulative N3.4 ACC ≈ 0.83 robust across seeds (small variance).
- 6-mo cum ACC ≈ 0.61, 18-mo ≈ 0.47, 24+ mo collapses.
- Spatial pattern correlation peaks ~0.4-0.5 then drops. Captures dominant
  ENSO mode but not finer pattern.
- **Reservoir τ=30 at coarse layer + monthly step = 2-yr memory** is the key
  ingredient. Daily step + same τ wouldn't have given enough memory.
**Outputs:** `results/enso_monthly_preds_three_layer_seed*.jld2` (12 seeds).

## E0 — Daily resolution attempt (PRE-2026-04-24, abandoned)
**Setup:** main_enso.jl on daily SST anomalies.
**Outcome:** failed completely. Reservoir τ couldn't bridge the 2–7 yr ENSO
timescale at 1-day step.
**Lesson:** match reservoir step to phenomenon's natural timescale.
