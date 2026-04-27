# ENSO Experiments Log

A running record of every ENSO experiment we've tried. Goal: never repeat a
mistake, never re-run a settled question. Newest experiments at the top.

Conventions:
- One row per discrete experiment.
- "Outcome" is what we learned, not just numbers.
- Output dirs are namespaced under `results/<tag>/`.

---

## Settled defaults (state of the art for monthly 3L pixel)

After E1-E10, the best 3L configuration on both 12-mo Niño 3.4 ACC and 12-mo
spatial pattern correlation is:

```
mode = three_layer
τ_coarse=15, τ_mid=10, τ_fine=3
ridge_coarse=1e-3, ridge_mid=1e-2, ridge_fine=1.0
g_layer_A_exp=-1.0, g_layer_B_exp=-1.0
N_coarse=N_mid=N_fine=500
rad_coarse=0.85, rad_mid=0.75, rad_fine=0.55
grids: (3,1) → (9,3) → (9,4)
mixing=2
regression=quadratic
```

→ **12-mo Niño 3.4 ACC = 0.831 ± 0.014** (range 0.819-0.847 across 8 seeds)
→ **12-mo spatial pc = 0.424** (ensemble mean across 8 seeds)

**Hyperparameter sensitivity ranking (high → low):**
1. τ_fine — sharp sweet spot at 3; 10+ collapses model.
2. g_layer_B — counter-intuitive: weaker coupling → ACC↑/pc↓.
3. τ_coarse — modest; sweet spot at 15-30.
4. Spatial grids — all refinements hurt pc.
5. ridge_fine — well-tuned at 1.0.
6. N (any layer) — insensitive in [300, 800].
7. rad (any layer) — insensitive in [0.4, 0.95].

**Structural trade-off (NOT breakable by tuning):** spatial cross-scale wiring
trades index-ACC for spatial fidelity. Pure single-layer (no wiring) reaches
ACC=0.891 but pc=0.11. Multi-layer with default wiring reaches pc=0.42 but
ACC=0.83. Cannot improve both simultaneously by tuning parameters of this
architecture. Going beyond requires structural change — see project memory
`project_temporal_multiscale_idea.md`.

---

## E10 — 3L capacity + spectral-radius sweep (2026-04-27)
**Question:** does scaling N_coarse, N_fine, rad_coarse, rad_fine break the
ceiling? Final coverage of the remaining hyperparameter axes.
**Setup:** 4 axes × 2 levels × 3 seeds at tauC15 default.
**Outcome:** all configs within seed noise of tauC15 baseline; no measurable
improvement on either ACC or pc. **N and rad are the lowest-sensitivity
axes.** Defaults (N=500/500/500, rad=0.85/0.75/0.55) are fine.
**Output:** `results/tune_3L/{Ncoarse,Nfine,radCoarse,radFine}*/`

## E9 — 3L grid-structure sweep (2026-04-27)
**Question:** is 12-mo spatial pc capped by the (3,1) coarse-lat blocking
at 18°? The pre-existing 3L code comment says finer lat blocks tank ACC,
but the test was done before tauC15 was the default and before our 8-seed
protocol. Re-test with current setup.
**Setup:** 4 configs × 3 seeds at tauC15 default. (3 lat strips at 18° fails
divisibility — 18° has 4 lat cells, only divisors {1,2,4} are valid.)
**Outcome:**
- All grid refinements REDUCE pc (0.32-0.36 vs 3-seed baseline 0.365).
- grid_2lat6 lifts 12-mo ACC slightly to 0.842 but loses pc.
- **The (3,1) coarse-lat decision was correct** — it's not a bottleneck.
- Spatial-pc ceiling is *not* set by coarse-block resolution.
**Output:** `results/tune_3L/grid_*/`
**Question:** is 12-mo spatial pc capped by the (3,1) coarse-lat blocking
at 18°? The pre-existing 3L code comment says finer lat blocks tank ACC,
but the test was done before tauC15 was the default and before our 8-seed
protocol. Re-test with current setup.
**Setup:** 4 configs × 3 seeds at tauC15 default:
- grid_18lat2: g18_lat=2 (was 1)
- grid_18lat3: g18_lat=3
- grid_2lat6:  g2_lat=6 (was 4)
- grid_18lat2_2lat6: combined refinement
**Output:** `results/tune_3L/grid_*/`
**Outcome:** TBD.

## E8 — tauC15 alone × 8 seeds (2026-04-27)
**Question:** does τ_coarse=15 (E6's only ACC+pc improvement) hold up at 8
seeds? Joint config gBm2p0+tauC15 (E7) failed — gBm2p0 dominated and degraded
spatial pc. Test tauC15 alone.
**Setup:** main_enso_monthly.jl, mode=three_layer, ENSO_TAU_COARSE=15,
8 seeds {1,2,3,7,11,23,42,99}.
**Outcome:**
- 12-mo ACC: 0.831 vs baseline 0.828 (+0.003, within noise).
- 12-mo pc: 0.424 vs baseline 0.415 (+0.009).
- 18-mo ACC: 0.495 vs 0.473 (+0.022).
- **24-mo ACC: 0.507 vs 0.442 (+0.065), pc: 0.170 vs 0.070 (+0.100).**
- Net: small but clean improvement on both axes, biggest at long leads.
- **Adopted as new 3L default (τ_coarse: 30 → 15).**
- The 12-mo pc ceiling is still ~0.42 — hyperparameter axes alone can't
  break it. Move to grid structure (E9).
**Output:** `results/tune_3L/tauC15_8seed/`

## E7 — 3L joint best config × 8 seeds (2026-04-27)
**Question:** do gBm2p0 (best E6 ACC) + tauC15 (best E6 pc) compose into a
clean win on both metrics?
**Setup:** ENSO_GLAYER_B_EXP=-2.0, ENSO_TAU_COARSE=15, 8 seeds.
**Outcome (vs baseline 12-seed):**
- 12-mo ACC: 0.854 vs 0.829 (+0.025, real improvement)
- 12-mo pc:  0.324 vs 0.433 (−0.109, big regression)
- 18-mo ACC: 0.435 vs 0.510 (−0.075)
- 24-mo ACC: 0.372 vs 0.528 (−0.156)
- **Mixed**: weakening cross-scale wiring (gBm2p0) makes 3L behave like 1L —
  gains the index-skill, loses spatial fidelity. Same trade-off as E5.
- Goal "improve both ACC AND pc" not achieved by this joint config.
**Output:** `results/tune_3L/joint_gBm2p0_tauC15/`

## E6 — 3L hyperparameter sensitivity sweep (2026-04-27)
**Question:** which 3L hyperparameters move 12-mo Niño 3.4 ACC and 12-mo
spatial pattern correlation? Goal: improve both for the multi-layer model.
**Setup:** 1D scans around current center (τ_coarse=30, τ_mid=10, τ_fine=3,
ridge_fine=1.0, g_layer_B_exp=-1.0). Four parameters × 2 perturbed levels ×
3 seeds {1,42,99} = 24 runs in `results/tune_3L/`.
**Outcome (per-config 12-mo ACC / spatial pc, vs 3-seed baseline 0.842 / 0.365):**
- **gBm2p0 (g_layer_B exp = −2.0):** ACC=0.857 (+0.015), pc=0.312 (−0.053)  ← best ACC
- **tauC15 (τ_coarse = 15):** ACC=0.845 (+0.003), pc=0.380 (+0.015)  ← best pc
- tauF10 (τ_fine = 10): ACC=0.749 (−0.093) — disaster
- tauF20 (τ_fine = 20): ACC=0.840 neutral, pc=0.186 hurts
- All others within seed noise.
**Lessons:**
- Cross-scale wiring strength is the *dominant* axis. Counter-intuitively,
  *weaker* coupling (gBm2p0) helps; stronger (gB0p0) hurts.
- τ_fine has a sharp sweet spot at 3. The E5 1L lesson (τ=30 best) does NOT
  transfer to 3L because the coarse layer already carries the slow memory.
  The fine layer must stay fast.
- τ_coarse weakly sensitive in [15, 60]; just having it long suffices.
- ridge_fine is well-tuned at 1.0.
**Next:** test joint (gBm2p0 + tauC15) at 8 seeds — see E7.

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
