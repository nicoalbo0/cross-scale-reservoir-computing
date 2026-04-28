# ENSO Experiments Log

A running record of every ENSO experiment we've tried. Goal: never repeat a
mistake, never re-run a settled question. Newest experiments at the top.

Conventions:
- One row per discrete experiment.
- "Outcome" is what we learned, not just numbers.
- Output dirs are namespaced under `results/<tag>/`.

---

## ⚠ Correction (2026-04-27): always check RMSE + std_ratio alongside ACC

The E5 result "1L τ=30 wins on Niño 3.4 ACC at 0.891" is a metric artifact.
1L τ=30 produces a forecast with only 18% of the truth's amplitude
(std_ratio = 0.18, range ~[−0.3, 0.3] vs truth ~[−2.0, 1.1]). High ACC
because the phase is right; low actual fidelity because magnitude is crushed.

**By RMSE (actual error in physical units), 3L tauC15 dominates:**

| Architecture | 12-mo ACC | 12-mo RMSE | std_ratio |
|---|---|---|---|
| 3L tauC15    | 0.831     | **0.469**  | 1.35      |
| 3L baseline  | 0.828     | 0.480      | 1.37      |
| 1L τ=30      | 0.891     | 0.613      | 0.18 ⚠   |

The honest claim: **3L is the better ENSO forecaster on every meaningful
metric**. 1L τ=30 just exploits ACC's amplitude-invariance.

Mechanism: τ=30 in a *single* reservoir over-damps the closed-loop dynamics
(α=1/30). The autonomous attractor decays toward zero; phase is preserved
but magnitude isn't. 3L avoids this because the fine layer (τ=3) keeps the
dynamics "alive" while the coarse layer carries the slow envelope.

**Lesson:** when comparing forecast configurations, always report RMSE and
std_ratio. ACC alone hides the damped-prediction failure mode.

---

## Settled defaults (state of the art for monthly 3L pixel) — UPDATED E12

After the E12 early-lead sweep, the best 3L configuration is:

```
mode = three_layer
τ_coarse=15, τ_mid=10, τ_fine=2                    ← τ_fine: was 3
ridge_coarse=1e-3, ridge_mid=1e-2, ridge_fine=3.0
g_layer_A_exp=-0.5, g_layer_B_exp=-1.0
N_coarse=N_mid=N_fine=500
rad_coarse=0.85, rad_mid=0.75, rad_fine=0.55
grids: (3,1) → (9,3) → (9,4)
mixing=2
regression=quadratic
warmup=12
```

→ **12-mo Niño 3.4 ACC = 0.847** (was 0.836 at gAm0p5_rF3; baseline 0.828)
→ **12-mo Niño 3.4 RMSE = 0.406** (was 0.455; baseline 0.480 — 15.4% lower)
→ **12-mo spatial pc = 0.449** (was 0.450 — essentially equal; baseline 0.415)
→ **std_ratio = 1.15** (close to perfect 1.0)

Run as:
`ENSO_TAU_COARSE=15 ENSO_GLAYER_A_EXP=-0.5 ENSO_RIDGE_FINE=3.0 ENSO_TAU_FINE=2`

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

## E13 — Temporal cross-scale RC on Niño 3.4 (Stage A: 1D scalar) (2026-04-27)
**Question:** does temporal-band decomposition + cross-scale wiring on the
1D Niño 3.4 series outperform a single un-decomposed 1D reservoir? Tests
the architectural prior cleanly without spatial confound. Plan: Stage A (1D)
→ Stage B (per-pixel SST) → Stage C (dyadic 7-band scalar) → Stage D
(dyadic per-pixel).

### A.0 — Bandpass utility + unit tests
**Setup:** New `src/data/temporal_decomposition.jl` using DSP.jl Butterworth
+ filtfilt. **Critical design decision:** bands constructed via cumulative
lowpass differences (slow=LP(f1), mid=LP(f2)-LP(f1), fast=signal-LP(f2)),
which gives EXACT reconstruction `signal == sum(bands)` (linearity of
filtfilt). The textbook LP+BP+HP design does NOT reconstruct exactly because
Butterworth's gentle rolloff at cutoffs leaves a residual.
**Outcome:** 24/24 unit tests pass — reconstruction <1e-6, energy isolation
>95%, zero phase ±1 sample. Filter is solid.

### A.1 — Canary: un-decomposed 1D reservoir (mode=single_reservoir)
**Question:** baseline ACC for a 1D reservoir on the scalar N3.4 series.
**Setup:** After E5 leaky regime (ρ=0.55, ridge=1.0) collapsed std_ratio to
0.04, switched to classic 1D ESN regime: N=1000, ρ=0.95, τ=1 (no leak),
ridge=1e-7. 4 seeds {1,7,42,99}.
**Outcome:**
- 12-mo cum ACC = 0.45 ± 0.30 across seeds (highly seed-variant).
- Persistent NEGATIVE 3-mo ACC (~−0.5) across all seeds — closed-loop
  transient artifact (autonomous attractor takes ~6 months to lock onto
  the trained dynamics).
- Long-lead diverges; std_ratio explodes (up to 11×) at 18+ months.
- Below E5 spatial 1L (0.891 ACC) — expected, since the spatial reservoir
  has 2880 dim/timestep vs our 1 dim per timestep.
- Establishes canary baseline for A.2/A.3/A.4 to beat.

### A.2 — Per-band INDEPENDENT reservoirs (mode=no_xscale)
**Question:** does decomposition alone (without cross-scale) help?
**Setup:** Cutoffs (1/24, 1/3) cyc/mo, Butterworth order 4. Slow τ=30 ρ=0.85
ridge=1e-3, mid τ=8 ρ=0.75 ridge=1e-2, fast τ=2 ρ=0.55 ridge=1.0; all N=500.
4 seeds.
**Outcome:**
- **12-mo cum ACC = 0.853 ± 0.002** — essentially zero seed variance,
  hallmark of a converged closed-loop attractor. Beats canary (0.45).
- 6-mo: 0.824, 9-mo: 0.852, 12-mo: 0.853, 18-mo: 0.84, 24-mo: 0.55.
- **std_ratio = 0.36 at 12-mo** ⚠ — amplitude collapsed by ~3×, the same
  failure mode as E5's 1L τ=30 (per memory `feedback_acc_vs_rmse.md`).
  Phase-correct but under-magnitude: would underestimate El Niño strengths.
- Per-band own-skill (full 96-mo window): mid ACC=0.53, slow ACC≈0, fast
  ACC≈0.09. **The MID reservoir carries the prediction**, NOT the slow band.
- Variance fractions: slow=74%, mid=18%, fast=0.3% of total Niño 3.4 variance.
  But over short test windows the slow band is approximately constant (its
  period exceeds the window), so MID dominates the predictability there.
- Long-window full ACC negative because slow's bad prediction tanks 24+ mo.
**Lessons:**
- Decomposition is doing real work — splitting the problem into easier
  sub-problems (smooth slow, oscillating mid, noise fast) and the mid band
  is genuinely predictable.
- The slow reservoir is poorly tuned for narrow-band slow signals; needs
  separate optimization (try canary regime ρ=0.95, τ=1 for slow band too?).
- Amplitude-collapse is a structural feature of summing damped predictions.

**Output:** `results/temporal_multiscale/{single_reservoir,no_xscale}/`

### A.3 — Two-band cross-scale (slow → mid)  [pending]
### A.4 — Full 3-band cascade (slow → mid → fast)  [pending]

## E12 — 3L early-lead optimization sweep (2026-04-28)
**Question:** the user observed in the n34_champion plot that the gAm0p5_rF3
forecast was visually "stretched right" (phase lag ≈ 4 months) and the post-
12-month error was unacceptable. Can we improve early-lead RMSE by tuning
τ_coarse, τ_fine, or warmup?
**Setup:** at gAm0p5_rF3 baseline, sweep:
- τ_coarse ∈ {5, 7, 10}  (untested below 15)
- τ_fine  ∈ {1, 2}        (untested below 3 — E6 only tested {3, 10, 20})
- warmup  ∈ {18, 24, 36}  (default 12)
3 seeds × 8 configs = 24 runs.
Then 8-seed confirmation of best candidates.
**Outcome — major win on RMSE:**
- τ_fine=2 (8 seeds): ACC=0.847, RMSE=0.406, pc=0.449, std_ratio=1.15
  - **15.4% RMSE reduction over E1 baseline** (0.480 → 0.406)
  - 10.8% RMSE reduction over E11 champion (0.455 → 0.406)
  - pc essentially unchanged (0.450 → 0.449)
- τ_fine=1 (8 seeds): RMSE 0.381 (even lower) but pc drops to 0.409
- τ_coarse < 15: all worse on RMSE; phase lag goes to +5 (worse).
- warmup > 12: all worse; warmup=36 catastrophic (RMSE 0.594).
**Key insight:** the E6 conclusion "τ_fine sweet spot at 3" was wrong because
we never tested below 3. True sweet spot is at 2. With τ_fine=2 (α=0.5), the
fine layer has minimal memory — letting the coarse-mid layers carry the slow
envelope while the fine layer applies spatial detail. τ_fine=1 (memoryless)
goes too far and pc drops; τ_fine=3 has unnecessary memory and adds RMSE.
**Phase lag of +4 months is structural** — same in all 3L configs we tested.
Reducing τ_coarse below 15 doesn't help; warmup variation doesn't help.
Likely a fundamental property of training a reservoir on ~5 ENSO cycles to
forecast a system with that periodicity.
**Adopted as new 3L default.**
**Output:** `results/tune_3L/champion_tauF2_8seed/`

## E11 — 3L joint multi-axis sweep + Pareto analysis (2026-04-27/28)
**Question:** can simultaneous improvement on all three metrics (ACC, RMSE,
spatial pc) be found in the joint hyperparameter space, beyond what 1D
scans showed?
**Setup:**
- Phase 1: 2D grid (g_layer_B_exp × ridge_fine), 14 cells × 3 seeds.
- Phase 2: 1D explorations of g_layer_A_exp (3 levels) and τ_mid (2 levels),
  each × 3 seeds.
- Phase 3: confirm Pareto candidates at 8 seeds.
- Aggregator computes ACC, RMSE, std_ratio, 12-mo spatial pc per config and
  finds Pareto frontier on (RMSE↓, pc↑).
**Outcome:** **clean win on all three metrics**.

8-seed Pareto-optimal config: g_layer_A_exp=-0.5, ridge_fine=3.0, tauC15
(`gAm0p5_rF3_8seed`):
- ACC=0.836 (+0.005 vs tauC15 baseline)
- RMSE=0.455 (-3% vs 0.469)
- spatial pc=0.450 (+6% vs 0.424) — first 8-seed config above 0.45
- std_ratio=1.03 (was 1.35) — nearly perfect amplitude match

Architectural insight: the optimal 3L cross-scale wiring is **asymmetric**.
- A-side (18°→6°): **stronger** than default (g_layer_A_exp=-0.5 vs -1.0)
- B-side (6°→2°): default (g_layer_B_exp=-1.0)
- Asymmetric A-strong + B-weak (gAm0p5_gBm2) reverts to the same trade-off
  as gBm2p0 alone (RMSE OK, pc collapses to 0.324). The strong-A win works
  only when paired with default-B.

ridge_fine=3.0 prevents amplitude over-shoot — std_ratio drops from 1.35 to
1.03 (matching truth's amplitude almost exactly).

**Adopted as new 3L default.**
**Output:** `results/tune_3L/gAm0p5_rF3_8seed/`, `results/tune_3L_pareto.png`.

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
