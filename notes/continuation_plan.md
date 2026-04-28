# ENSO Continuation Plan

Session-handoff document for resuming ENSO forecasting work on the
`cross-scale-reservoir-computing` package (branch `dev-giulio`).

To pick this up in a fresh session: read this file plus the top
"Settled defaults" block of `notes/experiments.md`, plus the most recent
experiment entry. That gives full context.

---

## Where we are

### Architecture and code state
- Three-layer pixel-monthly 3L pipeline in `main_enso_monthly.jl`, fully
  parameterized via env vars.
- All hyperparameters explored across E1–E12 (1D scans + 2D joint grid +
  Pareto analysis): τ_coarse/mid/fine, ridge per layer, g_layer per layer,
  N per layer, spectral radius per layer, spatial grids, regression mode,
  overlap mode.
- E14 added paper-faithful mode (linear readout, `:exclude` overlap, N=1250,
  aligned grids) and made it the **default** so fresh runs reproduce
  arXiv:2510.11209.

### Two state-of-the-art configs

| Config | When to use | 12-mo ACC | 12-mo RMSE | spatial pc | std_ratio |
|---|---|---|---|---|---|
| **Paper-faithful** (default) | reproduce paper | 0.859 | 0.521 | 0.086 | 0.90 |
| **E12 champion** | best on our metrics | 0.847 | 0.406 | 0.449 | 1.15 |

E12 champion wins on every meaningful metric except basin-mean phase
coherence and very-long-lead per-pixel RMSE. Run the champion via:

```bash
ENSO_REGRESSION=quadratic ENSO_OVERLAP_MODE=include \
  ENSO_N_COARSE=500 ENSO_N_MID=500 ENSO_N_FINE=500 \
  ENSO_GRID_2LAT=4 \
  ENSO_TAU_COARSE=15 ENSO_GLAYER_A_EXP=-0.5 \
  ENSO_RIDGE_FINE=3.0 ENSO_TAU_FINE=2 \
  julia --threads 4 --project=. main_enso_monthly.jl
```

### Open structural ceiling
- **+4 month phase lag** in every 3L config. Cannot be tuned away —
  reducing τ_coarse, varying warmup, all confirmed in E12 don't move it.
- Likely a fundamental property of training a reservoir on ~5 ENSO cycles
  to forecast a system with that periodicity.
- Breaking it requires a *structural* change, not parameter tuning.

### Idea on the table — temporal cross-scale RC
- The package implements *spatial* cross-scale (coarse→fine in space).
  ENSO is **spatially coherent / temporally nonlinear**, so spatial
  cross-scale is the wrong axis. Temporal multi-scale would decompose the
  signal into bands (slow / mid / fast) and feed slow→mid→fast through the
  same `run_multi_layer` wiring along the time axis.
- E13 (logged in `notes/experiments.md`) started Stage A of this on the
  scalar Niño 3.4 series:
  - A.1 canary (un-decomposed 1D reservoir): 0.45 ACC, highly seed-variant.
  - A.2 (per-band independent reservoirs, no cross-scale wiring): 0.853
    ACC, near-zero seed variance — but std_ratio collapsed to 0.36
    (amplitude crushed by ~3×).
  - A.3 (slow→mid cross-scale wiring): WORSE than A.2 (0.829 ACC). Slow
    band's noise-grade prediction poisoned the mid via cross-scale input.
- Memory note: `project_temporal_multiscale_idea.md`.

---

## Three concrete next directions

### Direction 1 — Continue E13 temporal multi-scale, fix the slow band
**What:** the A.3 negative result was caused by the slow band's reservoir
being mis-tuned (own-skill = 0.046 = noise). Re-tune slow band with
canary-style hyperparams (ρ=0.95, τ=1, no leak), then retry the cascade.
Also try the dyadic 7-band variant (Stage C) which gives finer scale
separation. Then move to per-pixel SST (Stage B).

**Cost:** medium. ~1-2 days of tuning + sweeps.

**Reward:** if the A.2 amplitude collapse can be fixed (e.g., via
amplitude-corrected reconstruction or scale-specific normalization), this
becomes a genuine new architecture validated on ENSO.

### Direction 2 — Break the +4 mo phase lag in spatial 3L
**Two specific levers, neither tested yet:**
- **Rolling forecast** (operational mode): re-train and re-warmup at every
  test step instead of one autonomous 96-month forecast. This is how real
  ENSO operational systems work and would dramatically improve early-lead
  skill. Requires a new evaluation script.
- **Lagged-input features** (Takens embedding): explicitly feed `u(t),
  u(t-3), u(t-6), u(t-12)` to each block's reservoir. Gives phase
  information directly instead of inferring it.

**Cost:** medium. Each lever is ~1 day of code + sweeps.

**Reward:** if either fixes the +4 mo lag, current pc+ACC numbers improve
across the board. If neither, we've ruled out two plausible structural fixes.

### Direction 3 — Write up
Current results are publishable:
- Spatial cross-scale validated on KS (paper).
- Partial transfer to ENSO documented (paper-faithful is competitive on
  basin ACC; quadratic+include better on spatial fidelity).
- New tuned baseline with E12 hyperparameters (1.5pp ACC, 15% RMSE
  reduction over original).
- Temporal multi-scale proposed as orthogonal extension; preliminary E13
  results show decomposition matters but cascade wiring needs work.

**Cost:** focused writing time, no new compute.

**Reward:** completed paper.

---

## Recommended sequence

1. **Direction 1 first** (continue E13). The temporal multi-scale idea is
   the most novel architectural contribution. Stage A.4 (proper slow-band
   tuning) and Stage C (dyadic) are cheap and likely to break the A.3
   negative result.
2. **Direction 2 in parallel or after** — rolling forecast is operationally
   meaningful and a clean test of one structural fix. Easy to bolt onto
   existing pipeline.
3. **Direction 3 (write-up)** once 1 or 2 produces a clean positive result,
   or when you decide to ship what we have.

---

## Quick reference

- **Repo:** `/home/giulio/Documents/research/climate/cross-scale-reservoir-computing`
- **Branch:** `dev-giulio` (currently 17 commits ahead of origin)
- **Paper:** arXiv:2510.11209 (PDF cached locally; full text in `/tmp/paper.txt`
  if `pdftotext` was already run)
- **Latest commit:** `acde467` "feat(enso-3L): paper-faithful mode"
- **Experiments log:** `notes/experiments.md`
- **Memory:** `~/.claude/projects/-home-giulio-Documents-research-climate-cross-scale-reservoir-computing/memory/`
- **Champion artifacts:** `results/tune_3L/champion_tauF2_8seed/`
- **Paper-faithful artifacts:** `results/tune_3L/paperfaithful_8seed/`

## Standing preferences (from memory)
- Cap CPU: `nice -n 19 julia --threads 4` for long runs.
- Suppress GKS popups: `ENV["GKSwstype"] = "100"` before `using Plots`.
- Always report RMSE and std_ratio alongside ACC. ACC alone hides
  amplitude-collapsed forecasts.
