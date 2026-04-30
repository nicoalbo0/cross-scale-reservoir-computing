#!/usr/bin/env bash
# Tier 1 coarse sweeps for Stage E (frequency-band vs timescale comparison).
#
# Each cell = (architecture, parameter setting). Each cell runs SEEDS seeds.
# After all seeds for a cell finish, aggregate_sweep_cell.jl appends one CSV
# row with mean ± std across seeds and a composite score.
#
# Usage:
#   bash scripts/sweep_arch_coarse.sh                # run all (A + B + D)
#   bash scripts/sweep_arch_coarse.sh A              # only architecture A
#   bash scripts/sweep_arch_coarse.sh B              # only B
#   bash scripts/sweep_arch_coarse.sh D              # only D
#
# Env overrides:
#   SEEDS  — space-separated seed list (default "1 42 99")
#   ROOT   — output root (default results/temporal_multiscale_compare/tier1)

set -euo pipefail

WHICH="${1:-all}"
SEEDS="${SEEDS:-1 42 99}"
ROOT="${ROOT:-results/temporal_multiscale_compare/tier1}"
THREADS="${THREADS:-4}"
JL="nice -n 19 env OPENBLAS_NUM_THREADS=1 julia --threads ${THREADS} --project=."

mkdir -p "$ROOT"
LOGDIR="$ROOT/_logs"
mkdir -p "$LOGDIR"

run_cell () {
    local mode="$1"; local cell_id="$2"; shift 2
    local celldir="$ROOT/$mode/$cell_id"
    mkdir -p "$celldir"
    local csv="$ROOT/${mode}_summary.csv"
    echo ""
    echo "### [$mode] cell=$cell_id  $@"
    for seed in $SEEDS; do
        local pred_file="$celldir/enso_temporal_field_preds_${mode}_seed${seed}.jld2"
        if [[ -f "$pred_file" ]]; then
            echo "  seed=$seed  (cached, skipping)"
            continue
        fi
        echo "  seed=$seed ..."
        $JL -e 'include("main_enso_temporal_multiscale_field.jl")' \
            ENSO_TM_MODE="$mode" \
            ENSO_OUTDIR="$celldir" \
            ENSO_SEED="$seed" \
            "$@" \
            >"$LOGDIR/${mode}_${cell_id}_seed${seed}.log" 2>&1 || {
                echo "    !! run failed; see $LOGDIR/${mode}_${cell_id}_seed${seed}.log" >&2
                continue
            }
    done
    $JL scripts/aggregate_sweep_cell.jl "$celldir" "$csv" "$cell_id" \
        >>"$LOGDIR/_aggregate.log" 2>&1 || true
}

# The launcher Julia process expects env vars BEFORE 'julia ...'. Wrap accordingly.
launch () {
    local mode="$1"; local cell_id="$2"; shift 2
    local celldir="$ROOT/$mode/$cell_id"
    mkdir -p "$celldir"
    local csv="$ROOT/${mode}_summary.csv"
    echo ""
    echo "### [$mode] cell=$cell_id  $*"
    for seed in $SEEDS; do
        local pred_file="$celldir/enso_temporal_field_preds_${mode}_seed${seed}.jld2"
        if [[ -f "$pred_file" ]]; then
            echo "  seed=$seed  (cached)"
            continue
        fi
        echo "  seed=$seed ..."
        nice -n 19 env OPENBLAS_NUM_THREADS=1 \
            ENSO_TM_MODE="$mode" \
            ENSO_OUTDIR="$celldir" \
            ENSO_SEED="$seed" \
            "$@" \
            julia --threads "$THREADS" --project=. main_enso_temporal_multiscale_field.jl \
            >"$LOGDIR/${mode}_${cell_id}_seed${seed}.log" 2>&1 || {
                echo "    !! run failed; see $LOGDIR/${mode}_${cell_id}_seed${seed}.log" >&2
                continue
            }
    done
    nice -n 19 julia --project=. scripts/aggregate_sweep_cell.jl \
        "$celldir" "$csv" "$cell_id" >>"$LOGDIR/_aggregate.log" 2>&1 || true
}

# ===========================================================================
# Architecture A — :no_xscale_field
# 1-D probes around baseline (1/24, 1/3) cutoffs / (1e-3, 1e-2, 1.0) ridges /
# (0.85, 0.75, 0.55) ρ. Baseline is reused via on-disk results when possible.
# ===========================================================================

if [[ "$WHICH" == "all" || "$WHICH" == "A" ]]; then
    MODE=no_xscale_field

    # cutoffs probe (4 cells; baseline reused)
    launch $MODE "cuts_36_3"  ENSO_TM_CUTOFFS="1/36,1/3"
    launch $MODE "cuts_24_3"  ENSO_TM_CUTOFFS="1/24,1/3"          # baseline
    launch $MODE "cuts_24_6"  ENSO_TM_CUTOFFS="1/24,1/6"
    launch $MODE "cuts_12_3"  ENSO_TM_CUTOFFS="1/12,1/3"

    # ridge probe (3 cells; baseline reused)
    launch $MODE "ridge_lo"   ENSO_TM_RIDGE_SLOW=1e-4 ENSO_TM_RIDGE_MID=1e-3 ENSO_TM_RIDGE_FAST=1e-1
    launch $MODE "ridge_hi"   ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=1e1

    # ρ probe (2 cells; baseline reused)
    launch $MODE "rho_hi"     ENSO_TM_RHO_SLOW=0.95 ENSO_TM_RHO_MID=0.85 ENSO_TM_RHO_FAST=0.65
fi

# ===========================================================================
# Architecture B — :multi_tau_3_field
# τ-tuple, g_layer, ρ-tuple probes around best-known config from B-mt-3-tune.
# ===========================================================================

if [[ "$WHICH" == "all" || "$WHICH" == "B" ]]; then
    MODE=multi_tau_3_field

    # τ-tuple (4 cells)
    launch $MODE "tau_30_8_2"   ENSO_TM_TAU_SLOW=30 ENSO_TM_TAU_MID=8  ENSO_TM_TAU_FAST=2  ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RHO_MID=0.55 ENSO_TM_RHO_FAST=0.55 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
    launch $MODE "tau_24_6_2"   ENSO_TM_TAU_SLOW=24 ENSO_TM_TAU_MID=6  ENSO_TM_TAU_FAST=2  ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RHO_MID=0.55 ENSO_TM_RHO_FAST=0.55 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
    launch $MODE "tau_40_12_4"  ENSO_TM_TAU_SLOW=40 ENSO_TM_TAU_MID=12 ENSO_TM_TAU_FAST=4  ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RHO_MID=0.55 ENSO_TM_RHO_FAST=0.55 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
    launch $MODE "tau_60_15_4"  ENSO_TM_TAU_SLOW=60 ENSO_TM_TAU_MID=15 ENSO_TM_TAU_FAST=4  ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RHO_MID=0.55 ENSO_TM_RHO_FAST=0.55 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0

    # g_layer probe at baseline τ
    launch $MODE "glayer_-1_-1"  ENSO_TM_GLAYER_MID_EXP=-1.0 ENSO_TM_GLAYER_FAST_EXP=-1.0 ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RIDGE_SLOW=1.0
    launch $MODE "glayer_-2_-2"  ENSO_TM_GLAYER_MID_EXP=-2.0 ENSO_TM_GLAYER_FAST_EXP=-2.0 ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RIDGE_SLOW=1.0
    launch $MODE "glayer_-2_-1"  ENSO_TM_GLAYER_MID_EXP=-2.0 ENSO_TM_GLAYER_FAST_EXP=-1.0 ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RIDGE_SLOW=1.0
    launch $MODE "glayer_-1_-2"  ENSO_TM_GLAYER_MID_EXP=-1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0 ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RIDGE_SLOW=1.0
    launch $MODE "glayer_-3_-2"  ENSO_TM_GLAYER_MID_EXP=-3.0 ENSO_TM_GLAYER_FAST_EXP=-2.0 ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RIDGE_SLOW=1.0

    # ρ-tuple probe at baseline τ + champion g_layer
    launch $MODE "rho_55_55_55"  ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RHO_MID=0.55 ENSO_TM_RHO_FAST=0.55 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
    launch $MODE "rho_65_65_65"  ENSO_TM_RHO_SLOW=0.65 ENSO_TM_RHO_MID=0.65 ENSO_TM_RHO_FAST=0.65 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
    launch $MODE "rho_85_75_55"  ENSO_TM_RHO_SLOW=0.85 ENSO_TM_RHO_MID=0.75 ENSO_TM_RHO_FAST=0.55 ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
fi

# ===========================================================================
# Architecture D — :full_cascade_field with overlap_mode=:include (BUG-FIXED)
# Stripped sweep on g_layer at baseline cutoffs/ρ/ridge.
# ===========================================================================

if [[ "$WHICH" == "all" || "$WHICH" == "D" ]]; then
    MODE=full_cascade_field

    launch $MODE "incl_glayer_-1_-1"  ENSO_FULL_CASCADE_INCLUDE=true ENSO_TM_GLAYER_MID_EXP=-1.0 ENSO_TM_GLAYER_FAST_EXP=-1.0
    launch $MODE "incl_glayer_-2_-2"  ENSO_FULL_CASCADE_INCLUDE=true ENSO_TM_GLAYER_MID_EXP=-2.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
    launch $MODE "incl_glayer_-1_-2"  ENSO_FULL_CASCADE_INCLUDE=true ENSO_TM_GLAYER_MID_EXP=-1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
fi

echo ""
echo "Tier 1 coarse sweep complete. Summaries:"
for f in $ROOT/*_summary.csv; do
    [[ -f "$f" ]] && echo "  $f"
done
