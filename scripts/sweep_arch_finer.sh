#!/usr/bin/env bash
# Tier 2 finer follow-ups around Tier 1 champions.
#
# Champions (from Tier 1):
#   A: ridge_hi (1e-2, 1e-1, 1e1) at default cutoffs (1/24, 1/3) → composite 0.768
#   A 2nd: cuts_36_3 at default ridge → composite 0.734
#   B: tau_40_12_4 + glayer_fast=-2 + ρ=0.55 → composite 0.731
#   D: incl_glayer_-2_-2 + default ridge → composite 0.547 (amplitude-inflated)

set -euo pipefail

WHICH="${1:-all}"
SEEDS="${SEEDS:-1 42 99}"
ROOT="${ROOT:-results/temporal_multiscale_compare/tier2_finer}"
THREADS="${THREADS:-4}"

mkdir -p "$ROOT"
LOGDIR="$ROOT/_logs"; mkdir -p "$LOGDIR"

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
            echo "  seed=$seed (cached)"; continue
        fi
        echo "  seed=$seed ..."
        nice -n 19 env OPENBLAS_NUM_THREADS=1 \
            ENSO_TM_MODE="$mode" \
            ENSO_OUTDIR="$celldir" \
            ENSO_SEED="$seed" \
            "$@" \
            julia --threads "$THREADS" --project=. main_enso_temporal_multiscale_field.jl \
            >"$LOGDIR/${mode}_${cell_id}_seed${seed}.log" 2>&1 || {
                echo "    !! run failed" >&2; continue; }
    done
    nice -n 19 julia --project=. scripts/aggregate_sweep_cell.jl \
        "$celldir" "$csv" "$cell_id" >>"$LOGDIR/_aggregate.log" 2>&1 || true
}

# ===========================================================================
# A — ridge_hi micro-sweep + cross with best-cuts
# ===========================================================================

if [[ "$WHICH" == "all" || "$WHICH" == "A" ]]; then
    MODE=no_xscale_field

    # ridge_fast micro at fixed ridge_slow=1e-2, ridge_mid=1e-1 (cuts default)
    launch $MODE "ridge_fast_3"   ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=3.0
    launch $MODE "ridge_fast_30"  ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=30.0
    launch $MODE "ridge_fast_100" ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=100.0

    # ridge_slow micro at fixed others
    launch $MODE "ridge_slow_5e3" ENSO_TM_RIDGE_SLOW=5e-3 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0
    launch $MODE "ridge_slow_3e2" ENSO_TM_RIDGE_SLOW=3e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0

    # cross with cuts_36_3 (lower slow cutoff) + ridge_hi
    launch $MODE "cuts36_ridgehi"  ENSO_TM_CUTOFFS="1/36,1/3"  ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0

    # cross with cuts_24_6 (higher fast cutoff, best pc3) + ridge_hi
    launch $MODE "cuts246_ridgehi" ENSO_TM_CUTOFFS="1/24,1/6"  ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0
fi

# ===========================================================================
# B — τ-tuple ±1 around (40, 12, 4); g_layer fast at -2 fixed
# ===========================================================================

if [[ "$WHICH" == "all" || "$WHICH" == "B" ]]; then
    MODE=multi_tau_3_field
    BASE=( ENSO_TM_RHO_SLOW=0.55 ENSO_TM_RHO_MID=0.55 ENSO_TM_RHO_FAST=0.55
           ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0 )

    launch $MODE "tau_35_10_3"  "${BASE[@]}" ENSO_TM_TAU_SLOW=35 ENSO_TM_TAU_MID=10 ENSO_TM_TAU_FAST=3
    launch $MODE "tau_45_14_5"  "${BASE[@]}" ENSO_TM_TAU_SLOW=45 ENSO_TM_TAU_MID=14 ENSO_TM_TAU_FAST=5
    launch $MODE "tau_40_10_4"  "${BASE[@]}" ENSO_TM_TAU_SLOW=40 ENSO_TM_TAU_MID=10 ENSO_TM_TAU_FAST=4
    launch $MODE "tau_40_12_3"  "${BASE[@]}" ENSO_TM_TAU_SLOW=40 ENSO_TM_TAU_MID=12 ENSO_TM_TAU_FAST=3
    launch $MODE "tau_50_14_4"  "${BASE[@]}" ENSO_TM_TAU_SLOW=50 ENSO_TM_TAU_MID=14 ENSO_TM_TAU_FAST=4

    # τ_40_12_4 + slightly higher ρ (might help amplitude further)
    launch $MODE "tau_40_12_4_rho65" \
        ENSO_TM_TAU_SLOW=40 ENSO_TM_TAU_MID=12 ENSO_TM_TAU_FAST=4 \
        ENSO_TM_RHO_SLOW=0.65 ENSO_TM_RHO_MID=0.6 ENSO_TM_RHO_FAST=0.55 \
        ENSO_TM_RIDGE_SLOW=1.0 ENSO_TM_GLAYER_FAST_EXP=-2.0
fi

# ===========================================================================
# D — apply A-style ridge taming (slow ridge=1e-2 + mid=1e-1 + fast=10) to
# the bug-fixed full_cascade_field. If amplitude tames, D can compete.
# ===========================================================================

if [[ "$WHICH" == "all" || "$WHICH" == "D" ]]; then
    MODE=full_cascade_field
    BASE=( ENSO_FULL_CASCADE_INCLUDE=true ENSO_TM_GLAYER_MID_EXP=-2.0 ENSO_TM_GLAYER_FAST_EXP=-2.0 )

    launch $MODE "incl_glayer22_ridgehi" \
        "${BASE[@]}" ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0
    launch $MODE "incl_glayer22_ridgehi_rho85" \
        "${BASE[@]}" ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0 \
        ENSO_TM_RHO_SLOW=0.85
    launch $MODE "incl_glayer11_ridgehi" \
        ENSO_FULL_CASCADE_INCLUDE=true ENSO_TM_GLAYER_MID_EXP=-1.0 ENSO_TM_GLAYER_FAST_EXP=-1.0 \
        ENSO_TM_RIDGE_SLOW=1e-2 ENSO_TM_RIDGE_MID=1e-1 ENSO_TM_RIDGE_FAST=10.0
fi

echo ""
echo "Tier 2 finer sweep complete. Summaries:"
ls $ROOT/*_summary.csv 2>/dev/null || true
