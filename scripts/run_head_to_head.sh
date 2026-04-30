#!/usr/bin/env bash
# Tier 4 — 8-seed head-to-head ensemble at champion config on W1.
#
# Reuses cached results when present.
#
# Usage env:
#   A_CHAMP / B_CHAMP / D_CHAMP : semicolon-separated env-var assignments
#   SEEDS                       : default "1 2 3 7 11 23 42 99"

set -euo pipefail

A_CHAMP="${A_CHAMP:-}"
B_CHAMP="${B_CHAMP:-}"
D_CHAMP="${D_CHAMP:-}"
SEEDS="${SEEDS:-1 2 3 7 11 23 42 99}"
ROOT="${ROOT:-results/temporal_multiscale_compare/tier4_head_to_head}"
THREADS="${THREADS:-4}"

mkdir -p "$ROOT"
LOGDIR="$ROOT/_logs"; mkdir -p "$LOGDIR"

run_champ () {
    local mode="$1"; local champ_kv="$2"
    [[ -z "$champ_kv" ]] && { echo "skip $mode (empty)"; return; }
    local celldir="$ROOT/${mode}_W1"
    mkdir -p "$celldir"
    local csv="$ROOT/${mode}_summary.csv"
    echo ""
    echo "### [$mode] W1 head-to-head (8 seeds)"
    for seed in $SEEDS; do
        local pred_file="$celldir/enso_temporal_field_preds_${mode}_seed${seed}.jld2"
        if [[ -f "$pred_file" ]]; then
            echo "  seed=$seed (cached)"; continue
        fi
        echo "  seed=$seed ..."
        IFS=';' read -ra champ_vars <<<"$champ_kv"
        nice -n 19 env OPENBLAS_NUM_THREADS=1 \
            ENSO_TM_MODE="$mode" \
            ENSO_OUTDIR="$celldir" \
            ENSO_SEED="$seed" \
            ENSO_TRAIN_START=1 \
            ENSO_TRAIN_LEN=288 \
            ENSO_PREDICT_LEN=96 \
            ENSO_WINDOW_LABEL="W1" \
            "${champ_vars[@]}" \
            julia --threads "$THREADS" --project=. main_enso_temporal_multiscale_field.jl \
            >"$LOGDIR/${mode}_seed${seed}.log" 2>&1 || {
                echo "    !! run failed" >&2; continue; }
    done
    nice -n 19 julia --project=. scripts/aggregate_sweep_cell.jl \
        "$celldir" "$csv" "W1" >>"$LOGDIR/_aggregate.log" 2>&1 || true
}

run_champ "no_xscale_field"   "$A_CHAMP"
run_champ "multi_tau_3_field" "$B_CHAMP"
[[ -n "$D_CHAMP" ]] && run_champ "full_cascade_field" "$D_CHAMP"

echo ""
echo "Tier 4 ensembles done. Run paired stat test:"
echo "  julia --project=. scripts/stat_test_head_to_head.jl \\"
echo "    $ROOT/no_xscale_field_W1 $ROOT/multi_tau_3_field_W1${D_CHAMP:+ \\\\\\n    $ROOT/full_cascade_field_W1}"
