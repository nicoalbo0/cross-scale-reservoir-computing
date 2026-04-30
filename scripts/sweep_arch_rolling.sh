#!/usr/bin/env bash
# Tier 3 rolling-window robustness for Stage E.
# Runs each champion config across 4 windows × 4 seeds.
#
# Champions are passed as semicolon-separated env-var assignments. Example:
#   A_CHAMP="ENSO_TM_RIDGE_SLOW=1e-2;ENSO_TM_RIDGE_MID=1e-1;ENSO_TM_RIDGE_FAST=1e1"
#   B_CHAMP="ENSO_TM_RHO_SLOW=0.55;ENSO_TM_RIDGE_SLOW=1.0;ENSO_TM_GLAYER_FAST_EXP=-2.0"
#   D_CHAMP=""   # empty = skip
#
# Windows (months in monthly cube starting 1982-01):
#   W1: train_start=1   (1982-01..2005-12 train, 2006-01..2013-12 verif)  [288 train / 96 predict]
#   W2: train_start=37  (1985-01..2008-12, 2009-01..2015-12)
#   W3: train_start=1   (1982-01..2001-12, 2002-01..2009-12) [240 train]
#   W4: train_start=1   (1982-01..2009-12, 2010-01..2015-12) [336 train]

set -euo pipefail

A_CHAMP="${A_CHAMP:-}"
B_CHAMP="${B_CHAMP:-}"
D_CHAMP="${D_CHAMP:-}"
SEEDS="${SEEDS:-1 23 42 99}"
ROOT="${ROOT:-results/temporal_multiscale_compare/tier3_rolling}"
THREADS="${THREADS:-4}"

mkdir -p "$ROOT"
LOGDIR="$ROOT/_logs"; mkdir -p "$LOGDIR"

# Window definitions: id, train_start, train_len, predict_len
# All windows must satisfy: train_start + train_len + predict_len + warmup(12) - 1 ≤ 408
# (the SST archive ends 2015-12; 408 monthly samples from 1982-01).
WINDOWS=(
    "W1 1 288 96"
    "W2 37 288 72"
    "W3 1 240 96"
    "W4 1 336 60"
)

run_champ () {
    local mode="$1"; local champ_label="$2"; local champ_kv="$3"
    [[ -z "$champ_kv" ]] && { echo "skip $champ_label (empty config)"; return; }
    local csv="$ROOT/${mode}_${champ_label}_summary.csv"
    for win in "${WINDOWS[@]}"; do
        read -r wid wstart wtrain wpred <<<"$win"
        local celldir="$ROOT/${mode}_${champ_label}/${wid}"
        mkdir -p "$celldir"
        echo ""
        echo "### [$mode/$champ_label] $wid (start=$wstart train=$wtrain predict=$wpred)"
        for seed in $SEEDS; do
            local pred_file="$celldir/enso_temporal_field_preds_${mode}_seed${seed}.jld2"
            if [[ -f "$pred_file" ]]; then
                echo "  seed=$seed (cached)"; continue
            fi
            echo "  seed=$seed ..."
            # Inject the champion's env vars
            IFS=';' read -ra champ_vars <<<"$champ_kv"
            nice -n 19 env OPENBLAS_NUM_THREADS=1 \
                ENSO_TM_MODE="$mode" \
                ENSO_OUTDIR="$celldir" \
                ENSO_SEED="$seed" \
                ENSO_TRAIN_START="$wstart" \
                ENSO_TRAIN_LEN="$wtrain" \
                ENSO_PREDICT_LEN="$wpred" \
                ENSO_WINDOW_LABEL="$wid" \
                "${champ_vars[@]}" \
                julia --threads "$THREADS" --project=. main_enso_temporal_multiscale_field.jl \
                >"$LOGDIR/${mode}_${champ_label}_${wid}_seed${seed}.log" 2>&1 || {
                    echo "    !! run failed; see log" >&2; continue; }
        done
        nice -n 19 julia --project=. scripts/aggregate_sweep_cell.jl \
            "$celldir" "$csv" "$wid" >>"$LOGDIR/_aggregate.log" 2>&1 || true
    done
}

run_champ "no_xscale_field"    "champ" "$A_CHAMP"
run_champ "multi_tau_3_field"  "champ" "$B_CHAMP"
[[ -n "$D_CHAMP" ]] && run_champ "full_cascade_field" "champ" "$D_CHAMP"

echo ""
echo "Tier 3 rolling-window sweep complete. Summaries:"
ls $ROOT/*_summary.csv 2>/dev/null || true
