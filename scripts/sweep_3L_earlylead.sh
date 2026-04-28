#!/usr/bin/env bash
# E12 — Optimize for early-lead skill: reduce phase lag and initial amplitude
# offset. Sweeps τ_coarse, τ_fine, and warmup, all at the gAm0p5_rF3 champion
# defaults.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(1 42 99)

run_one() {
    local cfg_tag="$1"
    local seed="$2"
    shift 2
    local logdir="/tmp/enso_tune"
    mkdir -p "$logdir"
    local log="$logdir/${cfg_tag}_seed${seed}.log"
    local outdir="results/tune_3L/${cfg_tag}"
    echo "[$(date '+%H:%M:%S')] cfg=${cfg_tag} seed=${seed}"
    if env ENSO_MODE=three_layer ENSO_OUTDIR="$outdir" ENSO_SEED="$seed" \
        ENSO_GLAYER_A_EXP=-0.5 ENSO_RIDGE_FINE=3.0 \
        "$@" \
        nice -n 19 julia --threads 4 --project=. main_enso_monthly.jl > "$log" 2>&1; then
        echo "    OK"
    else
        echo "    FAILED — see $log"
    fi
}

# τ_coarse phase: try smaller values for faster coarse-layer response
for tC in 5 7 10; do
    for s in "${SEEDS[@]}"; do
        run_one "early_tauC${tC}" "$s" ENSO_TAU_COARSE="$tC"
    done
done

# τ_fine: explore below sweet spot at 3
for tF in 1 2; do
    for s in "${SEEDS[@]}"; do
        run_one "early_tauF${tF}" "$s" ENSO_TAU_COARSE=15 ENSO_TAU_FINE="$tF"
    done
done

# warmup: longer state warm-up before closed loop
for w in 18 24 36; do
    for s in "${SEEDS[@]}"; do
        run_one "early_warmup${w}" "$s" ENSO_TAU_COARSE=15 ENSO_WARMUP="$w"
    done
done

echo "EARLY_SWEEP_DONE  $(date '+%Y-%m-%d %H:%M:%S')"
