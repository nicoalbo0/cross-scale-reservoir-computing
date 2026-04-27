#!/usr/bin/env bash
# E10 — Reservoir capacity (N) and spectral radius (rad) sweeps at tauC15 default.
# 4 axes × 2 levels × 3 seeds = 24 runs.
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
        ENSO_TAU_COARSE=15 \
        "$@" \
        nice -n 19 julia --threads 4 --project=. main_enso_monthly.jl > "$log" 2>&1; then
        local acc12=$(grep -E "^  12 mo" "$log" | head -1 | awk -F'ACC=' '{print $2}' | awk '{print $1}')
        echo "    OK — 12-mo ACC=${acc12}"
    else
        echo "    FAILED — see $log"
    fi
}

# N_coarse ∈ {300, 800}
for nC in 300 800; do
    for s in "${SEEDS[@]}"; do
        run_one "Ncoarse${nC}" "$s" ENSO_N_COARSE="$nC"
    done
done

# N_fine ∈ {300, 800}
for nF in 300 800; do
    for s in "${SEEDS[@]}"; do
        run_one "Nfine${nF}" "$s" ENSO_N_FINE="$nF"
    done
done

# rad_coarse ∈ {0.7, 0.95}
for rC in 0.7 0.95; do
    for s in "${SEEDS[@]}"; do
        rC_tag=$(echo "$rC" | tr '.' 'p')
        run_one "radCoarse${rC_tag}" "$s" ENSO_RAD_COARSE="$rC"
    done
done

# rad_fine ∈ {0.4, 0.7}
for rF in 0.4 0.7; do
    for s in "${SEEDS[@]}"; do
        rF_tag=$(echo "$rF" | tr '.' 'p')
        run_one "radFine${rF_tag}" "$s" ENSO_RAD_FINE="$rF"
    done
done

echo "CAPACITY_SWEEP_DONE  $(date '+%Y-%m-%d %H:%M:%S')"
