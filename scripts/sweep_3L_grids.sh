#!/usr/bin/env bash
# E9 — Spatial block grid sensitivity, at the E8 tauC15 default.
# 4 configs × 3 seeds = 12 runs. Output to results/tune_3L/<cfg>/.
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

# Cfg 1: g18_lat = 2  (double 18° lat resolution)
for s in "${SEEDS[@]}"; do
    run_one "grid_18lat2" "$s" ENSO_GRID_18LAT=2
done

# Cfg 2: g18_lat = 3  (triple 18° lat)
for s in "${SEEDS[@]}"; do
    run_one "grid_18lat3" "$s" ENSO_GRID_18LAT=3
done

# Cfg 3: g2_lat = 6  (refine fine layer)
for s in "${SEEDS[@]}"; do
    run_one "grid_2lat6" "$s" ENSO_GRID_2LAT=6
done

# Cfg 4: joint  18lat=2 + 2lat=6
for s in "${SEEDS[@]}"; do
    run_one "grid_18lat2_2lat6" "$s" ENSO_GRID_18LAT=2 ENSO_GRID_2LAT=6
done

echo "GRID_SWEEP_DONE  $(date '+%Y-%m-%d %H:%M:%S')"
