#!/usr/bin/env bash
# 1D sensitivity sweeps around the current 3L hyperparameter center point.
# Each sweep varies one parameter while holding others at default.
# Output goes to results/tune_3L/<config_tag>/.
#
# Usage:  scripts/sweep_3L.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Center point (= main_enso_monthly.jl defaults): τ_coarse=30 τ_mid=10
# τ_fine=3 ridge_fine=1.0 glayer_B_exp=-1.0. Compare against the 8-seed
# results/enso_monthly_preds_three_layer_seed*.jld2 data already on disk.
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
    if env ENSO_MODE=three_layer ENSO_OUTDIR="$outdir" ENSO_SEED="$seed" "$@" \
        nice -n 19 julia --threads 4 --project=. main_enso_monthly.jl > "$log" 2>&1; then
        local acc12=$(grep -E "^  12 mo" "$log" | head -1 | awk '{print $4}')
        echo "    OK — 12-mo ACC=${acc12}"
    else
        echo "    FAILED — see $log"
    fi
}

# Sweep 1: τ_fine ∈ {10, 20}  (default 3 already covered by existing 3L runs)
for tauF in 10 20; do
    for s in "${SEEDS[@]}"; do
        run_one "tauF${tauF}" "$s" ENSO_TAU_FINE="$tauF"
    done
done

# Sweep 2: τ_coarse ∈ {15, 60}  (default 30 already covered)
for tauC in 15 60; do
    for s in "${SEEDS[@]}"; do
        run_one "tauC${tauC}" "$s" ENSO_TAU_COARSE="$tauC"
    done
done

# Sweep 3: ridge_fine ∈ {0.1, 10}  (default 1.0 already covered)
for rF in 0.1 10; do
    for s in "${SEEDS[@]}"; do
        # File-system-safe tag for fractional values
        rF_tag=$(echo "$rF" | tr '.' 'p')
        run_one "rF${rF_tag}" "$s" ENSO_RIDGE_FINE="$rF"
    done
done

# Sweep 4: glayer_B_exp ∈ {-2.0, 0.0}  (default -1.0 already covered)
for gE in -2.0 0.0; do
    for s in "${SEEDS[@]}"; do
        gE_tag=$(echo "$gE" | tr '.-' 'pm')
        run_one "gB${gE_tag}" "$s" ENSO_GLAYER_B_EXP="$gE"
    done
done

echo "SWEEP_DONE  $(date '+%Y-%m-%d %H:%M:%S')"
