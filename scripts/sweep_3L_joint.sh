#!/usr/bin/env bash
# E11 — Joint multi-axis sweep, all on top of tauC15 default.
# Phase 1: 2D grid (g_layer_B_exp × ridge_fine), 14 new cells.
# Phase 2: 1D for g_layer_A_exp (3 levels) and τ_mid (2 levels).
# 3 seeds per cell.
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

tag_num() { echo "$1" | tr '.-' 'pm'; }

# Phase 1: 2D grid (g_layer_B_exp × ridge_fine), excluding cells we already
# have data for (gBm1p0+rF1p0 = current default, gBm2p0+rF1p0 = joint E7).
declare -a P1_CELLS=(
    # g_B=-2.0
    "-2.0 0.3"
    "-2.0 3.0"
    "-2.0 10"
    # g_B=-1.5  (all new)
    "-1.5 0.3"
    "-1.5 1.0"
    "-1.5 3.0"
    "-1.5 10"
    # g_B=-1.0  (current g, varying rF)
    "-1.0 0.3"
    "-1.0 3.0"
    "-1.0 10"
    # g_B=-0.5  (all new)
    "-0.5 0.3"
    "-0.5 1.0"
    "-0.5 3.0"
    "-0.5 10"
)

for cell in "${P1_CELLS[@]}"; do
    read gB rF <<< "$cell"
    cfg="gB$(tag_num $gB)_rF$(tag_num $rF)"
    for s in "${SEEDS[@]}"; do
        run_one "$cfg" "$s" ENSO_GLAYER_B_EXP="$gB" ENSO_RIDGE_FINE="$rF"
    done
done

# Phase 2a: g_layer_A_exp ∈ {-2.0, -0.5, 0.0}  (default is -1.0, untested)
for gA in -2.0 -0.5 0.0; do
    cfg="gA$(tag_num $gA)"
    for s in "${SEEDS[@]}"; do
        run_one "$cfg" "$s" ENSO_GLAYER_A_EXP="$gA"
    done
done

# Phase 2b: τ_mid ∈ {5, 20}  (default is 10)
for tM in 5 20; do
    cfg="tauM${tM}"
    for s in "${SEEDS[@]}"; do
        run_one "$cfg" "$s" ENSO_TAU_MID="$tM"
    done
done

echo "JOINT_SWEEP_DONE  $(date '+%Y-%m-%d %H:%M:%S')"
