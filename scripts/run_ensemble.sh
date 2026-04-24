#!/usr/bin/env bash
# Run main_enso.jl sequentially with different RNG seeds. Outputs land in
# results/enso_preds_three_layer_seed<N>.jld2 for later aggregation.
# Runs serially (not parallel) to keep peak memory bounded.

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(${ENSO_SEEDS:-42 1 2 3 7 11 23 99})
LOGDIR=/tmp
echo "Running ensemble over seeds: ${SEEDS[@]}"

for seed in "${SEEDS[@]}"; do
    log="${LOGDIR}/enso_seed${seed}.log"
    echo "=== seed=${seed}  log=${log}  start=$(date -Iseconds) ==="
    ENSO_SEED="${seed}" nice -n 19 julia --threads 4 --project=. main_enso.jl \
        > "${log}" 2>&1 || {
            echo "seed=${seed} FAILED — see ${log}" >&2
            tail -20 "${log}" >&2
            continue
        }
    # Quick summary line
    grep -a "ACC            =" "${log}" | tr '\r' '\n' | tail -1 || true
    grep -a "Lead-time ACCs" "${log}" | tr '\r' '\n' | tail -1 || true
    echo
done

echo "=== Ensemble complete $(date -Iseconds) ==="
ls -1 results/enso_preds_three_layer_seed*.jld2 2>/dev/null | sort
