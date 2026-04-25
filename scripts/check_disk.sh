#!/usr/bin/env bash
# Quick disk audit: where the project is using space, and what's safe to prune.
# Usage:  scripts/check_disk.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Top-level project usage ==="
du -sh data results 2>/dev/null | sort -h
echo
echo "=== data/sst breakdown ==="
du -sh data/sst/sst_final_*.jld2 data/sst/sst_clean_*.jld2 2>/dev/null | sort -h
raw_size=$(du -sh data/sst/19?? data/sst/20?? 2>/dev/null | tail -1 | awk '{print $1}')
raw_total=$(du -shc data/sst/19?? data/sst/20?? 2>/dev/null | tail -1 | awk '{print $1}')
echo "  raw daily NetCDF directories total: ${raw_total} (1982-2015 inclusive)"
echo "  → these were the inputs to the regrid step. If sst_final_*.jld2 are"
echo "    intact, the raw NetCDFs are recoverable (re-download via cdsapi)."
echo
echo "=== results/ breakdown ==="
du -sh results/*.png results/*.jld2 2>/dev/null | sort -h | tail -20
echo
echo "=== /tmp logs from this project ==="
du -sh /tmp/enso*.log /tmp/ensemble*.log 2>/dev/null | sort -h | tail -10
echo
echo "=== filesystem free space ==="
df -h "$PWD" | head -2
echo
echo "Cleanup hints (none of these run automatically):"
echo "  rm /tmp/enso*.log /tmp/ensemble*.log         # safe, all reproducible"
echo "  rm results/enso_*_seed*.png                   # per-seed plots, regenerable"
echo "  rm results/enso_*_preds_*.jld2                # per-seed JLD2s, regenerable"
echo "  rm -rf data/sst/19?? data/sst/20??            # raw NetCDFs, ~145 GB,"
echo "                                                #   but you'd need to re-download to re-regrid."
