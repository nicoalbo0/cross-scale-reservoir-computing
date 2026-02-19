#!/usr/bin/env bash
# Run tests with code coverage, print coverage summary to terminal, then delete .cov files.
# Usage: from project root, run:
#   ./scripts/run_tests_coverage.sh
# or:
#   bash scripts/run_tests_coverage.sh

set -e
cd "$(dirname "$0")/.."
julia --project=. -e 'using Pkg; Pkg.test("CrossScaleRC"; coverage=true)'
