#!/usr/bin/env bash
set -euo pipefail

# Source checkout entry point. The scorer ships with the Python package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../src/wrbench/eval/score_runtime_v2_d3d6.sh" "$@"
