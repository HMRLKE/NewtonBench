#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-outputs/hypothesis_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hypothesis_runs/h3-log-analysis-$(date +%Y%m%d-%H%M%S)}"

echo "=== H3 observed KB-error propagation analysis ==="
echo "Input root: ${INPUT_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

"${PYTHON_BIN}" result_analysis/analyze_kb_error_propagation.py \
  --input_root "${INPUT_ROOT}" \
  --output_dir "${OUTPUT_DIR}"
