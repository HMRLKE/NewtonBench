#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-all-hypotheses-g4s-$(date +%Y%m%d-%H%M%S)}"

echo "=== Running all G4S hypothesis suites with prefix: ${RUN_GROUP_PREFIX} ==="

RUN_GROUP_TAG="${RUN_GROUP_PREFIX}-h1" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
MODELS_FILE="${MODELS_FILE:-configs/models_g4s.txt}" \
MODELS_CSV="${MODELS_CSV:-}" \
MODULES="${MODULES:-m0_gravity,m1_coulomb_force,m2_magnetic_force}" \
EQUATION_DIFFICULTIES="${EQUATION_DIFFICULTIES:-easy}" \
MODEL_SYSTEMS="${MODEL_SYSTEMS:-vanilla_equation}" \
LAW_VERSIONS="${LAW_VERSIONS:-v0,v1,v2}" \
SCIENTIST_POPULATION="${SCIENTIST_POPULATION:-4}" \
NOISE="${NOISE:-0.0}" \
MAX_SCIENTIST_TURNS="${MAX_SCIENTIST_TURNS:-8}" \
MAX_REVIEWER_TURNS="${MAX_REVIEWER_TURNS:-4}" \
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}" \
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt5mini}" \
JUDGE_API_SOURCE="${JUDGE_API_SOURCE:-oa}" \
DRY_RUN="${DRY_RUN:-0}" \
bash "${SCRIPT_DIR}/H1_runner.sh"

RUN_GROUP_TAG="${RUN_GROUP_PREFIX}-h2" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
MODELS_FILE="${MODELS_FILE:-configs/models_g4s.txt}" \
MODELS_CSV="${MODELS_CSV:-}" \
OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-gpt5mini}" \
MODULES="${MODULES:-m0_gravity,m1_coulomb_force,m2_magnetic_force}" \
EQUATION_DIFFICULTIES="${EQUATION_DIFFICULTIES:-easy}" \
MODEL_SYSTEMS="${MODEL_SYSTEMS:-vanilla_equation}" \
LAW_VERSIONS="${LAW_VERSIONS:-v0,v1,v2}" \
SCIENTIST_POPULATION="${SCIENTIST_POPULATION:-4}" \
NOISE="${NOISE:-0.0}" \
MAX_SCIENTIST_TURNS="${MAX_SCIENTIST_TURNS:-8}" \
MAX_REVIEWER_TURNS="${MAX_REVIEWER_TURNS:-4}" \
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}" \
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt5mini}" \
JUDGE_API_SOURCE="${JUDGE_API_SOURCE:-oa}" \
REVIEWER_CAN_RUN_EXPERIMENTS="${REVIEWER_CAN_RUN_EXPERIMENTS:-0}" \
DRY_RUN="${DRY_RUN:-0}" \
bash "${SCRIPT_DIR}/H2_runner.sh"

echo
echo "All hypothesis runs finished."
echo "- H1 aggregate directory: outputs/hypothesis_runs/${RUN_GROUP_PREFIX}-h1"
echo "- H2 aggregate directory: outputs/hypothesis_runs/${RUN_GROUP_PREFIX}-h2"
