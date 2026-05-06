#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-all-hypotheses-g4s-$(date +%Y%m%d-%H%M%S)}"

echo "=== Compatibility wrapper: running H1-H4 for G4S models ==="
echo "For mixed providers, call scripts/hypotheses/run_all_hypotheses_all_models.sh directly."

DEFAULT_API_SOURCE="${DEFAULT_API_SOURCE:-g4s}" \
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX}" \
MODELS_FILE="${MODELS_FILE:-configs/models_g4s.txt}" \
MODELS_CSV="${MODELS_CSV:-}" \
HYPOTHESES="${HYPOTHESES:-H1,H2,H3,H4}" \
MAX_PARALLEL_RUNS="${MAX_PARALLEL_RUNS:-4}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
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
POISON_RATE="${POISON_RATE:-0.1}" \
POISON_EDIT_DISTANCE="${POISON_EDIT_DISTANCE:-1}" \
POISON_OPERATIONS="${POISON_OPERATIONS:-distance_exponent,drop_factor,operator_flip,add_term}" \
POISON_SEED="${POISON_SEED:-42}" \
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt5mini}" \
JUDGE_API_SOURCE="${JUDGE_API_SOURCE:-oa}" \
REVIEWER_CAN_RUN_EXPERIMENTS="${REVIEWER_CAN_RUN_EXPERIMENTS:-0}" \
DRY_RUN="${DRY_RUN:-0}" \
bash "${SCRIPT_DIR}/run_all_hypotheses_all_models.sh"
