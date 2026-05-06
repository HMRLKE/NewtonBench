#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS_FILE="${MODELS_FILE:-configs/models_g4s.txt}"
MODELS_CSV="${MODELS_CSV:-}"
OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-gpt5mini}"
DEFAULT_API_SOURCE="${DEFAULT_API_SOURCE:-g4s}"
HYPOTHESES="${HYPOTHESES:-H1,H2,H3,H4}"
MAX_PARALLEL_RUNS="${MAX_PARALLEL_RUNS:-4}"
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-all-hypotheses-all-models-$(date +%Y%m%d-%H%M%S)}"
MODULES="${MODULES:-m0_gravity,m1_coulomb_force,m2_magnetic_force}"
EQUATION_DIFFICULTIES="${EQUATION_DIFFICULTIES:-easy}"
MODEL_SYSTEMS="${MODEL_SYSTEMS:-vanilla_equation}"
LAW_VERSIONS="${LAW_VERSIONS:-v0,v1,v2}"
SCIENTIST_POPULATION="${SCIENTIST_POPULATION:-4}"
NOISE="${NOISE:-0.0}"
MAX_SCIENTIST_TURNS="${MAX_SCIENTIST_TURNS:-8}"
MAX_REVIEWER_TURNS="${MAX_REVIEWER_TURNS:-4}"
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}"
POISON_RATE="${POISON_RATE:-0.1}"
POISON_EDIT_DISTANCE="${POISON_EDIT_DISTANCE:-1}"
POISON_OPERATIONS="${POISON_OPERATIONS:-distance_exponent,drop_factor,operator_flip,add_term}"
POISON_SEED="${POISON_SEED:-42}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt5mini}"
JUDGE_API_SOURCE="${JUDGE_API_SOURCE:-oa}"
REVIEWER_CAN_RUN_EXPERIMENTS="${REVIEWER_CAN_RUN_EXPERIMENTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${MODELS_CSV}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
else
  mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "${MODELS_FILE}")
fi
IFS=',' read -r -a HYP_ARRAY <<< "${HYPOTHESES}"

AGG_DIR="outputs/hypothesis_runs/${RUN_GROUP_PREFIX}"
mkdir -p "${AGG_DIR}"
printf 'hypothesis,model_name,api_source,run_tag\n' > "${AGG_DIR}/job_map.csv"

wait_for_slot() {
  while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL_RUNS}" ]]; do
    sleep 5
  done
}

run_job() {
  local hypothesis="$1"
  local model="$2"
  local safe_model="$3"
  local api_source="$4"
  local run_tag="${RUN_GROUP_PREFIX}-${hypothesis,,}--${safe_model}"
  local dry_run_args=()
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_run_args=(--dry_run)
  fi
  local reviewer_experiment_args=()
  if [[ "${REVIEWER_CAN_RUN_EXPERIMENTS}" == "1" ]]; then
    reviewer_experiment_args=(--reviewer_can_run_experiments)
  fi
  printf '%s,%s,%s,%s\n' "${hypothesis}" "${model}" "${api_source}" "${run_tag}" >> "${AGG_DIR}/job_map.csv"

  case "${hypothesis}" in
    H1)
      "${PYTHON_BIN}" scripts/hypotheses/run_h1_reviewer_experiments.py \
        --run_tag "${run_tag}" \
        --scientist_model_name "${model}" --scientist_api_source "${api_source}" \
        --reviewer_model_name "${model}" --reviewer_api_source "${api_source}" \
        --modules "${MODULES}" --equation_difficulties "${EQUATION_DIFFICULTIES}" \
        --model_systems "${MODEL_SYSTEMS}" --law_versions "${LAW_VERSIONS}" \
        --scientist_population "${SCIENTIST_POPULATION}" --noise "${NOISE}" \
        --max_scientist_turns "${MAX_SCIENTIST_TURNS}" --max_reviewer_turns "${MAX_REVIEWER_TURNS}" \
        --max_review_rounds "${MAX_REVIEW_ROUNDS}" --judge_model_name "${JUDGE_MODEL_NAME}" \
        --judge_api_source "${JUDGE_API_SOURCE}" "${dry_run_args[@]}"
      ;;
    H2)
      "${PYTHON_BIN}" scripts/hypotheses/run_h2_cross_provider_review.py \
        --run_tag "${run_tag}" \
        --openai_model_name "${OPENAI_MODEL_NAME}" --open_model_name "${model}" \
        --open_api_source "${api_source}" \
        --modules "${MODULES}" --equation_difficulties "${EQUATION_DIFFICULTIES}" \
        --model_systems "${MODEL_SYSTEMS}" --law_versions "${LAW_VERSIONS}" \
        --scientist_population "${SCIENTIST_POPULATION}" --noise "${NOISE}" \
        --max_scientist_turns "${MAX_SCIENTIST_TURNS}" --max_reviewer_turns "${MAX_REVIEWER_TURNS}" \
        --max_review_rounds "${MAX_REVIEW_ROUNDS}" --judge_model_name "${JUDGE_MODEL_NAME}" \
        --judge_api_source "${JUDGE_API_SOURCE}" "${reviewer_experiment_args[@]}" "${dry_run_args[@]}"
      ;;
    H3)
      "${PYTHON_BIN}" scripts/hypotheses/run_h3_poisoned_kb.py \
        --run_tag "${run_tag}" \
        --scientist_model_name "${model}" --scientist_api_source "${api_source}" \
        --reviewer_model_name "${model}" --reviewer_api_source "${api_source}" \
        --modules "${MODULES}" --equation_difficulties "${EQUATION_DIFFICULTIES}" \
        --model_systems "${MODEL_SYSTEMS}" --law_versions "${LAW_VERSIONS}" \
        --scientist_population "${SCIENTIST_POPULATION}" --noise "${NOISE}" \
        --max_scientist_turns "${MAX_SCIENTIST_TURNS}" --max_reviewer_turns "${MAX_REVIEWER_TURNS}" \
        --max_review_rounds "${MAX_REVIEW_ROUNDS}" --poison_rate "${POISON_RATE}" \
        --poison_edit_distance "${POISON_EDIT_DISTANCE}" --poison_operations "${POISON_OPERATIONS}" \
        --poison_seed "${POISON_SEED}" --judge_model_name "${JUDGE_MODEL_NAME}" \
        --judge_api_source "${JUDGE_API_SOURCE}" "${reviewer_experiment_args[@]}" "${dry_run_args[@]}"
      ;;
    H4)
      "${PYTHON_BIN}" scripts/hypotheses/run_h4_thinking_mode.py \
        --run_tag "${run_tag}" \
        --scientist_model_name "${model}" --scientist_api_source "${api_source}" \
        --reviewer_model_name "${model}" --reviewer_api_source "${api_source}" \
        --modules "${MODULES}" --equation_difficulties "${EQUATION_DIFFICULTIES}" \
        --model_systems "${MODEL_SYSTEMS}" --law_versions "${LAW_VERSIONS}" \
        --scientist_population "${SCIENTIST_POPULATION}" --noise "${NOISE}" \
        --max_scientist_turns "${MAX_SCIENTIST_TURNS}" --max_reviewer_turns "${MAX_REVIEWER_TURNS}" \
        --max_review_rounds "${MAX_REVIEW_ROUNDS}" --judge_model_name "${JUDGE_MODEL_NAME}" \
        --judge_api_source "${JUDGE_API_SOURCE}" "${reviewer_experiment_args[@]}" "${dry_run_args[@]}"
      ;;
    *)
      echo "Unknown hypothesis: ${hypothesis}" >&2
      return 1
      ;;
  esac
}

echo "=== All-hypotheses runner: ${RUN_GROUP_PREFIX} ==="
echo "Models: ${MODELS[*]}"
echo "Hypotheses: ${HYP_ARRAY[*]}"
echo "Max parallel runs: ${MAX_PARALLEL_RUNS}"

for model_spec in "${MODELS[@]}"; do
  if [[ "${model_spec}" == *"@"* ]]; then
    model="${model_spec%@*}"
    api_source="${model_spec##*@}"
  else
    model="${model_spec}"
    api_source="${DEFAULT_API_SOURCE}"
  fi
  safe_model="$(printf '%s-%s' "${api_source}" "${model}" | sed 's/[^A-Za-z0-9._-]/-/g')"
  for hypothesis_raw in "${HYP_ARRAY[@]}"; do
    hypothesis="$(printf '%s' "${hypothesis_raw}" | tr -d '[:space:]')"
    if [[ -z "${hypothesis}" ]]; then
      continue
    fi
    wait_for_slot
    echo ">>> Dispatching ${hypothesis} for ${model} via ${api_source}"
    run_job "${hypothesis}" "${model}" "${safe_model}" "${api_source}" &
  done
done

failed=0
for job in $(jobs -pr); do
  wait "${job}" || failed=1
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - "${AGG_DIR}" "${RUN_GROUP_PREFIX}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

agg_dir = Path(sys.argv[1])
prefix = sys.argv[2]
base_dir = Path("outputs/hypothesis_runs")
targets = [
    "h1_summary.csv", "h2_summary.csv", "h3_summary.csv", "h4_summary.csv",
    "scenario_summary.csv", "paper_results.csv", "h3_paper_results.csv", "paper_rounds.csv",
]
for target in targets:
    frames = []
    for src in base_dir.glob(f"{prefix}-*--*/{target}"):
        df = pd.read_csv(src)
        df.insert(0, "source_run_tag", src.parent.name)
        frames.append(df)
    if frames:
        out = agg_dir / target.replace(".csv", "_all.csv")
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
PY

  "${PYTHON_BIN}" result_analysis/analyze_kb_error_propagation.py \
    --input_root "outputs/hypothesis_runs" \
    --output_dir "${AGG_DIR}/h3_observed_log_analysis"
fi

echo "Aggregate directory: ${AGG_DIR}"
exit "${failed}"
