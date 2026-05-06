#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODELS_FILE="${MODELS_FILE:-configs/models_g4s.txt}"
MODELS_CSV="${MODELS_CSV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_API_SOURCE="${DEFAULT_API_SOURCE:-g4s}"
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
RUN_GROUP_TAG="${RUN_GROUP_TAG:-h3-g4s-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${MODELS_CSV}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
else
  mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "${MODELS_FILE}")
fi

AGG_DIR="outputs/hypothesis_runs/${RUN_GROUP_TAG}"
mkdir -p "${AGG_DIR}"
printf 'model_name,api_source,run_tag\n' > "${AGG_DIR}/model_run_map.csv"

declare -a RUN_TAGS=()

echo "=== H3 runner group: ${RUN_GROUP_TAG} ==="
for model_spec in "${MODELS[@]}"; do
  if [[ "${model_spec}" == *"@"* ]]; then
    model="${model_spec%@*}"
    api_source="${model_spec##*@}"
  else
    model="${model_spec}"
    api_source="${DEFAULT_API_SOURCE}"
  fi
  safe_model="$(printf '%s-%s' "${api_source}" "${model}" | sed 's/[^A-Za-z0-9._-]/-/g')"
  run_tag="${RUN_GROUP_TAG}--${safe_model}"
  RUN_TAGS+=("${run_tag}")
  printf '%s,%s,%s\n' "${model}" "${api_source}" "${run_tag}" >> "${AGG_DIR}/model_run_map.csv"

  cmd=(
    "${PYTHON_BIN}" scripts/hypotheses/run_h3_poisoned_kb.py
    --run_tag "${run_tag}"
    --scientist_model_name "${model}"
    --scientist_api_source "${api_source}"
    --reviewer_model_name "${model}"
    --reviewer_api_source "${api_source}"
    --modules "${MODULES}"
    --equation_difficulties "${EQUATION_DIFFICULTIES}"
    --model_systems "${MODEL_SYSTEMS}"
    --law_versions "${LAW_VERSIONS}"
    --scientist_population "${SCIENTIST_POPULATION}"
    --noise "${NOISE}"
    --max_scientist_turns "${MAX_SCIENTIST_TURNS}"
    --max_reviewer_turns "${MAX_REVIEWER_TURNS}"
    --max_review_rounds "${MAX_REVIEW_ROUNDS}"
    --poison_rate "${POISON_RATE}"
    --poison_edit_distance "${POISON_EDIT_DISTANCE}"
    --poison_operations "${POISON_OPERATIONS}"
    --poison_seed "${POISON_SEED}"
    --judge_model_name "${JUDGE_MODEL_NAME}"
    --judge_api_source "${JUDGE_API_SOURCE}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry_run)
  fi
  if [[ "${REVIEWER_CAN_RUN_EXPERIMENTS}" == "1" ]]; then
    cmd+=(--reviewer_can_run_experiments)
  fi

  echo
  echo ">>> Running H3 for ${model} via ${api_source}"
  printf '>>> Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${PYTHON_BIN}" - "${AGG_DIR}" "${RUN_TAGS[@]}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

agg_dir = Path(sys.argv[1])
run_tags = sys.argv[2:]
base_dir = Path("outputs/hypothesis_runs")
targets = {
    "h3_summary.csv": "h3_summary_all.csv",
    "scenario_summary.csv": "scenario_summary_all.csv",
    "h3_paper_results.csv": "h3_paper_results_all.csv",
    "poison_manifest.csv": "poison_manifest_all.csv",
    "paper_rounds.csv": "paper_rounds_all.csv",
}

for src_name, dst_name in targets.items():
    frames = []
    for run_tag in run_tags:
        src = base_dir / run_tag / src_name
        if src.exists():
            df = pd.read_csv(src)
            df.insert(0, "source_run_tag", run_tag)
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(agg_dir / dst_name, index=False)
        try:
            (agg_dir / dst_name.replace(".csv", ".md")).write_text(combined.to_markdown(index=False), encoding="utf-8")
        except Exception:
            pass
PY

echo
echo "H3 aggregate outputs:"
echo "- ${AGG_DIR}/h3_summary_all.csv"
echo "- ${AGG_DIR}/h3_paper_results_all.csv"
