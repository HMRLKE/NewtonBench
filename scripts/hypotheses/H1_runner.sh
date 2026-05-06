#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODELS_FILE="${MODELS_FILE:-configs/models_g4s.txt}"
MODELS_CSV="${MODELS_CSV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODULES="${MODULES:-m0_gravity,m1_coulomb_force,m2_magnetic_force}"
EQUATION_DIFFICULTIES="${EQUATION_DIFFICULTIES:-easy}"
MODEL_SYSTEMS="${MODEL_SYSTEMS:-vanilla_equation}"
LAW_VERSIONS="${LAW_VERSIONS:-v0,v1,v2}"
SCIENTIST_POPULATION="${SCIENTIST_POPULATION:-4}"
NOISE="${NOISE:-0.0}"
MAX_SCIENTIST_TURNS="${MAX_SCIENTIST_TURNS:-8}"
MAX_REVIEWER_TURNS="${MAX_REVIEWER_TURNS:-4}"
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt5mini}"
JUDGE_API_SOURCE="${JUDGE_API_SOURCE:-oa}"
RUN_GROUP_TAG="${RUN_GROUP_TAG:-h1-g4s-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${MODELS_CSV}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
else
  if [[ ! -f "${MODELS_FILE}" ]]; then
    echo "Models file not found: ${MODELS_FILE}" >&2
    exit 1
  fi
  mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "${MODELS_FILE}")
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models resolved for H1 runner." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

AGG_DIR="outputs/hypothesis_runs/${RUN_GROUP_TAG}"
mkdir -p "${AGG_DIR}"
MAP_FILE="${AGG_DIR}/model_run_map.csv"
printf 'model_name,run_tag\n' > "${MAP_FILE}"

declare -a RUN_TAGS=()

echo "=== H1 runner group: ${RUN_GROUP_TAG} ==="
echo "Repo root: ${REPO_ROOT}"
echo "Models: ${MODELS[*]}"

for model in "${MODELS[@]}"; do
  safe_model="$(printf '%s' "${model}" | sed 's/[^A-Za-z0-9._-]/-/g')"
  run_tag="${RUN_GROUP_TAG}--${safe_model}"
  RUN_TAGS+=("${run_tag}")
  printf '%s,%s\n' "${model}" "${run_tag}" >> "${MAP_FILE}"

  cmd=(
    "${PYTHON_BIN}" scripts/hypotheses/run_h1_reviewer_experiments.py
    --run_tag "${run_tag}"
    --scientist_model_name "${model}"
    --scientist_api_source g4s
    --reviewer_model_name "${model}"
    --reviewer_api_source g4s
    --modules "${MODULES}"
    --equation_difficulties "${EQUATION_DIFFICULTIES}"
    --model_systems "${MODEL_SYSTEMS}"
    --law_versions "${LAW_VERSIONS}"
    --scientist_population "${SCIENTIST_POPULATION}"
    --noise "${NOISE}"
    --max_scientist_turns "${MAX_SCIENTIST_TURNS}"
    --max_reviewer_turns "${MAX_REVIEWER_TURNS}"
    --max_review_rounds "${MAX_REVIEW_ROUNDS}"
    --judge_model_name "${JUDGE_MODEL_NAME}"
    --judge_api_source "${JUDGE_API_SOURCE}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry_run)
  fi

  echo
  echo ">>> Running H1 for ${model}"
  printf '>>> Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run completed. No aggregate CSVs were generated."
  exit 0
fi

"${PYTHON_BIN}" - "${AGG_DIR}" "${RUN_GROUP_TAG}" "${RUN_TAGS[@]}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

agg_dir = Path(sys.argv[1])
group_tag = sys.argv[2]
run_tags = sys.argv[3:]
base_dir = Path("outputs/hypothesis_runs")

targets = {
    "h1_summary.csv": "h1_summary_all.csv",
    "scenario_summary.csv": "scenario_summary_all.csv",
    "paper_results.csv": "paper_results_all.csv",
    "paper_rounds.csv": "paper_rounds_all.csv",
}

for src_name, dst_name in targets.items():
    frames = []
    for run_tag in run_tags:
        src = base_dir / run_tag / src_name
        if not src.exists():
            continue
        df = pd.read_csv(src)
        df.insert(0, "source_run_tag", run_tag)
        frames.append(df)
    if not frames:
        continue
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(agg_dir / dst_name, index=False)
    try:
        (agg_dir / dst_name.replace(".csv", ".md")).write_text(combined.to_markdown(index=False), encoding="utf-8")
    except Exception:
        (agg_dir / dst_name.replace(".csv", ".md")).write_text(combined.to_string(index=False), encoding="utf-8")

(agg_dir / "run_group_tag.txt").write_text(group_tag + "\n", encoding="utf-8")
PY

echo
echo "H1 aggregate outputs:"
echo "- ${AGG_DIR}/h1_summary_all.csv"
echo "- ${AGG_DIR}/scenario_summary_all.csv"
echo "- ${AGG_DIR}/paper_results_all.csv"
echo "- ${AGG_DIR}/paper_rounds_all.csv"
