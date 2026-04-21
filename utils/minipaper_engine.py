import csv
import importlib
import json
import os
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils.call_llm_api import call_llm_api, normalize_api_source
from utils.minipaper_protocol import (
    MiniPaper,
    ReviewDecision,
    REVIEWER_SYSTEM_PROMPT_NO_EXPERIMENTS,
    REVIEWER_SYSTEM_PROMPT_WITH_EXPERIMENTS,
    SCIENTIST_SYSTEM_PROMPT,
    build_reviewer_prompt,
    build_scientist_prompt,
    load_minipaper_kb,
    parse_minipaper,
    parse_review_decision,
    save_minipaper_kb,
)
from utils.vanilla_agent import parse_experiment_request


@dataclass
class AgentSpec:
    model_name: str
    api_source: str

    def normalized(self) -> "AgentSpec":
        return AgentSpec(
            model_name=self.model_name,
            api_source=normalize_api_source(self.api_source) or self.api_source,
        )


@dataclass
class ScenarioSpec:
    scenario_id: str
    scientist: AgentSpec
    reviewer: AgentSpec
    reviewer_can_run_experiments: bool
    hypothesis_name: str
    reviewer_relation: str
    description: str

    def normalized(self) -> "ScenarioSpec":
        return ScenarioSpec(
            scenario_id=self.scenario_id,
            scientist=self.scientist.normalized(),
            reviewer=self.reviewer.normalized(),
            reviewer_can_run_experiments=self.reviewer_can_run_experiments,
            hypothesis_name=self.hypothesis_name,
            reviewer_relation=self.reviewer_relation,
            description=self.description,
        )


@dataclass
class TaskSpec:
    module_name: str
    equation_difficulty: str
    model_system: str
    law_version: str


@dataclass
class RunSuiteConfig:
    run_tag: str
    hypothesis_name: str
    scenarios: List[ScenarioSpec]
    tasks: List[TaskSpec]
    scientist_population: int = 1
    noise_level: float = 0.0
    max_scientist_turns: int = 8
    max_reviewer_turns: int = 4
    output_root: str = "outputs/hypothesis_runs"
    judge_model_name: Optional[str] = None
    judge_api_source: Optional[str] = None
    dry_run: bool = False


def _error_dict(error: Exception) -> Dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def sanitize_run_tag(run_tag: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in run_tag.strip())
    return safe or datetime.now().strftime("run-%Y%m%d-%H%M%S")


def discover_modules(modules_root: Path) -> List[str]:
    return sorted([p.name for p in modules_root.iterdir() if p.is_dir() and p.name.startswith("m")])


def expand_task_specs(
    *,
    repo_root: Path,
    modules: Optional[Sequence[str]],
    equation_difficulties: Sequence[str],
    model_systems: Sequence[str],
    law_versions: Optional[Sequence[str]],
) -> List[TaskSpec]:
    module_names = list(modules) if modules else discover_modules(repo_root / "modules")
    tasks: List[TaskSpec] = []

    for module_name in module_names:
        module = importlib.import_module(f"modules.{module_name}")
        for difficulty in equation_difficulties:
            available_versions = module.get_available_law_versions(difficulty)
            selected_versions = list(law_versions) if law_versions else [v for v in available_versions if v != "v_unchanged"]
            for version in selected_versions:
                if version not in available_versions:
                    continue
                if version == "v_unchanged":
                    continue
                for model_system in model_systems:
                    tasks.append(
                        TaskSpec(
                            module_name=module_name,
                            equation_difficulty=difficulty,
                            model_system=model_system,
                            law_version=version,
                        )
                    )
    return tasks


def _format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for idx, message in enumerate(chat_history):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"--- Round {idx + 1} ({role}) ---\n{content}\n")
    return "\n".join(lines)


def _scientific_string_if_needed(result: Any, model_system: str) -> Any:
    if model_system == "vanilla_equation" and isinstance(result, (float, int)):
        return "{:.15e}".format(result)
    return result


def _run_experiments_for_requests(
    *,
    module: Any,
    experiments_to_run: List[Dict[str, Any]],
    noise_level: float,
    difficulty: str,
    model_system: str,
    law_version: str,
) -> List[Any]:
    results: List[Any] = []
    for exp in experiments_to_run:
        result = module.run_experiment_for_module(
            **exp,
            noise_level=noise_level,
            difficulty=difficulty,
            system=model_system,
            law_version=law_version,
            consistency=True,
        )
        results.append(_scientific_string_if_needed(result, model_system))
    return results


def _append_experiment_results(messages: List[Dict[str, str]], experiment_results: List[Any]) -> None:
    output_str = f"<experiment_output>\n{json.dumps(experiment_results)}\n</experiment_output>"
    messages.append({"role": "user", "content": output_str})


def _call_agent(messages: List[Dict[str, str]], agent: AgentSpec, trial_info: Dict[str, Any]) -> Tuple[str, int]:
    response_text, reasoning_content, tokens = call_llm_api(
        messages,
        model_name=agent.model_name,
        trial_info={**trial_info, "api_source_override": agent.api_source},
    )
    response_text = response_text or ""
    if reasoning_content and str(reasoning_content).strip():
        content = f"**Reasoning Process:**\n{reasoning_content}\n\n**Main Response:**\n{response_text}"
    else:
        content = response_text
    messages.append({"role": "assistant", "content": content})
    return response_text, int(tokens or 0)


def choose_judge(
    *,
    scientist: AgentSpec,
    explicit_judge_model_name: Optional[str],
    explicit_judge_api_source: Optional[str],
) -> Tuple[str, str]:
    if explicit_judge_model_name:
        return explicit_judge_model_name, normalize_api_source(explicit_judge_api_source) or scientist.api_source

    if os.getenv("OPENAI_API_KEY"):
        return "gpt41", "oa"

    return scientist.model_name, scientist.api_source


def run_scientist_session(
    *,
    module: Any,
    agent: AgentSpec,
    task: TaskSpec,
    knowledge_base: Dict[str, Any],
    noise_level: float,
    max_turns: int,
    artifact_dir: Path,
    episode_id: str,
) -> Dict[str, Any]:
    system_prompt = SCIENTIST_SYSTEM_PROMPT
    user_prompt = build_scientist_prompt(
        module=module,
        system=task.model_system,
        noise_level=noise_level,
        knowledge_base=knowledge_base,
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    total_tokens = 0
    experiments_run = 0

    trial_info = {
        "trial_id": f"{episode_id}-scientist",
        "trial_dir": str(artifact_dir),
    }

    for turn in range(max_turns):
        response_text, tokens = _call_agent(messages, agent, trial_info)
        total_tokens += tokens

        minipaper = parse_minipaper(response_text)
        if minipaper:
            return {
                "status": "completed",
                "minipaper": minipaper,
                "rounds": turn + 1,
                "total_tokens": total_tokens,
                "num_experiments": experiments_run,
                "chat_history": messages,
            }

        experiments_to_run = parse_experiment_request(response_text)
        if experiments_to_run:
            experiment_results = _run_experiments_for_requests(
                module=module,
                experiments_to_run=experiments_to_run,
                noise_level=noise_level,
                difficulty=task.equation_difficulty,
                model_system=task.model_system,
                law_version=task.law_version,
            )
            experiments_run += len(experiments_to_run)
            _append_experiment_results(messages, experiment_results)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid response. Use <run_experiment> to gather data or submit a complete <mini_paper>.",
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "You have exhausted the available turns. Submit your final <mini_paper> now.",
        }
    )
    response_text, tokens = _call_agent(messages, agent, trial_info)
    total_tokens += tokens
    minipaper = parse_minipaper(response_text)
    return {
        "status": "max_turns_reached",
        "minipaper": minipaper,
        "rounds": max_turns,
        "total_tokens": total_tokens,
        "num_experiments": experiments_run,
        "chat_history": messages,
    }


def run_reviewer_session(
    *,
    module: Any,
    agent: AgentSpec,
    task: TaskSpec,
    knowledge_base: Dict[str, Any],
    scientist_paper: MiniPaper,
    reviewer_can_run_experiments: bool,
    noise_level: float,
    max_turns: int,
    artifact_dir: Path,
    episode_id: str,
) -> Dict[str, Any]:
    system_prompt = (
        REVIEWER_SYSTEM_PROMPT_WITH_EXPERIMENTS
        if reviewer_can_run_experiments
        else REVIEWER_SYSTEM_PROMPT_NO_EXPERIMENTS
    )
    user_prompt = build_reviewer_prompt(
        module=module,
        system=task.model_system,
        noise_level=noise_level,
        knowledge_base=knowledge_base,
        scientist_paper=scientist_paper,
        reviewer_can_experiment=reviewer_can_run_experiments,
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    total_tokens = 0
    experiments_run = 0

    trial_info = {
        "trial_id": f"{episode_id}-reviewer",
        "trial_dir": str(artifact_dir),
    }

    for turn in range(max_turns):
        response_text, tokens = _call_agent(messages, agent, trial_info)
        total_tokens += tokens

        review = parse_review_decision(response_text)
        if review:
            return {
                "status": "completed",
                "review": review,
                "rounds": turn + 1,
                "total_tokens": total_tokens,
                "num_experiments": experiments_run,
                "chat_history": messages,
            }

        experiments_to_run = parse_experiment_request(response_text)
        if experiments_to_run and reviewer_can_run_experiments:
            experiment_results = _run_experiments_for_requests(
                module=module,
                experiments_to_run=experiments_to_run,
                noise_level=noise_level,
                difficulty=task.equation_difficulty,
                model_system=task.model_system,
                law_version=task.law_version,
            )
            experiments_run += len(experiments_to_run)
            _append_experiment_results(messages, experiment_results)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "Return a valid <review_decision> JSON block. "
                    + ("You may also use <run_experiment> first." if reviewer_can_run_experiments else "You may not run experiments in this configuration."),
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "You have exhausted the available turns. Return your final <review_decision> now.",
        }
    )
    response_text, tokens = _call_agent(messages, agent, trial_info)
    total_tokens += tokens
    review = parse_review_decision(response_text)
    return {
        "status": "max_turns_reached",
        "review": review,
        "rounds": max_turns,
        "total_tokens": total_tokens,
        "num_experiments": experiments_run,
        "chat_history": messages,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scenario_output_dir(run_dir: Path, scenario: ScenarioSpec) -> Path:
    return run_dir / "scenarios" / scenario.scenario_id


def _episode_artifact_dir(run_dir: Path, scenario: ScenarioSpec, task: TaskSpec, episode_index: int) -> Path:
    task_id = f"{task.module_name}__{task.equation_difficulty}__{task.model_system}__{task.law_version}"
    return _scenario_output_dir(run_dir, scenario) / "artifacts" / task_id / f"episode_{episode_index:03d}"


def _append_to_kb(
    *,
    knowledge_base: Dict[str, Any],
    scenario: ScenarioSpec,
    task: TaskSpec,
    paper: MiniPaper,
    review: ReviewDecision,
    evaluation: Dict[str, Any],
) -> None:
    bucket = "accepted_papers" if review.is_accept() else "rejected_papers"
    knowledge_base.setdefault(bucket, []).append(
        {
            "scenario_id": scenario.scenario_id,
            "module": task.module_name,
            "equation_difficulty": task.equation_difficulty,
            "model_system": task.model_system,
            "law_version": task.law_version,
            "scientist_model_name": scenario.scientist.model_name,
            "scientist_api_source": scenario.scientist.api_source,
            "reviewer_model_name": scenario.reviewer.model_name,
            "reviewer_api_source": scenario.reviewer.api_source,
            "reviewer_can_run_experiments": scenario.reviewer_can_run_experiments,
            "equation": paper.equation,
            "justification": paper.justification,
            "review_decision": review.decision,
            "review_rationale": review.rationale,
            "review_confidence": review.confidence,
            "exact_accuracy": evaluation.get("exact_accuracy"),
            "rmsle": evaluation.get("rmsle"),
            "symbolic_equivalent": evaluation.get("symbolic_equivalent"),
        }
    )


def run_minipaper_suite(config: RunSuiteConfig) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    run_tag = sanitize_run_tag(config.run_tag)
    run_dir = repo_root / config.output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    normalized_scenarios = [scenario.normalized() for scenario in config.scenarios]
    manifest = {
        "run_tag": run_tag,
        "hypothesis_name": config.hypothesis_name,
        "scenarios": [asdict(s) for s in normalized_scenarios],
        "tasks": [asdict(t) for t in config.tasks],
        "scientist_population": config.scientist_population,
        "noise_level": config.noise_level,
        "max_scientist_turns": config.max_scientist_turns,
        "max_reviewer_turns": config.max_reviewer_turns,
        "started_at": datetime.now().isoformat(),
    }
    _write_json(run_dir / "manifest.json", manifest)

    if config.dry_run:
        return {"run_dir": str(run_dir), "manifest": manifest, "paper_results": [], "scenario_summary": []}

    rows: List[Dict[str, Any]] = []

    for scenario in normalized_scenarios:
        scenario_dir = _scenario_output_dir(run_dir, scenario)
        kb_path = scenario_dir / "knowledge_base.json"
        knowledge_base = load_minipaper_kb(kb_path)

        for task in config.tasks:
            module = importlib.import_module(f"modules.{task.module_name}")
            judge_model_name, judge_api_source = choose_judge(
                scientist=scenario.scientist,
                explicit_judge_model_name=config.judge_model_name,
                explicit_judge_api_source=config.judge_api_source,
            )

            for scientist_index in range(config.scientist_population):
                episode_index = len(rows) + 1
                artifact_dir = _episode_artifact_dir(run_dir, scenario, task, episode_index)
                artifact_dir.mkdir(parents=True, exist_ok=True)

                scientist_error: Optional[Dict[str, str]] = None
                reviewer_error: Optional[Dict[str, str]] = None
                evaluation_error: Optional[Dict[str, str]] = None
                fallback_equation = f"{module.FUNCTION_SIGNATURE} return float('nan')"

                try:
                    scientist_result = run_scientist_session(
                        module=module,
                        agent=scenario.scientist,
                        task=task,
                        knowledge_base=knowledge_base,
                        noise_level=config.noise_level,
                        max_turns=config.max_scientist_turns,
                        artifact_dir=artifact_dir,
                        episode_id=f"{scenario.scenario_id}-{episode_index}",
                    )
                except Exception as error:
                    scientist_error = _error_dict(error)
                    scientist_result = {
                        "status": "error",
                        "minipaper": None,
                        "rounds": 0,
                        "total_tokens": 0,
                        "num_experiments": 0,
                        "chat_history": [],
                    }

                scientist_paper = scientist_result.get("minipaper")
                if scientist_paper is None:
                    rationale = "Scientist agent failed to produce a valid minipaper."
                    if scientist_error:
                        rationale += f" Provider/runtime error: {scientist_error['error_type']}: {scientist_error['error_message']}"
                    scientist_paper = MiniPaper(
                        equation=fallback_equation,
                        justification=rationale,
                        raw_content="",
                    )

                try:
                    evaluation = module.evaluate_law(
                        scientist_paper.equation,
                        param_description=module.PARAM_DESCRIPTION,
                        difficulty=task.equation_difficulty,
                        law_version=task.law_version,
                        judge_model_name=judge_model_name,
                        trial_info={
                            "trial_id": f"{scenario.scenario_id}-{episode_index}-evaluation",
                            "trial_dir": str(artifact_dir),
                            "api_source_override": judge_api_source,
                            "judge_api_source": judge_api_source,
                        },
                        consistency=True,
                    )
                except Exception as error:
                    evaluation_error = _error_dict(error)
                    evaluation = {
                        "rmsle": float("nan"),
                        "exact_accuracy": 0.0,
                        "symbolic_equivalent": False,
                        "symbolic_msg": "Evaluation failed due to infrastructure/runtime error.",
                        "error": evaluation_error["error_message"],
                    }

                try:
                    reviewer_result = run_reviewer_session(
                        module=module,
                        agent=scenario.reviewer,
                        task=task,
                        knowledge_base=knowledge_base,
                        scientist_paper=scientist_paper,
                        reviewer_can_run_experiments=scenario.reviewer_can_run_experiments,
                        noise_level=config.noise_level,
                        max_turns=config.max_reviewer_turns,
                        artifact_dir=artifact_dir,
                        episode_id=f"{scenario.scenario_id}-{episode_index}",
                    )
                except Exception as error:
                    reviewer_error = _error_dict(error)
                    reviewer_result = {
                        "status": "error",
                        "review": None,
                        "rounds": 0,
                        "total_tokens": 0,
                        "num_experiments": 0,
                        "chat_history": [],
                    }

                review = reviewer_result.get("review") or ReviewDecision(
                    decision="reject",
                    rationale=(
                        "Reviewer agent failed to produce a valid review decision."
                        if not reviewer_error
                        else f"Reviewer failed due to {reviewer_error['error_type']}: {reviewer_error['error_message']}"
                    ),
                    confidence="low",
                    raw_content="",
                )

                _append_to_kb(
                    knowledge_base=knowledge_base,
                    scenario=scenario,
                    task=task,
                    paper=scientist_paper,
                    review=review,
                    evaluation=evaluation,
                )
                save_minipaper_kb(kb_path, knowledge_base)

                _write_json(
                    artifact_dir / "scientist_minipaper.json",
                    {
                        "paper": scientist_paper.to_dict(),
                        "status": scientist_result.get("status"),
                        "rounds": scientist_result.get("rounds"),
                        "total_tokens": scientist_result.get("total_tokens"),
                        "num_experiments": scientist_result.get("num_experiments"),
                        "error": scientist_error,
                    },
                )
                _write_json(
                    artifact_dir / "review_decision.json",
                    {
                        "review": review.to_dict(),
                        "status": reviewer_result.get("status"),
                        "rounds": reviewer_result.get("rounds"),
                        "total_tokens": reviewer_result.get("total_tokens"),
                        "num_experiments": reviewer_result.get("num_experiments"),
                        "error": reviewer_error,
                    },
                )
                _write_json(
                    artifact_dir / "evaluation.json",
                    {
                        "metrics": evaluation,
                        "error": evaluation_error,
                    },
                )
                _write_text(artifact_dir / "scientist_chat_history.log", _format_chat_history(scientist_result.get("chat_history", [])))
                _write_text(artifact_dir / "reviewer_chat_history.log", _format_chat_history(reviewer_result.get("chat_history", [])))

                rows.append(
                    {
                        "run_tag": run_tag,
                        "hypothesis_name": scenario.hypothesis_name,
                        "scenario_id": scenario.scenario_id,
                        "scenario_description": scenario.description,
                        "reviewer_relation": scenario.reviewer_relation,
                        "module": task.module_name,
                        "equation_difficulty": task.equation_difficulty,
                        "model_system": task.model_system,
                        "law_version": task.law_version,
                        "scientist_index": scientist_index,
                        "scientist_model_name": scenario.scientist.model_name,
                        "scientist_api_source": scenario.scientist.api_source,
                        "reviewer_model_name": scenario.reviewer.model_name,
                        "reviewer_api_source": scenario.reviewer.api_source,
                        "reviewer_can_run_experiments": scenario.reviewer_can_run_experiments,
                        "scientist_status": scientist_result.get("status"),
                        "reviewer_status": reviewer_result.get("status"),
                        "scientist_error_type": scientist_error["error_type"] if scientist_error else "",
                        "scientist_error_message": scientist_error["error_message"] if scientist_error else "",
                        "reviewer_error_type": reviewer_error["error_type"] if reviewer_error else "",
                        "reviewer_error_message": reviewer_error["error_message"] if reviewer_error else "",
                        "evaluation_error_type": evaluation_error["error_type"] if evaluation_error else "",
                        "evaluation_error_message": evaluation_error["error_message"] if evaluation_error else "",
                        "review_decision": review.decision,
                        "review_confidence": review.confidence,
                        "accepted": review.is_accept(),
                        "equation": scientist_paper.equation,
                        "justification": scientist_paper.justification,
                        "exact_accuracy": evaluation.get("exact_accuracy"),
                        "symbolic_equivalent": evaluation.get("symbolic_equivalent"),
                        "rmsle": evaluation.get("rmsle"),
                        "scientist_total_tokens": scientist_result.get("total_tokens"),
                        "reviewer_total_tokens": reviewer_result.get("total_tokens"),
                        "scientist_num_experiments": scientist_result.get("num_experiments"),
                        "reviewer_num_experiments": reviewer_result.get("num_experiments"),
                        "judge_model_name": judge_model_name,
                        "judge_api_source": judge_api_source,
                        "artifact_dir": str(artifact_dir),
                    }
                )

    paper_results_df = pd.DataFrame(rows)
    paper_results_path = run_dir / "paper_results.csv"
    paper_results_df.to_csv(paper_results_path, index=False, encoding="utf-8")

    scenario_summary_df = build_scenario_summary(paper_results_df)
    scenario_summary_df.to_csv(run_dir / "scenario_summary.csv", index=False, encoding="utf-8")
    (run_dir / "scenario_summary.md").write_text(dataframe_to_markdown(scenario_summary_df), encoding="utf-8")

    manifest["finished_at"] = datetime.now().isoformat()
    _write_json(run_dir / "manifest.json", manifest)
    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "paper_results": paper_results_df,
        "scenario_summary": scenario_summary_df,
    }


def build_scenario_summary(paper_results_df: pd.DataFrame) -> pd.DataFrame:
    if paper_results_df.empty:
        return pd.DataFrame()

    summary_rows: List[Dict[str, Any]] = []
    group_cols = [
        "run_tag",
        "hypothesis_name",
        "scenario_id",
        "scenario_description",
        "reviewer_relation",
        "scientist_model_name",
        "scientist_api_source",
        "reviewer_model_name",
        "reviewer_api_source",
        "reviewer_can_run_experiments",
    ]

    for keys, group in paper_results_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        total = len(group)
        accepted = int(group["accepted"].sum())
        accepted_group = group[group["accepted"]]
        correct_group = group[group["exact_accuracy"] == 1.0]
        accepted_correct = int(((group["accepted"]) & (group["exact_accuracy"] == 1.0)).sum())
        false_accepts = int(((group["accepted"]) & (group["exact_accuracy"] != 1.0)).sum())

        row.update(
            {
                "num_papers": total,
                "accepted_papers": accepted,
                "acceptance_rate_pct": 100.0 * accepted / total if total else 0.0,
                "mean_exact_accuracy_all_pct": 100.0 * group["exact_accuracy"].mean() if total else 0.0,
                "mean_exact_accuracy_accepted_pct": 100.0 * accepted_group["exact_accuracy"].mean() if not accepted_group.empty else 0.0,
                "accepted_correct_rate_pct": 100.0 * accepted_correct / total if total else 0.0,
                "false_accept_rate_pct": 100.0 * false_accepts / accepted if accepted else 0.0,
                "mean_rmsle_all": group["rmsle"].mean(),
                "mean_rmsle_accepted": accepted_group["rmsle"].mean() if not accepted_group.empty else float("nan"),
                "avg_scientist_tokens": group["scientist_total_tokens"].mean(),
                "avg_reviewer_tokens": group["reviewer_total_tokens"].mean(),
                "avg_scientist_experiments": group["scientist_num_experiments"].mean(),
                "avg_reviewer_experiments": group["reviewer_num_experiments"].mean(),
                "correct_papers": len(correct_group),
            }
        )
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(["scenario_id"]).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)
