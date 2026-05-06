import importlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from utils.consistency import get_group_data, get_related_modules
from utils.minipaper_engine import TaskSpec


@dataclass
class PoisonedPaperRecord:
    module: str
    equation_difficulty: str
    model_system: str
    law_version: str
    clean_equation: str
    poisoned_equation: str
    poison_operations: List[str]


def parse_signature_args(function_signature: str) -> List[str]:
    match = re.search(r"def\s+discovered_law\s*\(([^)]*)\)", function_signature)
    if not match:
        return []
    return [part.strip().split(":")[0].strip() for part in match.group(1).split(",") if part.strip()]


def get_distance_exponent(module_name: str, difficulty: str, law_version: str) -> float:
    group_data = get_group_data(module_name)
    axes_mapping = group_data.get("axes_mapping", {})
    version_mapping = axes_mapping.get(difficulty, {}).get(law_version, {})
    return float(version_mapping.get("distance_exponent", 2.0))


def build_structural_equation(
    *,
    function_signature: str,
    exponent: float,
    drop_second_source: bool = False,
    multiply_by_distance: bool = False,
    add_distance_term: bool = False,
) -> str:
    params = parse_signature_args(function_signature)
    if not params:
        raise ValueError(f"Could not parse function signature: {function_signature}")

    distance_param = "distance" if "distance" in params else params[-1]
    source_params = [param for param in params if param != distance_param]
    source_terms = source_params[:1] if drop_second_source else source_params[:2]
    source_product = " * ".join(["C", *source_terms]) if source_terms else "C"
    distance_expr = f"({distance_param} ** {exponent:g})"
    if multiply_by_distance:
        return_expr = f"({source_product}) * {distance_expr}"
    else:
        return_expr = f"({source_product}) / {distance_expr}"
    if add_distance_term:
        return_expr = f"({return_expr}) + {distance_param}"

    return "\n".join(
        [
            function_signature,
            "    C = 1.0",
            f"    return {return_expr}",
        ]
    )


def perturb_equation(
    *,
    function_signature: str,
    exponent: float,
    edit_distance: int,
    rng: random.Random,
    operations: Sequence[str],
) -> Tuple[str, List[str]]:
    candidate_ops = list(operations) or ["distance_exponent", "drop_factor", "operator_flip", "add_term"]
    rng.shuffle(candidate_ops)
    selected_ops = candidate_ops[: max(1, min(edit_distance, 3, len(candidate_ops)))]

    poisoned_exponent = exponent
    drop_second_source = False
    multiply_by_distance = False
    add_distance_term = False

    for op in selected_ops:
        if op == "distance_exponent":
            poisoned_exponent = exponent + rng.choice([-0.5, 0.5, 1.0])
            if poisoned_exponent <= 0:
                poisoned_exponent = exponent + 0.5
        elif op == "drop_factor":
            drop_second_source = True
        elif op == "operator_flip":
            multiply_by_distance = True
        elif op == "add_term":
            add_distance_term = True

    return (
        build_structural_equation(
            function_signature=function_signature,
            exponent=poisoned_exponent,
            drop_second_source=drop_second_source,
            multiply_by_distance=multiply_by_distance,
            add_distance_term=add_distance_term,
        ),
        selected_ops,
    )


def build_seed_papers(repo_root: Path, tasks: Iterable[TaskSpec]) -> List[Dict[str, Any]]:
    seen = set()
    papers: List[Dict[str, Any]] = []
    for task in tasks:
        group_data = get_group_data(task.module_name)
        if "distance_exponent" not in group_data.get("prompt_relevance_axes", []) and not group_data.get("axes_mapping"):
            continue
        key = (task.module_name, task.equation_difficulty, task.model_system, task.law_version)
        if key in seen:
            continue
        seen.add(key)
        module = importlib.import_module(f"modules.{task.module_name}")
        exponent = get_distance_exponent(task.module_name, task.equation_difficulty, task.law_version)
        equation = build_structural_equation(
            function_signature=module.FUNCTION_SIGNATURE,
            exponent=exponent,
        )
        papers.append(
            {
                "scenario_id": "seeded_kb",
                "module": task.module_name,
                "equation_difficulty": task.equation_difficulty,
                "model_system": task.model_system,
                "law_version": task.law_version,
                "scientist_model_name": "seeded_oracle",
                "scientist_api_source": "seeded",
                "reviewer_model_name": "seeded_oracle",
                "reviewer_api_source": "seeded",
                "reviewer_can_run_experiments": False,
                "equation": equation,
                "justification": (
                    "Seeded structurally related law used as prior shared context. "
                    "It preserves the declared distance-exponent consistency axis."
                ),
                "review_decision": "accept",
                "review_rationale": "Seeded clean prior.",
                "review_confidence": "high",
                "exact_accuracy": 1.0,
                "rmsle": 0.0,
                "symbolic_equivalent": True,
                "is_seeded": True,
                "is_poisoned": False,
            }
        )
    return papers


def build_clean_and_poisoned_kbs(
    *,
    repo_root: Path,
    tasks: Sequence[TaskSpec],
    poison_rate: float,
    edit_distance: int,
    seed: int,
    operations: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[PoisonedPaperRecord]]:
    clean_papers = build_seed_papers(repo_root, tasks)
    poisoned_papers = [dict(paper) for paper in clean_papers]
    poisonable_indices = list(range(len(poisoned_papers)))
    rng = random.Random(seed)
    rng.shuffle(poisonable_indices)

    poison_count = int(round(len(poisonable_indices) * poison_rate))
    if poison_rate > 0 and poison_count == 0 and poisonable_indices:
        poison_count = 1
    poison_count = min(poison_count, len(poisonable_indices))

    poison_records: List[PoisonedPaperRecord] = []
    for idx in poisonable_indices[:poison_count]:
        paper = poisoned_papers[idx]
        module = importlib.import_module(f"modules.{paper['module']}")
        exponent = get_distance_exponent(paper["module"], paper["equation_difficulty"], paper["law_version"])
        poisoned_equation, selected_ops = perturb_equation(
            function_signature=module.FUNCTION_SIGNATURE,
            exponent=exponent,
            edit_distance=edit_distance,
            rng=rng,
            operations=operations,
        )
        clean_equation = paper["equation"]
        paper.update(
            {
                "equation": poisoned_equation,
                "justification": (
                    "Seeded poisoned prior. This minipaper is intentionally close to a plausible "
                    "consistency-axis law but contains a small algebraic perturbation."
                ),
                "review_rationale": "Seeded poisoned prior accepted for H3 intervention.",
                "review_confidence": "high",
                "exact_accuracy": 0.0,
                "rmsle": float("nan"),
                "symbolic_equivalent": False,
                "is_poisoned": True,
                "poison_operations": selected_ops,
                "clean_equation": clean_equation,
            }
        )
        poison_records.append(
            PoisonedPaperRecord(
                module=paper["module"],
                equation_difficulty=paper["equation_difficulty"],
                model_system=paper["model_system"],
                law_version=paper["law_version"],
                clean_equation=clean_equation,
                poisoned_equation=poisoned_equation,
                poison_operations=selected_ops,
            )
        )

    return (
        {"accepted_papers": clean_papers, "rejected_papers": []},
        {"accepted_papers": poisoned_papers, "rejected_papers": []},
        poison_records,
    )


def has_related_poison(target_module: str, poisoned_records: Sequence[PoisonedPaperRecord]) -> bool:
    related_modules = set(get_related_modules(target_module))
    return any(record.module in related_modules for record in poisoned_records)
