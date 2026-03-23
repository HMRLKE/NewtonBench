import os
import json
from typing import Any, Dict, List, Optional

from .consistency import get_group_data, get_module_group, get_related_modules

def _load_global_kg(accumulation_dir: str) -> Dict[str, Any]:
    kg_path = os.path.join(accumulation_dir, "global_kg.json")
    if not os.path.exists(kg_path):
        return {}

    try:
        with open(kg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading discovered laws: {e}")
        return {}

def _law_sort_key(law: Dict[str, Any]) -> tuple:
    exact_accuracy = float(law.get("exact_accuracy", 0.0) or 0.0)
    similarity = float(law.get("similarity", 0.0) or 0.0)
    rmsle = float(law.get("rmsle", float("inf")) or float("inf"))
    return (-exact_accuracy, -similarity, rmsle, law.get("task", ""), law.get("difficulty", ""), law.get("version", ""))

def _format_quality_hint(law: Dict[str, Any]) -> str:
    exact_accuracy = float(law.get("exact_accuracy", 0.0) or 0.0)
    rmsle = float(law.get("rmsle", float("inf")) or float("inf"))
    if exact_accuracy >= 1.0:
        return "quality: exact symbolic match"
    if rmsle != float("inf"):
        return f"quality: best-so-far candidate (rmsle={rmsle:.4f})"
    return "quality: best-so-far candidate"

def get_discovered_laws_context(
    target_module: Optional[str] = None,
    consistency: Optional[bool] = None,
    accumulation_dir: str = "accumulation",
    max_related_laws: int = 3
) -> str:
    """Read global_kg.json and return a relevance-filtered string of discovered laws."""
    if not target_module:
        return ""

    data = _load_global_kg(accumulation_dir)
    laws = data.get("laws", [])
    if not laws:
        return (
            "**Related Previously Discovered Laws:**\n"
            "No prior laws have been discovered in this universe yet.\n"
            "Treat this as an initial discovery attempt: assume we currently know nothing about the universe "
            "beyond the task description and the experiments you perform."
        )

    group_name = get_module_group(target_module)
    group_data = get_group_data(target_module)
    related_modules = get_related_modules(target_module)
    if not group_name or not related_modules:
        return ""

    selected_laws: List[Dict[str, Any]] = []
    for law in laws:
        if law.get("task") not in related_modules:
            continue
        if consistency is not None and bool(law.get("consistency", False)) != bool(consistency):
            continue
        if not law.get("equation"):
            continue
        selected_laws.append(law)

    if not selected_laws:
        return (
            "**Related Previously Discovered Laws:**\n"
            f"No previously discovered laws from the same conceptual family are available yet for `{target_module}`.\n"
            "Do not assume any shared structure from other phenomena unless your own experiments justify it."
        )

    selected_laws.sort(key=_law_sort_key)
    selected_laws = selected_laws[:max_related_laws]

    consistency_axes = group_data.get("consistency_shared_axes", group_data.get("shared_axes", []))
    prompt_axes = group_data.get("prompt_relevance_axes", [])
    rationale = group_data.get("prompt_relation_rationale", "")
    mode_label = "consistency-matched" if consistency else "same-family"

    lines = [
        "**Related Previously Discovered Laws:**",
        f"You are discovering a law for `{target_module}`.",
        f"Only a small set of conceptually related laws from the `{group_name}` family is shown below.",
    ]

    if rationale:
        lines.append(rationale)
    if consistency_axes:
        lines.append(f"Strict shared consistency axes in this family: {', '.join(consistency_axes)}.")
    if prompt_axes:
        lines.append(f"Additional prompt-level relevance cues: {', '.join(prompt_axes)}.")

    lines.append(
        "Treat these as soft structural hints rather than guaranteed templates. "
        "Focus on transferable aspects such as distance dependence, multiplicative structure, "
        "dominant variables, exponents, and whether a source factor disappears or survives."
    )

    for law in selected_laws:
        task = law.get("task", "unknown_task")
        difficulty = law.get("difficulty", "unknown")
        version = law.get("version", "unknown")
        equation = law.get("equation", "Unknown Equation")
        quality_hint = _format_quality_hint(law)
        lines.append(
            f"- Related phenomenon: {task} (difficulty={difficulty}, version={version}, {mode_label})\n"
            f"  Best discovered equation: {equation}\n"
            f"  {quality_hint}"
        )

    return "\n".join(lines)
