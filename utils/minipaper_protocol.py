import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .consistency import get_group_data, get_related_modules


DEFAULT_MINIPAPER_KB = {
    "accepted_papers": [],
    "rejected_papers": [],
}


@dataclass
class MiniPaper:
    equation: str
    justification: str
    raw_content: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewDecision:
    decision: str
    rationale: str
    confidence: str
    raw_content: str

    def is_accept(self) -> bool:
        return self.decision.strip().lower() == "accept"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SCIENTIST_SYSTEM_PROMPT = """You are a scientist agent working in a simulated universe.
Your objective is to discover the governing law for the current task and to communicate your conclusion as a short minipaper.

Rules:
1. You may use <run_experiment> to request experiments.
2. You may submit only one final <mini_paper>.
3. A minipaper is not automatically accepted into the shared knowledge base.
4. Your minipaper must contain:
   - an <equation> block with a valid Python implementation of `def discovered_law(...)`
   - a <justification> block of at most one paragraph
5. The justification should state why you believe the proposed law is correct and, when relevant, which already accepted regularities informed the proposal.
6. Use experiments to validate exponents, constants, and variable dependence before writing the minipaper.
7. Your minipaper may be rejected by a reviewer. If reviewer feedback is provided, revise the proposal directly and address the objections explicitly in the next draft.
8. The final minipaper format must be:

<mini_paper>
<equation>
def discovered_law(...):
    ...
</equation>
<justification>
One paragraph only.
</justification>
</mini_paper>
"""


REVIEWER_SYSTEM_PROMPT_WITH_EXPERIMENTS = """You are a reviewer agent evaluating a scientist's minipaper.
Your task is to decide whether the minipaper should enter the shared knowledge base.

Rules:
1. You may use <run_experiment> to verify the proposal against the simulator.
2. You must end by returning exactly one <review_decision> block in JSON.
3. The JSON must contain the fields:
   - decision: accept or reject
   - rationale: short explanation
   - confidence: low, medium, or high
4. Accept only when the proposed law and its justification are sufficiently well supported.
5. Reject when the evidence is weak, the justification is inconsistent with the data, or the law appears structurally implausible.

Required output format:
<review_decision>
{"decision": "accept", "rationale": "...", "confidence": "medium"}
</review_decision>
"""


REVIEWER_SYSTEM_PROMPT_NO_EXPERIMENTS = """You are a reviewer agent evaluating a scientist's minipaper.
Your task is to decide whether the minipaper should enter the shared knowledge base.

Rules:
1. You may NOT run experiments in this configuration.
2. You must end by returning exactly one <review_decision> block in JSON.
3. The JSON must contain the fields:
   - decision: accept or reject
   - rationale: short explanation
   - confidence: low, medium, or high
4. Base your decision only on the minipaper, the task description, and the accepted shared context.

Required output format:
<review_decision>
{"decision": "accept", "rationale": "...", "confidence": "medium"}
</review_decision>
"""


def strip_final_submission_section(task_prompt: str) -> str:
    marker = "**Final Submission:**"
    if marker not in task_prompt:
        return task_prompt.strip()
    return task_prompt.split(marker, 1)[0].strip()


def load_minipaper_kb(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_MINIPAPER_KB))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_MINIPAPER_KB))


def save_minipaper_kb(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_last_tag_content(text: str, tag_name: str) -> Optional[str]:
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    start = text.rfind(start_tag)
    if start == -1:
        return None
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start + len(start_tag):end].strip()


def parse_minipaper(response_text: str) -> Optional[MiniPaper]:
    content = _extract_last_tag_content(response_text or "", "mini_paper")
    if not content:
        return None

    equation = _extract_last_tag_content(content, "equation")
    justification = _extract_last_tag_content(content, "justification")
    if not equation or not justification:
        return None

    return MiniPaper(
        equation=equation.strip(),
        justification=" ".join(justification.split()),
        raw_content=content.strip(),
    )


def parse_review_decision(response_text: str) -> Optional[ReviewDecision]:
    content = _extract_last_tag_content(response_text or "", "review_decision")
    if not content:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    decision = str(data.get("decision", "")).strip().lower()
    if decision not in {"accept", "reject"}:
        return None

    return ReviewDecision(
        decision=decision,
        rationale=" ".join(str(data.get("rationale", "")).split()),
        confidence=str(data.get("confidence", "medium")).strip().lower() or "medium",
        raw_content=content.strip(),
    )


def _format_paper_context_lines(papers: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for paper in papers:
        lines.append(
            f"- Accepted paper: {paper.get('module')} "
            f"(difficulty={paper.get('equation_difficulty')}, system={paper.get('model_system')}, "
            f"law_version={paper.get('law_version')})"
        )
        lines.append(f"  Equation: {paper.get('equation')}")
        lines.append(f"  Justification: {paper.get('justification')}")
    return lines


def format_related_accepted_papers_context(
    *,
    target_module: str,
    knowledge_base: Dict[str, Any],
    max_papers: int = 3,
) -> str:
    papers = knowledge_base.get("accepted_papers", [])
    if not papers:
        return (
            "**Accepted Shared Context:**\n"
            "No minipapers have been accepted into the shared knowledge base yet.\n"
            "Treat this as an initial discovery setting and rely primarily on your own experiments."
        )

    related_modules = set(get_related_modules(target_module))
    group_data = get_group_data(target_module)
    prompt_axes = group_data.get("prompt_relevance_axes", [])
    rationale = group_data.get("prompt_relation_rationale", "")

    selected = [paper for paper in papers if paper.get("module") in related_modules]
    selected = selected[:max_papers]

    if not selected:
        return (
            "**Accepted Shared Context:**\n"
            "The shared knowledge base currently contains accepted minipapers, but none from the most relevant conceptual family.\n"
            "Do not assume cross-task transfer unless your experiments justify it."
        )

    lines = [
        "**Accepted Shared Context:**",
        f"The following accepted minipapers are conceptually related to `{target_module}`.",
    ]
    if rationale:
        lines.append(rationale)
    if prompt_axes:
        lines.append(f"Relevant transfer cues: {', '.join(prompt_axes)}.")
    lines.extend(_format_paper_context_lines(selected))
    return "\n".join(lines)


def build_scientist_prompt(
    *,
    module: Any,
    system: str,
    noise_level: float,
    knowledge_base: Dict[str, Any],
    revision_context: Optional[Dict[str, Any]] = None,
) -> str:
    base_task_prompt = module.get_task_prompt(system, noise_level=noise_level, prompt_set="original")
    apparatus_prompt = strip_final_submission_section(base_task_prompt)
    context = format_related_accepted_papers_context(
        target_module=module.__name__.split(".")[-1],
        knowledge_base=knowledge_base,
    )
    minipaper_instructions = f"""**Scientist Deliverable:**
You must conclude with a short minipaper rather than a bare function.

The minipaper must use the exact function signature:
`{module.FUNCTION_SIGNATURE}`

Required final format:
<mini_paper>
<equation>
{module.FUNCTION_SIGNATURE}
    ...
</equation>
<justification>
One paragraph explaining why the proposed law is plausible, and when relevant which accepted prior regularities informed your reasoning.
</justification>
</mini_paper>
"""
    sections = [context]

    if revision_context:
        previous_paper = revision_context.get("previous_paper")
        previous_review = revision_context.get("previous_review")
        current_round = revision_context.get("current_round")
        max_rounds = revision_context.get("max_rounds")
        if previous_paper and previous_review:
            sections.append(
                f"""**Revision Context:**
You are now preparing a revised minipaper draft.
Current review round: {current_round} / {max_rounds}

Previous draft:
<mini_paper>
<equation>
{previous_paper.equation}
</equation>
<justification>
{previous_paper.justification}
</justification>
</mini_paper>

Reviewer feedback:
- decision: {previous_review.decision}
- confidence: {previous_review.confidence}
- rationale: {previous_review.rationale}

Revise the law and/or justification so that the new draft directly addresses the reviewer criticism.
If you keep the same law, the justification still has to respond to the reviewer's objections explicitly."""
            )

    sections.extend([apparatus_prompt, minipaper_instructions])
    return "\n\n".join(sections).strip()


def build_reviewer_prompt(
    *,
    module: Any,
    system: str,
    noise_level: float,
    knowledge_base: Dict[str, Any],
    scientist_paper: MiniPaper,
    reviewer_can_experiment: bool,
) -> str:
    base_task_prompt = module.get_task_prompt(system, noise_level=noise_level, prompt_set="original")
    apparatus_prompt = strip_final_submission_section(base_task_prompt)
    context = format_related_accepted_papers_context(
        target_module=module.__name__.split(".")[-1],
        knowledge_base=knowledge_base,
    )
    reviewer_mode = (
        "You may run experiments to verify the proposal before deciding."
        if reviewer_can_experiment
        else "You may not run experiments in this review configuration."
    )
    paper_block = f"""**Scientist Minipaper Under Review:**
<mini_paper>
<equation>
{scientist_paper.equation}
</equation>
<justification>
{scientist_paper.justification}
</justification>
</mini_paper>
"""
    reviewer_instruction = f"""**Reviewer Task:**
Decide whether this minipaper should enter the shared knowledge base.
{reviewer_mode}
Only accepted minipapers become part of the shared memory used by future scientist agents."""
    return "\n\n".join([context, apparatus_prompt, paper_block, reviewer_instruction]).strip()
