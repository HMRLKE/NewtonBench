import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.minipaper_engine import (  # noqa: E402
    AgentSpec,
    RunSuiteConfig,
    ScenarioSpec,
    dataframe_to_markdown,
    expand_task_specs,
    run_minipaper_suite,
    sanitize_run_tag,
)


def _split_csv(arg: str) -> list[str]:
    return [item.strip() for item in arg.split(",") if item.strip()]


def _metric(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean()) if not df.empty else 0.0


def build_h4_summary(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    base = scenario_summary[
        (scenario_summary["scientist_thinking_enabled"] == False)  # noqa: E712
        & (scenario_summary["reviewer_thinking_enabled"] == False)  # noqa: E712
    ]
    scientist_only = scenario_summary[
        (scenario_summary["scientist_thinking_enabled"] == True)  # noqa: E712
        & (scenario_summary["reviewer_thinking_enabled"] == False)  # noqa: E712
    ]
    reviewer_only = scenario_summary[
        (scenario_summary["scientist_thinking_enabled"] == False)  # noqa: E712
        & (scenario_summary["reviewer_thinking_enabled"] == True)  # noqa: E712
    ]
    both = scenario_summary[
        (scenario_summary["scientist_thinking_enabled"] == True)  # noqa: E712
        & (scenario_summary["reviewer_thinking_enabled"] == True)  # noqa: E712
    ]

    base_acc = _metric(base, "accepted_correct_rate_pct")
    scientist_acc = _metric(scientist_only, "accepted_correct_rate_pct")
    reviewer_acc = _metric(reviewer_only, "accepted_correct_rate_pct")
    both_acc = _metric(both, "accepted_correct_rate_pct")
    condition_scores = {
        "baseline": base_acc,
        "scientist_only": scientist_acc,
        "reviewer_only": reviewer_acc,
        "both": both_acc,
    }

    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H4",
                "statement": "Thinking-mode on the scientist and/or reviewer side changes accepted-paper quality.",
                "baseline_accepted_correct_rate_pct": base_acc,
                "scientist_only_accepted_correct_rate_pct": scientist_acc,
                "reviewer_only_accepted_correct_rate_pct": reviewer_acc,
                "both_accepted_correct_rate_pct": both_acc,
                "scientist_main_effect_pct": scientist_acc - base_acc,
                "reviewer_main_effect_pct": reviewer_acc - base_acc,
                "both_effect_pct": both_acc - base_acc,
                "best_condition": max(condition_scores, key=condition_scores.get),
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="H4: 2x2 thinking-mode experiment for scientist and reviewer agents.")
    parser.add_argument("--run_tag", default="h4-thinking-mode")
    parser.add_argument("--scientist_model_name", default="gpt5mini")
    parser.add_argument("--scientist_api_source", default="oa", choices=["oa", "or", "g4s"])
    parser.add_argument("--reviewer_model_name", default="", help="Defaults to scientist_model_name.")
    parser.add_argument("--reviewer_api_source", default="", help="Defaults to scientist_api_source.")
    parser.add_argument("--reviewer_can_run_experiments", action="store_true")
    parser.add_argument("--modules", default="m0_gravity,m1_coulomb_force,m2_magnetic_force")
    parser.add_argument("--equation_difficulties", default="easy")
    parser.add_argument("--model_systems", default="vanilla_equation")
    parser.add_argument("--law_versions", default="v0,v1,v2")
    parser.add_argument("--scientist_population", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--max_scientist_turns", type=int, default=8)
    parser.add_argument("--max_reviewer_turns", type=int, default=4)
    parser.add_argument("--max_review_rounds", type=int, default=2)
    parser.add_argument("--judge_model_name", default=None)
    parser.add_argument("--judge_api_source", default=None, choices=["oa", "or", "g4s"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    reviewer_model_name = args.reviewer_model_name or args.scientist_model_name
    reviewer_api_source = args.reviewer_api_source or args.scientist_api_source

    tasks = expand_task_specs(
        repo_root=REPO_ROOT,
        modules=_split_csv(args.modules),
        equation_difficulties=_split_csv(args.equation_difficulties),
        model_systems=_split_csv(args.model_systems),
        law_versions=_split_csv(args.law_versions),
    )

    scenario_defs = [
        ("thinking_off_off", False, False, "Neither scientist nor reviewer receives thinking-mode instruction."),
        ("thinking_scientist_only", True, False, "Only the scientist receives thinking-mode instruction."),
        ("thinking_reviewer_only", False, True, "Only the reviewer receives thinking-mode instruction."),
        ("thinking_both", True, True, "Both scientist and reviewer receive thinking-mode instruction."),
    ]
    scenarios = [
        ScenarioSpec(
            scenario_id=scenario_id,
            scientist=AgentSpec(args.scientist_model_name, args.scientist_api_source, scientist_thinking),
            reviewer=AgentSpec(reviewer_model_name, reviewer_api_source, reviewer_thinking),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H4",
            reviewer_relation="thinking_factorial",
            description=description,
        )
        for scenario_id, scientist_thinking, reviewer_thinking, description in scenario_defs
    ]

    config = RunSuiteConfig(
        run_tag=sanitize_run_tag(args.run_tag),
        hypothesis_name="H4",
        scenarios=scenarios,
        tasks=tasks,
        scientist_population=args.scientist_population,
        noise_level=args.noise,
        max_scientist_turns=args.max_scientist_turns,
        max_reviewer_turns=args.max_reviewer_turns,
        max_review_rounds=args.max_review_rounds,
        judge_model_name=args.judge_model_name,
        judge_api_source=args.judge_api_source,
        reset_knowledge_base=True,
        dry_run=args.dry_run,
    )

    result = run_minipaper_suite(config)
    if args.dry_run:
        print(json.dumps(result["manifest"], indent=2))
        return 0

    run_dir = Path(result["run_dir"])
    summary_df = build_h4_summary(result["scenario_summary"])
    summary_df.to_csv(run_dir / "h4_summary.csv", index=False, encoding="utf-8")
    (run_dir / "h4_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print("Scenario summary:")
    print(result["scenario_summary"].to_string(index=False))
    print("\nH4 summary:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
