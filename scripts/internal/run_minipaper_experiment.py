import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.minipaper_engine import (  # noqa: E402
    AgentSpec,
    RunSuiteConfig,
    ScenarioSpec,
    expand_task_specs,
    run_minipaper_suite,
    sanitize_run_tag,
)


def _split_csv(arg: str) -> list[str]:
    return [item.strip() for item in arg.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one minipaper scientist-reviewer scenario over a task sweep.")
    parser.add_argument("--run_tag", required=True, help="Logical run tag.")
    parser.add_argument("--scenario_id", default="single_scenario", help="Scenario identifier.")
    parser.add_argument("--description", default="Single minipaper scenario", help="Scenario description.")
    parser.add_argument("--scientist_model_name", required=True)
    parser.add_argument("--scientist_api_source", required=True, choices=["oa", "or", "g4s"])
    parser.add_argument("--reviewer_model_name", required=True)
    parser.add_argument("--reviewer_api_source", required=True, choices=["oa", "or", "g4s"])
    parser.add_argument("--reviewer_can_run_experiments", action="store_true")
    parser.add_argument("--reviewer_relation", default="custom", help="Scenario label such as same_provider or cross_provider.")
    parser.add_argument("--modules", default="", help="Comma-separated module filter. Empty means all modules.")
    parser.add_argument("--equation_difficulties", default="easy", help="Comma-separated difficulties.")
    parser.add_argument("--model_systems", default="vanilla_equation", help="Comma-separated model systems.")
    parser.add_argument("--law_versions", default="", help="Comma-separated law versions. Empty means all changed versions.")
    parser.add_argument("--scientist_population", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--max_scientist_turns", type=int, default=8)
    parser.add_argument("--max_reviewer_turns", type=int, default=4)
    parser.add_argument("--max_review_rounds", type=int, default=2)
    parser.add_argument("--judge_model_name", default=None)
    parser.add_argument("--judge_api_source", default=None, choices=["oa", "or", "g4s"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tasks = expand_task_specs(
        repo_root=REPO_ROOT,
        modules=_split_csv(args.modules) if args.modules else None,
        equation_difficulties=_split_csv(args.equation_difficulties),
        model_systems=_split_csv(args.model_systems),
        law_versions=_split_csv(args.law_versions) if args.law_versions else None,
    )

    scenario = ScenarioSpec(
        scenario_id=args.scenario_id,
        scientist=AgentSpec(args.scientist_model_name, args.scientist_api_source),
        reviewer=AgentSpec(args.reviewer_model_name, args.reviewer_api_source),
        reviewer_can_run_experiments=args.reviewer_can_run_experiments,
        hypothesis_name="custom_minipaper_run",
        reviewer_relation=args.reviewer_relation,
        description=args.description,
    )

    config = RunSuiteConfig(
        run_tag=sanitize_run_tag(args.run_tag),
        hypothesis_name="custom_minipaper_run",
        scenarios=[scenario],
        tasks=tasks,
        scientist_population=args.scientist_population,
        noise_level=args.noise,
        max_scientist_turns=args.max_scientist_turns,
        max_reviewer_turns=args.max_reviewer_turns,
        max_review_rounds=args.max_review_rounds,
        judge_model_name=args.judge_model_name,
        judge_api_source=args.judge_api_source,
        dry_run=args.dry_run,
    )

    result = run_minipaper_suite(config)
    if args.dry_run:
        print(json.dumps(result["manifest"], indent=2))
    else:
        print(f"Run directory: {result['run_dir']}")
        print("Scenario summary:")
        print(result["scenario_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
