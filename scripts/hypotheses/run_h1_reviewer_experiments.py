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


def build_h1_summary(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    passive = scenario_summary[scenario_summary["reviewer_can_run_experiments"] == False]  # noqa: E712
    active = scenario_summary[scenario_summary["reviewer_can_run_experiments"] == True]  # noqa: E712

    passive_metric = float(passive["accepted_correct_rate_pct"].mean()) if not passive.empty else 0.0
    active_metric = float(active["accepted_correct_rate_pct"].mean()) if not active.empty else 0.0
    absolute_gain_pct = active_metric - passive_metric
    if passive_metric > 0:
        relative_gain_pct = absolute_gain_pct / passive_metric * 100.0
        supports_h1 = relative_gain_pct >= 10.0
        gain_interpretation = "relative_gain_defined"
    elif active_metric > 0:
        relative_gain_pct = float("nan")
        supports_h1 = True
        gain_interpretation = "passive_zero_active_positive_relative_gain_undefined"
    else:
        relative_gain_pct = 0.0
        supports_h1 = False
        gain_interpretation = "both_zero_no_observed_gain"

    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H1",
                "statement": "Allowing reviewer-side experiments improves accepted_correct_rate_pct by at least 10% relative.",
                "passive_mean_accepted_correct_rate_pct": passive_metric,
                "active_mean_accepted_correct_rate_pct": active_metric,
                "absolute_gain_pct": absolute_gain_pct,
                "relative_gain_pct": relative_gain_pct,
                "gain_interpretation": gain_interpretation,
                "supports_hypothesis": supports_h1,
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="H1: reviewer-side experimentation improves minipaper outcomes.")
    parser.add_argument("--run_tag", default="h1-reviewer-experiments")
    parser.add_argument("--scientist_model_name", default="gpt5mini")
    parser.add_argument("--scientist_api_source", default="oa", choices=["oa", "or", "g4s"])
    parser.add_argument("--reviewer_model_name", default="", help="Defaults to scientist_model_name.")
    parser.add_argument("--reviewer_api_source", default="", help="Defaults to scientist_api_source.")
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

    scenarios = [
        ScenarioSpec(
            scenario_id="reviewer_passive",
            scientist=AgentSpec(args.scientist_model_name, args.scientist_api_source),
            reviewer=AgentSpec(reviewer_model_name, reviewer_api_source),
            reviewer_can_run_experiments=False,
            hypothesis_name="H1",
            reviewer_relation="same_pair",
            description="Reviewer must decide from the minipaper only.",
        ),
        ScenarioSpec(
            scenario_id="reviewer_active",
            scientist=AgentSpec(args.scientist_model_name, args.scientist_api_source),
            reviewer=AgentSpec(reviewer_model_name, reviewer_api_source),
            reviewer_can_run_experiments=True,
            hypothesis_name="H1",
            reviewer_relation="same_pair",
            description="Reviewer may run experiments before accepting or rejecting.",
        ),
    ]

    config = RunSuiteConfig(
        run_tag=sanitize_run_tag(args.run_tag),
        hypothesis_name="H1",
        scenarios=scenarios,
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
        return 0

    run_dir = Path(result["run_dir"])
    summary_df = build_h1_summary(result["scenario_summary"])
    summary_df.to_csv(run_dir / "h1_summary.csv", index=False, encoding="utf-8")
    (run_dir / "h1_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print("Scenario summary:")
    print(result["scenario_summary"].to_string(index=False))
    print("\nH1 summary:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
