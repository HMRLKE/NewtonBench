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


def build_h2_summary(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    same_provider = scenario_summary[scenario_summary["reviewer_relation"] == "same_provider"]
    cross_provider = scenario_summary[scenario_summary["reviewer_relation"] == "cross_provider"]

    same_acc = float(same_provider["mean_exact_accuracy_accepted_pct"].mean()) if not same_provider.empty else 0.0
    cross_acc = float(cross_provider["mean_exact_accuracy_accepted_pct"].mean()) if not cross_provider.empty else 0.0
    same_false_accept = float(same_provider["false_accept_rate_pct"].mean()) if not same_provider.empty else 0.0
    cross_false_accept = float(cross_provider["false_accept_rate_pct"].mean()) if not cross_provider.empty else 0.0

    supports_h2 = (cross_acc > same_acc) and (cross_false_accept < same_false_accept)

    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H2",
                "statement": "Cross-provider review is stricter and yields higher accepted-paper accuracy than same-provider review.",
                "same_provider_mean_exact_accuracy_accepted_pct": same_acc,
                "cross_provider_mean_exact_accuracy_accepted_pct": cross_acc,
                "same_provider_false_accept_rate_pct": same_false_accept,
                "cross_provider_false_accept_rate_pct": cross_false_accept,
                "supports_hypothesis": supports_h2,
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="H2: cross-provider review filters errors more aggressively than same-provider review.")
    parser.add_argument("--run_tag", default="h2-cross-provider-review")
    parser.add_argument("--openai_model_name", default="gpt5mini")
    parser.add_argument("--g4s_model_name", default="gemma4:31b")
    parser.add_argument("--modules", default="m0_gravity,m1_coulomb_force,m2_magnetic_force")
    parser.add_argument("--equation_difficulties", default="easy")
    parser.add_argument("--model_systems", default="vanilla_equation")
    parser.add_argument("--law_versions", default="v0,v1,v2")
    parser.add_argument("--reviewer_can_run_experiments", action="store_true")
    parser.add_argument("--scientist_population", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--max_scientist_turns", type=int, default=8)
    parser.add_argument("--max_reviewer_turns", type=int, default=4)
    parser.add_argument("--judge_model_name", default=None)
    parser.add_argument("--judge_api_source", default=None, choices=["oa", "or", "g4s"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tasks = expand_task_specs(
        repo_root=REPO_ROOT,
        modules=_split_csv(args.modules),
        equation_difficulties=_split_csv(args.equation_difficulties),
        model_systems=_split_csv(args.model_systems),
        law_versions=_split_csv(args.law_versions),
    )

    scenarios = [
        ScenarioSpec(
            scenario_id="oa_to_oa",
            scientist=AgentSpec(args.openai_model_name, "oa"),
            reviewer=AgentSpec(args.openai_model_name, "oa"),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H2",
            reviewer_relation="same_provider",
            description="Closed-source scientist reviewed by the same provider family.",
        ),
        ScenarioSpec(
            scenario_id="g4s_to_g4s",
            scientist=AgentSpec(args.g4s_model_name, "g4s"),
            reviewer=AgentSpec(args.g4s_model_name, "g4s"),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H2",
            reviewer_relation="same_provider",
            description="Open-weight scientist reviewed by the same provider family.",
        ),
        ScenarioSpec(
            scenario_id="oa_to_g4s",
            scientist=AgentSpec(args.openai_model_name, "oa"),
            reviewer=AgentSpec(args.g4s_model_name, "g4s"),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H2",
            reviewer_relation="cross_provider",
            description="Closed-source scientist reviewed by open-weight model.",
        ),
        ScenarioSpec(
            scenario_id="g4s_to_oa",
            scientist=AgentSpec(args.g4s_model_name, "g4s"),
            reviewer=AgentSpec(args.openai_model_name, "oa"),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H2",
            reviewer_relation="cross_provider",
            description="Open-weight scientist reviewed by closed-source model.",
        ),
    ]

    config = RunSuiteConfig(
        run_tag=sanitize_run_tag(args.run_tag),
        hypothesis_name="H2",
        scenarios=scenarios,
        tasks=tasks,
        scientist_population=args.scientist_population,
        noise_level=args.noise,
        max_scientist_turns=args.max_scientist_turns,
        max_reviewer_turns=args.max_reviewer_turns,
        judge_model_name=args.judge_model_name,
        judge_api_source=args.judge_api_source,
        dry_run=args.dry_run,
    )

    result = run_minipaper_suite(config)
    if args.dry_run:
        print(json.dumps(result["manifest"], indent=2))
        return 0

    run_dir = Path(result["run_dir"])
    summary_df = build_h2_summary(result["scenario_summary"])
    summary_df.to_csv(run_dir / "h2_summary.csv", index=False, encoding="utf-8")
    (run_dir / "h2_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print("Scenario summary:")
    print(result["scenario_summary"].to_string(index=False))
    print("\nH2 summary:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
