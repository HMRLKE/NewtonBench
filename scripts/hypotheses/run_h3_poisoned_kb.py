import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.kb_poisoning import build_clean_and_poisoned_kbs, has_related_poison  # noqa: E402
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


def build_h3_summary(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    clean = scenario_summary[scenario_summary["scenario_id"] == "clean_kb"]
    poisoned = scenario_summary[scenario_summary["scenario_id"] == "poisoned_kb"]

    def metric(df: pd.DataFrame, column: str) -> float:
        return float(df[column].mean()) if not df.empty else 0.0

    clean_acc = metric(clean, "mean_exact_accuracy_all_pct")
    poisoned_acc = metric(poisoned, "mean_exact_accuracy_all_pct")
    clean_accepted_correct = metric(clean, "accepted_correct_rate_pct")
    poisoned_accepted_correct = metric(poisoned, "accepted_correct_rate_pct")
    clean_false_accept = metric(clean, "false_accept_rate_pct")
    poisoned_false_accept = metric(poisoned, "false_accept_rate_pct")

    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H3",
                "statement": "Incorrect accepted knowledge-base entries degrade later related discoveries.",
                "clean_mean_exact_accuracy_all_pct": clean_acc,
                "poisoned_mean_exact_accuracy_all_pct": poisoned_acc,
                "delta_exact_accuracy_pct": poisoned_acc - clean_acc,
                "clean_accepted_correct_rate_pct": clean_accepted_correct,
                "poisoned_accepted_correct_rate_pct": poisoned_accepted_correct,
                "delta_accepted_correct_rate_pct": poisoned_accepted_correct - clean_accepted_correct,
                "clean_false_accept_rate_pct": clean_false_accept,
                "poisoned_false_accept_rate_pct": poisoned_false_accept,
                "delta_false_accept_rate_pct": poisoned_false_accept - clean_false_accept,
                "supports_hypothesis": (poisoned_acc < clean_acc) or (poisoned_false_accept > clean_false_accept),
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="H3: poisoned accepted KB entries affect later minipaper discovery.")
    parser.add_argument("--run_tag", default="h3-poisoned-kb")
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
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--poison_edit_distance", type=int, default=1)
    parser.add_argument("--poison_operations", default="distance_exponent,drop_factor,operator_flip,add_term")
    parser.add_argument("--poison_seed", type=int, default=42)
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
    clean_kb, poisoned_kb, poison_records = build_clean_and_poisoned_kbs(
        repo_root=REPO_ROOT,
        tasks=tasks,
        poison_rate=args.poison_rate,
        edit_distance=args.poison_edit_distance,
        seed=args.poison_seed,
        operations=_split_csv(args.poison_operations),
    )

    scenarios = [
        ScenarioSpec(
            scenario_id="clean_kb",
            scientist=AgentSpec(args.scientist_model_name, args.scientist_api_source),
            reviewer=AgentSpec(reviewer_model_name, reviewer_api_source),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H3",
            reviewer_relation="clean_seeded_kb",
            description="Agent sees clean accepted related minipapers.",
        ),
        ScenarioSpec(
            scenario_id="poisoned_kb",
            scientist=AgentSpec(args.scientist_model_name, args.scientist_api_source),
            reviewer=AgentSpec(reviewer_model_name, reviewer_api_source),
            reviewer_can_run_experiments=args.reviewer_can_run_experiments,
            hypothesis_name="H3",
            reviewer_relation="poisoned_seeded_kb",
            description="Agent sees intentionally perturbed accepted related minipapers.",
        ),
    ]

    config = RunSuiteConfig(
        run_tag=sanitize_run_tag(args.run_tag),
        hypothesis_name="H3",
        scenarios=scenarios,
        tasks=tasks,
        scientist_population=args.scientist_population,
        noise_level=args.noise,
        max_scientist_turns=args.max_scientist_turns,
        max_reviewer_turns=args.max_reviewer_turns,
        max_review_rounds=args.max_review_rounds,
        judge_model_name=args.judge_model_name,
        judge_api_source=args.judge_api_source,
        initial_knowledge_bases={"clean_kb": clean_kb, "poisoned_kb": poisoned_kb},
        reset_knowledge_base=True,
        dry_run=args.dry_run,
    )

    result = run_minipaper_suite(config)
    if args.dry_run:
        print(json.dumps(result["manifest"], indent=2))
        print(json.dumps([record.__dict__ for record in poison_records], indent=2))
        return 0

    run_dir = Path(result["run_dir"])
    poison_df = pd.DataFrame([record.__dict__ for record in poison_records])
    poison_df.to_csv(run_dir / "poison_manifest.csv", index=False, encoding="utf-8")

    paper_results = result["paper_results"].copy()
    paper_results["related_poison_in_context"] = paper_results.apply(
        lambda row: row["scenario_id"] == "poisoned_kb" and has_related_poison(row["module"], poison_records),
        axis=1,
    )
    paper_results.to_csv(run_dir / "h3_paper_results.csv", index=False, encoding="utf-8")

    summary_df = build_h3_summary(result["scenario_summary"])
    summary_df.to_csv(run_dir / "h3_summary.csv", index=False, encoding="utf-8")
    (run_dir / "h3_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print("Scenario summary:")
    print(result["scenario_summary"].to_string(index=False))
    print("\nH3 summary:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
