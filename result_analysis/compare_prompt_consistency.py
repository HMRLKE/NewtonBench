import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from summarize_results import build_config_summary, collect_trial_rows, dataframe_to_markdown


def _load_config_summary(result_dir: Path, run_tag: str) -> pd.DataFrame:
    rows = collect_trial_rows(result_dir=result_dir, run_tag=run_tag)
    trials_df = pd.DataFrame(rows)
    if trials_df.empty:
        return pd.DataFrame()

    numeric_cols = ["noise_level", "rmsle", "exact_accuracy", "rounds", "experiments", "total_tokens"]
    for col in numeric_cols:
        trials_df[col] = pd.to_numeric(trials_df[col], errors="coerce")
    trials_df["trial_success"] = trials_df["trial_success"].astype(bool)
    return build_config_summary(trials_df)


def _prepare_run_table(summary_df: pd.DataFrame, label: str, expected_prompt_set: str, expected_consistency: bool) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    table = summary_df.copy()
    table["accuracy_pct"] = table["mean_exact_accuracy"] * 100.0
    table["success_rate_pct"] = table["success_rate"] * 100.0
    table["law_id"] = (
        table["module"].astype(str)
        + "/"
        + table["equation_difficulty"].astype(str)
        + "/"
        + table["model_system"].astype(str)
        + "/"
        + table["law_version"].astype(str)
    )

    key_cols = [
        "api_source",
        "model_name",
        "agent_backend",
        "module",
        "equation_difficulty",
        "model_system",
        "law_version",
        "noise_level",
        "law_id",
    ]
    metric_cols = {
        "run_tag": f"run_tag_{label}",
        "prompt_set": f"prompt_set_{label}",
        "consistency": f"consistency_{label}",
        "num_trials": f"num_trials_{label}",
        "num_successful_trials": f"num_successful_trials_{label}",
        "success_rate_pct": f"success_rate_pct_{label}",
        "accuracy_pct": f"accuracy_pct_{label}",
        "mean_rmsle": f"mean_rmsle_{label}",
        "avg_total_tokens": f"avg_total_tokens_{label}",
    }
    selected = table[key_cols + list(metric_cols.keys())].rename(columns=metric_cols)
    selected[f"expected_prompt_set_{label}"] = expected_prompt_set
    selected[f"expected_consistency_{label}"] = expected_consistency
    return selected


def build_four_way_comparison(run_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    key_cols = [
        "api_source",
        "model_name",
        "agent_backend",
        "module",
        "equation_difficulty",
        "model_system",
        "law_version",
        "noise_level",
        "law_id",
    ]

    merged = None
    for label in ["original_inconsistent", "original_consistent", "modified_inconsistent", "modified_consistent"]:
        table = run_tables.get(label)
        if table is None or table.empty:
            continue
        if merged is None:
            merged = table.copy()
        else:
            merged = merged.merge(table, on=key_cols, how="outer")

    if merged is None:
        return pd.DataFrame()

    merged["delta_accuracy_pct_original"] = merged["accuracy_pct_original_consistent"] - merged["accuracy_pct_original_inconsistent"]
    merged["delta_accuracy_pct_modified"] = merged["accuracy_pct_modified_consistent"] - merged["accuracy_pct_modified_inconsistent"]
    merged["delta_rmsle_original"] = merged["mean_rmsle_original_consistent"] - merged["mean_rmsle_original_inconsistent"]
    merged["delta_rmsle_modified"] = merged["mean_rmsle_modified_consistent"] - merged["mean_rmsle_modified_inconsistent"]
    merged["delta_prompt_effect_inconsistent"] = merged["accuracy_pct_modified_inconsistent"] - merged["accuracy_pct_original_inconsistent"]
    merged["delta_prompt_effect_consistent"] = merged["accuracy_pct_modified_consistent"] - merged["accuracy_pct_original_consistent"]
    merged["delta_prompt_rmsle_inconsistent"] = merged["mean_rmsle_modified_inconsistent"] - merged["mean_rmsle_original_inconsistent"]
    merged["delta_prompt_rmsle_consistent"] = merged["mean_rmsle_modified_consistent"] - merged["mean_rmsle_original_consistent"]

    return merged.sort_values(
        ["api_source", "model_name", "agent_backend", "module", "equation_difficulty", "model_system", "law_version"]
    ).reset_index(drop=True)


def build_model_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()

    group_cols = ["api_source", "model_name", "agent_backend"]
    summary = (
        comparison_df.groupby(group_cols, dropna=False)
        .agg(
            num_tasks=("law_id", "count"),
            mean_accuracy_pct_original_inconsistent=("accuracy_pct_original_inconsistent", "mean"),
            mean_accuracy_pct_original_consistent=("accuracy_pct_original_consistent", "mean"),
            mean_accuracy_pct_modified_inconsistent=("accuracy_pct_modified_inconsistent", "mean"),
            mean_accuracy_pct_modified_consistent=("accuracy_pct_modified_consistent", "mean"),
            mean_rmsle_original_inconsistent=("mean_rmsle_original_inconsistent", "mean"),
            mean_rmsle_original_consistent=("mean_rmsle_original_consistent", "mean"),
            mean_rmsle_modified_inconsistent=("mean_rmsle_modified_inconsistent", "mean"),
            mean_rmsle_modified_consistent=("mean_rmsle_modified_consistent", "mean"),
        )
        .reset_index()
    )
    summary["delta_accuracy_pct_original"] = summary["mean_accuracy_pct_original_consistent"] - summary["mean_accuracy_pct_original_inconsistent"]
    summary["delta_accuracy_pct_modified"] = summary["mean_accuracy_pct_modified_consistent"] - summary["mean_accuracy_pct_modified_inconsistent"]
    summary["delta_prompt_effect_inconsistent"] = summary["mean_accuracy_pct_modified_inconsistent"] - summary["mean_accuracy_pct_original_inconsistent"]
    summary["delta_prompt_effect_consistent"] = summary["mean_accuracy_pct_modified_consistent"] - summary["mean_accuracy_pct_original_consistent"]
    return summary.sort_values(["api_source", "model_name", "agent_backend"]).reset_index(drop=True)


def write_markdown_report(output_path: Path, comparison_df: pd.DataFrame, model_summary_df: pd.DataFrame, run_tags: Dict[str, str]) -> None:
    lines: List[str] = [
        "# Prompt and Consistency Comparison",
        "",
        f"- Original inconsistent run tag: `{run_tags['original_inconsistent']}`",
        f"- Original consistent run tag: `{run_tags['original_consistent']}`",
        f"- Modified inconsistent run tag: `{run_tags['modified_inconsistent']}`",
        f"- Modified consistent run tag: `{run_tags['modified_consistent']}`",
        f"- Task-level rows: `{len(comparison_df)}`",
        "",
        "## Model Summary",
        "",
        dataframe_to_markdown(model_summary_df),
        "",
        "## Task-Level Comparison",
        "",
        dataframe_to_markdown(comparison_df.head(100)),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_reports(result_dir: Path, output_dir: Path, run_tags: Dict[str, str]):
    summaries = {
        "original_inconsistent": _load_config_summary(result_dir, run_tags["original_inconsistent"]),
        "original_consistent": _load_config_summary(result_dir, run_tags["original_consistent"]),
        "modified_inconsistent": _load_config_summary(result_dir, run_tags["modified_inconsistent"]),
        "modified_consistent": _load_config_summary(result_dir, run_tags["modified_consistent"]),
    }

    run_tables = {
        "original_inconsistent": _prepare_run_table(summaries["original_inconsistent"], "original_inconsistent", "original", False),
        "original_consistent": _prepare_run_table(summaries["original_consistent"], "original_consistent", "original", True),
        "modified_inconsistent": _prepare_run_table(summaries["modified_inconsistent"], "modified_inconsistent", "modified", False),
        "modified_consistent": _prepare_run_table(summaries["modified_consistent"], "modified_consistent", "modified", True),
    }

    comparison_df = build_four_way_comparison(run_tables)
    model_summary_df = build_model_summary(comparison_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "prompt_consistency_comparison.csv", index=False, encoding="utf-8")
    model_summary_df.to_csv(output_dir / "prompt_consistency_model_summary.csv", index=False, encoding="utf-8")
    (output_dir / "prompt_consistency_comparison.md").write_text(dataframe_to_markdown(comparison_df), encoding="utf-8")
    (output_dir / "prompt_consistency_model_summary.md").write_text(dataframe_to_markdown(model_summary_df), encoding="utf-8")
    write_markdown_report(output_dir / "prompt_consistency_report.md", comparison_df, model_summary_df, run_tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a four-way comparison across original/modified and inconsistent/consistent runs.")
    parser.add_argument("--result_dir", default="evaluation_results", help="Directory containing evaluation results.")
    parser.add_argument("--output_dir", required=True, help="Directory where the comparison outputs should be written.")
    parser.add_argument("--original_inconsistent_run_tag", required=True)
    parser.add_argument("--original_consistent_run_tag", required=True)
    parser.add_argument("--modified_inconsistent_run_tag", required=True)
    parser.add_argument("--modified_consistent_run_tag", required=True)
    args = parser.parse_args()

    generate_reports(
        result_dir=Path(args.result_dir),
        output_dir=Path(args.output_dir),
        run_tags={
            "original_inconsistent": args.original_inconsistent_run_tag,
            "original_consistent": args.original_consistent_run_tag,
            "modified_inconsistent": args.modified_inconsistent_run_tag,
            "modified_consistent": args.modified_consistent_run_tag,
        },
    )
