import argparse
from pathlib import Path
from typing import Tuple

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


def _prepare_side(summary_df: pd.DataFrame, label: str) -> pd.DataFrame:
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
        "model_name",
        "agent_backend",
        "prompt_set",
        "module",
        "equation_difficulty",
        "model_system",
        "law_version",
        "noise_level",
        "law_id",
    ]
    metric_cols = {
        "run_tag": f"run_tag_{label}",
        "consistency": f"consistency_{label}",
        "num_trials": f"num_trials_{label}",
        "num_successful_trials": f"num_successful_trials_{label}",
        "success_rate_pct": f"success_rate_pct_{label}",
        "accuracy_pct": f"accuracy_pct_{label}",
        "mean_rmsle": f"mean_rmsle_{label}",
        "avg_total_tokens": f"avg_total_tokens_{label}",
    }
    selected_cols = key_cols + list(metric_cols.keys())
    renamed = table[selected_cols].rename(columns=metric_cols)
    return renamed


def build_consistency_comparison(
    inconsistent_df: pd.DataFrame,
    consistent_df: pd.DataFrame,
    inconsistent_run_tag: str,
    consistent_run_tag: str,
) -> pd.DataFrame:
    left = _prepare_side(inconsistent_df, "inconsistent")
    right = _prepare_side(consistent_df, "consistent")

    key_cols = [
        "model_name",
        "agent_backend",
        "prompt_set",
        "module",
        "equation_difficulty",
        "model_system",
        "law_version",
        "noise_level",
        "law_id",
    ]

    if left.empty and right.empty:
        return pd.DataFrame()
    if left.empty:
        merged = right.copy()
    elif right.empty:
        merged = left.copy()
    else:
        merged = left.merge(right, on=key_cols, how="outer")

    merged["expected_inconsistent_run_tag"] = inconsistent_run_tag
    merged["expected_consistent_run_tag"] = consistent_run_tag
    merged["delta_accuracy_pct"] = merged["accuracy_pct_consistent"] - merged["accuracy_pct_inconsistent"]
    merged["delta_success_rate_pct"] = merged["success_rate_pct_consistent"] - merged["success_rate_pct_inconsistent"]
    merged["delta_rmsle"] = merged["mean_rmsle_consistent"] - merged["mean_rmsle_inconsistent"]
    merged["delta_avg_total_tokens"] = merged["avg_total_tokens_consistent"] - merged["avg_total_tokens_inconsistent"]

    merged = merged.sort_values(
        ["model_name", "agent_backend", "module", "equation_difficulty", "model_system", "law_version"]
    ).reset_index(drop=True)
    return merged


def build_model_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()

    group_cols = ["model_name", "agent_backend", "prompt_set"]
    summary = (
        comparison_df.groupby(group_cols, dropna=False)
        .agg(
            num_tasks=("law_id", "count"),
            mean_accuracy_pct_inconsistent=("accuracy_pct_inconsistent", "mean"),
            mean_accuracy_pct_consistent=("accuracy_pct_consistent", "mean"),
            mean_success_rate_pct_inconsistent=("success_rate_pct_inconsistent", "mean"),
            mean_success_rate_pct_consistent=("success_rate_pct_consistent", "mean"),
            mean_rmsle_inconsistent=("mean_rmsle_inconsistent", "mean"),
            mean_rmsle_consistent=("mean_rmsle_consistent", "mean"),
            mean_avg_total_tokens_inconsistent=("avg_total_tokens_inconsistent", "mean"),
            mean_avg_total_tokens_consistent=("avg_total_tokens_consistent", "mean"),
        )
        .reset_index()
    )
    summary["delta_accuracy_pct"] = summary["mean_accuracy_pct_consistent"] - summary["mean_accuracy_pct_inconsistent"]
    summary["delta_success_rate_pct"] = summary["mean_success_rate_pct_consistent"] - summary["mean_success_rate_pct_inconsistent"]
    summary["delta_rmsle"] = summary["mean_rmsle_consistent"] - summary["mean_rmsle_inconsistent"]
    summary["delta_avg_total_tokens"] = summary["mean_avg_total_tokens_consistent"] - summary["mean_avg_total_tokens_inconsistent"]
    return summary.sort_values(["model_name", "agent_backend", "prompt_set"]).reset_index(drop=True)


def write_markdown_report(
    output_path: Path,
    comparison_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    inconsistent_run_tag: str,
    consistent_run_tag: str,
) -> None:
    lines = [
        "# Consistency Comparison",
        "",
        f"- Inconsistent run tag: `{inconsistent_run_tag}`",
        f"- Consistent run tag: `{consistent_run_tag}`",
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


def generate_consistency_reports(
    result_dir: Path,
    output_dir: Path,
    inconsistent_run_tag: str,
    consistent_run_tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inconsistent_summary = _load_config_summary(result_dir, inconsistent_run_tag)
    consistent_summary = _load_config_summary(result_dir, consistent_run_tag)
    comparison_df = build_consistency_comparison(
        inconsistent_summary,
        consistent_summary,
        inconsistent_run_tag=inconsistent_run_tag,
        consistent_run_tag=consistent_run_tag,
    )
    model_summary_df = build_model_summary(comparison_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "consistency_comparison.csv", index=False, encoding="utf-8")
    model_summary_df.to_csv(output_dir / "consistency_model_summary.csv", index=False, encoding="utf-8")
    (output_dir / "consistency_comparison.md").write_text(dataframe_to_markdown(comparison_df), encoding="utf-8")
    (output_dir / "consistency_model_summary.md").write_text(dataframe_to_markdown(model_summary_df), encoding="utf-8")
    write_markdown_report(
        output_dir / "consistency_report.md",
        comparison_df,
        model_summary_df,
        inconsistent_run_tag=inconsistent_run_tag,
        consistent_run_tag=consistent_run_tag,
    )
    return comparison_df, model_summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare inconsistent vs consistent NewtonBench runs side by side.")
    parser.add_argument("--result_dir", default="evaluation_results", help="Directory containing evaluation results.")
    parser.add_argument("--output_dir", required=True, help="Directory where the comparison report files should be written.")
    parser.add_argument("--inconsistent_run_tag", required=True, help="Run tag for the inconsistent sweep.")
    parser.add_argument("--consistent_run_tag", required=True, help="Run tag for the consistent sweep.")
    args = parser.parse_args()

    generate_consistency_reports(
        result_dir=Path(args.result_dir),
        output_dir=Path(args.output_dir),
        inconsistent_run_tag=args.inconsistent_run_tag,
        consistent_run_tag=args.consistent_run_tag,
    )
