import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def read_models_from_file(models_file: Path) -> List[str]:
    if not models_file.exists():
        return []
    models: List[str] = []
    with models_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    return models


def parse_experiment_name(experiment_name: str) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "noise_level": np.nan,
        "prompt_set": "original",
        "consistency": False,
        "file_version": "v_unknown",
    }

    noise_match = re.search(r'noise([0-9]+(?:_[0-9]+)*)', experiment_name)
    if noise_match:
        metadata["noise_level"] = float(noise_match.group(1).replace("_", "."))

    if "_prompt_modified_" in experiment_name:
        metadata["prompt_set"] = "modified"

    if "_inconsistent_" in experiment_name:
        metadata["consistency"] = False
    elif "_consistent_" in experiment_name:
        metadata["consistency"] = True

    version_match = re.search(r'_v(\d+)$', experiment_name)
    if version_match:
        metadata["file_version"] = f"v{version_match.group(1)}"

    return metadata


def iter_experiment_dirs(result_dir: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(result_dir):
        if "aggregated_results.json" in files:
            yield Path(root)


def load_experiment_config(experiment_dir: Path, result_dir: Path) -> Optional[Dict[str, object]]:
    aggregated_path = experiment_dir / "aggregated_results.json"
    try:
        with aggregated_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    config = payload.get("config", {}).copy()
    rel_parts = experiment_dir.relative_to(result_dir).parts
    if len(rel_parts) < 6:
        return None

    model_name, module_name, agent_backend, difficulty, law_version, experiment_name = rel_parts[:6]
    parsed_name = parse_experiment_name(experiment_name)

    config.setdefault("model_name", model_name)
    config.setdefault("module", module_name)
    config.setdefault("Agent backend", agent_backend)
    config.setdefault("equation_difficulty", difficulty)
    config.setdefault("law_version", law_version)
    config.setdefault("experiment_name", experiment_name)
    config.setdefault("noise_level", parsed_name["noise_level"])
    config.setdefault("prompt_set", parsed_name["prompt_set"])
    config.setdefault("consistency", parsed_name["consistency"])
    config.setdefault("run_tag", None)
    config["experiment_uid"] = str(experiment_dir.relative_to(result_dir)).replace("\\", "/")
    config["file_version"] = parsed_name["file_version"]
    return config


def collect_trial_rows(
    result_dir: Path,
    model_name: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for experiment_dir in iter_experiment_dirs(result_dir):
        config = load_experiment_config(experiment_dir, result_dir)
        if config is None:
            continue
        if model_name and config.get("model_name") != model_name:
            continue
        if run_tag is not None and config.get("run_tag") != run_tag:
            continue

        trials_dir = experiment_dir / "trials"
        if not trials_dir.is_dir():
            continue

        for trial_file in sorted(trials_dir.glob("trial*.json")):
            try:
                with trial_file.open("r", encoding="utf-8") as f:
                    trial = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            evaluation = trial.get("evaluation", {})
            row = {
                "trial_uid": str(trial_file.relative_to(result_dir)).replace("\\", "/"),
                "experiment_uid": config["experiment_uid"],
                "file_version": config["file_version"],
                "run_tag": config.get("run_tag"),
                "model_name": trial.get("model_name", config.get("model_name")),
                "module": trial.get("module_name", config.get("module")),
                "noise_level": trial.get("noise_level", config.get("noise_level")),
                "equation_difficulty": trial.get("equation_difficulty", config.get("equation_difficulty")),
                "model_system": trial.get("model_system", config.get("model_system")),
                "law_version": trial.get("law_version", config.get("law_version")),
                "agent_backend": trial.get("agent_backend", config.get("Agent backend")),
                "prompt_set": trial.get("prompt_set", config.get("prompt_set")),
                "consistency": bool(trial.get("consistency", config.get("consistency", False))),
                "trial_id": trial.get("trial_id"),
                "status": trial.get("status", "unknown"),
                "top_level_error": trial.get("error"),
                "evaluation_error": evaluation.get("error"),
                "symbolic_equivalent": evaluation.get("symbolic_equivalent"),
                "rmsle": evaluation.get("rmsle"),
                "exact_accuracy": evaluation.get("exact_accuracy"),
                "rounds": trial.get("rounds"),
                "experiments": trial.get("num_experiments"),
                "total_tokens": trial.get("total_tokens"),
                "trial_success": (
                    not trial_file.name.endswith("_fail.json")
                    and trial.get("error") in (None, "")
                    and evaluation.get("error") in (None, "")
                ),
            }
            rows.append(row)
    return rows


def build_config_summary(trials_df: pd.DataFrame) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()

    group_cols = [
        "run_tag",
        "model_name",
        "module",
        "agent_backend",
        "equation_difficulty",
        "model_system",
        "law_version",
        "noise_level",
        "prompt_set",
        "consistency",
    ]

    summary = (
        trials_df.groupby(group_cols, dropna=False)
        .agg(
            num_trials=("trial_uid", "count"),
            num_successful_trials=("trial_success", "sum"),
            success_rate=("trial_success", "mean"),
            mean_exact_accuracy=("exact_accuracy", "mean"),
            std_exact_accuracy=("exact_accuracy", "std"),
            mean_rmsle=("rmsle", "mean"),
            std_rmsle=("rmsle", "std"),
            avg_rounds=("rounds", "mean"),
            avg_experiments=("experiments", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            source_experiment_count=("experiment_uid", pd.Series.nunique),
        )
        .reset_index()
    )

    for col in ["std_exact_accuracy", "std_rmsle"]:
        summary[col] = summary[col].fillna(0.0)
    return summary


def _format_metric(mean_value: float, std_value: float, multiplier: float = 1.0, decimals: int = 1) -> str:
    if pd.isna(mean_value):
        return "N/A"
    mean_fmt = mean_value * multiplier
    std_fmt = std_value * multiplier
    return f"{mean_fmt:.{decimals}f} (±{std_fmt:.3f})"


def build_leaderboard(config_summary_df: pd.DataFrame) -> pd.DataFrame:
    if config_summary_df.empty:
        return pd.DataFrame()

    difficulties = ["easy", "medium", "hard"]
    systems = ["vanilla_equation", "simple_system", "complex_system"]
    group_cols = ["run_tag", "model_name", "agent_backend", "prompt_set", "consistency"]
    leaderboard_rows: List[Dict[str, object]] = []

    for keys, group in config_summary_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for system in systems:
            for difficulty in difficulties:
                mask = (
                    (group["model_system"] == system)
                    & (group["equation_difficulty"] == difficulty)
                )
                subset = group[mask]
                col_name = f"acc_{difficulty}_{system}"
                if subset.empty:
                    row[col_name] = "N/A"
                else:
                    row[col_name] = _format_metric(
                        subset["mean_exact_accuracy"].mean(),
                        subset["mean_exact_accuracy"].std(ddof=0) if len(subset) > 1 else 0.0,
                        multiplier=100.0,
                        decimals=1,
                    )

        row["overall_acc"] = _format_metric(
            group["mean_exact_accuracy"].mean(),
            group["mean_exact_accuracy"].std(ddof=0) if len(group) > 1 else 0.0,
            multiplier=100.0,
            decimals=1,
        )
        row["overall_rmsle"] = _format_metric(
            group["mean_rmsle"].mean(),
            group["mean_rmsle"].std(ddof=0) if len(group) > 1 else 0.0,
            multiplier=1.0,
            decimals=4,
        )
        row["overall_success_rate"] = _format_metric(
            group["success_rate"].mean(),
            group["success_rate"].std(ddof=0) if len(group) > 1 else 0.0,
            multiplier=100.0,
            decimals=1,
        )
        row["avg_total_tokens"] = f"{group['avg_total_tokens'].mean():.0f}" if group["avg_total_tokens"].notna().any() else "N/A"
        leaderboard_rows.append(row)

    column_order = group_cols + [
        f"acc_{difficulty}_{system}"
        for system in systems
        for difficulty in difficulties
    ] + ["overall_acc", "overall_rmsle", "overall_success_rate", "avg_total_tokens"]

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    return leaderboard_df.reindex(columns=column_order)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def build_law_accuracy_table(config_summary_df: pd.DataFrame) -> pd.DataFrame:
    if config_summary_df.empty:
        return pd.DataFrame()

    table = config_summary_df.copy()
    table["law_id"] = (
        table["module"].astype(str)
        + "/"
        + table["equation_difficulty"].astype(str)
        + "/"
        + table["model_system"].astype(str)
        + "/"
        + table["law_version"].astype(str)
    )
    table["accuracy_pct"] = table["mean_exact_accuracy"] * 100.0
    table["success_rate_pct"] = table["success_rate"] * 100.0
    ordered = table[
        [
            "run_tag",
            "model_name",
            "agent_backend",
            "prompt_set",
            "consistency",
            "law_id",
            "module",
            "equation_difficulty",
            "model_system",
            "law_version",
            "accuracy_pct",
            "success_rate_pct",
            "num_trials",
            "num_successful_trials",
            "mean_rmsle",
            "avg_total_tokens",
        ]
    ].sort_values(
        [
            "model_name",
            "agent_backend",
            "module",
            "equation_difficulty",
            "model_system",
            "law_version",
        ]
    )
    return ordered.reset_index(drop=True)


def write_markdown_report(
    output_path: Path,
    trials_df: pd.DataFrame,
    config_summary_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    law_accuracy_df: pd.DataFrame,
    run_tag: Optional[str],
) -> None:
    lines = [
        "# NewtonBench Summary",
        "",
        f"- Run tag: `{run_tag}`" if run_tag else "- Run tag: `all-runs`",
        f"- Trial rows: `{len(trials_df)}`",
        f"- Logical configurations: `{len(config_summary_df)}`",
        "",
        "## Leaderboard",
        "",
        dataframe_to_markdown(leaderboard_df),
        "",
        "## Configuration Summary",
        "",
        dataframe_to_markdown(config_summary_df.head(30)),
        "",
        "## Law Accuracy Table",
        "",
        dataframe_to_markdown(law_accuracy_df.head(50)),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_reports(
    result_dir: Path,
    output_dir: Path,
    model_name: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = collect_trial_rows(result_dir=result_dir, model_name=model_name, run_tag=run_tag)
    trials_df = pd.DataFrame(rows)
    if not trials_df.empty:
        numeric_cols = ["noise_level", "rmsle", "exact_accuracy", "rounds", "experiments", "total_tokens"]
        for col in numeric_cols:
            trials_df[col] = pd.to_numeric(trials_df[col], errors="coerce")
        trials_df["trial_success"] = trials_df["trial_success"].astype(bool)
        trials_df = trials_df.sort_values(["model_name", "module", "agent_backend", "equation_difficulty", "model_system", "law_version", "trial_uid"])

    config_summary_df = build_config_summary(trials_df)
    leaderboard_df = build_leaderboard(config_summary_df)
    law_accuracy_df = build_law_accuracy_table(config_summary_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(output_dir / "results_by_trial.csv", index=False, encoding="utf-8")
    config_summary_df.to_csv(output_dir / "config_summary.csv", index=False, encoding="utf-8")
    leaderboard_df.to_csv(output_dir / "aggregated_trial_summary.csv", index=False, encoding="utf-8")
    law_accuracy_df.to_csv(output_dir / "law_accuracy_summary.csv", index=False, encoding="utf-8")
    (output_dir / "law_accuracy_summary.md").write_text(dataframe_to_markdown(law_accuracy_df), encoding="utf-8")
    write_markdown_report(output_dir / "summary_report.md", trials_df, config_summary_df, leaderboard_df, law_accuracy_df, run_tag)
    return trials_df, config_summary_df, leaderboard_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize NewtonBench experiment outputs into trial, config, and leaderboard reports.")
    parser.add_argument("-m", "--model_name", default="all", help="Optional model filter.")
    parser.add_argument("-d", "--result_dir", default="evaluation_results", help="Directory containing the evaluation results.")
    parser.add_argument("-o", "--output_dir", default="result_analysis", help="Directory where the generated report files should be written.")
    parser.add_argument("--models_file", type=str, default="configs/models.txt", help="Path to newline-delimited models list when --model_name=all.")
    parser.add_argument("--run_tag", default=None, help="Only include results that were written with this run tag.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)

    if args.model_name == "all":
        generate_reports(result_dir=result_dir, output_dir=output_dir, run_tag=args.run_tag)
    else:
        generate_reports(result_dir=result_dir, output_dir=output_dir, model_name=args.model_name, run_tag=args.run_tag)
