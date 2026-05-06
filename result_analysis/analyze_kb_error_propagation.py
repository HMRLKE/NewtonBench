import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.consistency import get_related_modules


def _read_paper_results(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.rglob("paper_results.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df.insert(0, "source_file", str(path))
        df.insert(1, "source_run_dir", str(path.parent))
        df.insert(2, "source_row_index", range(len(df)))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _episode_order(df: pd.DataFrame) -> pd.Series:
    if "source_row_index" in df.columns:
        return df["source_row_index"].fillna(0).astype(int)
    if "scientist_index" in df.columns:
        return df["scientist_index"].fillna(0).astype(int)
    return pd.Series(range(len(df)), index=df.index)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def find_propagation_events(paper_results: pd.DataFrame) -> pd.DataFrame:
    if paper_results.empty:
        return pd.DataFrame()

    events: List[Dict[str, object]] = []
    group_cols = [
        col
        for col in [
            "source_run_dir",
            "run_tag",
            "scenario_id",
            "scientist_model_name",
            "scientist_api_source",
            "reviewer_model_name",
            "reviewer_api_source",
        ]
        if col in paper_results.columns
    ]

    for group_keys, group in paper_results.groupby(group_cols, dropna=False):
        group = group.copy()
        group["_order"] = _episode_order(group)
        group["accepted_bool"] = _bool_series(group["accepted"]) if "accepted" in group.columns else False
        group["exact_accuracy"] = pd.to_numeric(group.get("exact_accuracy"), errors="coerce")
        group = group.sort_values(["_order"]).reset_index(drop=True)
        accepted_wrong = group[group["accepted_bool"] & (group["exact_accuracy"] != 1.0)]
        if accepted_wrong.empty:
            continue

        group_key_dict = dict(zip(group_cols, group_keys if isinstance(group_keys, tuple) else (group_keys,)))
        for wrong_pos, wrong_row in accepted_wrong.iterrows():
            related_modules = set(get_related_modules(str(wrong_row["module"])))
            if not related_modules:
                continue
            later = group.iloc[wrong_pos + 1 :]
            later_related = later[later["module"].isin(related_modules)]
            for _, later_row in later_related.iterrows():
                events.append(
                    {
                        **group_key_dict,
                        "early_module": wrong_row.get("module"),
                        "early_law_version": wrong_row.get("law_version"),
                        "early_equation_difficulty": wrong_row.get("equation_difficulty"),
                        "early_model_system": wrong_row.get("model_system"),
                        "early_exact_accuracy": wrong_row.get("exact_accuracy"),
                        "early_accepted": wrong_row.get("accepted_bool"),
                        "early_equation": wrong_row.get("equation"),
                        "later_module": later_row.get("module"),
                        "later_law_version": later_row.get("law_version"),
                        "later_equation_difficulty": later_row.get("equation_difficulty"),
                        "later_model_system": later_row.get("model_system"),
                        "later_exact_accuracy": later_row.get("exact_accuracy"),
                        "later_accepted": later_row.get("accepted_bool"),
                        "later_false_accept": bool(later_row.get("accepted_bool")) and later_row.get("exact_accuracy") != 1.0,
                        "later_equation": later_row.get("equation"),
                    }
                )

    return pd.DataFrame(events)


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            [
                {
                    "num_potential_propagation_events": 0,
                    "later_incorrect_rate_pct": 0.0,
                    "later_false_accept_rate_pct": 0.0,
                }
            ]
        )

    later_incorrect = (events["later_exact_accuracy"] != 1.0).mean() * 100.0
    later_false_accept = events["later_false_accept"].mean() * 100.0
    return pd.DataFrame(
        [
            {
                "num_potential_propagation_events": len(events),
                "later_incorrect_rate_pct": float(later_incorrect),
                "later_false_accept_rate_pct": float(later_false_accept),
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze whether accepted wrong KB entries precede later related failures.")
    parser.add_argument("--input_root", default="outputs/hypothesis_runs")
    parser.add_argument("--output_dir", default="outputs/hypothesis_runs/h3-log-analysis")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_results = _read_paper_results(input_root)
    events = find_propagation_events(paper_results)
    summary = build_summary(events)

    events.to_csv(output_dir / "h3_observed_error_propagation_events.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "h3_observed_error_propagation_summary.csv", index=False, encoding="utf-8")
    try:
        (output_dir / "h3_observed_error_propagation_events.md").write_text(events.to_markdown(index=False), encoding="utf-8")
        (output_dir / "h3_observed_error_propagation_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    except Exception:
        pass

    print(f"Output directory: {output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
