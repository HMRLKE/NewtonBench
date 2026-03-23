import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def discover_modules(modules_root: Path) -> List[str]:
    modules = [p.name for p in modules_root.iterdir() if p.is_dir() and p.name.startswith("m")]
    return sorted(modules)


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


def sanitize_run_tag(run_tag: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in run_tag.strip())
    return safe or datetime.now().strftime("run-%Y%m%d-%H%M%S")


def timestamped_run_tag(prefix: str) -> str:
    return sanitize_run_tag(f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def stream_subprocess(command: List[str], workdir: Path, log_handle) -> int:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        log_handle.write(line)
    return process.wait()


def build_quick_command(args, run_tag: str) -> List[str]:
    command = [
        "python",
        "run_experiments.py",
        "--module",
        args.module or "m0_gravity",
        "--model_name",
        args.model_name,
        "--agent_backend",
        args.agent_backend or "vanilla_agent",
        "--equation_difficulty",
        args.equation_difficulty or "easy",
        "--model_system",
        args.model_system or "vanilla_equation",
        "--law_version",
        args.law_version or "v0",
        "--trials",
        str(args.trials),
        "--noise",
        str(args.noise),
        "--prompt_set",
        args.prompt_set,
        "--run_tag",
        run_tag,
    ]
    if args.consistency:
        command.append("--consistency")
    if args.dashboard or args.prompt_set == "modified":
        command.append("--dashboard")
    return command


def build_benchmark_commands(args, repo_root: Path, run_tag: str) -> List[List[str]]:
    models = [args.model_name] if args.model_name else read_models_from_file(repo_root / args.models_file)
    if not models:
        raise RuntimeError("No models resolved. Provide --model_name or populate configs/models.txt.")

    modules = [args.module] if args.module else [None]
    backends = [args.agent_backend] if args.agent_backend else ["vanilla_agent", "code_assisted_agent"]
    commands: List[List[str]] = []
    for model_name in models:
        for backend in backends:
            for module in modules:
                command = [
                    "python",
                    "run_all_evaluations.py",
                    "--model_name",
                    model_name,
                    "--agent_backend",
                    backend,
                    "--noise",
                    str(args.noise),
                    "--trials_per_law",
                    str(args.trials),
                    "--prompt_set",
                    args.prompt_set,
                    "--run_tag",
                    run_tag,
                    "--no_prompt",
                ]
                if module:
                    command.extend(["--module", module])
                if args.equation_difficulty:
                    command.extend(["--equation_difficulty", args.equation_difficulty])
                if args.model_system:
                    command.extend(["--model_system", args.model_system])
                if args.consistency:
                    command.append("--consistency")
                if args.include_unchanged:
                    command.append("--include_unchanged")
                if args.dashboard or args.prompt_set == "modified":
                    command.append("--dashboard")
                commands.append(command)
    return commands


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_results_index(run_dir: Path, report_dir: Path, run_tag: str) -> None:
    content = f"""# Results Index

Run tag: {run_tag}

Most useful files:
- Law-by-law accuracy table: {report_dir / 'law_accuracy_summary.csv'}
- Law-by-law accuracy table (Markdown): {report_dir / 'law_accuracy_summary.md'}
- Configuration summary: {report_dir / 'config_summary.csv'}
- Leaderboard summary: {report_dir / 'aggregated_trial_summary.csv'}
- Human-readable report: {report_dir / 'summary_report.md'}
- Raw trial rows: {report_dir / 'results_by_trial.csv'}
- Full console log: {run_dir / 'pipeline.log'}
"""
    (run_dir / "RESULTS_INDEX.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command NewtonBench pipeline with automatic logs and report files.")
    parser.add_argument("--preset", choices=["quick", "benchmark"], required=True, help="Quick smoke test or full benchmark pipeline.")
    parser.add_argument("--model_name", default="", help="Model to evaluate. For benchmark, omit to use configs/models.txt.")
    parser.add_argument("--models_file", default="configs/models.txt", help="Model list used when --model_name is omitted for benchmark mode.")
    parser.add_argument("--module", default=None, help="Optional single module filter. Default: quick uses m0_gravity, benchmark uses all modules.")
    parser.add_argument("--agent_backend", default=None, help="Optional backend filter. Default: quick uses vanilla_agent, benchmark uses both backends.")
    parser.add_argument("--equation_difficulty", default=None, choices=["easy", "medium", "hard"], help="Optional difficulty filter.")
    parser.add_argument("--model_system", default=None, choices=["vanilla_equation", "simple_system", "complex_system"], help="Optional system filter.")
    parser.add_argument("--law_version", default=None, help="Quick mode only. Default: v0.")
    parser.add_argument("--trials", type=int, default=None, help="Trials per configuration. Default: quick=1, benchmark=4.")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level.")
    parser.add_argument("--prompt_set", default="original", choices=["original", "modified"], help="Prompt set.")
    parser.add_argument("--consistency", action="store_true", help="Use consistent-law variants where supported.")
    parser.add_argument("--include_unchanged", action="store_true", help="Include v_unchanged control laws in benchmark mode.")
    parser.add_argument("--run_tag", default=None, help="Optional logical run identifier used for filtering reports.")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard updates during execution.")
    parser.add_argument("--dry_run", action="store_true", help="Print the commands and exit without running them.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    if args.trials is None:
        args.trials = 1 if args.preset == "quick" else 4

    if args.preset == "quick" and not args.model_name:
        args.model_name = "gpt41mini"

    run_tag = sanitize_run_tag(args.run_tag) if args.run_tag else timestamped_run_tag(args.preset)
    run_dir = repo_root / "outputs" / "pipeline_runs" / run_tag
    report_dir = run_dir / "report"
    run_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "pipeline_runs").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    manifest_path = run_dir / "manifest.json"

    if args.preset == "quick":
        commands = [build_quick_command(args, run_tag)]
    else:
        commands = build_benchmark_commands(args, repo_root, run_tag)

    manifest = {
        "run_tag": run_tag,
        "preset": args.preset,
        "started_at": datetime.now().isoformat(),
        "commands": commands,
        "log_path": str(log_path),
        "report_dir": str(report_dir),
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        write_manifest(manifest_path, manifest)
        return 0

    with log_path.open("w", encoding="utf-8") as log_handle:
        for index, command in enumerate(commands, start=1):
            header = f"\n{'=' * 100}\n[{index}/{len(commands)}] {' '.join(command)}\n{'=' * 100}\n"
            print(header, end="")
            log_handle.write(header)
            return_code = stream_subprocess(command, repo_root, log_handle)
            if return_code != 0:
                manifest["failed_command"] = command
                manifest["return_code"] = return_code
                manifest["finished_at"] = datetime.now().isoformat()
                write_manifest(manifest_path, manifest)
                print(f"\nPipeline stopped because one command failed with exit code {return_code}.")
                return return_code

        report_command = [
            "python",
            "result_analysis/summarize_results.py",
            "--result_dir",
            "evaluation_results",
            "--output_dir",
            str(report_dir),
            "--run_tag",
            run_tag,
        ]
        report_header = f"\n{'=' * 100}\n[report] {' '.join(report_command)}\n{'=' * 100}\n"
        print(report_header, end="")
        log_handle.write(report_header)
        report_return_code = stream_subprocess(report_command, repo_root, log_handle)
        if report_return_code != 0:
            manifest["failed_command"] = report_command
            manifest["return_code"] = report_return_code
            manifest["finished_at"] = datetime.now().isoformat()
            write_manifest(manifest_path, manifest)
            print(f"\nReport generation failed with exit code {report_return_code}.")
            return report_return_code

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["return_code"] = 0
    write_manifest(manifest_path, manifest)
    write_results_index(run_dir, report_dir, run_tag)
    (repo_root / "outputs" / "pipeline_runs" / "LATEST_RUN.txt").write_text(run_tag + "\n", encoding="utf-8")
    print(f"\nRun completed.")
    print(f"Run folder: {run_dir}")
    print(f"Law-by-law results: {report_dir / 'law_accuracy_summary.csv'}")
    print(f"Readable summary: {report_dir / 'summary_report.md'}")
    print(f"Full log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
