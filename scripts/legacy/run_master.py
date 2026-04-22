import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Deprecated wrapper. Use run_pipeline.py instead.")
    parser.add_argument("-m", "--model_name", type=str, default="", help="Single model to run.")
    parser.add_argument("--models_file", type=str, default="configs/models.txt", help="Models file.")
    parser.add_argument("-p", "--parallel", type=int, default=5, help="Legacy option. No longer used by this wrapper.")
    parser.add_argument("--print_only", action="store_true", help="Map to pipeline dry-run.")
    args = parser.parse_args()

    print("[DEPRECATED] run_master.py helyett hasznald a run_pipeline.py --preset benchmark parancsot.")
    if args.parallel != 5:
        print(f"[INFO] A legacy --parallel={args.parallel} opcio ebben a wrapperben mar nincs hasznalva.")

    command = ["python", "run_pipeline.py", "--preset", "benchmark"]
    if args.model_name:
        command.extend(["--model_name", args.model_name])
    else:
        command.extend(["--models_file", args.models_file])
    if args.print_only:
        command.append("--dry_run")

    print("Atiranyitas erre:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
