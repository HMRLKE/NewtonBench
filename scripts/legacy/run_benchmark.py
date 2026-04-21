import subprocess


def main() -> int:
    print("[DEPRECATED] run_benchmark.py helyett hasznald a run_pipeline.py --preset benchmark parancsot.")
    command = [
        "python",
        "run_pipeline.py",
        "--preset",
        "benchmark",
        "--model_name",
        "gpt-4o-mini",
        "--equation_difficulty",
        "easy",
        "--trials",
        "1",
    ]
    print("Javasolt helyettesito parancs:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
