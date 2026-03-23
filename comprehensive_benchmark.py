import subprocess


def main() -> int:
    print("[DEPRECATED] comprehensive_benchmark.py helyett hasznald a run_pipeline.py --preset benchmark parancsot.")
    print("A regi script sajat, nehezen kovetheto kombinatorikus sweepet futtatott.")
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
    print("Javasolt kiindulo parancs:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
