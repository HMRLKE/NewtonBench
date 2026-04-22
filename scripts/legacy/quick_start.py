import subprocess
import sys


def main() -> int:
    print("[DEPRECATED] quick_start.py helyett hasznald a run_pipeline.py --preset quick parancsot.")
    command = ["python", "run_pipeline.py", "--preset", "quick", "--model_name", "gpt41mini"]
    print("Atiranyitas erre:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
