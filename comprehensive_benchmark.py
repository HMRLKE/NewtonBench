import subprocess
import os

MODULES = [f"m{i}_{name}" for i, name in enumerate([
    "gravity", "coulomb_force", "magnetic_force", "fourier_law", "snell_law",
    "radioactive_decay", "underdamped_harmonic", "malus_law", "sound_speed",
    "hooke_law", "be_distribution", "heat_transfer"
])]

MODEL = "gpt-4o-mini"
TRIALS = 1  # Standard for large benchmarks, increase for more stability
DIFFICULTIES = ["easy"]
VERSIONS = ["v_unchanged", "v0", "v1", "v2"]
NOISES = [0.0, 0.1]
PROMPT_SETS = ["original", "modified"]
CONSISTENCIES = [True, False]

def run_benchmark():
    total = len(MODULES) * len(VERSIONS) * len(NOISES) * len(PROMPT_SETS) * len(CONSISTENCIES)
    current = 0
    
    print(f"Starting comprehensive benchmark: {total} experiments planned.")
    
    for module in MODULES:
        for version in VERSIONS:
            for noise in NOISES:
                for prompt in PROMPT_SETS:
                    for consistency in CONSISTENCIES:
                        current += 1
                        print(f"\n[{current}/{total}] Running: {module} | {version} | noise:{noise} | prompt:{prompt} | consistency:{consistency}")
                        
                        cmd = [
                            "python", "run_experiments.py",
                            "--module", module,
                            "--model_name", MODEL,
                            "--trials", str(TRIALS),
                            "--law_version", version,
                            "--noise", str(noise),
                            "--prompt_set", prompt
                        ]
                        
                        if consistency:
                            cmd.append("--consistency")
                            
                        # Run the experiment
                        try:
                            # We don't use check=True to allow benchmark to continue if one trial fails
                            subprocess.run(cmd)
                        except KeyboardInterrupt:
                            print("Benchmark aborted by user.")
                            return
                        except Exception as e:
                            print(f"Error running experiment: {e}")

    print("\nComprehensive benchmark finished.")
    print("Run 'python generate_benchmark_report.py' to see results.")

if __name__ == "__main__":
    run_benchmark()
