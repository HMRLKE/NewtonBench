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
                            # Use run instead of Popen for synchronous execution
                            subprocess.run(cmd, check=False)
                        except KeyboardInterrupt:
                            print("\n[ABORT] Benchmark interrupted by user.")
                            return
                        except Exception as e:
                            print(f"\n[ERROR] Failed to run trial: {e}")

    print("\n" + "="*50)
    print("NEWTONBENCH COMPREHENSIVE BENCHMARK COMPLETE")
    print("="*50)
    print("All results have been stored in 'evaluation_results/'.")
    print("To generate the final analytical table, run:")
    print("    python generate_benchmark_report.py")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
