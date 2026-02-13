
import os
import subprocess
import time

# List of modules to benchmark
# Exclude 'common' or any non-task directories
MODULES = [
    "m0_gravity",
    "m1_coulomb_force",
    "m2_magnetic_force",
    "m3_fourier_law",
    "m4_snell_law",
    "m5_radioactive_decay",
    "m6_underdamped_harmonic",
    "m7_malus_law",
    "m8_sound_speed",
    "m9_hooke_law",
    "m10_be_distribution",
    "m11_heat_transfer"
]

# Benchmark configuration
MODEL_NAME = "gpt-4o-mini" # Fast model for benchmark
TRIALS = 1 
DIFFICULTY = "easy"
LAW_VERSION = "v0" # Keep it simple for now

def run_benchmark():
    print("Starting NewtonBench Full Benchmark...")
    print(f"Modules: {len(MODULES)}")
    print(f"Configuration: Model={MODEL_NAME}, Trials={TRIALS}, Difficulty={DIFFICULTY}")
    
    # 1. Clear dashboard once at the beginning
    acc_path = os.path.join("accumulation", "global_kg.json")
    if os.path.exists(acc_path):
        os.remove(acc_path)
        print("Cleared previous dashboard data.")

    for i, module in enumerate(MODULES):
        print(f"\n[{i+1}/{len(MODULES)}] Running benchmark for: {module}")
        
        cmd = [
            "python", "run_experiments.py",
            "--module", module,
            "--model_name", MODEL_NAME,
            "-t", str(TRIALS),
            "-d", DIFFICULTY,
            "-l", LAW_VERSION,
            "--dashboard",
            "--keep_history" # Accumulate!
        ]
        
        try:
            # Run correctly using subprocess
            result = subprocess.run(cmd, check=True)
            print(f"Completed {module}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {module}: {e}")
        except KeyboardInterrupt:
            print("Benchmark interrupted by user.")
            break
            
    print("\nBenchmark Verification Complete.")
    print("Check dashboard at: http://localhost:8000/mini_scientist/dashboard/index.html")

if __name__ == "__main__":
    run_benchmark()
