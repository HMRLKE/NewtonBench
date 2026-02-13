import subprocess
import sys
import time
import os
import signal

def run_command(command):
    """Runs a command and prints its output in real-time."""
    print(f"\n{'='*80}\nRunning command: {' '.join(command)}\n{'='*80}")
    try:
        result = subprocess.run(command, check=True, text=True, encoding='utf-8')
        return result.returncode
    except FileNotFoundError:
        print(f"Error: Command not found. Please ensure '{command[0]}' is in your PATH.")
        return -1
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command exited with non-zero status {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

def main():
    """Runs the quick start experiments."""
    print("--- Starting Quick Start ---")

    # Start Dashboard Server
    print("\n--- Starting Human Eval Dashboard ---")
    dashboard_server = subprocess.Popen(
        ["python", "mini_scientist/server.py", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("Dashboard serving at http://localhost:8000/mini_scientist/dashboard/index.html")
    time.sleep(2) # Give server a moment to start

    # Command 1: Run vanilla agent with gpt41mini, equation difficulty as easy and model system as vanilla equation
    command1 = [
        "python", "run_experiments.py",
        "--model_name", "gpt41mini",
        "-b", "vanilla_agent", 
        "-t", "1",
        "--dashboard"
    ]
    
    # Command 2: Run code-assisted agent with gpt41mini, equation difficulty as easy and model system as vanilla equation
    command2 = [
        "python", "run_experiments.py", 
        "--model_name", "gpt41mini",
        "-b", "code_assisted_agent", 
        "-t", "1",
        "--dashboard"
    ]

    run_command(command1)
    run_command(command2)

    print("\n--- Quick Start Finished ---")
    print("\nDashboard is still running. Press Ctrl+C to stop the dashboard server.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Dashboard Server...")
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(dashboard_server.pid)])
        else:
            dashboard_server.terminate()

if __name__ == "__main__":
    main()
