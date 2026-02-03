import subprocess
import time
import sys
import os

def main():
    print("=== Mini Scientist System Launcher ===")
    
    # Start Server
    print("Starting Dashboard Server on port 8000...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "mini_scientist.server", "--port", "8000"],
        cwd=os.getcwd(),
        # Hide output to keep terminal clean? Or show it?
        # Let's show it but it might interleave.
        # stdout=subprocess.DEVNULL, 
        # stderr=subprocess.DEVNULL
    )
    
    # Wait a bit for server to start
    time.sleep(2)
    
    # Pass all arguments from this script to the loop script
    loop_args = sys.argv[1:]
    if not loop_args:
        # Default if no args provided
        loop_args = ["--tasks", "m0_gravity,m1_coulomb_force", "--n_samples", "300"]
    
    print(f"\nStarting Scientist Loop with args: {' '.join(loop_args)}")
    print("Open http://localhost:8000 to view progress.\n")
    
    try:
        # Run Loop (blocking call in this thread, but server running in background)
        subprocess.run(
            [sys.executable, "-m", "mini_scientist.loop"] + loop_args,
            cwd=os.getcwd()
        )
        print("\nDiscovery completed. Server is still running.")
        print("You can view results at http://localhost:8000")
        print("Press Ctrl+C to stop the server.")
        
        # Keep the process alive to keep the server process running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        pass
    finally:
        print("\nTerminating Server...")
        server_process.terminate()
        server_process.wait()
        print("Done.")

if __name__ == "__main__":
    main()
