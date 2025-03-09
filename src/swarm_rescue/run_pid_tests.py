import subprocess
import sys
import time

def run_launchers():
    launchers = [
        "pid_angle_launcher.py",
        "pid_lateral_launcher.py"
    ]
    
    for launcher in launchers:
        print(f"\n{'='*80}\nRunning {launcher}\n{'='*80}\n")
        try:
            # Run the launcher and capture its output
            process = subprocess.run([sys.executable, launcher], 
                                  cwd="/home/etienne/Desktop/swarm/main/src/swarm_rescue",
                                  check=True)
            
            if process.returncode != 0:
                print(f"Error running {launcher}")
                return False
                
            # Add a small delay between launches
            time.sleep(1)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {launcher}: {e}")
            return False
            
    return True

if __name__ == "__main__":
    success = run_launchers()
    sys.exit(0 if success else 1)