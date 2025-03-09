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

import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

def plot_pid_values(log_number: str):
    """Plot PID values from angle and lateral CSV files with PID parameters"""
    # Set up paths
    log_dir = "logs"
    angle_file = os.path.join(log_dir, f"{log_number}_pid_values_angle.csv")
    lateral_file = os.path.join(log_dir, f"{log_number}_pid_values_lateral.csv")
    params_file = os.path.join(log_dir, f"{log_number}_pid_params_drone.csv")
    
    # Read CSV files
    angle_df = pd.read_csv(angle_file)
    lateral_df = pd.read_csv(lateral_file)
    
    # Read PID parameters
    pid_params = pd.read_csv(params_file)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('PID Controller Performance')
    
    # Plot angle error
    ax1.plot(angle_df['timestep'], angle_df['epsilon_angle'], 'b-', label='Angle Error')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Angle Error (rad)')
    ax1.grid(True)
    ax1.legend()
    
    # Add angle PID parameters text
    angle_params_text = f'Angle PID:\nKp = {pid_params["Kp_angle"].iloc[0]:.4f}\nKd = {pid_params["Kd_angle"].iloc[0]:.4f}\nKi = {pid_params["Ki_angle"].iloc[0]:.4f}'
    ax1.text(0.02, 0.98, angle_params_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot lateral error
    ax2.plot(lateral_df['timestep'], lateral_df['epsilon_lateral'], 'r-', label='Lateral Error')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Lateral Error (pixels)')
    ax2.grid(True)
    ax2.legend()
    
    # Add lateral PID parameters text
    lateral_params_text = f'Lateral PID:\nKp = {pid_params["Kp_distance"].iloc[0]:.4f}\nKd = {pid_params["Kd_distance"].iloc[0]:.4f}\nKi = {pid_params["Ki_distance"].iloc[0]:.4f}'
    ax2.text(0.02, 0.98, lateral_params_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{log_number}_pid_performance.png"))
    plt.close(fig)

if __name__ == "__main__":
    success = run_launchers()
    time.sleep(1)
    log_files = glob("logs/*_pid_values_angle.csv")
    if log_files:
        latest_log = max(log_files)
        log_number = latest_log.split('/')[-1].split('_')[0]
        plot_pid_values(log_number)
    else:
        print("No log files found in ./logs directory")
    sys.exit(0 if success else 1)