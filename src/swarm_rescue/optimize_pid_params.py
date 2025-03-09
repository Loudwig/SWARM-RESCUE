import subprocess
import sys
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class OptimizationConfig:
    # Parameter ranges to test
    kp_distance_range: List[float] = [0.5, 1.0, 1.5, 2.0]
    kd_distance_range: List[float] = [0.5, 1.0, 1.5, 2.0]
    ki_distance_range: List[float] = [0.0, 0.01, 0.05, 0.1]
    
    # Number of tests per parameter combination
    tests_per_config: int = 1
    
    # Metric to optimize - 'mse', 'mae', or 'max_error'
    metric: str = 'mse'
    
    # Set to True to enable grid search across all parameter combinations
    grid_search: bool = True
    
    # Set to True to enable sequential testing (one parameter at a time)
    sequential_testing: bool = False
    
    # Base parameters (starting point for sequential testing)
    base_kp_distance: float = 1.0
    base_kd_distance: float = 1.0
    base_ki_distance: float = 0.0


def modify_pid_params(kp_distance: float, kd_distance: float, ki_distance: float) -> None:
    """Modifies the PID parameters in the dataclasses_config.py file"""
    filepath = "/home/etienne/Desktop/swarm/main/src/swarm_rescue/solutions/utils/dataclasses_config.py"
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Find PIDParams class definition
    pid_class_start = -1
    for i, line in enumerate(lines):
        if "@dataclass" in line and "class PIDParams" in lines[i+1]:
            pid_class_start = i + 2
            break
    
    if pid_class_start == -1:
        print("Error: Could not find PIDParams class in dataclasses_config.py")
        return
    
    # Update the distance-related PID parameters
    for i in range(pid_class_start, len(lines)):
        if "Kp_distance" in lines[i]:
            lines[i] = f"    Kp_distance : float = {kp_distance}\n"
        elif "Kd_distance" in lines[i]:
            lines[i] = f"    Kd_distance: float = {kd_distance}\n"
        elif "Ki_distance" in lines[i]:
            lines[i] = f"    Ki_distance: float = {ki_distance}\n"
            # Found all parameters, can break
            break
    
    # Write the updated file
    with open(filepath, 'w') as file:
        file.writelines(lines)


def run_pid_tests() -> str:
    """Run the PID test scripts and return the log ID used"""
    log_id = time.strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Run the test script with the specified log ID
        process = subprocess.run(
            [sys.executable, "pid_lateral_launcher.py"], 
            input=log_id.encode(),
            check=True
        )
        
        if process.returncode != 0:
            print(f"Error running PID tests")
            return ""
        
        # Return the log ID used for the test
        return log_id
        
    except subprocess.CalledProcessError as e:
        print(f"Error running PID tests: {e}")
        return ""


def cleanup_log_files(log_id: str, keep_best_logs=False) -> None:
    """Removes log files for a given log_id after they've been analyzed"""
    if not keep_best_logs:  # Only clean up if we don't want to keep the logs
        log_dir = "logs"
        patterns = [
            f"{log_id}_pid_values_lateral.csv",
            f"{log_id}_pid_values_angle.csv",
            f"{log_id}_pid_params_drone.csv"
        ]
        
        for pattern in patterns:
            file_path = os.path.join(log_dir, pattern)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")


def evaluate_performance(log_id: str) -> Dict[str, float]:
    """
    Evaluate the performance of the PID controller using the logs
    Returns a dictionary of metrics
    """
    log_dir = "logs"
    lateral_file = os.path.join(log_dir, f"{log_id}_pid_values_lateral.csv")
    
    # Check if file exists
    if not os.path.exists(lateral_file):
        print(f"Error: Log file {lateral_file} not found")
        return {"mse": float('inf'), "mae": float('inf'), "max_error": float('inf')}
    
    # Read the log file
    lateral_df = pd.read_csv(lateral_file)
    
    # Calculate metrics
    # We want to minimize these error metrics
    abs_errors = np.abs(lateral_df['epsilon_lateral'])
    squared_errors = abs_errors ** 2
    
    metrics = {
        "mse": np.mean(squared_errors),  # Mean Squared Error
        "mae": np.mean(abs_errors),      # Mean Absolute Error
        "max_error": np.max(abs_errors)  # Maximum Absolute Error
    }
    
    return metrics


def optimize_pid():
    """Main function to optimize PID parameters"""
    config = OptimizationConfig()
    results = []
    
    # Display optimization configuration
    print(f"\n{'='*80}")
    print(f"Starting PID Parameter Optimization")
    print(f"{'='*80}")
    print(f"Metric to optimize: {config.metric}")
    
    if config.grid_search:
        # Grid search across all parameter combinations
        print(f"\nPerforming grid search with {len(config.kp_distance_range) * len(config.kd_distance_range) * len(config.ki_distance_range)} combinations")
        
        for kp in config.kp_distance_range:
            for kd in config.kd_distance_range:
                for ki in config.ki_distance_range:
                    # Update PID parameters
                    modify_pid_params(kp, kd, ki)
                    
                    print(f"\n{'-'*60}")
                    print(f"Testing: Kp={kp}, Kd={kd}, Ki={ki}")
                    
                    # Run tests
                    log_id = run_pid_tests()
                    if log_id:
                        # Evaluate performance
                        metrics = evaluate_performance(log_id)
                        
                        result = {
                            'kp_distance': kp,
                            'kd_distance': kd,
                            'ki_distance': ki,
                            'log_id': log_id,
                            **metrics
                        }
                        
                        results.append(result)
                        print(f"Results: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, Max Error={metrics['max_error']:.4f}")
                        best_result = min(results, key=lambda x: x.get(config.metric, float('inf')))
        
                        print(f"Best parameters so far: Kp_distance = {best_result.get('kp_distance', 0)}, Kd_distance = {best_result.get('kd_distance', 0)}, Ki_distance = {best_result.get('ki_distance', 0)}")

                        cleanup_log_files(log_id)
                    
                    # Small delay between tests
                    time.sleep(1)
    
    elif config.sequential_testing:
        # Sequential testing (one parameter at a time)
        kp_best = config.base_kp_distance
        kd_best = config.base_kd_distance
        ki_best = config.base_ki_distance
        best_metric_value = float('inf')
        
        # Test Kp variations
        print("\nOptimizing Kp (proportional gain)...")
        for kp in config.kp_distance_range:
            # Update PID parameters
            modify_pid_params(kp, kd_best, ki_best)
            
            print(f"\n{'-'*60}")
            print(f"Testing: Kp={kp}, Kd={kd_best}, Ki={ki_best}")
            
            # Run tests
            log_id = run_pid_tests()
            if log_id:
                # Evaluate performance
                metrics = evaluate_performance(log_id)
                
                result = {
                    'kp_distance': kp,
                    'kd_distance': kd_best,
                    'ki_distance': ki_best,
                    'log_id': log_id,
                    **metrics
                }
                
                results.append(result)
                print(f"Results: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, Max Error={metrics['max_error']:.4f}")
                
                # Update best if improved
                if metrics[config.metric] < best_metric_value:
                    best_metric_value = metrics[config.metric]
                    kp_best = kp
            
            # Small delay between tests
            time.sleep(1)
        
        # Test Kd variations with best Kp
        print("\nOptimizing Kd (derivative gain)...")
        best_metric_value = float('inf')
        for kd in config.kd_distance_range:
            # Update PID parameters
            modify_pid_params(kp_best, kd, ki_best)
            
            print(f"\n{'-'*60}")
            print(f"Testing: Kp={kp_best}, Kd={kd}, Ki={ki_best}")
            
            # Run tests
            log_id = run_pid_tests()
            if log_id:
                # Evaluate performance
                metrics = evaluate_performance(log_id)
                
                result = {
                    'kp_distance': kp_best,
                    'kd_distance': kd,
                    'ki_distance': ki_best,
                    'log_id': log_id,
                    **metrics
                }
                
                results.append(result)
                print(f"Results: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, Max Error={metrics['max_error']:.4f}")
                
                # Update best if improved
                if metrics[config.metric] < best_metric_value:
                    best_metric_value = metrics[config.metric]
                    kd_best = kd
            
            # Small delay between tests
            time.sleep(1)
        
        # Test Ki variations with best Kp and Kd
        if any(ki != 0 for ki in config.ki_distance_range):
            print("\nOptimizing Ki (integral gain)...")
            best_metric_value = float('inf')
            for ki in config.ki_distance_range:
                # Update PID parameters
                modify_pid_params(kp_best, kd_best, ki)
                
                print(f"\n{'-'*60}")
                print(f"Testing: Kp={kp_best}, Kd={kd_best}, Ki={ki}")
                
                # Run tests
                log_id = run_pid_tests()
                if log_id:
                    # Evaluate performance
                    metrics = evaluate_performance(log_id)
                    
                    result = {
                        'kp_distance': kp_best,
                        'kd_distance': kd_best,
                        'ki_distance': ki,
                        'log_id': log_id,
                        **metrics
                    }
                    
                    results.append(result)
                    print(f"Results: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, Max Error={metrics['max_error']:.4f}")
                    
                    # Update best if improved
                    if metrics[config.metric] < best_metric_value:
                        best_metric_value = metrics[config.metric]
                        ki_best = ki
                
                # Small delay between tests
                time.sleep(1)
    
    # Plot results
    if results:
        # Find best result
        best_result = min(results, key=lambda x: x.get(config.metric, float('inf')))
        
        print(f"\n{'='*80}")
        print(f"Optimization Complete!")
        print(f"{'='*80}")
        print(f"Best parameters found:")
        print(f"Kp_distance = {best_result.get('kp_distance', 0)}")
        print(f"Kd_distance = {best_result.get('kd_distance', 0)}")
        print(f"Ki_distance = {best_result.get('ki_distance', 0)}")
        print(f"MSE = {best_result.get('mse', 0):.4f}")
        print(f"MAE = {best_result.get('mae', 0):.4f}")
        print(f"Max Error = {best_result.get('max_error', 0):.4f}")
        
        # Update to best parameters
        modify_pid_params(
            best_result.get('kp_distance', config.base_kp_distance),
            best_result.get('kd_distance', config.base_kd_distance),
            best_result.get('ki_distance', config.base_ki_distance)
        )
    else:
        print("No valid results were collected during optimization")


if __name__ == "__main__":
    optimize_pid()