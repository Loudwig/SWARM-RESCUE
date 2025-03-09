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
    kp_forward_range: List[float] = np.linspace(0.0, 2.0, 10)  # Adjusted for forward movement
    kd_forward_range: List[float] = np.linspace(0.0, 2.0, 10)    # Adjusted for forward movement
    ki_forward_range: List[float] = [0]
    
    # Number of tests per parameter combination
    tests_per_config: int = 1
    
    # Metric to optimize
    # Options: 'overshoot', 'settling_time', 'stability', 'composite'
    metric: str = 'composite'

    # Weights for composite metric (only used if metric='composite')
    # These weights determine the importance of each factor in the overall evaluation
    overshoot_weight: float = 0.4    # Lower weight for forward overshoot
    settling_time_weight: float = 0.6 # Higher weight for settling time
    stability_weight: float = 1.0
    
    # Set to True to enable grid search across all parameter combinations
    grid_search: bool = True
    
    # Set to True to enable sequential testing (one parameter at a time)
    sequential_testing: bool = False
    
    # Base parameters (starting point for sequential testing)
    base_kp_forward: float = 0.01
    base_kd_forward: float = 0.03
    base_ki_forward: float = 0.0

def calculate_composite_score(metrics: Dict[str, float], config) -> float:
    """
    Calculate a composite score from multiple metrics.
    Lower score is better.
    """
    # Get individual metrics with default values if not present
    overshoot = metrics.get('overshoot', float('inf'))
    settling_time = metrics.get('settling_time', float('inf'))
    stability = metrics.get('stability', float('inf'))
    
    # Calculate weighted composite score
    composite_score = (
        config.overshoot_weight * overshoot +
        config.settling_time_weight * settling_time +
        config.stability_weight * stability
    )
    
    return composite_score


def modify_pid_params(kp_forward: float, kd_forward: float, ki_forward: float) -> None:
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
    
    # Update the forward-related PID parameters
    for i in range(pid_class_start, len(lines)):
        if "Kp_forward" in lines[i]:
            lines[i] = f"    Kp_forward : float = {kp_forward}\n"
        elif "Kd_forward" in lines[i]:
            lines[i] = f"    Kd_forward: float = {kd_forward}\n"
        elif "Ki_forward" in lines[i]:
            lines[i] = f"    Ki_forward: float = {ki_forward}\n"
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
            [sys.executable, "pid_forward_launcher.py"], 
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


def evaluate_performance(log_id: str, config) -> Dict[str, float]:
    """
    Evaluate the performance of the forward PID controller using the logs
    Returns a dictionary of metrics including:
    - Overshoot: Maximum deviation beyond the target
    - Settling Time: Time to reach and stay within 5% of the final value
    - Stability: Measure of oscillation (lower is better)
    """
    log_dir = "logs"
    angle_file = os.path.join(log_dir, f"{log_id}_pid_values_angle.csv")
    
    # Check if file exists
    if not os.path.exists(angle_file):
        print(f"Error: Log file {angle_file} not found")
        return {"overshoot": float('inf'), "settling_time": float('inf'), "stability": float('inf')}
    
    # Read the log file - for forward control, we use the angle error values
    forward_df = pd.read_csv(angle_file)
    
    # Calculate basic metrics
    abs_errors = np.abs(forward_df['epsilon_angle'])  # In the forward case, we're logging the forward error to the angle column
    squared_errors = abs_errors ** 2
    
    metrics = {}
    
    # Advanced metrics
    try:
        errors = forward_df['epsilon_angle'].values
        timesteps = forward_df['timestep'].values
        
        # Calculate overshoot after the initial one
        if len(errors) > 10:  # Ensure we have enough data points
            # Calculate overshoot - for forward movement, this is passing the target point
            max_error = np.max(abs_errors)
            
            # Find the point where the error starts decreasing (approaching target)
            approaching_target = False
            for i in range(1, len(errors)):
                if abs_errors[i] < abs_errors[i-1]:
                    approaching_target = True
                    break
            
            if approaching_target:
                # Once approaching, find if it overshoots (error becomes negative)
                overshoot_idx = None
                original_sign = np.sign(errors[0])
                for i in range(1, len(errors)):
                    if np.sign(errors[i]) != original_sign:
                        overshoot_idx = i
                        break
                
                if overshoot_idx is not None:
                    # Measure maximum deviation after first sign change
                    metrics["overshoot"] = np.max(abs_errors[overshoot_idx:])
                else:
                    # No overshoot detected
                    metrics["overshoot"] = 0.0
            else:
                # Never approached target
                metrics["overshoot"] = max_error
        else:
            # Not enough data points
            metrics["overshoot"] = np.max(abs_errors)
        
        # Calculate settling time
        # Time to reach and stay within a certain percentage of the target value
        # We'll use 5% of the maximum error as our threshold
        if len(errors) > 10:  # Ensure we have enough data points
            threshold = 0.05 * np.max(abs_errors)  # 5% of maximum error
            
            # Find the point where error remains below threshold
            settled = False
            settling_time = len(errors)  # Default to max time if never settles
            
            for i in range(10, len(errors)):  # Start after initial transients
                # Check if error stays below threshold for the next 10 timesteps
                if all(abs_errors[i:min(i+10, len(errors))] < threshold):
                    settled = True
                    settling_time = timesteps[i] - timesteps[0]
                    break
            
            metrics["settling_time"] = settling_time if settled else float('inf')
        else:
            metrics["settling_time"] = float('inf')
        
        # Calculate stability (oscillation measure)
        # Count zero-crossings and compute the variance of error rate of change
        if len(errors) > 3:
            # Compute rate of change
            error_derivative = np.diff(errors)
            
            # Count sign changes (zero crossings)
            sign_changes = np.sum(np.diff(np.signbit(error_derivative)) != 0)
            
            # Normalize by length
            oscillation_rate = sign_changes / (len(error_derivative) - 1)
            
            # Variance of error derivative (higher variance = less stability)
            derivative_variance = np.var(error_derivative)
            
            # Combine into a stability metric (lower is better)
            metrics["stability"] = oscillation_rate * derivative_variance
        else:
            metrics["stability"] = float('inf')
        
        metrics["composite"] = calculate_composite_score(metrics, config)
            
    except Exception as e:
        print(f"Warning: Error calculating advanced metrics: {e}")
        metrics["overshoot"] = float('inf')
        metrics["settling_time"] = float('inf')
        metrics["stability"] = float('inf')
    
    return metrics


def optimize_pid():
    """Main function to optimize forward PID parameters"""
    config = OptimizationConfig()
    results = []
    
    # Display optimization configuration
    print(f"\n{'='*80}")
    print(f"Starting Forward PID Parameter Optimization")
    print(f"{'='*80}")
    print(f"Metric to optimize: {config.metric}")
    
    if config.grid_search:
        # Grid search across all parameter combinations
        print(f"\nPerforming grid search with {len(config.kp_forward_range) * len(config.kd_forward_range) * len(config.ki_forward_range)} combinations")
        
        for kp in config.kp_forward_range:
            for kd in config.kd_forward_range:
                for ki in config.ki_forward_range:
                    # Update PID parameters
                    modify_pid_params(kp, kd, ki)
                    
                    print(f"\n{'-'*60}")
                    print(f"Testing: Kp={kp}, Kd={kd}, Ki={ki}")
                    
                    # Run tests
                    log_id = run_pid_tests()
                    if log_id:
                        # Evaluate performance
                        metrics = evaluate_performance(log_id, config)
                        
                        result = {
                            'kp_forward': kp,
                            'kd_forward': kd,
                            'ki_forward': ki,
                            'log_id': log_id,
                            **metrics
                        }
                        
                        results.append(result)
                        print(f"Results: Overshoot={metrics.get('overshoot', float('inf')):.4f}, "
                              f"Settling Time={metrics.get('settling_time', float('inf')):.4f}, "
                              f"Stability={metrics.get('stability', float('inf')):.4f}")
                        
                        if results:
                            best_result = min(results, key=lambda x: x.get(config.metric, float('inf')))
                            print(f"Best parameters so far: "
                                  f"Kp_forward = {best_result.get('kp_forward', 0)}, "
                                  f"Kd_forward = {best_result.get('kd_forward', 0)}, "
                                  f"Ki_forward = {best_result.get('ki_forward', 0)}")

                        cleanup_log_files(log_id)
                    
                    # Small delay between tests
                    time.sleep(1)
    
    elif config.sequential_testing:
        # Sequential testing (one parameter at a time)
        kp_best = config.base_kp_forward
        kd_best = config.base_kd_forward
        ki_best = config.base_ki_forward
        best_metric_value = float('inf')
        
        # Test Kp variations
        print("\nOptimizing Kp (proportional gain)...")
        for kp in config.kp_forward_range:
            # Update PID parameters
            modify_pid_params(kp, kd_best, ki_best)
            
            print(f"\n{'-'*60}")
            print(f"Testing: Kp={kp}, Kd={kd_best}, Ki={ki_best}")
            
            # Run tests
            log_id = run_pid_tests()
            if log_id:
                # Evaluate performance
                metrics = evaluate_performance(log_id, config)
                
                result = {
                    'kp_forward': kp,
                    'kd_forward': kd_best,
                    'ki_forward': ki_best,
                    'log_id': log_id,
                    **metrics
                }
                
                results.append(result)
                print(f"Results: Overshoot={metrics.get('overshoot', float('inf')):.4f}, "
                      f"Settling Time={metrics.get('settling_time', float('inf')):.4f}, "
                      f"Stability={metrics.get('stability', float('inf')):.4f}")
                
                # Update best if improved
                if metrics[config.metric] < best_metric_value:
                    best_metric_value = metrics[config.metric]
                    kp_best = kp
            
            # Small delay between tests
            time.sleep(1)
        
        # Test Kd variations with best Kp
        print("\nOptimizing Kd (derivative gain)...")
        best_metric_value = float('inf')
        for kd in config.kd_forward_range:
            # Update PID parameters
            modify_pid_params(kp_best, kd, ki_best)
            
            print(f"\n{'-'*60}")
            print(f"Testing: Kp={kp_best}, Kd={kd}, Ki={ki_best}")
            
            # Run tests
            log_id = run_pid_tests()
            if log_id:
                # Evaluate performance
                metrics = evaluate_performance(log_id, config)
                
                result = {
                    'kp_forward': kp_best,
                    'kd_forward': kd,
                    'ki_forward': ki_best,
                    'log_id': log_id,
                    **metrics
                }
                
                results.append(result)
                print(f"Results: Overshoot={metrics.get('overshoot', float('inf')):.4f}, "
                      f"Settling Time={metrics.get('settling_time', float('inf')):.4f}, "
                      f"Stability={metrics.get('stability', float('inf')):.4f}")
                
                # Update best if improved
                if metrics[config.metric] < best_metric_value:
                    best_metric_value = metrics[config.metric]
                    kd_best = kd
            
            # Small delay between tests
            time.sleep(1)
        
        # Test Ki variations with best Kp and Kd
        if any(ki != 0 for ki in config.ki_forward_range):
            print("\nOptimizing Ki (integral gain)...")
            best_metric_value = float('inf')
            for ki in config.ki_forward_range:
                # Update PID parameters
                modify_pid_params(kp_best, kd_best, ki)
                
                print(f"\n{'-'*60}")
                print(f"Testing: Kp={kp_best}, Kd={kd_best}, Ki={ki}")
                
                # Run tests
                log_id = run_pid_tests()
                if log_id:
                    # Evaluate performance
                    metrics = evaluate_performance(log_id, config)
                    
                    result = {
                        'kp_forward': kp_best,
                        'kd_forward': kd_best,
                        'ki_forward': ki,
                        'log_id': log_id,
                        **metrics
                    }
                    
                    results.append(result)
                    print(f"Results: Overshoot={metrics.get('overshoot', float('inf')):.4f}, "
                          f"Settling Time={metrics.get('settling_time', float('inf')):.4f}, "
                          f"Stability={metrics.get('stability', float('inf')):.4f}")
                    
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
        print(f"Forward PID Optimization Complete!")
        print(f"{'='*80}")
        print(f"Best parameters found:")
        print(f"Kp_forward = {best_result.get('kp_forward', 0)}")
        print(f"Kd_forward = {best_result.get('kd_forward', 0)}")
        print(f"Ki_forward = {best_result.get('ki_forward', 0)}")
        print(f"Overshoot = {best_result.get('overshoot', 0):.4f}")
        print(f"Settling Time = {best_result.get('settling_time', 0):.4f}")
        print(f"Stability = {best_result.get('stability', 0):.4f}")
        
        # Update to best parameters
        modify_pid_params(
            best_result.get('kp_forward', config.base_kp_forward),
            best_result.get('kd_forward', config.base_kd_forward),
            best_result.get('ki_forward', config.base_ki_forward)
        )
        
        # Visualize parameter landscape if we have enough data points
        if len(config.kp_forward_range) > 1 and len(config.kd_forward_range) > 1:
            try:
                # Create a heatmap of PID performance with fixed ki (using the best ki value)
                best_ki = best_result.get('ki_forward', 0)
                
                # Filter results with the best ki
                filtered_results = [r for r in results if abs(r.get('ki_forward', 0) - best_ki) < 1e-6]
                
                if len(filtered_results) > 4:  # Need enough points for visualization
                    kp_values = sorted(set([r.get('kp_forward', 0) for r in filtered_results]))
                    kd_values = sorted(set([r.get('kd_forward', 0) for r in filtered_results]))
                    
                    # Create a grid for the heatmap
                    performance_grid = np.full((len(kp_values), len(kd_values)), np.nan)
                    
                    # Fill in the performance grid
                    for result in filtered_results:
                        kp_idx = kp_values.index(result.get('kp_forward', 0))
                        kd_idx = kd_values.index(result.get('kd_forward', 0))
                        performance_grid[kp_idx, kd_idx] = result.get(config.metric, np.nan)
                    
                    # Create the heatmap
                    plt.figure(figsize=(10, 8))
                    plt.pcolormesh(kd_values, kp_values, performance_grid, cmap='viridis_r', shading='auto')
                    plt.colorbar(label='Performance (lower is better)')
                    plt.xlabel('Kd (Derivative Gain)')
                    plt.ylabel('Kp (Proportional Gain)')
                    plt.title(f'Forward PID Parameter Performance (Ki={best_ki:.6f})')
                    
                    # Mark the best parameters
                    best_kp = best_result.get('kp_forward', 0)
                    best_kd = best_result.get('kd_forward', 0)
                    plt.scatter([best_kd], [best_kp], color='red', marker='x', s=100)
                    plt.annotate(f"Best (Kp={best_kp:.4f}, Kd={best_kd:.4f})", 
                                (best_kd, best_kp), 
                                xytext=(10, 10), 
                                textcoords='offset points',
                                color='white',
                                bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))
                    
                    # Save the plot
                    os.makedirs("logs/plots", exist_ok=True)
                    plt.savefig("logs/plots/forward_pid_optimization.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved parameter landscape visualization to logs/plots/forward_pid_optimization.png")
            except Exception as e:
                print(f"Warning: Could not create parameter landscape visualization: {e}")
    else:
        print("No valid results were collected during optimization")


if __name__ == "__main__":
    optimize_pid()