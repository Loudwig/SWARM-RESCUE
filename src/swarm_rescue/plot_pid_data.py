import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_multiple_pid_data(data_pairs, output_dir=None):
    """
    Plot multiple PID datasets on the same figure
    
    Args:
        data_pairs: List of tuples (params_file, values_file, label)
        output_dir: Directory to save the output plot (if None, just displays the plot)
    """
    # Create a figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Colors for different datasets
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Store parameter texts for legend
    param_texts = []
    
    for idx, (params_file, values_file, label) in enumerate(data_pairs):
        # Read the data
        params_df = pd.read_csv(params_file)
        values_df = pd.read_csv(values_file)
        
        # Extract the first row of parameters
        kp_angle = params_df['Kp_angle'].iloc[0]
        kd_angle = params_df['Kd_angle'].iloc[0]
        ki_angle = params_df['Ki_angle'].iloc[0]
        kp_distance = params_df['Kp_distance'].iloc[0]
        kd_distance = params_df['Kd_distance'].iloc[0]
        ki_distance = params_df['Ki_distance'].iloc[0]
        
        # Create parameter text for this dataset
        param_text = (
            f"{label}: "
            f"Angle PID (Kp={kp_angle:.3f}, Kd={kd_angle:.3f}, Ki={ki_angle:.3f}), "
            f"Distance PID (Kp={kp_distance:.3f}, Kd={kd_distance:.3f}, Ki={ki_distance:.3f})"
        )
        param_texts.append(param_text)
        
        # Get color for this dataset
        color = colors[idx % len(colors)]
        
        # Plot angle error
        ax1.plot(values_df['timestep'], values_df['epsilon_angle'], f'{color}-', 
                 linewidth=2, label=f'{label} - Angle Error')
        
        # Plot lateral error
        ax2.plot(values_df['timestep'], values_df['epsilon_lateral'], f'{color}--', 
                 linewidth=2, label=f'{label} - Lateral Error')
    
    # Set titles and labels
    ax1.set_title('PID Control Performance Comparison', fontsize=14)
    ax1.set_ylabel('Angle Error', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Lateral Error', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Add the parameter texts at the bottom
    full_param_text = '\n'.join(param_texts)
    fig.text(0.5, 0.01, full_param_text, ha='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Adjusted to make room for parameters
    
    # Save or display the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Choose file number automatically
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('pid_comparison') and f.endswith('.png')]
        file_number = len(existing_files) + 1
        output_path = os.path.join(output_dir, f'pid_comparison{file_number}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_pid_data(params_file, values_file, output_dir=None):
    """
    Plot PID control data from a single CSV file pair.
    
    Args:
        params_file: Path to the CSV with PID parameters
        values_file: Path to the CSV with timestep and error values
        output_dir: Directory to save the output plot (if None, just displays the plot)
    """
    # Read the data
    params_df = pd.read_csv(params_file)
    values_df = pd.read_csv(values_file)
    
    # Extract the first row of parameters (assuming they're constant)
    kp_angle = params_df['Kp_angle'].iloc[0]
    kd_angle = params_df['Kd_angle'].iloc[0]
    ki_angle = params_df['Ki_angle'].iloc[0]
    kp_distance = params_df['Kp_distance'].iloc[0]
    kd_distance = params_df['Kd_distance'].iloc[0]
    ki_distance = params_df['Ki_distance'].iloc[0]
    
    # Create a figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot angle error
    ax1.plot(values_df['timestep'], values_df['epsilon_angle'], 'b-', linewidth=2, label='Angle Error')
    ax1.set_ylabel('Angle Error', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_title('PID Control Performance', fontsize=14)
    
    # Plot lateral error
    ax2.plot(values_df['timestep'], values_df['epsilon_lateral'], 'r-', linewidth=2, label='Lateral Error')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Lateral Error', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Add the PID parameters as text on the figure
    param_text = (
        f"Angle PID: Kp={kp_angle:.3f}, Kd={kd_angle:.3f}, Ki={ki_angle:.3f}\n"
        f"Distance PID: Kp={kp_distance:.3f}, Kd={kd_distance:.3f}, Ki={ki_distance:.3f}"
    )
    
    fig.text(0.5, 0.01, param_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save or display the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Choisit le numéro du fichier automatiquement
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('pid_performance') and f.endswith('.png')]
        file_number = len(existing_files) + 1
        output_path = os.path.join(output_dir, f'pid_performance{file_number}.png')

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Define file paths
    base_dir = "/home/etienne/Desktop/swarm/main/src/swarm_rescue/logs"
    output_dir = os.path.join(base_dir, "plots_circuits")
    
    # Define data pairs for comparison
    data_pairs = [
        (os.path.join(base_dir, "10_pid_params_drone.csv"), 
         os.path.join(base_dir, "10_pid_values_drone.csv"),
         "Config 10"),
        (os.path.join(base_dir, "11_pid_params_drone.csv"), 
         os.path.join(base_dir, "11_pid_values_drone.csv"),
         "Config 11")
    ]
    
    # Plot both datasets for comparison
    plot_multiple_pid_data(data_pairs, output_dir)
    
    # Optionally, also plot individual datasets
    # plot_pid_data(data_pairs[0][0], data_pairs[0][1], output_dir)
    # plot_pid_data(data_pairs[1][0], data_pairs[1][1], output_dir)