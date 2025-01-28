#!/bin/bash

# Define your base directory (remote path) and local destination path
base_dir="rplanchon-23@gpu2:~/code/SWARM-RESCUE/src/swarm_rescue/solutions/trained_models"
destination="/Users/rplanchon/Documents/projet/swarmRescue/SWARM-RESCUE/src/swarm_rescue/solutions/trained_models"

# Check if the script received an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the target directory from the argument
target_dir=$1

# Concatenate the base directory with the target directory
full_path="${base_dir}/${target_dir}"

# SCP command to copy the directory from remote to local
scp -r "$full_path" "$destination"

# Check if the SCP command succeeded
if [ "$?" -ne 0 ]; then
    echo "SCP command failed. Exiting."
    exit 1
fi

# Define paths for loss file and rewards file
loss_file="${destination}/${target_dir}/losses.csv"
rewards_file="${destination}/${target_dir}/rewards_per_episode.csv"

# Verify if the loss file exists
if [ -f "$loss_file" ]; then
    # Launch the Python script to plot losses
    python3 /Users/rplanchon/Documents/projet/swarmRescue/SWARM-RESCUE/src/swarm_rescue/solutions/plot_losses.py "$loss_file"

    # Check if the Python script executed successfully for losses
    if [ "$?" -eq 0 ]; then
        echo "Python script to plot losses executed successfully."
    else
        echo "Python script to plot losses execution failed."
    fi
else
    echo "Error: Loss file '$loss_file' does not exist."
fi

# Verify if the rewards file exists
if [ -f "$rewards_file" ]; then
    # Launch the Python script to plot rewards
    python3 /Users/rplanchon/Documents/projet/swarmRescue/SWARM-RESCUE/src/swarm_rescue/solutions/plot_losses.py "$rewards_file"

    # Check if the Python script executed successfully for rewards
    if [ "$?" -eq 0 ]; then
        echo "Python script to plot rewards executed successfully."
    else
        echo "Python script to plot rewards execution failed."
    fi
else
    echo "Error: Rewards file '$rewards_file' does not exist."
fi
