#!/bin/bash

# Define your base directory and destination path
base_dir="rplanchon-23@gpu2:~/code/SWARM-RESCUE/src/swarm_rescue/solutions/trained_models/"
destination="/Users/rplanchon/Documents/projet/swarmRescue/SWARM-RESCUE/src/swarm_rescue/solutions/trained_models/"

# Check if the script received an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the directory from the argument
target_dir=$1

# Concatenate the base directory with the target directory
full_path="${base_dir}/${target_dir}"

# SCP command
scp -r "$full_path" "$destination"

# Check if the SCP command succeeded
if [ "$?" -ne 0 ]; then
    echo "SCP command failed. Exiting."
    exit 1
fi

# Launch the Python script
python3 /Users/rplanchon/Documents/projet/swarmRescue/SWARM-RESCUE/src/swarm_rescue/solutions/plot_losses.py ./trained_models/$1/losses.csv

# Optional: Check if the Python script executed successfully
if [ "$?" -eq 0 ]; then
    echo "Python script to plot losses executed successfully."
else
    echo "Python script to plot losses execution failed."
fi

python3 /Users/rplanchon/Documents/projet/swarmRescue/SWARM-RESCUE/src/swarm_rescue/solutions/plot_losses.py ./trained_models/$1/rewards_per_episode.csv


# Optional: Check if the Python script executed successfully
if [ "$?" -eq 0 ]; then
    echo "Python script to plot rewards executed successfully."
else
    echo "Python script to plot rewards execution failed."
fi