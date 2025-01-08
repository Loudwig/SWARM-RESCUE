import argparse
import csv
import os
import matplotlib.pyplot as plt
import math

def plot_losses(csv_file):
    steps = []
    columns = []
    data = {}

    # Read CSV headers and initialize data structures
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)  # First row is header
        
        if len(headers) < 2:
            print("CSV file must have at least two columns: 'Step' and at least one loss metric.")
            return
        if headers[0].strip().lower() != "step":
            print("The first column should be 'Step'.")
            return
        # The first column is "Step", subsequent columns are metrics
        columns = headers[1:]   # exclude "Step"

        # Initialize empty lists for each metric
        for col in columns:
            data[col] = []

        # Read rows
        for row_num, row in enumerate(reader, start=2):
            if len(row) < len(headers):
                print(f"Row {row_num} is incomplete. Expected {len(headers)} columns, got {len(row)}.")
                continue
            try:
                steps.append(float(row[0]))
                for i, col in enumerate(columns, start=1):
                    # Handle possible empty strings or invalid floats
                    try:
                        value = float(row[i])
                    except ValueError:
                        print(f"Invalid float value at row {row_num}, column {i} ('{col}'). Setting as NaN.")
                        value = math.nan
                    data[col].append(value)
            except ValueError as e:
                print(f"Error parsing row {row_num}: {e}")
                continue

    if not steps:
        print("No data to plot.")
        return

    # Plot each metric vs. steps individually
    for col in columns:
        plt.figure()
        plt.plot(steps, data[col], label=col)
        plt.title(col)
        plt.xlabel("Step")
        plt.ylabel(col)
        plt.grid(True)
        plt.legend()

        # Save individual figures next to CSV
        out_file = os.path.splitext(csv_file)[0] + f"_{col}.png"
        plt.savefig(out_file, dpi=150)
        print(f"Saved individual plot: {out_file}")
        plt.close()

    # Plot all metrics on separate subplots within the same figure
    num_metrics = len(columns)
    if num_metrics == 0 or num_metrics == 1:
        print("No metrics to plot in combined figure.")
        return

    # Determine subplot grid size (rows and cols)
    cols_grid = 3  # You can adjust this based on your preference
    rows_grid = math.ceil(num_metrics / cols_grid)

    fig, axs = plt.subplots(rows_grid, cols_grid, figsize=(5 * cols_grid, 4 * rows_grid))
    axs = axs.flatten()  # Flatten in case of multiple rows

    for idx, col in enumerate(columns):
        axs[idx].plot(steps, data[col], label=col)
        axs[idx].set_title(col)
        axs[idx].set_xlabel("Step")
        axs[idx].set_ylabel(col)
        axs[idx].grid(True)
        axs[idx].legend()

    # Hide any unused subplots
    for idx in range(num_metrics, len(axs)):
        fig.delaxes(axs[idx])

    fig.tight_layout()

    # Save combined subplot figure
    combined_out_file = os.path.splitext(csv_file)[0] + "_all_subplots.png"
    plt.savefig(combined_out_file, dpi=150)
    print(f"Saved combined subplots figure: {combined_out_file}")
    plt.close()

    # Optionally display the plots
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot CSV data from a training log.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: File '{args.csv_file}' does not exist.")
        return

    plot_losses(args.csv_file)

if __name__ == "__main__":
    main()