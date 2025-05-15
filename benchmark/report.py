#!/usr/bin/env python
# Pointblank Benchmark Report Generator

import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set plot style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


def load_benchmark_data(results_dir="benchmark/results", latest_only=True):
    """Load benchmark data from JSON files"""
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        print(f"No benchmark results found in {results_dir}")
        return None

    # Get the most recent file if latest_only is True
    if latest_only and json_files:
        latest_file = max(json_files, key=os.path.getmtime)
        json_files = [latest_file]
        print(f"Loading latest benchmark results: {os.path.basename(latest_file)}")

    # Load all benchmark results into a single dataframe
    all_data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    return pd.DataFrame(all_data)


def create_benchmark_report(df, output_dir="benchmark/reports"):
    """Generate visualizations and summary statistics from benchmark data"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract the benchmark configuration name for the report title
    config_name = df["config_name"].iloc[0]

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a figure for execution time comparison by backend and validation type
    plt.figure(figsize=(12, 8))

    # Group by backend, validation_type, and row_count for comparison
    grouped = (
        df.groupby(["backend", "validation_type", "row_count"])["execution_time"]
        .mean()
        .reset_index()
    )

    # Create the plot with seaborn
    g = sns.catplot(
        data=grouped,
        x="validation_type",
        y="execution_time",
        hue="backend",
        col="row_count",
        kind="bar",
        height=6,
        aspect=1.5,
        palette="viridis",
        legend_out=False,
    )

    # Customize the plot
    g.set_xticklabels(rotation=45, ha="right")
    g.fig.suptitle(f"Pointblank Validation Performance: {config_name}", fontsize=16)
    g.fig.subplots_adjust(top=0.9)

    # Save the plot
    plt.savefig(
        os.path.join(output_dir, f"{timestamp}_{config_name}_execution_time.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create a figure for memory usage comparison
    plt.figure(figsize=(12, 8))

    # Group by backend, validation_type, and row_count for memory usage comparison
    grouped = (
        df.groupby(["backend", "validation_type", "row_count"])["memory_usage"].mean().reset_index()
    )

    # Create the plot with seaborn
    g = sns.catplot(
        data=grouped,
        x="validation_type",
        y="memory_usage",
        hue="backend",
        col="row_count",
        kind="bar",
        height=6,
        aspect=1.5,
        palette="magma",
        legend_out=False,
    )

    # Customize the plot
    g.set_xticklabels(rotation=45, ha="right")
    g.fig.suptitle(f"Memory Usage by Validation Type: {config_name}", fontsize=16)
    g.fig.subplots_adjust(top=0.9)

    # Save the memory plot
    plt.savefig(
        os.path.join(output_dir, f"{timestamp}_{config_name}_memory_usage.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Generate a summary table
    summary = (
        df.groupby(["backend", "validation_type"])
        .agg(
            {
                "execution_time": ["mean", "min", "max"],
                "memory_usage": ["mean", "min", "max"],
                "validation_passed": "mean",
            }
        )
        .reset_index()
    )

    # Save the summary as CSV with timestamp
    summary.to_csv(os.path.join(output_dir, f"{timestamp}_{config_name}_summary.csv"), index=False)

    print(f"Report generated in {output_dir}")


if __name__ == "__main__":
    # Load the benchmark data
    df = load_benchmark_data()

    if df is not None:
        create_benchmark_report(df)
