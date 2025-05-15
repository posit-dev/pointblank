#!/usr/bin/env python
"""
Data Generator for Pointblank Benchmarks

This script creates compressed CSV files with benchmark data of various sizes
and characteristics, which can then be reused across benchmark runs.
"""

import argparse
import os

# Import the DataGenerator from benchmark.py
from benchmark import SEED, DataGenerator


def generate_benchmark_datasets(
    output_dir="benchmark/data", sizes=None, column_counts=None, compression="gzip"
):
    """Generate benchmark datasets and save as compressed CSVs"""
    if sizes is None:
        sizes = [1_000, 10_000, 100_000, 1_000_000]
    if column_counts is None:
        column_counts = [10, 20, 50]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create data generator
    generator = DataGenerator(seed=SEED)

    # Generate datasets for each size and column count combination
    for row_count in sizes:
        for column_count in column_counts:
            # Generate the dataset (using pandas as intermediate)
            data, metadata = generator.generate_dataset(
                row_count, column_count, backend="pandas", missing_pct=0.05, invalid_pct=0.10
            )

            # Create filename
            filename = f"benchmark_data_{row_count}rows_{column_count}cols.csv.gz"
            filepath = os.path.join(output_dir, filename)

            print(f"Generating {filename}...")

            # Save as compressed CSV
            data.to_csv(filepath, index=False, compression=compression)

            # Also save metadata
            metadata_file = os.path.join(
                output_dir, f"metadata_{row_count}rows_{column_count}cols.json"
            )
            # Convert any non-serializable values to strings
            for key, value in metadata.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if not isinstance(v, (str, int, float, bool, list, dict)) and v is not None:
                            metadata[key][k] = str(v)

            # Save as JSON
            import json

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    print(f"Generated {len(sizes) * len(column_counts)} datasets in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark datasets")
    parser.add_argument("--output", default="benchmark/data", help="Output directory")
    parser.add_argument("--sizes", nargs="+", type=int, help="List of row counts to generate")
    parser.add_argument("--columns", nargs="+", type=int, help="List of column counts to generate")

    args = parser.parse_args()

    sizes = args.sizes if args.sizes else None
    column_counts = args.columns if args.columns else None

    generate_benchmark_datasets(args.output, sizes, column_counts)
