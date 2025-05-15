#!/usr/bin/env python
# Pointblank Performance Benchmarking Suite

import gc
import json
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import polars as pl
import psutil

import pointblank as pb

# Configure benchmarking parameters
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""

    name: str
    backends: List[str] = field(default_factory=lambda: ["polars", "pandas", "duckdb"])
    row_sizes: List[int] = field(default_factory=lambda: [10_000, 100_000, 1_000_000])
    column_counts: List[int] = field(default_factory=lambda: [10, 20, 50])
    iterations: int = 3
    output_dir: str = "benchmark/results"
    missing_pct: float = 0.05
    invalid_pct: float = 0.10
    categorical_count: int = 20

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""

    config_name: str
    backend: str
    row_count: int
    column_count: int
    validation_type: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    validation_passed: bool
    failure_rate: float
    parameters: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "config_name": self.config_name,
            "backend": self.backend,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "validation_type": self.validation_type,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "validation_passed": self.validation_passed,
            "failure_rate": self.failure_rate,
            "parameters": self.parameters,
            "timestamp": datetime.now().isoformat(),
        }


class DataGenerator:
    """Generate synthetic data with controlled characteristics"""

    def __init__(self, seed=SEED):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_numeric_column(
        self, size, min_val=0, max_val=100, missing_pct=0.0, invalid_pct=0.0
    ):
        """Generate a numeric column with controlled characteristics"""
        values = np.random.uniform(min_val, max_val, size)

        # Add missing values
        if missing_pct > 0:
            missing_count = int(size * missing_pct)
            missing_indices = np.random.choice(size, missing_count, replace=False)
            values[missing_indices] = np.nan

        # Add invalid values (outside of min_val, max_val range)
        if invalid_pct > 0:
            invalid_count = int(size * invalid_pct)
            invalid_indices = np.random.choice(
                np.setdiff1d(np.arange(size), missing_indices), invalid_count, replace=False
            )
            # Generate values outside the range
            invalid_values = np.concatenate(
                [
                    np.random.uniform(max_val + 1, max_val * 2, invalid_count // 2),
                    np.random.uniform(
                        min_val * 2, min_val - 1, invalid_count // 2 + invalid_count % 2
                    ),
                ]
            )
            values[invalid_indices] = invalid_values

        return values

    def generate_categorical_column(
        self, size, categories=None, n_categories=10, missing_pct=0.0, invalid_pct=0.0
    ):
        """Generate a categorical column with controlled characteristics"""
        if categories is None:
            categories = [f"cat_{i}" for i in range(n_categories)]

        # Generate valid values
        indices = np.random.randint(0, len(categories), size)
        values = np.array(categories)[indices]

        # Add missing values
        if missing_pct > 0:
            missing_count = int(size * missing_pct)
            missing_indices = np.random.choice(size, missing_count, replace=False)
            values[missing_indices] = None

        # Add invalid values (not in the categories list)
        if invalid_pct > 0:
            invalid_count = int(size * invalid_pct)
            invalid_indices = np.random.choice(
                np.setdiff1d(np.arange(size), missing_indices), invalid_count, replace=False
            )
            invalid_values = [f"invalid_{i}" for i in range(invalid_count)]
            values[invalid_indices] = invalid_values

        return values

    def generate_string_column(
        self, size, pattern=None, min_length=5, max_length=10, missing_pct=0.0, invalid_pct=0.0
    ):
        """Generate a string column with controlled characteristics"""

        def random_string(min_len, max_len):
            length = random.randint(min_len, max_len)
            return "".join(random.choices(string.ascii_letters + string.digits, k=length))

        # Generate valid values
        values = np.array([random_string(min_length, max_length) for _ in range(size)])

        # Apply pattern if provided (e.g., "ABC-\d{3}-\w{2}")
        if pattern:
            # Placeholder - actual implementation would generate strings matching the pattern
            pass

        # Add missing values
        if missing_pct > 0:
            missing_count = int(size * missing_pct)
            missing_indices = np.random.choice(size, missing_count, replace=False)
            values[missing_indices] = None

        # Add invalid values (special characters, etc.)
        if invalid_pct > 0:
            invalid_count = int(size * invalid_pct)
            invalid_indices = np.random.choice(
                np.setdiff1d(np.arange(size), missing_indices), invalid_count, replace=False
            )
            # Generate invalid strings - implementation would depend on the pattern
            invalid_values = ["!@#$%^" + s for s in np.random.choice(values, invalid_count)]
            values[invalid_indices] = invalid_values

        return values

    def generate_datetime_column(
        self, size, start_date=None, end_date=None, missing_pct=0.0, invalid_pct=0.0
    ):
        """Generate a datetime column with controlled characteristics"""
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime(2023, 12, 31)

        # Calculate time delta in seconds
        delta = (end_date - start_date).total_seconds()

        # Generate random datetimes between start_date and end_date
        random_seconds = np.random.randint(0, int(delta), size)
        # Convert numpy.int64 to Python int
        values = np.array([start_date + timedelta(seconds=int(s)) for s in random_seconds])

        # Add missing values
        if missing_pct > 0:
            missing_count = int(size * missing_pct)
            missing_indices = np.random.choice(size, missing_count, replace=False)
            values[missing_indices] = np.datetime64("NaT")

        # Add invalid values (dates outside the range)
        if invalid_pct > 0:
            invalid_count = int(size * invalid_pct)
            invalid_indices = np.random.choice(
                np.setdiff1d(np.arange(size), missing_indices), invalid_count, replace=False
            )

            # Generate dates outside the range
            before_start = [
                start_date - timedelta(days=random.randint(1, 365))
                for _ in range(invalid_count // 2)
            ]
            after_end = [
                end_date + timedelta(days=random.randint(1, 365))
                for _ in range(invalid_count // 2 + invalid_count % 2)
            ]
            invalid_values = before_start + after_end
            values[invalid_indices] = invalid_values

        return values

    def generate_related_columns(
        self, size, relationship_type="linear", noise_pct=0.1, missing_pct=0.0, invalid_pct=0.0
    ):
        """Generate related columns with a specific relationship plus noise"""
        # Generate base column
        col1 = self.generate_numeric_column(size, 1, 100, missing_pct=0)

        # Generate related column based on relationship_type
        if relationship_type == "linear":
            slope = random.uniform(0.5, 2.0)
            intercept = random.uniform(-10, 10)
            col2 = slope * col1 + intercept
        elif relationship_type == "quadratic":
            col2 = col1**2
        elif relationship_type == "inverse":
            col2 = 100 / (col1 + 1)  # Add 1 to avoid division by zero
        else:
            raise ValueError(f"Unknown relationship type: {relationship_type}")

        # Add noise to the relationship
        noise = np.random.normal(0, noise_pct * np.std(col2), size)
        col2 = col2 + noise

        # Apply missing and invalid values to both columns
        if missing_pct > 0:
            missing_count = int(size * missing_pct)
            col1_missing = np.random.choice(size, missing_count, replace=False)
            col2_missing = np.random.choice(size, missing_count, replace=False)
            col1[col1_missing] = np.nan
            col2[col2_missing] = np.nan

        # Make some values invalid by breaking the relationship
        if invalid_pct > 0:
            invalid_count = int(size * invalid_pct)
            invalid_indices = np.random.choice(size, invalid_count, replace=False)

            # For invalid pairs, randomly assign values that break the relationship
            col2[invalid_indices] = np.random.uniform(
                np.max(col2) + 10, np.max(col2) + 100, invalid_count
            )

        return col1, col2

    def generate_dataset(
        self,
        rows,
        columns,
        backend="polars",
        column_types=None,
        missing_pct=0.05,
        invalid_pct=0.10,
        categorical_count=20,
    ):
        """
        Generate a synthetic dataset with the specified characteristics

        Parameters:
        -----------
        rows: int
            Number of rows in the dataset
        columns: int
            Number of columns in the dataset
        backend: str
            The backend to use for the dataset ("polars", "pandas", "duckdb")
        column_types: Dict[str, str], optional
            Dictionary mapping column names to types
            If None, will generate a mix of types based on columns count
        missing_pct: float
            Percentage of missing values in the dataset (0.0 to 1.0)
        invalid_pct: float
            Percentage of invalid values in the dataset (0.0 to 1.0)
        categorical_count: int
            Number of categories for categorical columns

        Returns:
        --------
        dataset: Object
            Dataset in the specified backend format
        metadata: Dict
            Metadata about the dataset including column types and validity info
        """
        # If no column_types provided, generate a mix of types
        if column_types is None:
            # Decide distribution of column types
            n_numeric = columns // 2
            n_categorical = max(1, columns // 4)
            n_string = max(1, columns // 8)
            n_datetime = max(1, columns // 8)

            # Adjust to match the exact column count
            remainder = columns - (n_numeric + n_categorical + n_string + n_datetime)
            n_numeric += remainder

            column_types = {}
            for i in range(n_numeric):
                column_types[f"num_{i}"] = "numeric"
            for i in range(n_categorical):
                column_types[f"cat_{i}"] = "categorical"
            for i in range(n_string):
                column_types[f"str_{i}"] = "string"
            for i in range(n_datetime):
                column_types[f"dt_{i}"] = "datetime"

            # Add a few special columns for specific tests
            if columns >= 10:
                # Add related columns for relationship testing
                column_types["linear_x"] = "linear_x"
                column_types["linear_y"] = "linear_y"
                column_types["quadratic_x"] = "quadratic_x"
                column_types["quadratic_y"] = "quadratic_y"

                # Make sure we don't exceed the requested column count
                excess = len(column_types) - columns
                if excess > 0:
                    keys_to_remove = list(column_types.keys())[-excess:]
                    for key in keys_to_remove:
                        del column_types[key]

        # Generate data for each column
        data = {}
        metadata = {"column_types": {}, "valid_ranges": {}, "categories": {}}

        # Categories to use for all categorical columns
        categories = [f"category_{i}" for i in range(categorical_count)]

        for col_name, col_type in column_types.items():
            if col_type == "numeric":
                min_val, max_val = 0, 100
                data[col_name] = self.generate_numeric_column(
                    rows, min_val, max_val, missing_pct, invalid_pct
                )
                metadata["column_types"][col_name] = "numeric"
                metadata["valid_ranges"][col_name] = (min_val, max_val)

            elif col_type == "categorical":
                data[col_name] = self.generate_categorical_column(
                    rows, categories, categorical_count, missing_pct, invalid_pct
                )
                metadata["column_types"][col_name] = "categorical"
                metadata["categories"][col_name] = categories

            elif col_type == "string":
                data[col_name] = self.generate_string_column(
                    rows, None, 5, 10, missing_pct, invalid_pct
                )
                metadata["column_types"][col_name] = "string"

            elif col_type == "datetime":
                start = datetime(2020, 1, 1)
                end = datetime(2023, 12, 31)
                data[col_name] = self.generate_datetime_column(
                    rows, start, end, missing_pct, invalid_pct
                )
                metadata["column_types"][col_name] = "datetime"
                metadata["valid_ranges"][col_name] = (start, end)

            elif col_type in ["linear_x", "quadratic_x"]:
                # These will be handled later with their corresponding y columns
                pass

            else:
                raise ValueError(f"Unknown column type: {col_type}")

        # Generate related columns
        if "linear_x" in column_types and "linear_y" in column_types:
            data["linear_x"], data["linear_y"] = self.generate_related_columns(
                rows, "linear", 0.1, missing_pct, invalid_pct
            )
            metadata["column_types"]["linear_x"] = "numeric"
            metadata["column_types"]["linear_y"] = "numeric"
            metadata["relationship"] = {"type": "linear", "columns": ["linear_x", "linear_y"]}

        if "quadratic_x" in column_types and "quadratic_y" in column_types:
            data["quadratic_x"], data["quadratic_y"] = self.generate_related_columns(
                rows, "quadratic", 0.1, missing_pct, invalid_pct
            )
            metadata["column_types"]["quadratic_x"] = "numeric"
            metadata["column_types"]["quadratic_y"] = "numeric"
            metadata["relationship"] = {
                "type": "quadratic",
                "columns": ["quadratic_x", "quadratic_y"],
            }

        # Convert to the specified backend format
        if backend == "polars":
            return pl.DataFrame(data), metadata
        elif backend == "pandas":
            return pd.DataFrame(data), metadata
        elif backend == "duckdb":
            import os
            import tempfile

            import duckdb

            # Create a polars DataFrame first
            pdf = pl.DataFrame(data)

            # Create a unique temporary file for the DuckDB database
            db_file = os.path.join(tempfile.gettempdir(), f"benchmark_{uuid.uuid4()}.ddb")

            # Create DuckDB connection
            conn = duckdb.connect(database=db_file, read_only=False)

            # Create a table from the DataFrame
            # Note: This pattern is closer to how the original x-02-duckdb.qmd example works
            conn.execute("CREATE TABLE benchmark_data AS SELECT * FROM pdf")

            # Create a relation object using the connection's table method
            rel = conn.table("benchmark_data")

            # Return the relation object directly
            return rel, metadata
        else:
            raise ValueError(f"Unknown backend: {backend}")


class ValidationBenchmark:
    """Run benchmarks for different validation methods across backends and data sizes"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.generator = DataGenerator(seed=SEED)
        self.results = []

    def load_dataset(self, row_count, column_count, backend="polars"):
        """Load a pre-generated dataset from disk"""
        import json

        # Construct file paths
        data_file = os.path.join(
            "benchmark/data", f"benchmark_data_{row_count}rows_{column_count}cols.csv.gz"
        )
        metadata_file = os.path.join(
            "benchmark/data", f"metadata_{row_count}rows_{column_count}cols.json"
        )

        if not os.path.exists(data_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Pre-generated dataset not found: {data_file}")

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Load data according to backend
        if backend == "polars":
            data = pl.read_csv(data_file)
        elif backend == "pandas":
            data = pd.read_csv(data_file)
        elif backend == "duckdb":
            import duckdb

            # For DuckDB, we'll use pandas as an intermediate
            pdf = pd.read_csv(data_file)
            conn = duckdb.connect(":memory:")
            conn.execute("CREATE TABLE benchmark_data AS SELECT * FROM pdf")
            data = conn.table("benchmark_data")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Let's fix any datetime columns that might have been converted to strings
        for col_name, col_type in metadata.get("column_types", {}).items():
            if col_type == "datetime" and col_name in data.columns:
                # Handle different backends
                if backend == "polars":
                    data = data.with_columns(pl.col(col_name).str.to_datetime())
                elif backend == "pandas":
                    data[col_name] = pd.to_datetime(data[col_name])
                elif backend == "duckdb":
                    data = data.with_columns(pl.col(col_name).str.to_datetime())
        return data, metadata

    def run_single_benchmark(
        self,
        validation_fn: Callable,
        row_count: int,
        column_count: int,
        backend: str,
        validation_type: str,
        parameters: Dict = None,
    ):
        """Run a single benchmark for a specific validation function"""
        print(
            f"Running benchmark: {validation_type} with {row_count} rows, {column_count} cols on {backend}"
        )

        # Load the pre-generated dataset instead of generating it
        try:
            data, metadata = self.load_dataset(row_count, column_count, backend)
        except FileNotFoundError:
            print("WARNING: Pre-generated dataset not found. Generating new data...")
            # Fall back to generating data
            data, metadata = self.generator.generate_dataset(
                row_count,
                column_count,
                backend,
                missing_pct=self.config.missing_pct,
                invalid_pct=self.config.invalid_pct,
                categorical_count=self.config.categorical_count,
            )

        # Collect baseline system stats
        process = psutil.Process(os.getpid())
        base_memory = process.memory_info().rss / (1024 * 1024)  # MB

        execution_times = []
        memory_usages = []
        cpu_usages = []
        validation_results = []

        # Run the validation multiple times and collect stats
        for i in range(self.config.iterations):
            # Clear caches and collect garbage to minimize interference
            gc.collect()

            # Record starting metrics
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            start_cpu = process.cpu_percent()
            start_time = time.time()

            # Run the validation
            validation_result = validation_fn(data, metadata, parameters)

            # Record ending metrics
            end_time = time.time()
            end_cpu = process.cpu_percent()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Calculate differences
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = end_cpu - start_cpu

            execution_times.append(execution_time)
            memory_usages.append(memory_usage)
            cpu_usages.append(cpu_usage)
            validation_results.append(validation_result)

            print(
                f"  Iteration {i + 1}: {execution_time:.4f}s, {memory_usage:.2f}MB, passed: {validation_result['passed']}"
            )

        # Calculate averages
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)

        # With this code that handles potential type issues:
        validation_passed = all(r["passed"] for r in validation_results)
        failure_rates = []
        for r in validation_results:
            # Ensure we're only using numeric failure rates
            fr = r["failure_rate"]
            if isinstance(fr, (int, float)):
                failure_rates.append(fr)
            else:
                # If failure_rate isn't numeric, use 1.0 to indicate failure
                print(f"  Warning: Non-numeric failure rate encountered: {fr}")
                failure_rates.append(1.0)

        avg_failure_rate = sum(failure_rates) / len(failure_rates) if failure_rates else 0.0

        # Create and save the result
        result = BenchmarkResult(
            config_name=self.config.name,
            backend=backend,
            row_count=row_count,
            column_count=column_count,
            validation_type=validation_type,
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            cpu_usage=avg_cpu_usage,
            validation_passed=validation_passed,
            failure_rate=avg_failure_rate,
            parameters=parameters or {},
        )

        # Cleanup DuckDB connection if necessary
        try:
            if backend == "duckdb" and isinstance(data, dict) and "conn" in data:
                data["conn"].close()
        except Exception as e:
            print(f"Error closing DuckDB connection: {e}")

        self.results.append(result)
        return result

    def run_validation_benchmarks(self):
        """Run all benchmarks specified in the configuration"""
        for backend in self.config.backends:
            for row_count in self.config.row_sizes:
                for column_count in self.config.column_counts:
                    # Run simple column value validations
                    self.run_single_benchmark(
                        validation_basic_numeric_range,
                        row_count,
                        column_count,
                        backend,
                        "col_vals_between",
                        {"range": (0, 100)},
                    )

                    self.run_single_benchmark(
                        validation_categorical_values,
                        row_count,
                        column_count,
                        backend,
                        "col_vals_in_set",
                        {"categories": self.config.categorical_count},
                    )

                    # Run more complex validations
                    self.run_single_benchmark(
                        validation_row_distinct,
                        row_count,
                        column_count,
                        backend,
                        "rows_distinct",
                        {},
                    )

                    self.run_single_benchmark(
                        validation_schema_match,
                        row_count,
                        column_count,
                        backend,
                        "col_schema_match",
                        {},
                    )

                    # Run validations with multiple columns
                    self.run_single_benchmark(
                        validation_column_relationship,
                        row_count,
                        column_count,
                        backend,
                        "col_vals_relationship",
                        {"relationship": "linear"},
                    )

                    # Run validations with pattern matching
                    if column_count >= 10:  # Only if we have enough columns
                        self.run_single_benchmark(
                            validation_regex_patterns,
                            row_count,
                            column_count,
                            backend,
                            "col_vals_regex",
                            {"complexity": "simple"},
                        )

                        self.run_single_benchmark(
                            validation_regex_patterns,
                            row_count,
                            column_count,
                            backend,
                            "col_vals_regex",
                            {"complexity": "complex"},
                        )

                    # Run validations with segmentation
                    if column_count >= 12:  # Only if we have enough columns
                        self.run_single_benchmark(
                            validation_with_segmentation,
                            row_count,
                            column_count,
                            backend,
                            "segmentation",
                            {"segments": 5},
                        )

        # Save all results
        self.save_results()

    def save_results(self):
        """Save the benchmark results to a JSON file"""
        results_file = os.path.join(
            self.config.output_dir,
            f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)

        print(f"Saved benchmark results to {results_file}")


# Definition of validation functions


def validation_basic_numeric_range(data, metadata, params):
    """Simple validation of numeric columns being within a range"""
    try:
        min_val, max_val = params.get("range", (0, 100))

        # Find numeric columns to validate
        numeric_cols = [
            col for col, type_ in metadata["column_types"].items() if type_ == "numeric"
        ][:3]  # Limit to first 3 numeric columns

        if not numeric_cols:
            return {"passed": False, "failure_rate": 1.0, "error": "No numeric columns found"}

        # Create and run validation
        validation = (
            pb.Validate(data=data)
            .col_vals_between(columns=numeric_cols, left=min_val, right=max_val, na_pass=True)
            .interrogate()
        )

        # Get failure rate - use index 1 to get the first step's rate
        # When multiple validations exist, we need to focus on a specific step
        # or aggregate the results
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_basic_numeric_range: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def validation_categorical_values(data, metadata, params):
    """Validation of categorical columns containing values from a defined set"""
    try:
        # Find categorical columns to validate
        categorical_cols = [
            col for col, type_ in metadata["column_types"].items() if type_ == "categorical"
        ][:3]  # Limit to first 3 categorical cols

        if not categorical_cols:
            return {"passed": False, "failure_rate": 1.0, "error": "No categorical columns found"}

        # Get the list of categories
        categories = []
        for col in categorical_cols:
            if col in metadata.get("categories", {}):
                categories.extend(metadata["categories"][col])
            else:
                # Default list if metadata doesn't contain categories
                categories.extend([f"category_{i}" for i in range(params.get("categories", 20))])

        # Create and run validation - removed na_pass parameter
        validation = (
            pb.Validate(data=data)
            .col_vals_in_set(
                columns=categorical_cols[0],  # Just test first column for simplicity
                set=list(set(categories)),
            )
            .interrogate()
        )

        # Get failure rate with scalar=True and specific step
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_categorical_values: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def validation_row_distinct(data, metadata, params):
    """Validation that rows are distinct"""
    try:
        # Create and run validation
        validation = pb.Validate(data=data).rows_distinct().interrogate()

        # Get failure rate
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_row_distinct: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def validation_schema_match(data, metadata, params):
    """Validation that the data matches a specified schema"""
    try:
        # Create a schema based on the metadata
        schema_columns = []
        for col_name, col_type in metadata["column_types"].items():
            if col_type == "numeric":
                schema_columns.append((col_name, ["float64", "int64"]))
            elif col_type == "categorical" or col_type == "string":
                schema_columns.append((col_name, "string"))
            elif col_type == "datetime":
                schema_columns.append((col_name, "datetime64"))
            else:
                schema_columns.append((col_name,))

        schema = pb.Schema(columns=schema_columns[:5])  # Limit to 5 columns for performance

        # Create and run validation
        validation = pb.Validate(data=data).col_schema_match(schema=schema).interrogate()

        # Get failure rate
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_schema_match: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def validation_column_relationship(data, metadata, params):
    """Validation that columns satisfy a relationship with each other"""
    try:
        relationship = params.get("relationship", "linear")

        # Find pairs of columns with relationship
        col_pairs = []
        if (
            relationship == "linear"
            and "linear_x" in metadata["column_types"]
            and "linear_y" in metadata["column_types"]
        ):
            col_pairs.append(("linear_x", "linear_y"))
        elif (
            relationship == "quadratic"
            and "quadratic_x" in metadata["column_types"]
            and "quadratic_y" in metadata["column_types"]
        ):
            col_pairs.append(("quadratic_x", "quadratic_y"))

        if not col_pairs:
            # Fall back to any two numeric columns
            numeric_cols = [
                col for col, type_ in metadata["column_types"].items() if type_ == "numeric"
            ]
            if len(numeric_cols) >= 2:
                col_pairs.append((numeric_cols[0], numeric_cols[1]))

        if not col_pairs:
            return {"passed": False, "failure_rate": 1.0, "error": "No suitable column pairs found"}

        # For this test, we'll just check that the first column is less than the second
        # This is a simple test but allows us to benchmark the performance
        col1, col2 = col_pairs[0]

        # Create and run validation
        validation = (
            pb.Validate(data=data)
            .col_vals_lt(columns=col1, value=pb.col(col2), na_pass=True)
            .interrogate()
        )

        # Get failure rate
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_column_relationship: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def validation_regex_patterns(data, metadata, params):
    """Validation that string columns match regex patterns"""
    try:
        complexity = params.get("complexity", "simple")

        # Find string columns to validate
        string_cols = [col for col, type_ in metadata["column_types"].items() if type_ == "string"][
            :2
        ]  # Limit to first 2 string columns

        if not string_cols:
            return {"passed": False, "failure_rate": 1.0, "error": "No string columns found"}

        # Choose a pattern based on complexity
        if complexity == "simple":
            pattern = r"^[a-zA-Z0-9]+$"  # Simple alphanumeric
        else:
            pattern = r"^[a-zA-Z0-9]{5,10}$"  # Between 5-10 alphanumeric chars

        # Create and run validation
        validation = (
            pb.Validate(data=data)
            .col_vals_regex(
                columns=string_cols[0],  # Just test first column for simplicity
                pattern=pattern,
                na_pass=True,
            )
            .interrogate()
        )

        # Get failure rate
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_regex_patterns: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def validation_with_segmentation(data, metadata, params):
    """Validation with segmentation by categorical columns"""
    try:
        n_segments = params.get("segments", 5)

        # Find categorical and numeric columns
        categorical_cols = [
            col for col, type_ in metadata["column_types"].items() if type_ == "categorical"
        ][:1]  # Use first categorical column
        numeric_cols = [
            col for col, type_ in metadata["column_types"].items() if type_ == "numeric"
        ][:1]  # Use first numeric column

        if not categorical_cols or not numeric_cols:
            return {
                "passed": False,
                "failure_rate": 1.0,
                "error": f"Need both category and numeric columns. Found {len(categorical_cols)} cat, {len(numeric_cols)} num",
            }

        # Get a list of unique categories to use for segmentation
        categories = metadata.get("categories", {}).get(categorical_cols[0], [])
        categories = (
            categories[:n_segments] if categories else [f"category_{i}" for i in range(n_segments)]
        )

        # Create and run validation with segmentation
        validation = (
            pb.Validate(data=data)
            .col_vals_gt(
                columns=numeric_cols[0],
                value=0,
                segments=(categorical_cols[0], categories),
                na_pass=True,
            )
            .interrogate()
        )

        # Get failure rate
        if len(validation.validation_info) > 0:
            failure_rate = validation.f_failed(i=1, scalar=True)
        else:
            failure_rate = 0.0

        return {"passed": validation.all_passed(), "failure_rate": failure_rate}
    except Exception as e:
        print(f"Error in validation_with_segmentation: {e}")
        return {"passed": False, "failure_rate": 1.0, "error": str(e)}


def main():
    """Main entry point for the benchmarking suite"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Pointblank benchmarks")
    parser.add_argument(
        "--config",
        default="small",
        choices=["small", "medium", "large", "xlarge"],
        help="Benchmark configuration size",
    )
    args = parser.parse_args()

    # Select configuration based on argument
    if args.config == "small":
        config = BenchmarkConfig(
            name="pointblank_benchmark_small",
            backends=["polars", "pandas"],
            row_sizes=[1_000, 10_000],  # Small data sizes
            column_counts=[10],
            iterations=2,
            missing_pct=0.05,
            invalid_pct=0.10,
        )
    elif args.config == "medium":
        config = BenchmarkConfig(
            name="pointblank_benchmark_medium",
            backends=["polars", "pandas"],
            row_sizes=[10_000, 100_000],
            column_counts=[10, 20],
            iterations=3,
            missing_pct=0.05,
            invalid_pct=0.10,
        )
    elif args.config == "large":
        config = BenchmarkConfig(
            name="pointblank_benchmark_large",
            backends=["polars", "pandas"],
            row_sizes=[100_000, 1_000_000],
            column_counts=[10, 20, 50],
            iterations=3,
            missing_pct=0.05,
            invalid_pct=0.10,
        )
    else:  # xlarge
        config = BenchmarkConfig(
            name="pointblank_benchmark_xlarge",
            backends=["polars", "pandas"],
            row_sizes=[1_000_000, 10_000_000],
            column_counts=[10, 20],
            iterations=2,
            missing_pct=0.05,
            invalid_pct=0.10,
        )

    # Run the benchmarks
    benchmark = ValidationBenchmark(config)
    benchmark.run_validation_benchmarks()

    print("Benchmarking complete!")


if __name__ == "__main__":
    main()
