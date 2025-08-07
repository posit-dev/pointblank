from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import pointblank as pb
from pointblank._utils import _get_tbl_type, _is_lib_present

console = Console()


class OrderedGroup(click.Group):
    """A Click Group that displays commands in a custom order."""

    def list_commands(self, ctx):
        """Return commands in the desired logical order."""
        # Define the desired order
        desired_order = [
            # Data Discovery/Exploration
            "info",
            "preview",
            "scan",
            "missing",
            # Validation
            "validate",
            "run",
            "make-template",
            # Data Manipulation
            "pl",
            # Utilities
            "datasets",
            "requirements",
        ]

        # Get all available commands
        available_commands = super().list_commands(ctx)

        # Return commands in desired order, followed by any not in the list
        ordered = []
        for cmd in desired_order:
            if cmd in available_commands:
                ordered.append(cmd)

        # Add any commands not in our desired order (safety fallback)
        for cmd in available_commands:
            if cmd not in ordered:
                ordered.append(cmd)

        return ordered


def _load_data_source(data_source: str) -> Any:
    """
    Centralized data loading function for CLI that handles all supported data source types.

    This function provides a consistent way to load data across all CLI commands by leveraging
    the _process_data() utility function and adding support for pointblank dataset names.

    Parameters
    ----------
    data_source : str
        The data source which could be:
        - A pointblank dataset name (small_table, game_revenue, nycflights, global_sales)
        - A GitHub URL pointing to a CSV or Parquet file
        - A database connection string (e.g., "duckdb:///path/to/file.ddb::table_name")
        - A CSV file path (string or Path object with .csv extension)
        - A Parquet file path, glob pattern, directory, or partitioned dataset

    Returns
    -------
    Any
        Loaded data as a DataFrame or other data object

    Raises
    ------
    ValueError
        If the pointblank dataset name is not recognized
    """
    # Check if it's a pointblank dataset name first
    if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
        return pb.load_dataset(data_source)

    # Otherwise, use the centralized _process_data() function for all other data sources
    from pointblank.validate import _process_data

    return _process_data(data_source)


def _is_piped_data_source(data_source: str) -> bool:
    """Check if the data source is from a piped pb command."""
    return (
        data_source
        and ("pb_pipe_" in data_source)
        and (data_source.startswith("/var/folders/") or data_source.startswith("/tmp/"))
    )


def _format_cell_value(
    value: Any, is_row_number: bool = False, max_width: int = 50, num_columns: int = 10
) -> str:
    """Format a cell value for Rich table display, highlighting None/NA values in red.

    Args:
        value: The raw cell value from the dataframe
        is_row_number: Whether this is a row number column value
        max_width: Maximum character width for text truncation
        num_columns: Number of columns in the table (affects truncation aggressiveness)

    Returns:
        Formatted string with Rich markup for None/NA values or row numbers
    """
    # Special formatting for row numbers: never truncate them
    if is_row_number:
        return f"[dim]{value}[/dim]"

    # Check for actual None/null values (not string representations)
    if value is None:
        return "[red]None[/red]"

    # Check for pandas/numpy specific NA values
    try:
        import numpy as np
        import pandas as pd

        # Check for pandas NA
        if pd.isna(value):
            # If it's specifically numpy.nan, show as NaN
            if isinstance(value, float) and np.isnan(value):
                return "[red]NaN[/red]"
            # If it's pandas NA, show as NA
            elif str(type(value)).find("pandas") != -1:
                return "[red]NA[/red]"
            # Generic NA for other pandas missing values
            else:
                return "[red]NA[/red]"

    except (ImportError, TypeError, ValueError):  # pragma: no cover
        # If pandas/numpy not available, value not compatible, or ambiguous array
        pass

    # Check for empty strings (but only actual empty strings, not whitespace)
    if isinstance(value, str) and value == "":
        return "[red][/red]"  # Empty string shown as red empty space

    # Convert to string and apply intelligent truncation
    str_value = str(value)

    # Adjust max_width based on number of columns to prevent overly wide tables
    if num_columns > 15:
        adjusted_max_width = min(max_width, 30)  # Be more aggressive with many columns
    elif num_columns > 10:
        adjusted_max_width = min(max_width, 40)
    else:
        adjusted_max_width = max_width

    # Apply truncation if the string is too long
    if len(str_value) > adjusted_max_width:
        # For very long text, truncate more aggressively
        if len(str_value) > adjusted_max_width * 2:
            # For extremely long text, use a shorter truncation
            truncated = str_value[: adjusted_max_width // 2] + "…"
        else:
            # For moderately long text, use a more generous truncation
            truncated = str_value[: adjusted_max_width - 1] + "…"

        return truncated

    return str_value


def _get_column_dtypes(df: Any, columns: list[str]) -> dict[str, str]:
    """Extract data types for columns and format them in a compact way.

    Args:
        df: The dataframe object
        columns: List of column names

    Returns:
        Dictionary mapping column names to formatted data type strings
    """
    dtypes_dict = {}

    try:
        if hasattr(df, "dtypes"):
            # Polars/Pandas style
            if hasattr(df.dtypes, "to_dict"):
                # Polars DataFrame dtypes
                raw_dtypes = df.dtypes.to_dict() if hasattr(df.dtypes, "to_dict") else {}
                for col in columns:
                    if col in raw_dtypes:
                        dtype_str = str(raw_dtypes[col])
                        # Convert to compact format similar to Polars glimpse()
                        dtypes_dict[col] = _format_dtype_compact(dtype_str)
                    else:
                        dtypes_dict[col] = "?"
            else:
                # Pandas DataFrame dtypes (Series-like)
                for i, col in enumerate(columns):
                    if i < len(df.dtypes):
                        dtype_str = str(
                            df.dtypes.iloc[i] if hasattr(df.dtypes, "iloc") else df.dtypes[i]
                        )
                        dtypes_dict[col] = _format_dtype_compact(dtype_str)
                    else:
                        dtypes_dict[col] = "?"
        elif hasattr(df, "schema"):
            # Other schema-based systems (e.g., Ibis)
            schema = df.schema
            if hasattr(schema, "to_dict"):  # pragma: no cover
                raw_dtypes = schema.to_dict()
                for col in columns:
                    if col in raw_dtypes:
                        dtypes_dict[col] = _format_dtype_compact(str(raw_dtypes[col]))
                    else:  # pragma: no cover
                        dtypes_dict[col] = "?"
            else:  # pragma: no cover
                for col in columns:
                    try:
                        dtype_str = str(getattr(schema, col, "Unknown"))
                        dtypes_dict[col] = _format_dtype_compact(dtype_str)
                    except Exception:  # pragma: no cover
                        dtypes_dict[col] = "?"
        else:
            # Fallback: no type information available
            for col in columns:
                dtypes_dict[col] = "?"

    except Exception:  # pragma: no cover
        # If any error occurs, fall back to unknown types
        for col in columns:
            dtypes_dict[col] = "?"

    return dtypes_dict


def _format_dtype_compact(dtype_str: str) -> str:
    """Format a data type string to a compact representation.

    Args:
        dtype_str: The raw data type string

    Returns:
        Compact formatted data type string
    """
    # Remove common prefixes and make compact
    dtype_str = dtype_str.lower()

    # Polars types
    if "utf8" in dtype_str or "string" in dtype_str:
        return "str"
    elif "int64" in dtype_str:
        return "i64"
    elif "int32" in dtype_str:
        return "i32"
    elif "float64" in dtype_str:
        return "f64"
    elif "float32" in dtype_str:
        return "f32"
    elif "boolean" in dtype_str or "bool" in dtype_str:
        return "bool"
    elif "datetime" in dtype_str:
        return "datetime"
    elif "date" in dtype_str and "datetime" not in dtype_str:
        return "date"
    elif "time" in dtype_str:
        return "time"

    # Pandas types
    elif "object" in dtype_str:
        return "obj"
    elif "category" in dtype_str:
        return "cat"

    # Generic fallbacks
    elif "int" in dtype_str:
        return "int"
    elif "float" in dtype_str:
        return "float"
    elif "str" in dtype_str:
        return "str"

    # Unknown or complex types: truncate if too long
    elif len(dtype_str) > 8:
        return dtype_str[:8] + "…"
    else:
        return dtype_str


def _rich_print_scan_table(
    scan_result: Any,
    data_source: str,
    source_type: str,
    table_type: str,
    total_rows: int | None = None,
    total_columns: int | None = None,
) -> None:
    """
    Display scan results as a Rich table in the terminal with statistical measures.

    Args:
        scan_result: The GT object from col_summary_tbl()
        data_source: Name of the data source being scanned
        source_type: Type of data source (e.g., "Pointblank dataset: small_table")
        table_type: Type of table (e.g., "polars.LazyFrame")
        total_rows: Total number of rows in the dataset
        total_columns: Total number of columns in the dataset
    """
    try:
        import re

        import narwhals as nw
        from rich.box import SIMPLE_HEAD

        # Extract the underlying DataFrame from the GT object
        # The GT object has a _tbl_data attribute that contains the DataFrame
        gt_data = scan_result._tbl_data

        # Convert to Narwhals DataFrame for consistent handling
        nw_data = nw.from_native(gt_data)

        # Convert to dictionary for easier access
        data_dict = nw_data.to_dict(as_series=False)

        # Create main scan table with missing data table styling
        # Create a comprehensive title with data source, source type, and table type
        title_text = f"Column Summary / {source_type} / {table_type}"

        # Add dimensions subtitle in gray if available
        if total_rows is not None and total_columns is not None:
            title_text += f"\n[dim]{total_rows:,} rows / {total_columns} columns[/dim]"

        scan_table = Table(
            title=title_text,
            show_header=True,
            header_style="bold magenta",
            box=SIMPLE_HEAD,
            title_style="bold cyan",
            title_justify="left",
        )

        # Add columns with specific styling and appropriate widths
        scan_table.add_column("Column", style="cyan", no_wrap=True, width=20)
        scan_table.add_column("Type", style="yellow", no_wrap=True, width=10)
        scan_table.add_column(
            "NA", style="red", width=6, justify="right"
        )  # Adjusted for better formatting
        scan_table.add_column(
            "UQ", style="green", width=8, justify="right"
        )  # Adjusted for boolean values

        # Add statistical columns if they exist with appropriate widths
        stat_columns = []
        column_mapping = {
            "mean": ("Mean", "blue", 9),
            "std": ("SD", "blue", 9),
            "min": ("Min", "yellow", 9),
            "median": ("Med", "yellow", 9),
            "max": ("Max", "yellow", 9),
            "q_1": ("Q₁", "magenta", 8),
            "q_3": ("Q₃", "magenta", 9),
            "iqr": ("IQR", "magenta", 8),
        }

        for col_key, (display_name, color, width) in column_mapping.items():
            if col_key in data_dict:
                scan_table.add_column(display_name, style=color, width=width, justify="right")
                stat_columns.append(col_key)

        # Helper function to extract column name and type from HTML
        def extract_column_info(html_content: str) -> tuple[str, str]:
            """Extract column name and type from HTML formatted content."""
            # Extract column name from first div
            name_match = re.search(r"<div[^>]*>([^<]+)</div>", html_content)
            column_name = name_match.group(1) if name_match else "Unknown"

            # Extract data type from second div (with gray color)
            type_match = re.search(r"<div[^>]*color: gray[^>]*>([^<]+)</div>", html_content)
            if type_match:
                data_type = type_match.group(1)
                # Convert to compact format using the existing function
                compact_type = _format_dtype_compact(data_type)
                data_type = compact_type
            else:
                data_type = "unknown"

            return column_name, data_type

        # Helper function to format values with improved number formatting
        def format_value(
            value: Any, is_missing: bool = False, is_unique: bool = False, max_width: int = 8
        ) -> str:
            """Format values for display with smart number formatting and HTML cleanup."""
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return "[dim]—[/dim]"

            # Handle missing values indicator
            if is_missing and str(value) == "0":
                return "[green]●[/green]"  # No missing values

            # Clean up HTML formatting from the raw data
            str_val = str(value)

            # Handle multi-line values with <br> tags FIRST: take the first line (absolute number)
            if "<br>" in str_val:
                str_val = str_val.split("<br>")[0].strip()
                # For unique values, we want just the integer part
                if is_unique:
                    try:
                        # Try to extract just the integer part for unique counts
                        num_val = float(str_val)
                        return str(int(num_val))
                    except (ValueError, TypeError):
                        pass

            # Now handle HTML content (especially from boolean unique values)
            if "<" in str_val and ">" in str_val:
                # Remove HTML tags completely for cleaner display
                str_val = re.sub(r"<[^>]+>", "", str_val).strip()
                # Clean up extra whitespace
                str_val = re.sub(r"\s+", " ", str_val).strip()

            # Handle values like "2<.01": extract the first number
            if "<" in str_val and not (str_val.startswith("<") and str_val.endswith(">")):
                # Extract number before the < symbol
                before_lt = str_val.split("<")[0].strip()
                if before_lt and before_lt.replace(".", "").replace("-", "").isdigit():
                    str_val = before_lt

            # Handle boolean unique values like "T0.62F0.38": extract the more readable format
            if re.match(r"^[TF]\d+\.\d+[TF]\d+\.\d+$", str_val):
                # Extract T and F values
                t_match = re.search(r"T(\d+\.\d+)", str_val)
                f_match = re.search(r"F(\d+\.\d+)", str_val)
                if t_match and f_match:
                    t_val = float(t_match.group(1))
                    f_val = float(f_match.group(1))
                    # Show as "T0.62F0.38" but truncated if needed
                    formatted = f"T{t_val:.2f}F{f_val:.2f}"
                    if len(formatted) > max_width:
                        # Truncate to fit, showing dominant value
                        if t_val > f_val:
                            return f"T{t_val:.1f}"
                        else:
                            return f"F{f_val:.1f}"
                    return formatted

            # Try to parse as a number for better formatting
            try:
                # Try to convert to float first
                num_val = float(str_val)

                # Handle special cases
                if num_val == 0:
                    return "0"
                elif abs(num_val) == int(abs(num_val)) and abs(num_val) < 10000:
                    # Simple integers under 10000
                    return str(int(num_val))
                elif abs(num_val) >= 10000000 and abs(num_val) < 100000000:
                    # Likely dates in YYYYMMDD format: format as date-like
                    int_val = int(num_val)
                    if 19000101 <= int_val <= 29991231:  # Reasonable date range
                        str_date = str(int_val)
                        if len(str_date) == 8:
                            return (
                                f"{str_date[:4]}-{str_date[4:6]}-{str_date[6:]}"[: max_width - 1]
                                + "…"
                            )
                    # Otherwise treat as large number
                    return f"{num_val / 1000000:.1f}M"
                elif abs(num_val) >= 1000000:
                    # Large numbers: use scientific notation or M/k notation

                    if abs(num_val) >= 1000000000:
                        return f"{num_val:.1e}"
                    else:
                        return f"{num_val / 1000000:.1f}M"
                elif abs(num_val) >= 10000:
                    # Numbers >= 10k: use compact notation
                    return f"{num_val / 1000:.1f}k"
                elif abs(num_val) >= 100:
                    # Numbers 100-9999: show with minimal decimals
                    return f"{num_val:.1f}"
                elif abs(num_val) >= 10:
                    # Numbers 10-99: show with one decimal
                    return f"{num_val:.1f}"
                elif abs(num_val) >= 1:
                    # Numbers 1-9: show with two decimals
                    return f"{num_val:.2f}"
                elif abs(num_val) >= 0.01:
                    # Small numbers: show with appropriate precision
                    return f"{num_val:.2f}"
                else:
                    # Very small numbers: use scientific notation

                    return f"{num_val:.1e}"

            except (ValueError, TypeError):
                # Not a number, handle as string
                pass

            # Handle date/datetime strings: show abbreviated format
            if len(str_val) > 10 and any(char in str_val for char in ["-", "/", ":"]):
                # Likely a date/datetime, show abbreviated
                if len(str_val) > max_width:
                    return str_val[: max_width - 1] + "…"

            # General string truncation with ellipsis
            if len(str_val) > max_width:
                return str_val[: max_width - 1] + "…"

            return str_val

        # Populate table rows
        num_rows = len(data_dict["colname"])
        for i in range(num_rows):
            row_data = []

            # Column name and type from HTML content
            colname_html = data_dict["colname"][i]
            column_name, data_type = extract_column_info(colname_html)
            row_data.append(column_name)
            row_data.append(data_type)

            # Missing values (NA)
            missing_val = data_dict.get("n_missing", [None] * num_rows)[i]
            row_data.append(format_value(missing_val, is_missing=True, max_width=6))

            # Unique values (UQ)
            unique_val = data_dict.get("n_unique", [None] * num_rows)[i]
            row_data.append(format_value(unique_val, is_unique=True, max_width=8))

            # Statistical columns
            for stat_col in stat_columns:
                stat_val = data_dict.get(stat_col, [None] * num_rows)[i]
                # Use appropriate width based on column type
                if stat_col in ["q_1", "iqr"]:
                    width = 8
                elif stat_col in ["mean", "std", "min", "median", "max", "q_3"]:
                    width = 9
                else:
                    width = 8
                row_data.append(format_value(stat_val, max_width=width))

            scan_table.add_row(*row_data)

        # Display the results
        console.print()
        console.print(scan_table)

    except Exception as e:
        # Fallback to simple message if table creation fails
        console.print(f"[yellow]Scan results available for {data_source}[/yellow]")
        console.print(f"[red]Error displaying table: {str(e)}[/red]")


def _rich_print_gt_table(
    gt_table: Any, preview_info: dict | None = None, show_summary: bool = True
) -> None:
    """Convert a GT table to Rich table and display it in the terminal.

    Args:
        gt_table: The GT table object to display
        preview_info: Optional dict with preview context info:
            - total_rows: Total rows in the dataset
            - total_columns: Total columns in the dataset
            - head_rows: Number of head rows shown
            - tail_rows: Number of tail rows shown
            - is_complete: Whether the entire dataset is shown
            - source_type: Type of data source (e.g., "External source: worldcities_new.csv")
            - table_type: Type of table (e.g., "polars")
        show_summary: Whether to show the row count summary at the bottom
    """
    try:
        # Try to extract the underlying data from the GT table
        df = None

        # Great Tables stores the original data in different places depending on how it was created
        # Let's try multiple approaches to get the data
        if hasattr(gt_table, "_tbl_data") and gt_table._tbl_data is not None:
            df = gt_table._tbl_data
        elif (
            hasattr(gt_table, "_body")
            and hasattr(gt_table._body, "body")
            and gt_table._body.body is not None
        ):
            df = gt_table._body.body
        elif hasattr(gt_table, "_data") and gt_table._data is not None:
            df = gt_table._data
        elif hasattr(gt_table, "data") and gt_table.data is not None:
            df = gt_table.data

        if df is not None:
            # Create a Rich table with horizontal lines
            from rich.box import SIMPLE_HEAD

            # Create enhanced title if preview_info contains metadata
            table_title = None
            if preview_info and "source_type" in preview_info and "table_type" in preview_info:
                source_type = preview_info["source_type"]
                table_type = preview_info["table_type"]
                table_title = f"Data Preview / {source_type} / {table_type}"

                # Add dimensions subtitle in gray if available
                total_rows = preview_info.get("total_rows")
                total_columns = preview_info.get("total_columns")
                if total_rows is not None and total_columns is not None:
                    table_title += f"\n[dim]{total_rows:,} rows / {total_columns} columns[/dim]"

            rich_table = Table(
                title=table_title,
                show_header=True,
                header_style="bold magenta",
                box=SIMPLE_HEAD,
                title_style="bold cyan",
                title_justify="left",
            )

            # Get column names
            columns = []
            if hasattr(df, "columns"):
                columns = list(df.columns)
            elif hasattr(df, "schema"):  # pragma: no cover
                columns = list(df.schema.names)
            elif hasattr(df, "column_names"):  # pragma: no cover
                columns = list(df.column_names)

            if not columns:  # pragma: no cover
                # Fallback: try to determine columns from first row
                try:
                    if hasattr(df, "to_dicts") and len(df) > 0:
                        first_dict = df.to_dicts()[0]
                        columns = list(first_dict.keys())
                    elif hasattr(df, "to_dict") and len(df) > 0:
                        first_dict = df.to_dict("records")[0]
                        columns = list(first_dict.keys())
                except Exception:  # pragma: no cover
                    columns = [f"Column {i + 1}" for i in range(10)]  # Default fallback

            # Add columns to Rich table
            # Handle wide tables by limiting columns displayed
            max_terminal_cols = 15  # Reasonable limit for terminal display

            # Get terminal width to adjust column behavior
            try:
                terminal_width = console.size.width
                # Estimate max column width based on terminal size and number of columns
                if len(columns) <= 5:
                    max_col_width = min(60, terminal_width // 4)
                elif len(columns) <= 10:
                    max_col_width = min(40, terminal_width // 6)
                else:
                    max_col_width = min(30, terminal_width // 8)
            except Exception:  # pragma: no cover
                # Fallback if we can't get terminal width
                max_col_width = 40 if len(columns) <= 10 else 25

            if len(columns) > max_terminal_cols:
                # For wide tables, show first few, middle indicator, and last few columns
                first_cols = 7
                last_cols = 7

                display_columns = columns[:first_cols] + ["...more..."] + columns[-last_cols:]

                console.print(
                    f"\n[yellow]⚠ Table has {len(columns)} columns. Showing first {first_cols} and last {last_cols} columns.[/yellow]"
                )
                console.print("[dim]Use --columns to specify which columns to display.[/dim]")
                console.print(
                    f"[dim]Full column list: {', '.join(columns[:5])}...{', '.join(columns[-5:])}[/dim]\n"
                )
            else:
                display_columns = columns

            # Get data types for columns
            dtypes_dict = _get_column_dtypes(df, columns)

            # Calculate row number column width if needed
            row_num_width = 6  # Default width
            if "_row_num_" in columns:
                try:
                    # Get the maximum row number to calculate appropriate width
                    if hasattr(df, "to_dicts"):
                        data_dict = df.to_dicts()
                        if data_dict:
                            row_nums = [row.get("_row_num_", 0) for row in data_dict]
                            max_row_num = max(row_nums) if row_nums else 0
                            row_num_width = max(len(str(max_row_num)) + 1, 6)  # +1 for padding
                    elif hasattr(df, "to_dict"):
                        data_dict = df.to_dict("records")
                        if data_dict:
                            row_nums = [row.get("_row_num_", 0) for row in data_dict]
                            max_row_num = max(row_nums) if row_nums else 0
                            row_num_width = max(len(str(max_row_num)) + 1, 6)  # +1 for padding
                except Exception:  # pragma: no cover
                    # If we can't determine max row number, use default
                    row_num_width = 8  # Slightly larger default for safety

            for i, col in enumerate(display_columns):
                if col == "...more...":
                    # Add a special indicator column
                    rich_table.add_column("···", style="dim", width=3, no_wrap=True)
                else:
                    # Handle row number column specially
                    if col == "_row_num_":
                        # Row numbers get no header, right alignment, and dim gray style
                        # Use dynamic width to prevent truncation
                        rich_table.add_column(
                            "", style="dim", justify="right", no_wrap=True, width=row_num_width
                        )
                    else:
                        display_col = str(col)

                        # Get data type for this column (if available)
                        if col in dtypes_dict:
                            dtype_display = f"<{dtypes_dict[col]}>"
                            # Create header with column name and data type
                            header_text = f"{display_col}\n[dim yellow]{dtype_display}[/dim yellow]"
                        else:
                            header_text = display_col

                        rich_table.add_column(
                            header_text,
                            style="cyan",
                            no_wrap=False,
                            overflow="ellipsis",
                            max_width=max_col_width,
                        )

            # Convert data to list of rows
            rows = []
            try:
                if hasattr(df, "to_dicts"):
                    # Polars interface
                    data_dict = df.to_dicts()
                    if len(columns) > max_terminal_cols:
                        # For wide tables, extract only the displayed columns
                        display_data_columns = (
                            columns[:7] + columns[-7:]
                        )  # Skip the "...more..." placeholder
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in display_data_columns
                            ]
                            for row in data_dict
                        ]
                        # Add the "..." column in the middle
                        for i, row in enumerate(rows):
                            rows[i] = row[:7] + ["···"] + row[7:]
                    else:
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in columns
                            ]
                            for row in data_dict
                        ]
                elif hasattr(df, "to_dict"):
                    # Pandas-like interface
                    data_dict = df.to_dict("records")
                    if len(columns) > max_terminal_cols:
                        # For wide tables, extract only the displayed columns
                        display_data_columns = columns[:7] + columns[-7:]
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in display_data_columns
                            ]
                            for row in data_dict
                        ]
                        # Add the "..." column in the middle
                        for i, row in enumerate(rows):
                            rows[i] = row[:7] + ["···"] + row[7:]
                    else:
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in columns
                            ]
                            for row in data_dict
                        ]
                elif hasattr(df, "iter_rows"):
                    # Polars lazy frame
                    rows = [
                        [
                            _format_cell_value(
                                val,
                                is_row_number=(i == 0 and columns[0] == "_row_num_"),
                                max_width=max_col_width,
                                num_columns=len(columns),
                            )
                            for i, val in enumerate(row)
                        ]
                        for row in df.iter_rows()
                    ]
                elif hasattr(df, "__iter__"):
                    # Try to iterate directly
                    rows = [
                        [
                            _format_cell_value(
                                val,
                                is_row_number=(i == 0 and columns[0] == "_row_num_"),
                                max_width=max_col_width,
                                num_columns=len(columns),
                            )
                            for i, val in enumerate(row)
                        ]
                        for row in df
                    ]
                else:
                    rows = [["Could not extract data from this format"]]  # pragma: no cover
            except Exception as e:
                rows = [[f"Error extracting data: {e}"]]  # pragma: no cover

            # Add rows to Rich table with separator between head and tail
            max_rows = 50  # Reasonable limit for terminal display

            # Get preview info to determine head/tail separation
            head_rows_count = 0
            tail_rows_count = 0
            total_dataset_rows = 0

            if preview_info:
                head_rows_count = preview_info.get("head_rows", 0)
                tail_rows_count = preview_info.get("tail_rows", 0)
                total_dataset_rows = preview_info.get("total_rows", len(rows))
                is_complete = preview_info.get("is_complete", False)
            else:
                # Fallback: assume all rows are shown
                is_complete = True

            # Add rows with optional separator
            for i, row in enumerate(rows[:max_rows]):
                try:
                    # Add separator between head and tail rows
                    if (
                        not is_complete
                        and head_rows_count > 0
                        and tail_rows_count > 0
                        and i == head_rows_count
                    ):
                        # Add a visual separator row with dashes
                        separator_row = [
                            "─" * 3 if col != "_row_num_" else "⋮"
                            for col in (
                                display_columns if "display_columns" in locals() else columns
                            )
                        ]
                        rich_table.add_row(*separator_row, style="dim")

                    rich_table.add_row(*row)
                except Exception as e:  # pragma: no cover
                    # If there's an issue with row data, show error
                    rich_table.add_row(*[f"Error: {e}" for _ in columns])  # pragma: no cover
                    break  # pragma: no cover

            # Show the table
            console.print()
            console.print(rich_table)

            # Show summary info (conditionally)
            if show_summary:
                total_rows = len(rows)

                # Use preview info if available, otherwise fall back to old logic
                if preview_info:
                    total_dataset_rows = preview_info.get("total_rows", total_rows)
                    head_rows = preview_info.get("head_rows", 0)
                    tail_rows = preview_info.get("tail_rows", 0)
                    is_complete = preview_info.get("is_complete", False)

                    if is_complete:
                        console.print(f"\n[dim]Showing all {total_rows} rows.[/dim]")
                    elif head_rows > 0 and tail_rows > 0:
                        console.print(
                            f"\n[dim]Showing first {head_rows} and last {tail_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                        )
                    elif head_rows > 0:
                        console.print(
                            f"\n[dim]Showing first {head_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                        )
                    elif tail_rows > 0:
                        console.print(
                            f"\n[dim]Showing last {tail_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                        )
                    else:
                        # Fallback for other cases
                        console.print(
                            f"\n[dim]Showing {total_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                        )
                else:
                    # Original logic as fallback
                    max_rows = 50  # This should match the limit used above
                    if total_rows > max_rows:
                        console.print(
                            f"\n[dim]Showing first {max_rows} of {total_rows} rows. Use --output-html to see all data.[/dim]"
                        )
                    else:
                        console.print(f"\n[dim]Showing all {total_rows} rows.[/dim]")

        else:
            # If we can't extract data, show the success message
            console.print(
                Panel(
                    "[green]✓[/green] Table rendered successfully. "
                    "Use --output-html to save the full interactive report.",
                    title="Table Preview",
                    border_style="green",
                )
            )

    except Exception as e:  # pragma: no cover
        console.print(f"[red]Error rendering table:[/red] {e}")
        console.print(
            f"[dim]GT table type: {type(gt_table) if 'gt_table' in locals() else 'undefined'}[/dim]"
        )

        # Fallback: show the success message
        console.print(
            Panel(
                "[green]✓[/green] Table rendered successfully. "
                "Use --output-html to save the full interactive report.",
                title="Table Preview",
                border_style="green",
            )
        )


def _display_validation_summary(validation: Any) -> None:
    """Display a validation summary in a compact Rich table format."""
    try:
        # Try to get the summary from the validation report
        if hasattr(validation, "validation_info") and validation.validation_info is not None:
            # Use the validation_info to create a summary
            info = validation.validation_info
            n_steps = len(info)

            # Count steps based on their threshold status
            n_passed = sum(
                1 for step in info if not step.warning and not step.error and not step.critical
            )
            n_all_passed = sum(1 for step in info if step.all_passed)
            n_failed = n_steps - n_passed

            # Calculate severity counts
            n_warning = sum(1 for step in info if step.warning)
            n_error = sum(1 for step in info if step.error)
            n_critical = sum(1 for step in info if step.critical)

            all_passed = n_failed == 0

            # Determine highest severity and its color
            if n_critical > 0:
                highest_severity = "critical"
                severity_color = "red"
            elif n_error > 0:
                highest_severity = "error"
                severity_color = "yellow"
            elif n_warning > 0:
                highest_severity = "warning"
                severity_color = "bright_black"  # gray
            elif n_all_passed == n_steps:
                # All steps passed AND all steps had 100% pass rate
                highest_severity = "all passed"
                severity_color = "bold green"
            else:
                # Steps passed (no threshold exceedances) but some had failing test units
                highest_severity = "passed"
                severity_color = "green"

            # Create compact summary header
            # Format: Steps: 6 / P: 3 (3 AP) / W: 3 / E: 0 / C: 0 / warning
            summary_header = (
                f"Steps: {n_steps} / P: {n_passed} ({n_all_passed} AP) / "
                f"W: {n_warning} / E: {n_error} / C: {n_critical} / "
                f"[{severity_color}]{highest_severity}[/{severity_color}]"
            )

            # Print the report title and summary
            console.print()
            console.print("[blue]Validation Report[/blue]")
            console.print(f"[white]{summary_header}[/white]")

            # Display step details
            if n_steps > 0:
                from rich.box import SIMPLE_HEAD

                steps_table = Table(
                    show_header=True,
                    header_style="bold cyan",
                    box=SIMPLE_HEAD,
                )
                steps_table.add_column("", style="dim")
                steps_table.add_column("Step", style="white")
                steps_table.add_column("Column", style="cyan")
                steps_table.add_column("Values", style="yellow")
                steps_table.add_column("Units", style="blue")
                steps_table.add_column("Pass", style="green")
                steps_table.add_column("Fail", style="red")
                steps_table.add_column("W", style="bright_black")
                steps_table.add_column("E", style="yellow")
                steps_table.add_column("C", style="red")
                steps_table.add_column("Ext", style="blue", justify="center")

                def format_units(n: int) -> str:
                    """Format large numbers with K, M, B abbreviations for values above 10,000."""
                    if n is None:
                        return "—"
                    if n >= 1000000000:  # Billions
                        return f"{n / 1000000000:.1f}B"
                    elif n >= 1000000:  # Millions
                        return f"{n / 1000000:.1f}M"
                    elif n >= 10000:  # Use K for 10,000 and above
                        return f"{n / 1000:.0f}K"
                    else:
                        return str(n)

                def format_pass_fail(passed: int, total: int) -> str:
                    """Format pass/fail counts with abbreviated numbers and fractions."""
                    if passed is None or total is None or total == 0:
                        return "—/—"

                    # Calculate fraction
                    fraction = passed / total

                    # Format fraction with special handling for very small and very large values
                    if fraction == 0.0:
                        fraction_str = "0.00"
                    elif fraction == 1.0:
                        fraction_str = "1.00"
                    elif fraction < 0.005:  # Less than 0.005 rounds to 0.00
                        fraction_str = "<0.01"
                    elif fraction > 0.995:  # Greater than 0.995 rounds to 1.00
                        fraction_str = ">0.99"
                    else:
                        fraction_str = f"{fraction:.2f}"

                    # Format absolute number with abbreviations
                    absolute_str = format_units(passed)

                    return f"{absolute_str}/{fraction_str}"

                for step in info:
                    # Extract values information for the Values column
                    values_str = "—"  # Default to em dash if no values

                    # Handle different validation types
                    if step.assertion_type == "col_schema_match":
                        values_str = "—"  # Schema is too complex to display inline
                    elif step.assertion_type == "col_vals_between":
                        # For between validations, try to get left and right bounds
                        if (
                            hasattr(step, "left")
                            and hasattr(step, "right")
                            and step.left is not None
                            and step.right is not None
                        ):
                            values_str = f"[{step.left}, {step.right}]"
                        elif hasattr(step, "values") and step.values is not None:
                            if isinstance(step.values, (list, tuple)) and len(step.values) >= 2:
                                values_str = f"[{step.values[0]}, {step.values[1]}]"
                            else:
                                values_str = str(step.values)
                    elif step.assertion_type in ["row_count_match", "col_count_match"]:
                        # For count match validations, extract the 'count' value from the dictionary
                        if hasattr(step, "values") and step.values is not None:
                            if isinstance(step.values, dict) and "count" in step.values:
                                values_str = str(step.values["count"])
                            else:
                                values_str = str(step.values)
                        else:
                            values_str = "—"
                    elif step.assertion_type in ["col_vals_expr", "conjointly"]:
                        values_str = "COLUMN EXPR"
                    elif step.assertion_type == "specially":
                        values_str = "EXPR"
                    elif hasattr(step, "values") and step.values is not None:
                        if isinstance(step.values, (list, tuple)):
                            if len(step.values) <= 3:
                                values_str = ", ".join(str(v) for v in step.values)
                            else:
                                values_str = f"{', '.join(str(v) for v in step.values[:3])}..."
                        else:
                            values_str = str(step.values)
                    elif hasattr(step, "value") and step.value is not None:
                        values_str = str(step.value)
                    elif hasattr(step, "set") and step.set is not None:
                        if isinstance(step.set, (list, tuple)):
                            if len(step.set) <= 3:
                                values_str = ", ".join(str(v) for v in step.set)
                            else:
                                values_str = f"{', '.join(str(v) for v in step.set[:3])}..."
                        else:
                            values_str = str(step.set)

                    # Determine threshold status for W, E, C columns
                    # Check if thresholds are set and whether they were exceeded

                    # Warning threshold
                    if (
                        hasattr(step, "thresholds")
                        and step.thresholds
                        and hasattr(step.thresholds, "warning")
                        and step.thresholds.warning is not None
                    ):
                        w_status = (
                            "[bright_black]●[/bright_black]"
                            if step.warning
                            else "[bright_black]○[/bright_black]"
                        )
                    else:
                        w_status = "—"

                    # Error threshold
                    if (
                        hasattr(step, "thresholds")
                        and step.thresholds
                        and hasattr(step.thresholds, "error")
                        and step.thresholds.error is not None
                    ):
                        e_status = "[yellow]●[/yellow]" if step.error else "[yellow]○[/yellow]"
                    else:
                        e_status = "—"

                    # Critical threshold
                    if (
                        hasattr(step, "thresholds")
                        and step.thresholds
                        and hasattr(step.thresholds, "critical")
                        and step.thresholds.critical is not None
                    ):
                        c_status = "[red]●[/red]" if step.critical else "[red]○[/red]"
                    else:
                        c_status = "—"

                    # Extract status, here we check if the step has any extract data
                    if (
                        hasattr(step, "extract")
                        and step.extract is not None
                        and hasattr(step.extract, "__len__")
                        and len(step.extract) > 0
                    ):
                        ext_status = "[blue]✓[/blue]"
                    else:
                        ext_status = "[bright_black]—[/bright_black]"

                    steps_table.add_row(
                        str(step.i),
                        step.assertion_type,
                        str(step.column) if step.column else "—",
                        values_str,
                        format_units(step.n),
                        format_pass_fail(step.n_passed, step.n),
                        format_pass_fail(step.n - step.n_passed, step.n),
                        w_status,
                        e_status,
                        c_status,
                        ext_status,
                    )

                console.print(steps_table)

            # Display status with appropriate color
            if highest_severity == "all passed":
                console.print(
                    Panel(
                        "[green]✓ All validations passed![/green]",
                        border_style="green",
                        expand=False,
                    )
                )
            elif highest_severity == "passed":
                console.print(
                    Panel(
                        "[dim green]⚠ Some steps had failing test units[/dim green]",
                        border_style="dim green",
                        expand=False,
                    )
                )
            elif highest_severity in ["warning", "error", "critical"]:
                if highest_severity == "warning":
                    color = "bright_black"  # gray
                elif highest_severity == "error":
                    color = "yellow"
                else:  # critical
                    color = "red"
                console.print(
                    Panel(
                        f"[{color}]✗ Validation failed with {highest_severity} severity[/{color}]",
                        border_style=color,
                        expand=False,
                    )
                )
        else:
            console.print("[yellow]Validation object does not contain validation results.[/yellow]")

    except Exception as e:  # pragma: no cover
        console.print(f"[red]Error displaying validation summary:[/red] {e}")
        import traceback  # pragma: no cover

        console.print(f"[dim]{traceback.format_exc()}[/dim]")  # pragma: no cover


@click.group(cls=OrderedGroup)
@click.version_option(pb.__version__, "-v", "--version", prog_name="pb")
@click.help_option("-h", "--help")
def cli():
    """
    Pointblank CLI: Data validation and quality tools for data engineers.

    Use this CLI to validate data quality, explore datasets, and generate comprehensive
    reports for CSV, Parquet, and database sources. Suitable for data pipelines, ETL
    validation, and exploratory data analysis from the command line.

    Quick Examples:

    \b
      pb preview data.csv              Preview your data
      pb scan data.csv                 Generate data profile
      pb validate data.csv             Run basic validation

    Use pb COMMAND --help for detailed help on any command.
    """
    pass


@cli.command()
@click.argument("data_source", type=str, required=False)
def info(data_source: str | None):
    """
    Display information about a data source.

    Shows table type, dimensions, column names, and data types.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
    """
    try:
        # Handle missing data_source with concise help
        if data_source is None:
            _show_concise_help("info", None)
            return

        with console.status("[bold green]Loading data..."):
            # Load the data source using the centralized function
            data = _load_data_source(data_source)

            # Get table information
            tbl_type = _get_tbl_type(data)
            row_count = pb.get_row_count(data)
            col_count = pb.get_column_count(data)

            # Import the box style
            from rich.box import SIMPLE_HEAD

            # Create info table
            info_table = Table(
                title="Data Source Information",
                show_header=True,
                header_style="bold magenta",
                box=SIMPLE_HEAD,
                title_style="bold cyan",
                title_justify="left",
            )
            info_table.add_column("Property", style="cyan", no_wrap=True)
            info_table.add_column("Value", style="green")

            info_table.add_row("Source", data_source)
            info_table.add_row("Table Type", tbl_type)
            info_table.add_row("Rows", f"{row_count:,}")
            info_table.add_row("Columns", f"{col_count:,}")

            console.print()
            console.print(info_table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str, required=False)
@click.option("--columns", help="Comma-separated list of columns to display")
@click.option("--col-range", help="Column range like '1:10' or '5:' or ':15' (1-based indexing)")
@click.option("--col-first", type=int, help="Show first N columns")
@click.option("--col-last", type=int, help="Show last N columns")
@click.option("--head", default=5, help="Number of rows from the top (default: 5)")
@click.option("--tail", default=5, help="Number of rows from the bottom (default: 5)")
@click.option("--limit", default=50, help="Maximum total rows to display (default: 50)")
@click.option("--no-row-numbers", is_flag=True, help="Hide row numbers")
@click.option("--max-col-width", default=250, help="Maximum column width in pixels (default: 250)")
@click.option("--min-table-width", default=500, help="Minimum table width in pixels (default: 500)")
@click.option("--no-header", is_flag=True, help="Hide table header")
@click.option("--output-html", type=click.Path(), help="Save HTML output to file")
def preview(
    data_source: str | None,
    columns: str | None,
    col_range: str | None,
    col_first: int | None,
    col_last: int | None,
    head: int,
    tail: int,
    limit: int,
    no_row_numbers: bool,
    max_col_width: int,
    min_table_width: int,
    no_header: bool,
    output_html: str | None,
):
    """
    Preview a data table showing head and tail rows.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
    - Piped data from pb pl command

    COLUMN SELECTION OPTIONS:

    For tables with many columns, use these options to control which columns are displayed:

    \b
    - --columns: Specify exact columns (--columns "name,age,email")
    - --col-range: Select column range (--col-range "1:10", --col-range "5:", --col-range ":15")
    - --col-first: Show first N columns (--col-first 5)
    - --col-last: Show last N columns (--col-last 3)

    Tables with >15 columns automatically show first 7 and last 7 columns with indicators.
    """
    try:
        import sys

        # Handle piped input
        if data_source is None:
            if not sys.stdin.isatty():
                # Data is being piped in - read the file path from stdin
                piped_input = sys.stdin.read().strip()
                if piped_input:
                    data_source = piped_input

                    # Determine the format from the file extension
                    if piped_input.endswith(".parquet"):
                        format_type = "Parquet"
                    elif piped_input.endswith(".csv"):
                        format_type = "CSV"
                    else:
                        format_type = "unknown"

                    console.print(f"[dim]Using piped data source in {format_type} format.[/dim]")
                else:
                    console.print("[red]Error:[/red] No data provided via pipe")
                    sys.exit(1)
            else:
                # Show concise help and exit
                _show_concise_help("preview", None)
                return

        with console.status("[bold green]Loading data..."):
            # Load the data source using the centralized function
            data = _load_data_source(data_source)

            # Check if this is a piped data source and create friendly display name
            is_piped_data = _is_piped_data_source(data_source)

            if is_piped_data:
                if data_source.endswith(".parquet"):
                    display_source = "Parquet file via `pb pl`"
                elif data_source.endswith(".csv"):
                    display_source = "CSV file via `pb pl`"
                else:
                    display_source = "File via `pb pl`"
                console.print(
                    f"[green]✓[/green] Loaded data source: {display_source} ({data_source})"
                )
            else:
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Parse columns if provided
        columns_list = None
        if columns:
            columns_list = [col.strip() for col in columns.split(",")]

            # If data has _row_num_ and it's not explicitly included, add it at the beginning
            try:
                # Data is already processed, just use it directly
                processed_data = data

                # Get column names from the processed data
                all_columns = []
                if hasattr(processed_data, "columns"):
                    all_columns = list(processed_data.columns)
                elif hasattr(processed_data, "schema"):
                    all_columns = list(processed_data.schema.names)

                # If _row_num_ exists in data but not in user selection, add it at beginning
                if all_columns and "_row_num_" in all_columns and "_row_num_" not in columns_list:
                    columns_list = ["_row_num_"] + columns_list
            except Exception:
                # If we can't process the data, just use the user's column list as-is
                pass
        elif col_range or col_first or col_last:
            # Need to get column names to apply range/first/last selection
            # Data is already processed, just use it directly
            processed_data = data

            # Get column names from the processed data
            all_columns = []
            if hasattr(processed_data, "columns"):
                all_columns = list(processed_data.columns)
            elif hasattr(processed_data, "schema"):
                all_columns = list(processed_data.schema.names)
            else:
                console.print(
                    "[yellow]Warning: Could not determine column names for range selection[/yellow]"
                )

            if all_columns:
                # Check if _row_num_ exists and preserve it
                has_row_num = "_row_num_" in all_columns

                if col_range:
                    # Parse range like "1:10", "5:", ":15"
                    if ":" in col_range:
                        parts = col_range.split(":")
                        start_idx = int(parts[0]) - 1 if parts[0] else 0  # Convert to 0-based
                        end_idx = int(parts[1]) if parts[1] else len(all_columns)

                        # Filter out _row_num_ from the range selection, we'll add it back later
                        columns_for_range = [col for col in all_columns if col != "_row_num_"]
                        selected_columns = columns_for_range[start_idx:end_idx]

                        # Always include _row_num_ at the beginning if it exists
                        if has_row_num:
                            columns_list = ["_row_num_"] + selected_columns
                        else:
                            columns_list = selected_columns
                    else:
                        console.print(
                            "[yellow]Warning: Invalid range format. Use 'start:end' format[/yellow]"
                        )
                elif col_first:
                    # Filter out _row_num_ from the first N selection, we'll add it back later
                    columns_for_first = [col for col in all_columns if col != "_row_num_"]
                    selected_columns = columns_for_first[:col_first]

                    # Always include _row_num_ at the beginning if it exists
                    if has_row_num:
                        columns_list = ["_row_num_"] + selected_columns
                    else:
                        columns_list = selected_columns
                elif col_last:
                    # Filter out _row_num_ from the last N selection, we'll add it back later
                    columns_for_last = [col for col in all_columns if col != "_row_num_"]
                    selected_columns = columns_for_last[-col_last:]

                    # Always include _row_num_ at the beginning if it exists
                    if has_row_num:
                        columns_list = ["_row_num_"] + selected_columns
                    else:
                        columns_list = selected_columns

        # Generate preview
        with console.status("[bold green]Generating preview..."):
            # Get total dataset size before preview and gather metadata
            try:
                # Data is already processed, just use it directly
                processed_data = data

                total_dataset_rows = pb.get_row_count(processed_data)
                total_dataset_columns = pb.get_column_count(processed_data)

                # Determine source type and table type for enhanced preview title
                if is_piped_data:
                    if data_source.endswith(".parquet"):
                        source_type = "Polars expression (serialized to Parquet) from `pb pl`"
                    elif data_source.endswith(".csv"):
                        source_type = "Polars expression (serialized to CSV) from `pb pl`"
                    else:
                        source_type = "Polars expression from `pb pl`"
                elif data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                    source_type = f"Pointblank dataset: {data_source}"
                else:
                    source_type = f"External source: {data_source}"

                table_type = _get_tbl_type(processed_data)
            except Exception:
                # If we can't get metadata, set defaults
                total_dataset_rows = None
                total_dataset_columns = None
                source_type = f"Data source: {data_source}"
                table_type = "unknown"

            gt_table = pb.preview(
                data=data,
                columns_subset=columns_list,
                n_head=head,
                n_tail=tail,
                limit=limit,
                show_row_numbers=not no_row_numbers,
                max_col_width=max_col_width,
                min_tbl_width=min_table_width,
                incl_header=not no_header,
            )

        if output_html:
            # Save HTML to file
            html_content = gt_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] HTML saved to: {output_html}")
        else:
            # Display in terminal with preview context info
            preview_info = None
            if total_dataset_rows is not None:
                # Determine if we're showing the complete dataset
                expected_rows = min(head + tail, limit, total_dataset_rows)
                is_complete = total_dataset_rows <= expected_rows

                preview_info = {
                    "total_rows": total_dataset_rows,
                    "total_columns": total_dataset_columns,
                    "head_rows": head,
                    "tail_rows": tail,
                    "is_complete": is_complete,
                    "source_type": source_type,
                    "table_type": table_type,
                }

            _rich_print_gt_table(gt_table, preview_info)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str, required=False)
@click.option("--output-html", type=click.Path(), help="Save HTML scan report to file")
@click.option("--columns", "-c", help="Comma-separated list of columns to scan")
def scan(
    data_source: str | None,
    output_html: str | None,
    columns: str | None,
):
    """
    Generate a data scan profile report.

    Produces a comprehensive data profile including:

    \b
    - Column types and distributions
    - Missing value patterns
    - Basic statistics
    - Data quality indicators

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
    - Piped data from pb pl command
    """
    try:
        import sys
        import time

        start_time = time.time()

        # Handle piped input
        if data_source is None:
            if not sys.stdin.isatty():
                # Data is being piped in - read the file path from stdin
                piped_input = sys.stdin.read().strip()
                if piped_input:
                    data_source = piped_input

                    # Determine the format from the file extension
                    if piped_input.endswith(".parquet"):
                        format_type = "Parquet"
                    elif piped_input.endswith(".csv"):
                        format_type = "CSV"
                    else:
                        format_type = "unknown"

                    console.print(f"[dim]Using piped data source in {format_type} format.[/dim]")
                else:
                    console.print("[red]Error:[/red] No data provided via pipe")
                    sys.exit(1)
            else:
                # Show concise help and exit
                _show_concise_help("scan", None)
                return

        with console.status("[bold green]Loading data..."):
            # Load the data source using the centralized function
            data = _load_data_source(data_source)

            # Check if this is a piped data source and create friendly display name
            is_piped_data = _is_piped_data_source(data_source)

            if is_piped_data:
                if data_source.endswith(".parquet"):
                    display_source = "Parquet file via `pb pl`"
                elif data_source.endswith(".csv"):
                    display_source = "CSV file via `pb pl`"
                else:
                    display_source = "File via `pb pl`"
                console.print(
                    f"[green]✓[/green] Loaded data source: {display_source} ({data_source})"
                )
            else:
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Parse columns if provided
        columns_list = None
        if columns:
            columns_list = [col.strip() for col in columns.split(",")]

        # Generate data scan
        with console.status("[bold green]Generating data scan..."):
            # Use col_summary_tbl for comprehensive column scanning
            # Data is already processed by _load_data_source
            scan_result = pb.col_summary_tbl(data=data)

            # Create friendly source type for display
            if is_piped_data:
                if data_source.endswith(".parquet"):
                    source_type = "Polars expression (serialized to Parquet) from `pb pl`"
                elif data_source.endswith(".csv"):
                    source_type = "Polars expression (serialized to CSV) from `pb pl`"
                else:
                    source_type = "Polars expression from `pb pl`"
            elif data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                source_type = f"Pointblank dataset: {data_source}"
            else:
                source_type = f"External source: {data_source}"

            table_type = _get_tbl_type(data)
            # Get row count and column count for header
            try:
                total_rows = pb.get_row_count(data)
                total_columns = pb.get_column_count(data)
            except Exception:
                total_rows = None
                total_columns = None

        scan_time = time.time() - start_time

        if output_html:
            # Save HTML to file
            try:
                html_content = scan_result.as_raw_html()
                Path(output_html).write_text(html_content, encoding="utf-8")
                console.print(f"[green]✓[/green] Data scan report saved to: {output_html}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save HTML report: {e}[/yellow]")
        else:
            # Display rich scan table in terminal
            console.print(f"[green]✓[/green] Data scan completed in {scan_time:.2f}s")
            console.print("Use --output-html to save the full interactive scan report.")

            # Display detailed column summary using rich formatting
            try:
                _rich_print_scan_table(
                    scan_result,
                    display_source if is_piped_data else data_source,
                    source_type,
                    table_type,
                    total_rows,
                    total_columns,
                )

            except Exception as e:
                console.print(f"[yellow]Could not display scan summary: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str, required=False)
@click.option("--output-html", type=click.Path(), help="Save HTML output to file")
def missing(data_source: str | None, output_html: str | None):
    """
    Generate a missing values report for a data table.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
    - Piped data from pb pl command
    """
    try:
        import sys

        # Handle piped input
        if data_source is None:
            if not sys.stdin.isatty():
                # Data is being piped in - read the file path from stdin
                piped_input = sys.stdin.read().strip()
                if piped_input:
                    data_source = piped_input

                    # Determine the format from the file extension
                    if piped_input.endswith(".parquet"):
                        format_type = "Parquet"
                    elif piped_input.endswith(".csv"):
                        format_type = "CSV"
                    else:
                        format_type = "unknown"

                    console.print(f"[dim]Using piped data source in {format_type} format.[/dim]")
                else:
                    console.print("[red]Error:[/red] No data provided via pipe")
                    sys.exit(1)
            else:
                # Show concise help and exit
                _show_concise_help("missing", None)
                return

        with console.status("[bold green]Loading data..."):
            # Load the data source using the centralized function
            data = _load_data_source(data_source)

            # Check if this is a piped data source and create friendly display name
            is_piped_data = _is_piped_data_source(data_source)

            if is_piped_data:
                if data_source.endswith(".parquet"):
                    display_source = "Parquet file via `pb pl`"
                elif data_source.endswith(".csv"):
                    display_source = "CSV file via `pb pl`"
                else:
                    display_source = "File via `pb pl`"
                console.print(
                    f"[green]✓[/green] Loaded data source: {display_source} ({data_source})"
                )
            else:
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Generate missing values table
        with console.status("[bold green]Analyzing missing values..."):
            gt_table = pb.missing_vals_tbl(data)

            # Data is already processed, just use it directly
            original_data = data

        if output_html:
            # Save HTML to file
            html_content = gt_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] Missing values report saved to: {output_html}")
        else:
            # Display in terminal with special missing values formatting
            # Create enhanced context info for missing table display
            missing_info = {}
            try:
                # Determine source type and table type for enhanced preview title
                if is_piped_data:
                    if data_source.endswith(".parquet"):
                        source_type = "Polars expression (serialized to Parquet) from `pb pl`"
                    elif data_source.endswith(".csv"):
                        source_type = "Polars expression (serialized to CSV) from `pb pl`"
                    else:
                        source_type = "Polars expression from `pb pl`"
                elif data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                    source_type = f"Pointblank dataset: {data_source}"
                else:
                    source_type = f"External source: {data_source}"

                missing_info = {
                    "source_type": source_type,
                    "table_type": _get_tbl_type(original_data),
                    "total_rows": pb.get_row_count(original_data),
                    "total_columns": pb.get_column_count(original_data),
                }
            except Exception:
                # Use defaults if metadata extraction fails
                missing_info = {
                    "source_type": f"Data source: {data_source}",
                    "table_type": "unknown",
                    "total_rows": None,
                    "total_columns": None,
                }

            _rich_print_missing_table_enhanced(gt_table, original_data, missing_info)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command(name="validate")
@click.argument("data_source", type=str, required=False)
@click.option("--list-checks", is_flag=True, help="List available validation checks and exit")
@click.option(
    "--check",
    "checks",
    type=click.Choice(
        [
            "rows-distinct",
            "col-vals-not-null",
            "rows-complete",
            "col-exists",
            "col-vals-in-set",
            "col-vals-gt",
            "col-vals-ge",
            "col-vals-lt",
            "col-vals-le",
        ]
    ),
    metavar="CHECK_TYPE",
    multiple=True,  # Allow multiple --check options
    help="Type of validation check to perform. Can be used multiple times for multiple checks.",
)
@click.option(
    "--column",
    "columns",
    multiple=True,  # Allow multiple --column options
    help="Column name or integer position as #N (1-based index) for validation.",
)
@click.option(
    "--set",
    "sets",
    multiple=True,  # Allow multiple --set options
    help="Comma-separated allowed values for col-vals-in-set checks.",
)
@click.option(
    "--value",
    "values",
    type=float,
    multiple=True,  # Allow multiple --value options
    help="Numeric value for comparison checks.",
)
@click.option(
    "--show-extract", is_flag=True, help="Show extract of failing rows if validation fails"
)
@click.option(
    "--write-extract", type=str, help="Save failing rows to folder. Provide base name for folder."
)
@click.option(
    "--limit", default=500, help="Maximum number of failing rows to save to CSV (default: 500)"
)
@click.option("--exit-code", is_flag=True, help="Exit with non-zero code if validation fails")
@click.pass_context
def validate(
    ctx: click.Context,
    data_source: str | None,
    checks: tuple[str, ...],
    columns: tuple[str, ...],
    sets: tuple[str, ...],
    values: tuple[float, ...],
    show_extract: bool,
    write_extract: str | None,
    limit: int,
    exit_code: bool,
    list_checks: bool,
):
    """
    Perform single or multiple data validations.

    Run one or more validation checks on your data in a single command.
    Use multiple --check options to perform multiple validations.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

    AVAILABLE CHECK_TYPES:

    Require no additional options:

    \b
    - rows-distinct: Check if all rows in the dataset are unique (no duplicates)
    - rows-complete: Check if all rows are complete (no missing values in any column)

    Require --column:

    \b
    - col-exists: Check if a specific column exists in the dataset
    - col-vals-not-null: Check if all values in a column are not null/missing

    Require --column and --value:

    \b
    - col-vals-gt: Check if column values are greater than a fixed value
    - col-vals-ge: Check if column values are greater than or equal to a fixed value
    - col-vals-lt: Check if column values are less than a fixed value
    - col-vals-le: Check if column values are less than or equal to a fixed value

    Require --column and --set:

    \b
    - col-vals-in-set: Check if column values are in an allowed set

    Use --list-checks to see all available validation methods with examples. The default CHECK_TYPE
    is 'rows-distinct' which checks for duplicate rows.

    Examples:

    \b
    pb validate data.csv                               # Uses default validation (rows-distinct)
    pb validate data.csv --list-checks                 # Show all available checks
    pb validate data.csv --check rows-distinct
    pb validate data.csv --check rows-distinct --show-extract
    pb validate data.csv --check rows-distinct --write-extract failing_rows_folder
    pb validate data.csv --check rows-distinct --exit-code
    pb validate data.csv --check col-exists --column price
    pb validate data.csv --check col-vals-not-null --column email
    pb validate data.csv --check col-vals-gt --column score --value 50
    pb validate data.csv --check col-vals-in-set --column status --set "active,inactive,pending"

    Multiple validations in one command:
    pb validate data.csv --check rows-distinct --check rows-complete
    """
    try:
        import sys

        # Handle --list-checks option early (doesn't need data source)
        if list_checks:
            console.print("[bold bright_cyan]Available Validation Checks:[/bold bright_cyan]")
            console.print()
            console.print("[bold magenta]Basic checks:[/bold magenta]")
            console.print(
                "  • [bold cyan]rows-distinct[/bold cyan]     Check for duplicate rows [yellow](default)[/yellow]"
            )
            console.print(
                "  • [bold cyan]rows-complete[/bold cyan]     Check for missing values in any column"
            )
            console.print()
            console.print(
                "[bold magenta]Column-specific checks [bright_black](require --column)[/bright_black]:[/bold magenta]"
            )
            console.print("  • [bold cyan]col-exists[/bold cyan]        Check if a column exists")
            console.print(
                "  • [bold cyan]col-vals-not-null[/bold cyan] Check for null values in a column"
            )
            console.print()
            console.print(
                "[bold magenta]Value comparison checks [bright_black](require --column and --value)[/bright_black]:[/bold magenta]"
            )
            console.print(
                "  • [bold cyan]col-vals-gt[/bold cyan]       Values greater than comparison value"
            )
            console.print(
                "  • [bold cyan]col-vals-ge[/bold cyan]       Values greater than or equal to comparison value"
            )
            console.print(
                "  • [bold cyan]col-vals-lt[/bold cyan]       Values less than comparison value"
            )
            console.print(
                "  • [bold cyan]col-vals-le[/bold cyan]       Values less than or equal to comparison value"
            )
            console.print()
            console.print(
                "[bold magenta]Set validation check [bright_black](requires --column and --set)[/bright_black]:[/bold magenta]"
            )
            console.print(
                "  • [bold cyan]col-vals-in-set[/bold cyan]   Values must be in allowed set"
            )
            console.print()
            console.print("[bold bright_yellow]Examples:[/bold bright_yellow]")
            console.print("  [bright_blue]pb validate data.csv --check rows-distinct[/bright_blue]")
            console.print(
                "  [bright_blue]pb validate data.csv --check col-vals-not-null --column price[/bright_blue]"
            )
            console.print(
                "  [bright_blue]pb validate data.csv --check col-vals-gt --column age --value 18[/bright_blue]"
            )
            import sys

            sys.exit(0)

        # Check if data_source is provided (required for all operations except --list-checks)
        # or if we have piped input
        if data_source is None:
            # Check if we have piped input
            if not sys.stdin.isatty():
                # Data is being piped in: read the file path from stdin
                piped_input = sys.stdin.read().strip()
                if piped_input:
                    data_source = piped_input

                    # Determine the format from the file extension
                    if piped_input.endswith(".parquet"):
                        format_type = "Parquet"
                    elif piped_input.endswith(".csv"):
                        format_type = "CSV"
                    else:
                        format_type = "unknown"

                    console.print(f"[dim]Using piped data source in {format_type} format.[/dim]")
                else:
                    console.print("[red]Error:[/red] No data provided via pipe")
                    sys.exit(1)
            else:
                # Show concise help and exit
                _show_concise_help("validate", None)
                return

        # Handle backward compatibility and parameter conversion
        import sys

        # Convert parameter tuples to lists, handling default case
        if not checks:
            # No --check options provided, use default
            checks_list = ["rows-distinct"]
            is_using_default_check = True
        else:
            checks_list = list(checks)
            is_using_default_check = False

        columns_list = list(columns) if columns else []
        sets_list = list(sets) if sets else []
        values_list = list(values) if values else []

        # Map parameters to checks intelligently
        mapped_columns, mapped_sets, mapped_values = _map_parameters_to_checks(
            checks_list, columns_list, sets_list, values_list
        )

        # Validate required parameters for different check types
        # Check parameters for each check in the list using mapped parameters
        for i, check in enumerate(checks_list):
            # Get corresponding mapped parameters for this check
            column = mapped_columns[i] if i < len(mapped_columns) else None
            set_val = mapped_sets[i] if i < len(mapped_sets) else None
            value = mapped_values[i] if i < len(mapped_values) else None

            if check == "col-vals-not-null" and not column:
                console.print(f"[red]Error:[/red] --column is required for {check} check")
                console.print(
                    "Example: pb validate data.csv --check col-vals-not-null --column email"
                )
                sys.exit(1)

            if check == "col-exists" and not column:
                console.print(f"[red]Error:[/red] --column is required for {check} check")
                console.print("Example: pb validate data.csv --check col-exists --column price")
                sys.exit(1)

            if check == "col-vals-in-set" and not column:
                console.print(f"[red]Error:[/red] --column is required for {check} check")
                console.print(
                    "Example: pb validate data.csv --check col-vals-in-set --column status --set 'active,inactive'"
                )
                sys.exit(1)

            if check == "col-vals-in-set" and not set_val:
                console.print(f"[red]Error:[/red] --set is required for {check} check")
                console.print(
                    "Example: pb validate data.csv --check col-vals-in-set --column status --set 'active,inactive'"
                )
                sys.exit(1)

            if check in ["col-vals-gt", "col-vals-ge", "col-vals-lt", "col-vals-le"] and not column:
                console.print(f"[red]Error:[/red] --column is required for {check} check")
                console.print(
                    f"Example: pb validate data.csv --check {check} --column score --value 50"
                )
                sys.exit(1)

            if (
                check in ["col-vals-gt", "col-vals-ge", "col-vals-lt", "col-vals-le"]
                and value is None
            ):
                console.print(f"[red]Error:[/red] --value is required for {check} check")
                console.print(
                    f"Example: pb validate data.csv --check {check} --column score --value 50"
                )
                sys.exit(1)

        with console.status("[bold green]Loading data..."):
            # Load the data source using the centralized function
            data = _load_data_source(data_source)

            # Get all column names for error reporting
            if hasattr(data, "columns"):
                all_columns = list(data.columns)
            elif hasattr(data, "schema"):
                all_columns = list(data.schema.names)
            else:
                all_columns = []

            # Resolve any '#N' column references to actual column names
            columns_list = _resolve_column_indices(columns_list, data)

            # Check for out-of-range #N columns and provide a helpful error
            for col in columns_list:
                if isinstance(col, str) and col.startswith("#"):
                    try:
                        idx = int(col[1:])
                        if idx < 1 or idx > len(all_columns):
                            console.print(
                                f"[red]Error:[/red] There is no column {idx} (the column position "
                                f"range is 1 to {len(all_columns)})"
                            )
                            sys.exit(1)
                    except Exception:
                        pass  # Let later validation handle other errors

            # Update mapped_columns to use resolved column names
            mapped_columns, mapped_sets, mapped_values = _map_parameters_to_checks(
                checks_list, columns_list, sets_list, values_list
            )

            # Check if this is a piped data source and create friendly display name
            is_piped_data = (
                data_source
                and data_source.startswith("/var/folders/")
                and ("pb_pipe_" in data_source or "/T/" in data_source)
            )

            if is_piped_data:
                if data_source.endswith(".parquet"):
                    display_source = "Parquet file via `pb pl`"
                elif data_source.endswith(".csv"):
                    display_source = "CSV file via `pb pl`"
                else:
                    display_source = "File via `pb pl`"
                console.print(
                    f"[green]✓[/green] Loaded data source: {display_source} ({data_source})"
                )
            else:
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Build a single validation object with chained checks
        with console.status(f"[bold green]Running {len(checks_list)} validation check(s)..."):
            # Initialize validation object
            validation = pb.Validate(
                data=data,
                tbl_name=f"Data from {data_source}",
                label=f"CLI Validation: {', '.join(checks_list)}",
            )

            # Add each check to the validation chain
            for i, check in enumerate(checks_list):
                # Get corresponding mapped parameters for this check
                column = mapped_columns[i] if i < len(mapped_columns) else None
                set_val = mapped_sets[i] if i < len(mapped_sets) else None
                value = mapped_values[i] if i < len(mapped_values) else None

                if check == "rows-distinct":
                    validation = validation.rows_distinct()
                elif check == "col-vals-not-null":
                    validation = validation.col_vals_not_null(columns=column)
                elif check == "rows-complete":
                    validation = validation.rows_complete()
                elif check == "col-exists":
                    validation = validation.col_exists(columns=column)
                elif check == "col-vals-in-set":
                    # Parse the comma-separated set values
                    allowed_values = [v.strip() for v in set_val.split(",")]
                    validation = validation.col_vals_in_set(columns=column, set=allowed_values)
                elif check == "col-vals-gt":
                    validation = validation.col_vals_gt(columns=column, value=value)
                elif check == "col-vals-ge":
                    validation = validation.col_vals_ge(columns=column, value=value)
                elif check == "col-vals-lt":
                    validation = validation.col_vals_lt(columns=column, value=value)
                elif check == "col-vals-le":
                    validation = validation.col_vals_le(columns=column, value=value)
                else:
                    console.print(f"[red]Error:[/red] Unknown check type: {check}")
                    sys.exit(1)

            # Execute all validations
            validation = validation.interrogate()
            all_passed = validation.all_passed()

            # Display completion message
            if len(checks_list) == 1:
                if is_using_default_check:
                    console.print(
                        f"[green]✓[/green] {checks_list[0]} validation completed [dim](default validation)[/dim]"
                    )
                else:
                    console.print(f"[green]✓[/green] {checks_list[0]} validation completed")
            else:
                console.print(f"[green]✓[/green] {len(checks_list)} validations completed")

        # Display results based on whether we have single or multiple checks
        if len(checks_list) == 1:
            # Single check: use current display format
            _display_validation_result(
                validation,
                checks_list,
                mapped_columns,
                mapped_sets,
                mapped_values,
                data_source,
                0,
                1,
                show_extract,
                write_extract,
                limit,
            )
        else:
            # Multiple checks: use stacked display format
            any_failed = False
            for i in range(len(checks_list)):
                console.print()  # Add spacing between results
                _display_validation_result(
                    validation,
                    checks_list,
                    mapped_columns,
                    mapped_sets,
                    mapped_values,
                    data_source,
                    i,
                    len(checks_list),
                    show_extract,
                    write_extract,
                    limit,
                )

                # Check if this validation failed
                if hasattr(validation, "validation_info") and len(validation.validation_info) > i:
                    step_info = validation.validation_info[i]
                    if step_info.n_failed > 0:
                        any_failed = True

            # Show tip about --show-extract if any failed and not already used
            if any_failed and not show_extract:
                console.print()
                console.print(
                    "[bright_blue]💡 Tip:[/bright_blue] [cyan]Use --show-extract to see the failing rows[/cyan]"
                )

        # Add informational hints when using default validation (only for single check)
        if len(checks_list) == 1 and is_using_default_check:
            console.print()
            console.print("[bold blue]ℹ️  Information:[/bold blue] Using default validation method")
            console.print("To specify a different validation, use the --check option.")
            console.print()
            console.print("[bold magenta]Common validation options:[/bold magenta]")
            console.print(
                "  • [bold cyan]--check rows-complete[/bold cyan]       Check for rows with missing values"
            )
            console.print(
                "  • [bold cyan]--check col-vals-not-null[/bold cyan]   Check for null values in a column [bright_black](requires --column)[/bright_black]"
            )
            console.print(
                "  • [bold cyan]--check col-exists[/bold cyan]          Check if a column exists [bright_black](requires --column)[/bright_black]"
            )
            console.print()
            console.print("[bold bright_yellow]Examples:[/bold bright_yellow]")
            console.print(
                f"  [bright_blue]pb validate {data_source} --check rows-complete[/bright_blue]"
            )
            console.print(
                f"  [bright_blue]pb validate {data_source} --check col-vals-not-null --column price[/bright_blue]"
            )

        # Exit with appropriate code if requested
        if exit_code and not all_passed:
            console.print("[dim]Exiting with non-zero code due to validation failure[/dim]")
            import sys

            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
def datasets():
    """
    List available built-in datasets.
    """
    from rich.box import SIMPLE_HEAD

    datasets_info = [
        ("small_table", "13 rows × 8 columns", "Small demo dataset for testing"),
        ("game_revenue", "2,000 rows × 11 columns", "Game development company revenue data"),
        ("nycflights", "336,776 rows × 18 columns", "NYC airport flights data from 2013"),
        ("global_sales", "50,000 rows × 20 columns", "Global sales data across regions"),
    ]

    table = Table(
        title="Available Pointblank Datasets", show_header=True, header_style="bold magenta"
    )

    # Create the datasets table
    table = Table(
        title="Available Pointblank Datasets",
        show_header=True,
        header_style="bold magenta",
        box=SIMPLE_HEAD,
        title_style="bold cyan",
        title_justify="left",
    )

    table.add_column("Dataset Name", style="cyan", no_wrap=True)
    table.add_column("Dimensions", style="green")
    table.add_column("Description", style="white")

    for name, dims, desc in datasets_info:
        table.add_row(name, dims, desc)

    console.print(table)
    console.print("\n[dim]Use these dataset names directly with any pb CLI command.[/dim]")
    console.print("[dim]Example: pb preview small_table[/dim]")


@cli.command()
def requirements():
    """
    Check installed dependencies and their availability.
    """
    from rich.box import SIMPLE_HEAD

    dependencies = [
        ("polars", "Polars DataFrame support"),
        ("pandas", "Pandas DataFrame support"),
        ("ibis", "Ibis backend support (DuckDB, etc.)"),
        ("duckdb", "DuckDB database support"),
        ("pyarrow", "Parquet file support"),
    ]

    # Create requirements table
    table = Table(
        title="Dependency Status",
        show_header=True,
        header_style="bold magenta",
        box=SIMPLE_HEAD,
        title_style="bold cyan",
        title_justify="left",
    )

    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Description", style="dim")

    for package, description in dependencies:
        if _is_lib_present(package):
            status = "[green]✓ Installed[/green]"
        else:
            status = "[red]✗ Not installed[/red]"

        table.add_row(package, status, description)

    console.print(table)
    console.print("\n[dim]Install missing packages to enable additional functionality.[/dim]")


def _rich_print_missing_table_enhanced(
    gt_table: Any, original_data: Any = None, missing_info: dict = None
) -> None:
    """Convert a missing values GT table to Rich table with enhanced formatting and metadata.

    Args:
        gt_table: The GT table object for missing values
        original_data: The original data source to extract column types
        missing_info: Dict with metadata including source_type, table_type, total_rows, total_columns
    """
    try:
        # Extract the underlying data from the GT table
        df = None

        if hasattr(gt_table, "_tbl_data") and gt_table._tbl_data is not None:
            df = gt_table._tbl_data
        elif hasattr(gt_table, "_data") and gt_table._data is not None:
            df = gt_table._data
        elif hasattr(gt_table, "data") and gt_table.data is not None:
            df = gt_table.data

        if df is not None:
            from rich.box import SIMPLE_HEAD

            # Extract metadata from missing_info or use defaults
            source_type = "Data source"
            table_type = "unknown"
            total_rows = None
            total_columns = None

            if missing_info:
                source_type = missing_info.get("source_type", "Data source")
                table_type = missing_info.get("table_type", "unknown")
                total_rows = missing_info.get("total_rows")
                total_columns = missing_info.get("total_columns")

            # Create enhanced title matching the scan table format
            title_text = f"Missing Values / {source_type} / {table_type}"

            # Add dimensions subtitle in gray if available
            if total_rows is not None and total_columns is not None:
                title_text += f"\n[dim]{total_rows:,} rows / {total_columns} columns[/dim]"

            # Get column names
            columns = []
            try:
                if hasattr(df, "columns"):
                    columns = list(df.columns)
                elif hasattr(df, "schema"):
                    columns = list(df.schema.names)
            except Exception as e:
                console.print(f"[red]Error getting columns:[/red] {e}")
                columns = []

            if not columns:
                columns = [f"Column {i + 1}" for i in range(10)]  # Fallback

            # Get original data to extract column types
            column_types = {}
            if original_data is not None:
                try:
                    # Get column types from original data
                    if hasattr(original_data, "columns"):
                        original_columns = list(original_data.columns)
                        column_types = _get_column_dtypes(original_data, original_columns)
                except Exception as e:
                    console.print(f"[red]Error getting column types:[/red] {e}")
                    pass  # Use empty dict as fallback

            # Add columns to Rich table with special formatting for missing values table
            sector_columns = [col for col in columns if col != "columns" and col.isdigit()]

            # Print the title first
            console.print()
            console.print(f"[bold cyan]{title_text}[/bold cyan]")

            # Show the custom spanner header if we have sector columns
            if sector_columns:
                # Create a custom header line that shows the spanner
                header_parts = []
                header_parts.append(" " * 20)  # Space for Column header
                header_parts.append(" " * 10)  # Space for Type header

                # Left-align "Row Sectors" with the first numbered column
                row_sectors_text = "Row Sectors"
                header_parts.append(row_sectors_text)

                # Print the custom spanner header
                console.print("[dim]" + "  ".join(header_parts) + "[/dim]")

                # Add a horizontal rule below the spanner
                rule_parts = []
                rule_parts.append(" " * 20)  # Space for Column header
                rule_parts.append(" " * 10)  # Space for Type header

                # Use a fixed width horizontal rule for "Row Sectors"
                horizontal_rule = "─" * 20
                rule_parts.append(horizontal_rule)

                # Print the horizontal rule
                console.print("[dim]" + "  ".join(rule_parts) + "[/dim]")

            # Create the missing values table WITHOUT the title (since we printed it above)
            rich_table = Table(
                show_header=True,
                header_style="bold magenta",
                box=SIMPLE_HEAD,
            )

            # Two separate columns: Column name (20 chars) and Data type (10 chars)
            rich_table.add_column("Column", style="cyan", no_wrap=True, width=20)
            rich_table.add_column("Type", style="yellow", no_wrap=True, width=10)

            # Sector columns: All same width, optimized for "100%" (4 chars + padding)
            for sector in sector_columns:
                rich_table.add_column(
                    sector,
                    style="cyan",
                    justify="center",
                    no_wrap=True,
                    width=5,  # Fixed width optimized for percentage values
                )

            # Convert data to rows with special formatting
            rows = []
            try:
                if hasattr(df, "to_dicts"):
                    data_dict = df.to_dicts()
                elif hasattr(df, "to_dict"):
                    data_dict = df.to_dict("records")
                else:
                    data_dict = []

                for i, row in enumerate(data_dict):
                    try:
                        # Each row should have: [column_name, data_type, sector1, sector2, ...]
                        column_name = str(row.get("columns", ""))

                        # Truncate column name to 20 characters with ellipsis if needed
                        if len(column_name) > 20:
                            truncated_name = column_name[:17] + "…"
                        else:
                            truncated_name = column_name

                        # Get data type for this column
                        if column_name in column_types:
                            dtype = column_types[column_name]
                            if len(dtype) > 10:
                                truncated_dtype = dtype[:9] + "…"
                            else:
                                truncated_dtype = dtype
                        else:
                            truncated_dtype = "?"

                        # Start building the row with column name and type
                        formatted_row = [truncated_name, truncated_dtype]

                        # Add sector values (formatted percentages)
                        for sector in sector_columns:
                            value = row.get(sector, 0.0)
                            if isinstance(value, (int, float)):
                                formatted_row.append(_format_missing_percentage(float(value)))
                            else:
                                formatted_row.append(str(value))

                        rows.append(formatted_row)

                    except Exception as e:
                        console.print(f"[red]Error processing row {i}:[/red] {e}")
                        continue

            except Exception as e:
                console.print(f"[red]Error extracting data:[/red] {e}")
                rows = [["Error extracting data", "?", *["" for _ in sector_columns]]]

            # Add rows to Rich table
            for row in rows:
                try:
                    rich_table.add_row(*row)
                except Exception as e:
                    console.print(f"[red]Error adding row:[/red] {e}")
                    break

            # Print the Rich table (without title since we already printed it)
            console.print(rich_table)

            footer_text = (
                "[dim]Symbols: [green]●[/green] = no missing vals in sector, "
                "[red]●[/red] = all vals completely missing, "
                "[cyan]x%[/cyan] = percentage missing[/dim]"
            )
            console.print(footer_text)

        else:
            # Fallback to regular table display
            _rich_print_gt_table(gt_table)

    except Exception as e:
        console.print(f"[red]Error rendering missing values table:[/red] {e}")
        # Fallback to regular table display
        _rich_print_gt_table(gt_table)


def _rich_print_scan_table(
    scan_result: Any,
    data_source: str,
    source_type: str,
    table_type: str,
    total_rows: int | None = None,
    total_columns: int | None = None,
) -> None:
    """
    Display scan results as a Rich table in the terminal with statistical measures.

    Args:
        scan_result: The GT object from col_summary_tbl()
        data_source: Name of the data source being scanned
        source_type: Type of data source (e.g., "Pointblank dataset: small_table")
        table_type: Type of table (e.g., "polars.LazyFrame")
        total_rows: Total number of rows in the dataset
        total_columns: Total number of columns in the dataset
    """
    try:
        import re

        import narwhals as nw
        from rich.box import SIMPLE_HEAD

        # Extract the underlying DataFrame from the GT object
        # The GT object has a _tbl_data attribute that contains the DataFrame
        gt_data = scan_result._tbl_data

        # Convert to Narwhals DataFrame for consistent handling
        nw_data = nw.from_native(gt_data)

        # Convert to dictionary for easier access
        data_dict = nw_data.to_dict(as_series=False)

        # Create main scan table with missing data table styling
        # Create a comprehensive title with data source, source type, and table type
        title_text = f"Column Summary / {source_type} / {table_type}"

        # Add dimensions subtitle in gray if available
        if total_rows is not None and total_columns is not None:
            title_text += f"\n[dim]{total_rows:,} rows / {total_columns} columns[/dim]"

        # Create the scan table
        scan_table = Table(
            title=title_text,
            show_header=True,
            header_style="bold magenta",
            box=SIMPLE_HEAD,
            title_style="bold cyan",
            title_justify="left",
        )

        # Add columns with specific styling and appropriate widths
        scan_table.add_column("Column", style="cyan", no_wrap=True, width=20)
        scan_table.add_column("Type", style="yellow", no_wrap=True, width=10)
        scan_table.add_column(
            "NA", style="red", width=6, justify="right"
        )  # Adjusted for better formatting
        scan_table.add_column(
            "UQ", style="green", width=8, justify="right"
        )  # Adjusted for boolean values

        # Add statistical columns if they exist with appropriate widths
        stat_columns = []
        column_mapping = {
            "mean": ("Mean", "blue", 9),
            "std": ("SD", "blue", 9),
            "min": ("Min", "yellow", 9),
            "median": ("Med", "yellow", 9),
            "max": ("Max", "yellow", 9),
            "q_1": ("Q₁", "magenta", 8),
            "q_3": ("Q₃", "magenta", 9),
            "iqr": ("IQR", "magenta", 8),
        }

        for col_key, (display_name, color, width) in column_mapping.items():
            if col_key in data_dict:
                scan_table.add_column(display_name, style=color, width=width, justify="right")
                stat_columns.append(col_key)

        # Helper function to extract column name and type from HTML
        def extract_column_info(html_content: str) -> tuple[str, str]:
            """Extract column name and type from HTML formatted content."""
            # Extract column name from first div
            name_match = re.search(r"<div[^>]*>([^<]+)</div>", html_content)
            column_name = name_match.group(1) if name_match else "Unknown"

            # Extract data type from second div (with gray color)
            type_match = re.search(r"<div[^>]*color: gray[^>]*>([^<]+)</div>", html_content)
            if type_match:
                data_type = type_match.group(1)
                # Convert to compact format using the existing function
                compact_type = _format_dtype_compact(data_type)
                data_type = compact_type
            else:
                data_type = "unknown"

            return column_name, data_type

        # Helper function to format values with improved number formatting
        def format_value(
            value: Any, is_missing: bool = False, is_unique: bool = False, max_width: int = 8
        ) -> str:
            """Format values for display with smart number formatting and HTML cleanup."""
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return "[dim]—[/dim]"

            # Handle missing values indicator
            if is_missing and str(value) == "0":
                return "[green]●[/green]"  # No missing values

            # Clean up HTML formatting from the raw data
            str_val = str(value)

            # Handle multi-line values with <br> tags FIRST: take the first line (absolute number)
            if "<br>" in str_val:
                str_val = str_val.split("<br>")[0].strip()
                # For unique values, we want just the integer part
                if is_unique:
                    try:
                        # Try to extract just the integer part for unique counts
                        num_val = float(str_val)
                        return str(int(num_val))
                    except (ValueError, TypeError):
                        pass

            # Now handle HTML content (especially from boolean unique values)
            if "<" in str_val and ">" in str_val:
                # Remove HTML tags completely for cleaner display
                str_val = re.sub(r"<[^>]+>", "", str_val).strip()
                # Clean up extra whitespace
                str_val = re.sub(r"\s+", " ", str_val).strip()

            # Handle values like "2<.01": extract the first number
            if "<" in str_val and not (str_val.startswith("<") and str_val.endswith(">")):
                # Extract number before the < symbol
                before_lt = str_val.split("<")[0].strip()
                if before_lt and before_lt.replace(".", "").replace("-", "").isdigit():
                    str_val = before_lt

            # Handle boolean unique values like "T0.62F0.38": extract the more readable format
            if re.match(r"^[TF]\d+\.\d+[TF]\d+\.\d+$", str_val):
                # Extract T and F values
                t_match = re.search(r"T(\d+\.\d+)", str_val)
                f_match = re.search(r"F(\d+\.\d+)", str_val)
                if t_match and f_match:
                    t_val = float(t_match.group(1))
                    f_val = float(f_match.group(1))
                    # Show as "T0.62F0.38" but truncated if needed
                    formatted = f"T{t_val:.2f}F{f_val:.2f}"
                    if len(formatted) > max_width:
                        # Truncate to fit, showing dominant value
                        if t_val > f_val:
                            return f"T{t_val:.1f}"
                        else:
                            return f"F{f_val:.1f}"
                    return formatted

            # Try to parse as a number for better formatting
            try:
                # Try to convert to float first
                num_val = float(str_val)

                # Handle special cases
                if num_val == 0:
                    return "0"
                elif abs(num_val) == int(abs(num_val)) and abs(num_val) < 10000:
                    # Simple integers under 10000
                    return str(int(num_val))
                elif abs(num_val) >= 10000000 and abs(num_val) < 100000000:
                    # Likely dates in YYYYMMDD format: format as date-like
                    int_val = int(num_val)
                    if 19000101 <= int_val <= 29991231:  # Reasonable date range
                        str_date = str(int_val)
                        if len(str_date) == 8:
                            return (
                                f"{str_date[:4]}-{str_date[4:6]}-{str_date[6:]}"[: max_width - 1]
                                + "…"
                            )
                    # Otherwise treat as large number
                    return f"{num_val / 1000000:.1f}M"
                elif abs(num_val) >= 1000000:
                    # Large numbers: use scientific notation or M/k notation

                    if abs(num_val) >= 1000000000:
                        return f"{num_val:.1e}"
                    else:
                        return f"{num_val / 1000000:.1f}M"
                elif abs(num_val) >= 10000:
                    # Numbers >= 10k: use compact notation
                    return f"{num_val / 1000:.1f}k"
                elif abs(num_val) >= 100:
                    # Numbers 100-9999: show with minimal decimals
                    return f"{num_val:.1f}"
                elif abs(num_val) >= 10:
                    # Numbers 10-99: show with one decimal
                    return f"{num_val:.1f}"
                elif abs(num_val) >= 1:
                    # Numbers 1-9: show with two decimals
                    return f"{num_val:.2f}"
                elif abs(num_val) >= 0.01:
                    # Small numbers: show with appropriate precision
                    return f"{num_val:.2f}"
                else:
                    # Very small numbers: use scientific notation

                    return f"{num_val:.1e}"

            except (ValueError, TypeError):
                # Not a number, handle as string
                pass

            # Handle date/datetime strings: show abbreviated format
            if len(str_val) > 10 and any(char in str_val for char in ["-", "/", ":"]):
                # Likely a date/datetime, show abbreviated
                if len(str_val) > max_width:
                    return str_val[: max_width - 1] + "…"

            # General string truncation with ellipsis
            if len(str_val) > max_width:
                return str_val[: max_width - 1] + "…"

            return str_val

        # Populate table rows
        num_rows = len(data_dict["colname"])
        for i in range(num_rows):
            row_data = []

            # Column name and type from HTML content
            colname_html = data_dict["colname"][i]
            column_name, data_type = extract_column_info(colname_html)
            row_data.append(column_name)
            row_data.append(data_type)

            # Missing values (NA)
            missing_val = data_dict.get("n_missing", [None] * num_rows)[i]
            row_data.append(format_value(missing_val, is_missing=True, max_width=6))

            # Unique values (UQ)
            unique_val = data_dict.get("n_unique", [None] * num_rows)[i]
            row_data.append(format_value(unique_val, is_unique=True, max_width=8))

            # Statistical columns
            for stat_col in stat_columns:
                stat_val = data_dict.get(stat_col, [None] * num_rows)[i]
                # Use appropriate width based on column type
                if stat_col in ["q_1", "iqr"]:
                    width = 8
                elif stat_col in ["mean", "std", "min", "median", "max", "q_3"]:
                    width = 9
                else:
                    width = 8
                row_data.append(format_value(stat_val, max_width=width))

            scan_table.add_row(*row_data)

        # Display the results
        console.print()
        console.print(scan_table)

    except Exception as e:
        # Fallback to simple message if table creation fails
        console.print(f"[yellow]Scan results available for {data_source}[/yellow]")
        console.print(f"[red]Error displaying table: {str(e)}[/red]")


def _rich_print_missing_table(gt_table: Any, original_data: Any = None) -> None:
    """Convert a missing values GT table to Rich table with special formatting.

    Args:
        gt_table: The GT table object for missing values
        original_data: The original data source to extract column types
    """
    try:
        # Extract the underlying data from the GT table
        df = None

        if hasattr(gt_table, "_tbl_data") and gt_table._tbl_data is not None:
            df = gt_table._tbl_data
        elif hasattr(gt_table, "_data") and gt_table._data is not None:
            df = gt_table._data
        elif hasattr(gt_table, "data") and gt_table.data is not None:
            df = gt_table.data

        if df is not None:
            from rich.box import SIMPLE_HEAD

            # Get metadata for enhanced missing table title
            total_rows = None
            total_columns = None
            source_type = "Data source"
            table_type = "unknown"

            if original_data is not None:
                try:
                    total_rows = pb.get_row_count(original_data)
                    total_columns = pb.get_column_count(original_data)
                    table_type = _get_tbl_type(original_data)
                except Exception:
                    pass

            # Create enhanced title matching the scan table format
            title_text = f"Missing Values / {source_type} / {table_type}"

            # Add dimensions subtitle in gray if available
            if total_rows is not None and total_columns is not None:
                title_text += f"\n[dim]{total_rows:,} rows / {total_columns} columns[/dim]"

            # Create the missing values table with enhanced title
            rich_table = Table(
                title=title_text,
                show_header=True,
                header_style="bold magenta",
                box=SIMPLE_HEAD,
                title_style="bold cyan",
                title_justify="left",
            )

            # Get column names
            columns = []
            try:
                if hasattr(df, "columns"):
                    columns = list(df.columns)
                elif hasattr(df, "schema"):
                    columns = list(df.schema.names)
            except Exception as e:
                console.print(f"[red]Error getting columns:[/red] {e}")
                columns = []

            if not columns:
                columns = [f"Column {i + 1}" for i in range(10)]  # Fallback

            # Get original data to extract column types
            column_types = {}
            if original_data is not None:
                try:
                    # Get column types from original data
                    if hasattr(original_data, "columns"):
                        original_columns = list(original_data.columns)
                        column_types = _get_column_dtypes(original_data, original_columns)
                except Exception as e:
                    console.print(f"[red]Error getting column types:[/red] {e}")
                    pass  # Use empty dict as fallback

            # Add columns to Rich table with special formatting for missing values table
            sector_columns = [col for col in columns if col != "columns" and col.isdigit()]

            # Two separate columns: Column name (20 chars) and Data type (10 chars)
            rich_table.add_column("Column", style="cyan", no_wrap=True, width=20)
            rich_table.add_column("Type", style="yellow", no_wrap=True, width=10)

            # Sector columns: All same width, optimized for "100%" (4 chars + padding)
            for sector in sector_columns:
                rich_table.add_column(
                    sector,
                    style="cyan",
                    justify="center",
                    no_wrap=True,
                    width=5,  # Fixed width optimized for percentage values
                )

            # Convert data to rows with special formatting
            rows = []
            try:
                if hasattr(df, "to_dicts"):
                    data_dict = df.to_dicts()
                elif hasattr(df, "to_dict"):
                    data_dict = df.to_dict("records")
                else:
                    data_dict = []

                for i, row in enumerate(data_dict):
                    try:
                        # Each row should have: [column_name, data_type, sector1, sector2, ...]
                        column_name = str(row.get("columns", ""))

                        # Truncate column name to 20 characters with ellipsis if needed
                        if len(column_name) > 20:
                            truncated_name = column_name[:17] + "…"
                        else:
                            truncated_name = column_name

                        # Get data type for this column
                        if column_name in column_types:
                            dtype = column_types[column_name]
                            if len(dtype) > 10:
                                truncated_dtype = dtype[:9] + "…"
                            else:
                                truncated_dtype = dtype
                        else:
                            truncated_dtype = "?"

                        # Start building the row with column name and type
                        formatted_row = [truncated_name, truncated_dtype]

                        # Add sector values (formatted percentages)
                        for sector in sector_columns:
                            value = row.get(sector, 0.0)
                            if isinstance(value, (int, float)):
                                formatted_row.append(_format_missing_percentage(float(value)))
                            else:
                                formatted_row.append(str(value))

                        rows.append(formatted_row)

                    except Exception as e:
                        console.print(f"[red]Error processing row {i}:[/red] {e}")
                        continue

            except Exception as e:
                console.print(f"[red]Error extracting data:[/red] {e}")
                rows = [["Error extracting data", "?", *["" for _ in sector_columns]]]

            # Add rows to Rich table
            for row in rows:
                try:
                    rich_table.add_row(*row)
                except Exception as e:
                    console.print(f"[red]Error adding row:[/red] {e}")
                    break

            # Show the table with custom spanner header if we have sector columns
            if sector_columns:
                # Create a custom header line that shows the spanner
                header_parts = []
                header_parts.append(" " * 20)  # Space for Column header
                header_parts.append(" " * 10)  # Space for Type header

                # Left-align "Row Sectors" with the first numbered column
                row_sectors_text = "Row Sectors"
                header_parts.append(row_sectors_text)

                # Print the custom spanner header
                console.print("[dim]" + "  ".join(header_parts) + "[/dim]")

                # Add a horizontal rule below the spanner
                rule_parts = []
                rule_parts.append(" " * 20)  # Space for Column header
                rule_parts.append(" " * 10)  # Space for Type header

                # Use a fixed width horizontal rule for "Row Sectors"
                horizontal_rule = "─" * 20
                rule_parts.append(horizontal_rule)

                # Print the horizontal rule
                console.print("[dim]" + "  ".join(rule_parts) + "[/dim]")

            # Print the Rich table (will handle terminal width automatically)
            console.print()
            console.print(rich_table)
            footer_text = (
                "[dim]Symbols: [green]●[/green] = no missing vals in sector, "
                "[red]●[/red] = all vals completely missing, "
                "[cyan]x%[/cyan] = percentage missing[/dim]"
            )
            console.print(footer_text)

        else:
            # Fallback to regular table display
            _rich_print_gt_table(gt_table)

    except Exception as e:
        console.print(f"[red]Error rendering missing values table:[/red] {e}")
        # Fallback to regular table display
        _rich_print_gt_table(gt_table)


def _map_parameters_to_checks(
    checks_list: list[str], columns_list: list[str], sets_list: list[str], values_list: list[float]
) -> tuple[list[str], list[str], list[float]]:
    """
    Map parameters to checks intelligently, handling flexible parameter ordering.

    This function distributes the provided parameters across checks based on what each check needs.
    For checks that don't need certain parameters, None/empty values are assigned.

    Args:
        checks_list: List of validation check types
        columns_list: List of column names provided by user
        sets_list: List of set values provided by user
        values_list: List of numeric values provided by user

    Returns:
        Tuple of (mapped_columns, mapped_sets, mapped_values) where each list
        has the same length as checks_list
    """
    mapped_columns = []
    mapped_sets = []
    mapped_values = []

    # Keep track of which parameters we've used
    column_index = 0
    set_index = 0
    value_index = 0

    for check in checks_list:
        # Determine what parameters this check needs
        needs_column = check in [
            "col-vals-not-null",
            "col-exists",
            "col-vals-in-set",
            "col-vals-gt",
            "col-vals-ge",
            "col-vals-lt",
            "col-vals-le",
        ]
        needs_set = check == "col-vals-in-set"
        needs_value = check in ["col-vals-gt", "col-vals-ge", "col-vals-lt", "col-vals-le"]

        # Assign column parameter if needed
        if needs_column:
            if column_index < len(columns_list):
                mapped_columns.append(columns_list[column_index])
                column_index += 1
            else:
                mapped_columns.append(None)  # Will cause validation error later
        else:
            mapped_columns.append(None)

        # Assign set parameter if needed
        if needs_set:
            if set_index < len(sets_list):
                mapped_sets.append(sets_list[set_index])
                set_index += 1
            else:
                mapped_sets.append(None)  # Will cause validation error later
        else:
            mapped_sets.append(None)

        # Assign value parameter if needed
        if needs_value:
            if value_index < len(values_list):
                mapped_values.append(values_list[value_index])
                value_index += 1
            else:
                mapped_values.append(None)  # Will cause validation error later
        else:
            mapped_values.append(None)

    return mapped_columns, mapped_sets, mapped_values


def _resolve_column_indices(columns_list, data):
    """
    Replace any '#N' entries in columns_list with the actual column name from data (1-based).
    """
    # Get column names from the data
    if hasattr(data, "columns"):
        all_columns = list(data.columns)
    elif hasattr(data, "schema"):
        all_columns = list(data.schema.names)
    else:
        return columns_list  # Can't resolve, return as-is

    resolved = []
    for col in columns_list:
        if isinstance(col, str) and col.startswith("#"):
            try:
                idx = int(col[1:]) - 1  # 1-based to 0-based
                if 0 <= idx < len(all_columns):
                    resolved.append(all_columns[idx])
                else:
                    resolved.append(col)  # Out of range, keep as-is
            except Exception:
                resolved.append(col)  # Not a valid number, keep as-is
        else:
            resolved.append(col)
    return resolved


def _display_validation_result(
    validation: Any,
    checks_list: list[str],
    columns_list: list[str],
    sets_list: list[str],
    values_list: list[float],
    data_source: str,
    step_index: int,
    total_checks: int,
    show_extract: bool,
    write_extract: str | None,
    limit: int,
) -> None:
    """Display a single validation result with proper formatting for single or multiple checks."""
    from rich.box import SIMPLE_HEAD

    # Get parameters for this specific check
    check = checks_list[step_index]
    column = columns_list[step_index] if step_index < len(columns_list) else None
    set_val = sets_list[step_index] if step_index < len(sets_list) else None
    value = values_list[step_index] if step_index < len(values_list) else None

    # Check if this is piped data
    is_piped_data = _is_piped_data_source(data_source)

    # Create friendly display name for data source
    if is_piped_data:
        if data_source.endswith(".parquet"):
            display_source = "Polars expression (serialized to Parquet) from `pb pl`"
        elif data_source.endswith(".csv"):
            display_source = "Polars expression (serialized to CSV) from `pb pl`"
        else:
            display_source = "Polars expression from `pb pl`"
    else:
        display_source = data_source

    # Get validation step info
    step_info = None
    if hasattr(validation, "validation_info") and len(validation.validation_info) > step_index:
        step_info = validation.validation_info[step_index]

    # Create friendly title for table
    if total_checks == 1:
        # Single check: use original title format
        if check == "rows-distinct":
            table_title = "Validation Result: Rows Distinct"
        elif check == "col-vals-not-null":
            table_title = "Validation Result: Column Values Not Null"
        elif check == "rows-complete":
            table_title = "Validation Result: Rows Complete"
        elif check == "col-exists":
            table_title = "Validation Result: Column Exists"
        elif check == "col-vals-in-set":
            table_title = "Validation Result: Column Values In Set"
        elif check == "col-vals-gt":
            table_title = "Validation Result: Column Values Greater Than"
        elif check == "col-vals-ge":
            table_title = "Validation Result: Column Values Greater Than Or Equal"
        elif check == "col-vals-lt":
            table_title = "Validation Result: Column Values Less Than"
        elif check == "col-vals-le":
            table_title = "Validation Result: Column Values Less Than Or Equal"
        else:
            table_title = f"Validation Result: {check.replace('-', ' ').title()}"
    else:
        # Multiple checks: add numbering
        if check == "rows-distinct":
            base_title = "Rows Distinct"
        elif check == "col-vals-not-null":
            base_title = "Column Values Not Null"
        elif check == "rows-complete":
            base_title = "Rows Complete"
        elif check == "col-exists":
            base_title = "Column Exists"
        elif check == "col-vals-in-set":
            base_title = "Column Values In Set"
        elif check == "col-vals-gt":
            base_title = "Column Values Greater Than"
        elif check == "col-vals-ge":
            base_title = "Column Values Greater Than Or Equal"
        elif check == "col-vals-lt":
            base_title = "Column Values Less Than"
        elif check == "col-vals-le":
            base_title = "Column Values Less Than Or Equal"
        else:
            base_title = check.replace("-", " ").title()

        table_title = f"Validation Result ({step_index + 1} of {total_checks}): {base_title}"

    # Create the validation results table
    result_table = Table(
        title=table_title,
        show_header=True,
        header_style="bold magenta",
        box=SIMPLE_HEAD,
        title_style="bold cyan",
        title_justify="left",
    )
    result_table.add_column("Property", style="cyan", no_wrap=True)
    result_table.add_column("Value", style="white")

    # Add basic info
    result_table.add_row("Data Source", display_source)
    result_table.add_row("Check Type", check)

    # Add column info for column-specific checks
    if check in [
        "col-vals-not-null",
        "col-exists",
        "col-vals-in-set",
        "col-vals-gt",
        "col-vals-ge",
        "col-vals-lt",
        "col-vals-le",
    ]:
        result_table.add_row("Column", column)

    # Add set info for col-vals-in-set check
    if check == "col-vals-in-set" and set_val:
        allowed_values = [v.strip() for v in set_val.split(",")]
        result_table.add_row("Allowed Values", ", ".join(allowed_values))

    # Add value info for range checks
    if check in ["col-vals-gt", "col-vals-ge", "col-vals-lt", "col-vals-le"] and value is not None:
        if check == "col-vals-gt":
            operator = ">"
        elif check == "col-vals-ge":
            operator = ">="
        elif check == "col-vals-lt":
            operator = "<"
        elif check == "col-vals-le":
            operator = "<="
        result_table.add_row("Comparison Value", f"{operator} {value}")

    # Get validation details
    if step_info:
        result_table.add_row("Total Rows Tested", f"{step_info.n:,}")
        result_table.add_row("Passing Rows", f"{step_info.n_passed:,}")
        result_table.add_row("Failing Rows", f"{step_info.n_failed:,}")

        # Check if this step passed
        step_passed = step_info.n_failed == 0

        # Overall result with color coding
        if step_passed:
            result_table.add_row("Result", "[green]✓ PASSED[/green]")
            if check == "rows-distinct":
                result_table.add_row("Duplicate Rows", "[green]None found[/green]")
            elif check == "col-vals-not-null":
                result_table.add_row("Null Values", "[green]None found[/green]")
            elif check == "rows-complete":
                result_table.add_row("Incomplete Rows", "[green]None found[/green]")
            elif check == "col-exists":
                result_table.add_row("Column Status", "[green]Column exists[/green]")
            elif check == "col-vals-in-set":
                result_table.add_row("Values Status", "[green]All values in allowed set[/green]")
            elif check == "col-vals-gt":
                result_table.add_row("Values Status", f"[green]All values > {value}[/green]")
            elif check == "col-vals-ge":
                result_table.add_row("Values Status", f"[green]All values >= {value}[/green]")
            elif check == "col-vals-lt":
                result_table.add_row("Values Status", f"[green]All values < {value}[/green]")
            elif check == "col-vals-le":
                result_table.add_row("Values Status", f"[green]All values <= {value}[/green]")
        else:
            result_table.add_row("Result", "[red]✗ FAILED[/red]")
            if check == "rows-distinct":
                result_table.add_row("Duplicate Rows", f"[red]{step_info.n_failed:,} found[/red]")
            elif check == "col-vals-not-null":
                result_table.add_row("Null Values", f"[red]{step_info.n_failed:,} found[/red]")
            elif check == "rows-complete":
                result_table.add_row("Incomplete Rows", f"[red]{step_info.n_failed:,} found[/red]")
            elif check == "col-exists":
                result_table.add_row("Column Status", "[red]Column does not exist[/red]")
            elif check == "col-vals-in-set":
                result_table.add_row("Invalid Values", f"[red]{step_info.n_failed:,} found[/red]")
            elif check == "col-vals-gt":
                result_table.add_row(
                    "Invalid Values", f"[red]{step_info.n_failed:,} values <= {value}[/red]"
                )
            elif check == "col-vals-ge":
                result_table.add_row(
                    "Invalid Values", f"[red]{step_info.n_failed:,} values < {value}[/red]"
                )
            elif check == "col-vals-lt":
                result_table.add_row(
                    "Invalid Values", f"[red]{step_info.n_failed:,} values >= {value}[/red]"
                )
            elif check == "col-vals-le":
                result_table.add_row(
                    "Invalid Values", f"[red]{step_info.n_failed:,} values > {value}[/red]"
                )

    console.print()
    console.print(result_table)

    # Show extract and summary for single check only, or if this is a failed step in multiple checks
    if total_checks == 1:
        # For single check, show extract and summary as before
        _show_extract_and_summary(
            validation,
            check,
            column,
            set_val,
            value,
            data_source,
            step_index,
            step_info,
            show_extract,
            write_extract,
            limit,
        )
    else:
        # For multiple checks, show summary panel and handle extract if needed
        if step_info:
            step_passed = step_info.n_failed == 0
            if step_passed:
                # Create success message for this step
                if check == "rows-distinct":
                    success_message = f"[green]✓ Validation PASSED: No duplicate rows found in {data_source}[/green]"
                elif check == "col-vals-not-null":
                    success_message = f"[green]✓ Validation PASSED: No null values found in column '{column}' in {data_source}[/green]"
                elif check == "rows-complete":
                    success_message = f"[green]✓ Validation PASSED: All rows are complete (no missing values) in {data_source}[/green]"
                elif check == "col-exists":
                    success_message = f"[green]✓ Validation PASSED: Column '{column}' exists in {data_source}[/green]"
                elif check == "col-vals-in-set":
                    success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are in the allowed set in {data_source}[/green]"
                elif check == "col-vals-gt":
                    success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are > {value} in {data_source}[/green]"
                elif check == "col-vals-ge":
                    success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are >= {value} in {data_source}[/green]"
                elif check == "col-vals-lt":
                    success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are < {value} in {data_source}[/green]"
                elif check == "col-vals-le":
                    success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are <= {value} in {data_source}[/green]"
                else:
                    success_message = f"[green]✓ Validation PASSED: {check} check passed for {data_source}[/green]"

                console.print(
                    Panel(
                        success_message,
                        border_style="green",
                        expand=False,
                    )
                )
            else:
                # Create failure message for this step (without tip)
                if check == "rows-distinct":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} duplicate rows found in {data_source}[/red]"
                elif check == "col-vals-not-null":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} null values found in column '{column}' in {data_source}[/red]"
                elif check == "rows-complete":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} incomplete rows found in {data_source}[/red]"
                elif check == "col-exists":
                    failure_message = f"[red]✗ Validation FAILED: Column '{column}' does not exist in {data_source}[/red]"
                elif check == "col-vals-in-set":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} invalid values found in column '{column}' in {data_source}[/red]"
                elif check == "col-vals-gt":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values <= {value} found in column '{column}' in {data_source}[/red]"
                elif check == "col-vals-ge":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values < {value} found in column '{column}' in {data_source}[/red]"
                elif check == "col-vals-lt":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values >= {value} found in column '{column}' in {data_source}[/red]"
                elif check == "col-vals-le":
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values > {value} found in column '{column}' in {data_source}[/red]"
                else:
                    failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} failing rows found in {data_source}[/red]"

                console.print(
                    Panel(
                        failure_message,
                        border_style="red",
                        expand=False,
                    )
                )

                # For multiple checks, show extract if requested and this step failed
                if (show_extract or write_extract) and not step_passed:
                    _show_extract_for_multi_check(
                        validation,
                        check,
                        column,
                        set_val,
                        value,
                        data_source,
                        step_index,
                        step_info,
                        show_extract,
                        write_extract,
                        limit,
                    )


def _show_extract_for_multi_check(
    validation: Any,
    check: str,
    column: str | None,
    set_val: str | None,
    value: float | None,
    data_source: str,
    step_index: int,
    step_info: Any,
    show_extract: bool,
    write_extract: str | None,
    limit: int,
) -> None:
    """Show extract for a single validation step in multiple checks scenario."""
    # Dynamic message based on check type
    if check == "rows-distinct":
        extract_message = "[yellow]Extract of failing rows (duplicates):[/yellow]"
        row_type = "duplicate rows"
    elif check == "rows-complete":
        extract_message = "[yellow]Extract of failing rows (incomplete rows):[/yellow]"
        row_type = "incomplete rows"
    elif check == "col-exists":
        extract_message = f"[yellow]Column '{column}' does not exist in the dataset[/yellow]"
        row_type = "missing column"
    elif check == "col-vals-not-null":
        extract_message = f"[yellow]Extract of failing rows (null values in '{column}'):[/yellow]"
        row_type = "rows with null values"
    elif check == "col-vals-in-set":
        extract_message = (
            f"[yellow]Extract of failing rows (invalid values in '{column}'):[/yellow]"
        )
        row_type = "rows with invalid values"
    elif check == "col-vals-gt":
        extract_message = (
            f"[yellow]Extract of failing rows (values in '{column}' <= {value}):[/yellow]"
        )
        row_type = f"rows with values <= {value}"
    elif check == "col-vals-ge":
        extract_message = (
            f"[yellow]Extract of failing rows (values in '{column}' < {value}):[/yellow]"
        )
        row_type = f"rows with values < {value}"
    elif check == "col-vals-lt":
        extract_message = (
            f"[yellow]Extract of failing rows (values in '{column}' >= {value}):[/yellow]"
        )
        row_type = f"rows with values >= {value}"
    elif check == "col-vals-le":
        extract_message = (
            f"[yellow]Extract of failing rows (values in '{column}' > {value}):[/yellow]"
        )
        row_type = f"rows with values > {value}"
    else:
        extract_message = "[yellow]Extract of failing rows:[/yellow]"
        row_type = "failing rows"

    if show_extract:
        console.print()
        console.print(extract_message)

    # Special handling for col-exists check: no rows to show when column doesn't exist
    if check == "col-exists":
        if show_extract:
            console.print(f"[dim]The column '{column}' was not found in the dataset.[/dim]")
            console.print(
                "[dim]Use --show-extract with other check types to see failing data rows.[/dim]"
            )
        if write_extract:
            console.print("[yellow]Cannot save failing rows when column doesn't exist[/yellow]")
    else:
        try:
            # Get failing rows extract: use step_index + 1 since extracts are 1-indexed
            failing_rows = validation.get_data_extracts(i=step_index + 1, frame=True)

            if failing_rows is not None and len(failing_rows) > 0:
                if show_extract:
                    # Always limit to 10 rows for display, regardless of limit option
                    display_limit = 10
                    if len(failing_rows) > display_limit:
                        display_rows = failing_rows.head(display_limit)
                        console.print(
                            f"[dim]Showing first {display_limit} of {len(failing_rows)} {row_type}[/dim]"
                        )
                    else:
                        display_rows = failing_rows
                        console.print(f"[dim]Showing all {len(failing_rows)} {row_type}[/dim]")

                    # Create a preview table using pointblank's preview function
                    import pointblank as pb

                    preview_table = pb.preview(
                        data=display_rows,
                        n_head=min(display_limit, len(display_rows)),
                        n_tail=0,
                        limit=display_limit,
                        show_row_numbers=True,
                    )

                    # Display using our Rich table function
                    _rich_print_gt_table(preview_table, show_summary=False)

                if write_extract:
                    try:
                        from pathlib import Path

                        folder_name = write_extract

                        # Create the output folder
                        output_folder = Path(folder_name)
                        output_folder.mkdir(parents=True, exist_ok=True)

                        # Create safe filename from check type
                        safe_check_type = check.replace("-", "_")
                        filename = f"step_{step_index + 1:02d}_{safe_check_type}.csv"
                        filepath = output_folder / filename

                        # Use limit option for write_extract
                        write_rows = failing_rows
                        if len(failing_rows) > limit:
                            write_rows = failing_rows.head(limit)

                        # Save to CSV
                        if hasattr(write_rows, "write_csv"):
                            # Polars
                            write_rows.write_csv(str(filepath))
                        elif hasattr(write_rows, "to_csv"):
                            # Pandas
                            write_rows.to_csv(str(filepath), index=False)
                        else:
                            # Try converting to pandas as fallback
                            import pandas as pd

                            pd_data = pd.DataFrame(write_rows)
                            pd_data.to_csv(str(filepath), index=False)

                        rows_saved = len(write_rows) if hasattr(write_rows, "__len__") else limit
                        console.print(
                            f"[green]✓[/green] Failing rows saved to folder: {output_folder}"
                        )
                        console.print(f"[dim]  - {filename}: {rows_saved} rows[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not save failing rows: {e}[/yellow]")
            else:
                if show_extract:
                    console.print("[yellow]No failing rows could be extracted[/yellow]")
                if write_extract:
                    console.print("[yellow]No failing rows could be extracted to save[/yellow]")
        except Exception as e:
            if show_extract:
                console.print(f"[yellow]Could not extract failing rows: {e}[/yellow]")
            if write_extract:
                console.print(f"[yellow]Could not extract failing rows to save: {e}[/yellow]")


def _show_extract_and_summary(
    validation: Any,
    check: str,
    column: str | None,
    set_val: str | None,
    value: float | None,
    data_source: str,
    step_index: int,
    step_info: Any,
    show_extract: bool,
    write_extract: str | None,
    limit: int,
) -> None:
    """Show extract and summary for a validation step (used for single checks)."""
    step_passed = step_info.n_failed == 0 if step_info else True

    # Get the friendly display name
    is_piped_data = _is_piped_data_source(data_source)
    if is_piped_data:
        if data_source.endswith(".parquet"):
            display_source = "Polars expression (serialized to Parquet) from `pb pl`"
        elif data_source.endswith(".csv"):
            display_source = "Polars expression (serialized to CSV) from `pb pl`"
        else:
            display_source = "Polars expression from `pb pl`"
    else:
        display_source = data_source

    # Show extract if requested and validation failed
    if (show_extract or write_extract) and not step_passed:
        console.print()

        # Dynamic message based on check type
        if check == "rows-distinct":
            extract_message = "[yellow]Extract of failing rows (duplicates):[/yellow]"
            row_type = "duplicate rows"
        elif check == "rows-complete":
            extract_message = "[yellow]Extract of failing rows (incomplete rows):[/yellow]"
            row_type = "incomplete rows"
        elif check == "col-exists":
            extract_message = f"[yellow]Column '{column}' does not exist in the dataset[/yellow]"
            row_type = "missing column"
        elif check == "col-vals-not-null":
            extract_message = (
                f"[yellow]Extract of failing rows (null values in '{column}'):[/yellow]"
            )
            row_type = "rows with null values"
        elif check == "col-vals-in-set":
            extract_message = (
                f"[yellow]Extract of failing rows (invalid values in '{column}'):[/yellow]"
            )
            row_type = "rows with invalid values"
        elif check == "col-vals-gt":
            extract_message = (
                f"[yellow]Extract of failing rows (values in '{column}' <= {value}):[/yellow]"
            )
            row_type = f"rows with values <= {value}"
        elif check == "col-vals-ge":
            extract_message = (
                f"[yellow]Extract of failing rows (values in '{column}' < {value}):[/yellow]"
            )
            row_type = f"rows with values < {value}"
        elif check == "col-vals-lt":
            extract_message = (
                f"[yellow]Extract of failing rows (values in '{column}' >= {value}):[/yellow]"
            )
            row_type = f"rows with values >= {value}"
        elif check == "col-vals-le":
            extract_message = (
                f"[yellow]Extract of failing rows (values in '{column}' > {value}):[/yellow]"
            )
            row_type = f"rows with values > {value}"
        else:
            extract_message = "[yellow]Extract of failing rows:[/yellow]"
            row_type = "failing rows"

        if show_extract:
            console.print(extract_message)

        # Special handling for col-exists check: no rows to show when column doesn't exist
        if check == "col-exists" and not step_passed:
            if show_extract:
                console.print(f"[dim]The column '{column}' was not found in the dataset.[/dim]")
                console.print(
                    "[dim]Use --show-extract with other check types to see failing data rows.[/dim]"
                )
            if write_extract:
                console.print("[yellow]Cannot save failing rows when column doesn't exist[/yellow]")
        else:
            try:
                # Get failing rows extract: use step_index + 1 since extracts are 1-indexed
                failing_rows = validation.get_data_extracts(i=step_index + 1, frame=True)

                if failing_rows is not None and len(failing_rows) > 0:
                    if show_extract:
                        # Always limit to 10 rows for display, regardless of limit option
                        display_limit = 10
                        if len(failing_rows) > display_limit:
                            display_rows = failing_rows.head(display_limit)
                            console.print(
                                f"[dim]Showing first {display_limit} of {len(failing_rows)} {row_type}[/dim]"
                            )
                        else:
                            display_rows = failing_rows
                            console.print(f"[dim]Showing all {len(failing_rows)} {row_type}[/dim]")

                        # Create a preview table using pointblank's preview function
                        import pointblank as pb

                        preview_table = pb.preview(
                            data=display_rows,
                            n_head=min(display_limit, len(display_rows)),
                            n_tail=0,
                            limit=display_limit,
                            show_row_numbers=True,
                        )

                        # Display using our Rich table function
                        _rich_print_gt_table(preview_table, show_summary=False)

                    if write_extract:
                        try:
                            from pathlib import Path

                            folder_name = write_extract

                            # Create the output folder
                            output_folder = Path(folder_name)
                            output_folder.mkdir(parents=True, exist_ok=True)

                            # Create safe filename from check type
                            safe_check_type = check.replace("-", "_")
                            filename = f"step_{step_index + 1:02d}_{safe_check_type}.csv"
                            filepath = output_folder / filename

                            # Use limit option for write_extract
                            write_rows = failing_rows
                            if len(failing_rows) > limit:
                                write_rows = failing_rows.head(limit)

                            # Save to CSV
                            if hasattr(write_rows, "write_csv"):
                                # Polars
                                write_rows.write_csv(str(filepath))
                            elif hasattr(write_rows, "to_csv"):
                                # Pandas
                                write_rows.to_csv(str(filepath), index=False)
                            else:
                                # Try converting to pandas as fallback
                                import pandas as pd

                                pd_data = pd.DataFrame(write_rows)
                                pd_data.to_csv(str(filepath), index=False)

                            rows_saved = (
                                len(write_rows) if hasattr(write_rows, "__len__") else limit
                            )
                            console.print(
                                f"[green]✓[/green] Failing rows saved to folder: {output_folder}"
                            )
                            console.print(f"[dim]  - {filename}: {rows_saved} rows[/dim]")
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning: Could not save failing rows: {e}[/yellow]"
                            )
                else:
                    if show_extract:
                        console.print("[yellow]No failing rows could be extracted[/yellow]")
                    if write_extract:
                        console.print("[yellow]No failing rows could be extracted to save[/yellow]")
            except Exception as e:
                if show_extract:
                    console.print(f"[yellow]Could not extract failing rows: {e}[/yellow]")
                if write_extract:
                    console.print(f"[yellow]Could not extract failing rows to save: {e}[/yellow]")

    # Summary message
    console.print()
    if step_passed:
        if check == "rows-distinct":
            success_message = (
                f"[green]✓ Validation PASSED: No duplicate rows found in {display_source}[/green]"
            )
        elif check == "col-vals-not-null":
            success_message = f"[green]✓ Validation PASSED: No null values found in column '{column}' in {display_source}[/green]"
        elif check == "rows-complete":
            success_message = f"[green]✓ Validation PASSED: All rows are complete (no missing values) in {display_source}[/green]"
        elif check == "col-exists":
            success_message = (
                f"[green]✓ Validation PASSED: Column '{column}' exists in {display_source}[/green]"
            )
        elif check == "col-vals-in-set":
            success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are in the allowed set in {display_source}[/green]"
        elif check == "col-vals-gt":
            success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are > {value} in {display_source}[/green]"
        elif check == "col-vals-ge":
            success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are >= {value} in {display_source}[/green]"
        elif check == "col-vals-lt":
            success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are < {value} in {display_source}[/green]"
        elif check == "col-vals-le":
            success_message = f"[green]✓ Validation PASSED: All values in column '{column}' are <= {value} in {display_source}[/green]"
        else:
            success_message = (
                f"[green]✓ Validation PASSED: {check} check passed for {display_source}[/green]"
            )

        console.print(Panel(success_message, border_style="green", expand=False))
    else:
        if step_info:
            if check == "rows-distinct":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} duplicate rows found in {display_source}[/red]"
            elif check == "col-vals-not-null":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} null values found in column '{column}' in {display_source}[/red]"
            elif check == "rows-complete":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} incomplete rows found in {display_source}[/red]"
            elif check == "col-exists":
                failure_message = f"[red]✗ Validation FAILED: Column '{column}' does not exist in {display_source}[/red]"
            elif check == "col-vals-in-set":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} invalid values found in column '{column}' in {display_source}[/red]"
            elif check == "col-vals-gt":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values <= {value} found in column '{column}' in {display_source}[/red]"
            elif check == "col-vals-ge":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values < {value} found in column '{column}' in {display_source}[/red]"
            elif check == "col-vals-lt":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values >= {value} found in column '{column}' in {display_source}[/red]"
            elif check == "col-vals-le":
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} values > {value} found in column '{column}' in {display_source}[/red]"
            else:
                failure_message = f"[red]✗ Validation FAILED: {step_info.n_failed:,} failing rows found in {display_source}[/red]"

            # Add hint about --show-extract if not already used (except for col-exists which has no rows to show)
            if not show_extract and check != "col-exists":
                failure_message += "\n[bright_blue]💡 Tip:[/bright_blue] [cyan]Use --show-extract to see the failing rows[/cyan]"

            console.print(Panel(failure_message, border_style="red", expand=False))
        else:
            if check == "rows-distinct":
                failure_message = (
                    f"[red]✗ Validation FAILED: Duplicate rows found in {display_source}[/red]"
                )
            elif check == "rows-complete":
                failure_message = (
                    f"[red]✗ Validation FAILED: Incomplete rows found in {display_source}[/red]"
                )
            else:
                failure_message = (
                    f"[red]✗ Validation FAILED: {check} check failed for {display_source}[/red]"
                )

            # Add hint about --show-extract if not already used
            if not show_extract:
                failure_message += "\n[bright_blue]💡 Tip:[/bright_blue] [cyan]Use --show-extract to see the failing rows[/cyan]"

            console.print(Panel(failure_message, border_style="red", expand=False))


@cli.command()
@click.argument("output_file", type=click.Path(), required=False)
def make_template(output_file: str | None):
    """
    Create a validation script or YAML configuration template.

    Creates a sample Python script or YAML configuration with examples showing how to use Pointblank
    for data validation. The template type is determined by the file extension:
    - .py files create Python script templates
    - .yaml/.yml files create YAML configuration templates

    Edit the template to add your own data loading and validation rules, then run it with 'pb run'.

    OUTPUT_FILE is the path where the template will be created.

    Examples:

    \b
    pb make-template my_validation.py        # Creates Python script template
    pb make-template my_validation.yaml      # Creates YAML config template
    pb make-template validation_template.yml # Creates YAML config template
    """
    # Handle missing output_file with concise help
    if output_file is None:
        _show_concise_help("make-template", None)
        return

    # Detect file type based on extension
    file_path = Path(output_file)
    file_extension = file_path.suffix.lower()

    is_yaml_file = file_extension in [".yaml", ".yml"]
    is_python_file = file_extension == ".py"

    if not is_yaml_file and not is_python_file:
        console.print(
            f"[yellow]Warning:[/yellow] Unknown file extension '{file_extension}'. "
            "Creating Python template by default. Use .py, .yaml, or .yml extensions for specific template types."
        )
        is_python_file = True

    if is_yaml_file:
        # Create YAML template
        example_yaml = """# Example Pointblank YAML validation configuration
#
# This YAML file demonstrates how to create validation rules for your data.
# Modify the data source and validation steps below to match your requirements.
#
# When using 'pb run' with --data option, the CLI will automatically replace
# the 'tbl' field with the provided data source.

# Data source configuration
tbl: small_table  # Replace with your data source
                  # Can be: dataset name, CSV file, Parquet file, database connection, etc.

# Optional: Table name for reporting (defaults to filename if not specified)
tbl_name: "Example Validation"

# Optional: Label for this validation run
label: "Validation Template"

# Optional: Validation thresholds (defaults shown below)
# thresholds:
#   warning: 0.05   # 5% failure rate triggers warning
#   error: 0.10     # 10% failure rate triggers error
#   critical: 0.15  # 15% failure rate triggers critical

# Validation steps to perform
steps:
  # Check for duplicate rows across all columns
  - rows_distinct

  # Check that required columns exist
  - col_exists:
      columns: [column1, column2]  # Replace with your actual column names

  # Check for null values in important columns
  - col_vals_not_null:
      columns: important_column    # Replace with your actual column name

  # Check value ranges (uncomment and modify as needed)
  # - col_vals_gt:
  #     columns: amount
  #     value: 0

  # - col_vals_between:
  #     columns: score
  #     left: 0
  #     right: 100

  # Check string patterns (uncomment and modify as needed)
  # - col_vals_regex:
  #     columns: email
  #     pattern: "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"

  # Check for unique values (uncomment and modify as needed)
  # - col_vals_unique:
  #     columns: id

  # Check values are in allowed set (uncomment and modify as needed)
  # - col_vals_in_set:
  #     columns: status
  #     set: [active, inactive, pending]

# Add more validation steps as needed
# See the Pointblank documentation for the full list of available validation functions
"""

        Path(output_file).write_text(example_yaml)
        console.print(f"[green]✓[/green] YAML validation template created: {output_file}")
        console.print("\nEdit the template to add your data source and validation rules, then run:")
        console.print(f"[cyan]pb run {output_file}[/cyan]")
        console.print(
            f"[cyan]pb run {output_file} --data your_data.csv[/cyan]  [dim]# Override data source[/dim]"
        )

    else:
        # Create Python template
        example_script = '''"""
Example Pointblank validation script.

This script demonstrates how to create validation rules for your data.
Modify the data loading and validation rules below to match your requirements.

When using 'pb run' with --data option, the CLI will automatically replace
the data source in your validation object with the provided data.
"""

import pointblank as pb

# Load your data (replace this with your actual data source)
# You can load from various sources:
# data = pb.load_dataset("small_table")  # Built-in dataset
# data = pd.read_csv("your_data.csv")    # CSV file
# data = pl.read_parquet("data.parquet") # Parquet file
# data = pb.load_data("database://connection") # Database

data = pb.load_dataset("small_table")  # Example with built-in dataset

# Create a validation object
validation = (
    pb.Validate(
        data=data,
        tbl_name="Example Data",
        label="Validation Example",
        thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
    )
    # Add your validation rules here
    # Example rules (modify these based on your data structure):

    # Check that specific columns exist
    # .col_exists(["column1", "column2"])

    # Check for null values
    # .col_vals_not_null(columns="important_column")

    # Check value ranges
    # .col_vals_gt(columns="amount", value=0)
    # .col_vals_between(columns="score", left=0, right=100)

    # Check string patterns
    # .col_vals_regex(columns="email", pattern=r"^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$")

    # Check unique values
    # .col_vals_unique(columns="id")

    # Finalize the validation
    .interrogate()
)
'''

        Path(output_file).write_text(example_script)
        console.print(f"[green]✓[/green] Python validation template created: {output_file}")
        console.print(
            "\nEdit the template to add your data loading and validation rules, then run:"
        )
        console.print(f"[cyan]pb run {output_file}[/cyan]")
        console.print(
            f"[cyan]pb run {output_file} --data your_data.csv[/cyan]  [dim]# Replace data source automatically[/dim]"
        )


@cli.command()
@click.argument("validation_file", type=click.Path(exists=True), required=False)
@click.option(
    "--data",
    type=str,
    help="Data source to replace in validation objects (Python scripts and YAML configs)",
)
@click.option("--output-html", type=click.Path(), help="Save HTML validation report to file")
@click.option("--output-json", type=click.Path(), help="Save JSON validation summary to file")
@click.option(
    "--show-extract", is_flag=True, help="Show extract of failing rows if validation fails"
)
@click.option(
    "--write-extract",
    type=str,
    help="Save failing rows to folders (one CSV per step). Provide base name for folder.",
)
@click.option(
    "--limit", default=500, help="Maximum number of failing rows to save to CSV (default: 500)"
)
@click.option(
    "--fail-on",
    type=click.Choice(["critical", "error", "warning", "any"], case_sensitive=False),
    help="Exit with non-zero code when validation reaches this threshold level",
)
def run(
    validation_file: str | None,
    data: str | None,
    output_html: str | None,
    output_json: str | None,
    show_extract: bool,
    write_extract: str | None,
    limit: int,
    fail_on: str | None,
):
    """
    Run a Pointblank validation script or YAML configuration.

    VALIDATION_FILE can be:
    - A Python file (.py) that defines validation logic
    - A YAML configuration file (.yaml, .yml) that defines validation steps

    Python scripts should load their own data and create validation objects.
    YAML configurations define data sources and validation steps declaratively.

    If --data is provided, it will automatically replace the data source in your
    validation objects (Python scripts) or override the 'tbl' field (YAML configs).

    To get started quickly, use 'pb make-template' to create templates.

    DATA can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

    Examples:

    \b
    pb make-template my_validation.py  # Create a Python template
    pb run validation_script.py
    pb run validation_config.yaml
    pb run validation_script.py --data data.csv
    pb run validation_config.yaml --data small_table --output-html report.html
    pb run validation_script.py --show-extract --fail-on error
    pb run validation_config.yaml --write-extract extracts_folder --fail-on critical
    """
    try:
        # Handle missing validation_file with concise help
        if validation_file is None:
            _show_concise_help("run", None)
            return

        # Detect file type based on extension
        file_path = Path(validation_file)
        file_extension = file_path.suffix.lower()

        is_yaml_file = file_extension in [".yaml", ".yml"]
        is_python_file = file_extension == ".py"

        if not is_yaml_file and not is_python_file:
            console.print(
                f"[red]Error:[/red] Unsupported file type '{file_extension}'. "
                "Only .py (Python scripts) and .yaml/.yml (YAML configs) are supported."
            )
            sys.exit(1)

        # Load optional data override if provided
        cli_data = None
        if data:
            with console.status(f"[bold green]Loading data from {data}..."):
                cli_data = _load_data_source(data)
                console.print(f"[green]✓[/green] Loaded data override: {data}")

        # Process based on file type
        validations = []

        if is_yaml_file:
            # Handle YAML configuration file
            from pointblank.yaml import YAMLValidationError, YAMLValidator, yaml_interrogate

            with console.status("[bold green]Running YAML validation..."):
                try:
                    if cli_data is not None:
                        # Load and modify YAML config to use CLI data
                        console.print(
                            "[yellow]Replacing data source in YAML config with CLI data[/yellow]"
                        )

                        validator = YAMLValidator()
                        config = validator.load_config(validation_file)

                        # Replace the 'tbl' field with our CLI data
                        # Note: We pass the CLI data object directly instead of a string
                        config["tbl"] = cli_data

                        # Build and execute validation with modified config
                        validation = validator.execute_workflow(config)

                    else:
                        # Use YAML config as-is
                        validation = yaml_interrogate(validation_file)

                    validations.append(validation)

                except YAMLValidationError as e:
                    console.print(f"[red]YAML validation error:[/red] {e}")
                    sys.exit(1)

        else:
            # Handle Python script file
            with console.status("[bold green]Running Python validation script..."):
                # Read and execute the validation script
                script_content = Path(validation_file).read_text()

                # Create a namespace with pointblank and optional CLI data
                namespace = {
                    "pb": pb,
                    "pointblank": pb,
                    "cli_data": cli_data,  # Available if --data was provided
                    "__name__": "__main__",
                    "__file__": str(Path(validation_file).resolve()),
                }

                # Execute the script
                try:
                    exec(script_content, namespace)
                except Exception as e:
                    console.print(f"[red]Error executing validation script:[/red] {e}")
                    sys.exit(1)

                # Look for validation objects in the namespace
                # Look for the 'validation' variable specifically first
                if "validation" in namespace:
                    validations.append(namespace["validation"])

                # Also look for any other validation objects
                for key, value in namespace.items():
                    if (
                        key != "validation"
                        and hasattr(value, "interrogate")
                        and hasattr(value, "validation_info")
                    ):
                        validations.append(value)
                    # Also check if it's a Validate object that has been interrogated
                    elif key != "validation" and str(type(value)).find("Validate") != -1:
                        validations.append(value)

                if not validations:
                    raise ValueError(
                        "No validation objects found in script. "
                        "Script should create Validate objects and call .interrogate() on them."
                    )

        console.print(f"[green]✓[/green] Found {len(validations)} validation object(s)")

        # Implement automatic data replacement for Python scripts only (YAML configs handle this differently)
        if cli_data is not None and is_python_file:
            # Check if we have multiple validations (this is not supported for Python scripts)
            if len(validations) > 1:
                console.print(
                    f"[red]Error: Found {len(validations)} validation objects in the Python script.[/red]"
                )
                console.print(
                    "[yellow]The --data option replaces data in ALL validation objects,[/yellow]"
                )
                console.print(
                    "[yellow]which may cause failures if validations expect different schemas.[/yellow]"
                )
                console.print("\n[cyan]Options:[/cyan]")
                console.print("  1. Split your script into separate files with one validation each")
                console.print(
                    "  2. Remove the --data option to use each validation's original data"
                )
                sys.exit(1)

            console.print(
                f"[yellow]Replacing data in {len(validations)} validation object(s) with CLI data[/yellow]"
            )

            for idx, validation in enumerate(validations, 1):
                # Check if it's a Validate object with data attribute
                if hasattr(validation, "data") and hasattr(validation, "interrogate"):
                    console.print("[cyan]Updating validation with new data source...[/cyan]")

                    # Store the original validation_info as our "plan"
                    original_validation_info = validation.validation_info.copy()

                    # Replace the data
                    validation.data = cli_data

                    # Re-process the data (same as what happens in __post_init__)
                    from pointblank.validate import _process_data

                    validation.data = _process_data(validation.data)

                    # Reset validation results but keep the plan
                    validation.validation_info = []

                    # Re-add each validation step from the original plan
                    for val_info in original_validation_info:
                        # Create a copy and reset any interrogation results
                        new_val_info = copy.deepcopy(val_info)
                        # Reset interrogation-specific attributes if they exist
                        if hasattr(new_val_info, "n_passed"):
                            new_val_info.n_passed = None
                        if hasattr(new_val_info, "n_failed"):
                            new_val_info.n_failed = None
                        if hasattr(new_val_info, "all_passed"):
                            new_val_info.all_passed = None
                        if hasattr(new_val_info, "warning"):
                            new_val_info.warning = None
                        if hasattr(new_val_info, "error"):
                            new_val_info.error = None
                        if hasattr(new_val_info, "critical"):
                            new_val_info.critical = None
                        validation.validation_info.append(new_val_info)

                    # Re-interrogate with the new data
                    console.print("[cyan]Re-interrogating with new data...[/cyan]")
                    validation.interrogate()

        # Process each validation
        overall_failed = False
        overall_critical = False
        overall_error = False
        overall_warning = False

        for i, validation in enumerate(validations, 1):
            if len(validations) > 1:
                console.print(f"\n[bold cyan]Validation {i}:[/bold cyan]")

            # Display summary
            _display_validation_summary(validation)

            # Check failure status
            validation_failed = False
            has_critical = False
            has_error = False
            has_warning = False

            if hasattr(validation, "validation_info") and validation.validation_info:
                for step_info in validation.validation_info:
                    if step_info.critical:
                        has_critical = True
                        overall_critical = True
                    if step_info.error:
                        has_error = True
                        overall_error = True
                    if step_info.warning:
                        has_warning = True
                        overall_warning = True
                    if step_info.n_failed > 0:
                        validation_failed = True
                        overall_failed = True

            # Handle extract functionality for failed validations
            failed_steps = []
            if (
                validation_failed
                and hasattr(validation, "validation_info")
                and validation.validation_info
            ):
                for j, step_info in enumerate(validation.validation_info, 1):
                    if step_info.n_failed > 0:
                        failed_steps.append((j, step_info))

            if validation_failed and failed_steps and (show_extract or write_extract):
                console.print()

                if show_extract:
                    extract_title = "Extract of failing rows from validation steps"
                    if len(validations) > 1:
                        extract_title += f" (Validation {i})"
                    console.print(f"[yellow]{extract_title}:[/yellow]")

                    for step_num, step_info in failed_steps:
                        try:
                            failing_rows = validation.get_data_extracts(i=step_num, frame=True)

                            if failing_rows is not None and len(failing_rows) > 0:
                                console.print(
                                    f"\n[cyan]Step {step_num}:[/cyan] {step_info.assertion_type}"
                                )

                                # Always limit to 10 rows for display, regardless of limit option
                                display_limit = 10
                                if len(failing_rows) > display_limit:
                                    display_rows = failing_rows.head(display_limit)
                                    console.print(
                                        f"[dim]Showing first {display_limit} of {len(failing_rows)} failing rows[/dim]"
                                    )
                                else:
                                    display_rows = failing_rows
                                    console.print(
                                        f"[dim]Showing all {len(failing_rows)} failing rows[/dim]"
                                    )

                                # Create a preview table using pointblank's preview function
                                preview_table = pb.preview(
                                    data=display_rows,
                                    n_head=min(display_limit, len(display_rows)),
                                    n_tail=0,
                                    limit=display_limit,
                                    show_row_numbers=True,
                                )

                                # Display using our Rich table function
                                _rich_print_gt_table(preview_table, show_summary=False)
                            else:
                                console.print(
                                    f"\n[cyan]Step {step_num}:[/cyan] {step_info.assertion_type}"
                                )
                                console.print("[yellow]No failing rows could be extracted[/yellow]")
                        except Exception as e:
                            console.print(
                                f"\n[cyan]Step {step_num}:[/cyan] {step_info.assertion_type}"
                            )
                            console.print(f"[yellow]Could not extract failing rows: {e}[/yellow]")

                if write_extract:
                    try:
                        folder_name = write_extract

                        # Add validation number if multiple validations
                        if len(validations) > 1:
                            folder_name = f"{folder_name}_validation_{i}"

                        # Create the output folder
                        output_folder = Path(folder_name)
                        output_folder.mkdir(parents=True, exist_ok=True)

                        saved_files = []

                        # Save each failing step to its own CSV file
                        for step_num, step_info in failed_steps:
                            try:
                                failing_rows = validation.get_data_extracts(i=step_num, frame=True)
                                if failing_rows is not None and len(failing_rows) > 0:
                                    # Create safe filename from assertion type
                                    safe_assertion_type = (
                                        step_info.assertion_type.replace(" ", "_")
                                        .replace("/", "_")
                                        .replace("\\", "_")
                                        .replace(":", "_")
                                        .replace("<", "_")
                                        .replace(">", "_")
                                        .replace("|", "_")
                                        .replace("?", "_")
                                        .replace("*", "_")
                                        .replace('"', "_")
                                    )

                                    filename = f"step_{step_num:02d}_{safe_assertion_type}.csv"
                                    filepath = output_folder / filename

                                    # Use limit for CSV output
                                    save_rows = failing_rows
                                    if hasattr(failing_rows, "head") and len(failing_rows) > limit:
                                        save_rows = failing_rows.head(limit)

                                    # Save to CSV
                                    if hasattr(save_rows, "write_csv"):
                                        # Polars
                                        save_rows.write_csv(str(filepath))
                                    elif hasattr(save_rows, "to_csv"):
                                        # Pandas
                                        save_rows.to_csv(str(filepath), index=False)
                                    else:
                                        # Try converting to pandas as fallback
                                        import pandas as pd

                                        pd_data = pd.DataFrame(save_rows)
                                        pd_data.to_csv(str(filepath), index=False)

                                    # Record the actual number of rows saved
                                    rows_saved = (
                                        len(save_rows) if hasattr(save_rows, "__len__") else limit
                                    )
                                    saved_files.append((filename, rows_saved))

                            except Exception as e:
                                console.print(
                                    f"[yellow]Warning: Could not save failing rows from step {step_num}: {e}[/yellow]"
                                )

                        if saved_files:
                            console.print(
                                f"[green]✓[/green] Failing rows saved to folder: {output_folder}"
                            )
                            for filename, row_count in saved_files:
                                console.print(f"[dim]  - {filename}: {row_count} rows[/dim]")
                        else:
                            console.print(
                                "[yellow]No failing rows could be extracted to save[/yellow]"
                            )

                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Could not save failing rows to CSV: {e}[/yellow]"
                        )

        # Save HTML and JSON outputs (combine multiple validations if needed)
        if output_html:
            try:
                if len(validations) == 1:
                    # Single validation: save directly
                    html_content = validations[0]._repr_html_()
                    Path(output_html).write_text(html_content, encoding="utf-8")
                else:
                    # Multiple validations: combine them
                    html_parts = []
                    html_parts.append("<html><body>")
                    html_parts.append("<h1>Pointblank Validation Report</h1>")

                    for i, validation in enumerate(validations, 1):
                        html_parts.append(f"<h2>Validation {i}</h2>")
                        html_parts.append(validation._repr_html_())

                    html_parts.append("</body></html>")
                    html_content = "\n".join(html_parts)
                    Path(output_html).write_text(html_content, encoding="utf-8")

                console.print(f"[green]✓[/green] HTML report saved to: {output_html}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save HTML report: {e}[/yellow]")

        if output_json:
            try:
                if len(validations) == 1:
                    # Single validation: save directly
                    json_report = validations[0].get_json_report()
                    Path(output_json).write_text(json_report, encoding="utf-8")
                else:
                    # Multiple validations: combine them
                    import json

                    combined_report = {"validations": []}

                    for i, validation in enumerate(validations, 1):
                        validation_json = json.loads(validation.get_json_report())
                        validation_json["validation_id"] = i
                        combined_report["validations"].append(validation_json)

                    Path(output_json).write_text(
                        json.dumps(combined_report, indent=2), encoding="utf-8"
                    )

                console.print(f"[green]✓[/green] JSON summary saved to: {output_json}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save JSON report: {e}[/yellow]")

        # Check if we should fail based on threshold
        if fail_on:
            should_exit = False
            exit_reason = ""

            if fail_on.lower() == "critical" and overall_critical:
                should_exit = True
                exit_reason = "critical validation failures"
            elif fail_on.lower() == "error" and (overall_critical or overall_error):
                should_exit = True
                exit_reason = "error or critical validation failures"
            elif fail_on.lower() == "warning" and (
                overall_critical or overall_error or overall_warning
            ):
                should_exit = True
                exit_reason = "warning, error, or critical validation failures"
            elif fail_on.lower() == "any" and overall_failed:
                should_exit = True
                exit_reason = "validation failures"

            if should_exit:
                console.print(f"[red]Exiting with error due to {exit_reason}[/red]")
                sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _format_missing_percentage(value: float) -> str:
    """Format missing value percentages for display.

    Args:
        value: The percentage value (0-100)

    Returns:
        Formatted string with proper percentage display
    """
    if value == 0.0:
        return "[green]●[/green]"  # Large green circle for no missing values
    elif value == 100.0:
        return "[red]●[/red]"  # Large red circle for completely missing values
    elif value < 1.0 and value > 0:
        return "<1%"  # Less than 1%
    elif value > 99.0 and value < 100.0:
        return ">99%"  # More than 99%
    else:
        return f"{int(round(value))}%"  # Round to nearest integer with % sign


@cli.command()
@click.argument("polars_expression", type=str, required=False)
@click.option("--edit", "-e", is_flag=True, help="Open editor for multi-line input")
@click.option("--file", "-f", type=click.Path(exists=True), help="Read query from file")
@click.option(
    "--editor", help="Editor to use for --edit mode (overrides $EDITOR and auto-detection)"
)
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["preview", "scan", "missing", "info"]),
    default="preview",
    help="Output format for the result",
)
@click.option("--preview-head", default=5, help="Number of head rows for preview")
@click.option("--preview-tail", default=5, help="Number of tail rows for preview")
@click.option("--output-html", type=click.Path(), help="Save HTML output to file")
@click.option(
    "--pipe", is_flag=True, help="Output data in a format suitable for piping to other pb commands"
)
@click.option(
    "--pipe-format",
    type=click.Choice(["parquet", "csv"]),
    default="parquet",
    help="Format for piped output (default: parquet)",
)
def pl(
    polars_expression: str | None,
    edit: bool,
    file: str | None,
    editor: str | None,
    output_format: str,
    preview_head: int,
    preview_tail: int,
    output_html: str | None,
    pipe: bool,
    pipe_format: str,
):
    """
    Execute Polars expressions and display results.

    Execute Polars DataFrame operations from the command line and display
    the results using Pointblank's visualization tools.

    POLARS_EXPRESSION should be a valid Polars expression that returns a DataFrame.
    The 'pl' module is automatically imported and available.

    Examples:

    \b
    # Direct expression
    pb pl "pl.read_csv('data.csv')"
    pb pl "pl.read_csv('data.csv').select(['name', 'age'])"
    pb pl "pl.read_csv('data.csv').filter(pl.col('age') > 25)"

    \b
    # Multi-line with editor (supports multiple statements)
    pb pl --edit

    \b
    # Multi-statement code example in editor:
    # csv = pl.read_csv('data.csv')
    # result = csv.select(['name', 'age']).filter(pl.col('age') > 25)

    \b
    # Multi-line with a specific editor
    pb pl --edit --editor nano
    pb pl --edit --editor code
    pb pl --edit --editor micro

    \b
    # From file
    pb pl --file query.py

    \b
    Piping to other pb commands
    pb pl "pl.read_csv('data.csv').head(20)" --pipe | pb validate --check rows-distinct
    pb pl --edit --pipe | pb preview --head 10
    pb pl --edit --pipe | pb scan --output-html report.html
    pb pl --edit --pipe | pb missing --output-html missing_report.html

    \b
    Use --output-format to change how results are displayed:
    pb pl "pl.read_csv('data.csv')" --output-format scan
    pb pl "pl.read_csv('data.csv')" --output-format missing
    pb pl "pl.read_csv('data.csv')" --output-format info

    Note: For multi-statement code, assign your final result to a variable like 'result', 'df',
    'data', or ensure it's the last expression.
    """
    try:
        # Check if Polars is available
        if not _is_lib_present("polars"):
            console.print("[red]Error:[/red] Polars is not installed")
            console.print("\nThe 'pb pl' command requires Polars to be installed.")
            console.print("Install it with: [cyan]pip install polars[/cyan]")
            console.print("\nTo check all dependency status, run: [cyan]pb requirements[/cyan]")
            sys.exit(1)

        import polars as pl

        # Determine the source of the query
        query_code = None

        if file:
            # Read from file
            query_code = Path(file).read_text()
        elif edit:
            # Determine which editor to use
            chosen_editor = editor or _get_best_editor()

            # When piping, send editor message to stderr
            if pipe:
                print(f"Using editor: {chosen_editor}", file=sys.stderr)
            else:
                console.print(f"[dim]Using editor: {chosen_editor}[/dim]")

            # Interactive editor with custom editor
            if chosen_editor == "code":
                # Special handling for VS Code
                query_code = _edit_with_vscode()
            else:
                # Use click.edit() for terminal editors
                query_code = click.edit(
                    "# Enter your Polars query here\n"
                    "# Example:\n"
                    "# pl.read_csv('data.csv').select(['name', 'age'])\n"
                    "# pl.read_csv('data.csv').filter(pl.col('age') > 25)\n"
                    "# \n"
                    "# The result should be a Polars DataFrame or LazyFrame\n"
                    "\n",
                    editor=chosen_editor,
                )

            if query_code is None:
                if pipe:
                    print("No query entered", file=sys.stderr)
                else:
                    console.print("[yellow]No query entered[/yellow]")
                sys.exit(1)
        elif polars_expression:
            # Direct argument
            query_code = polars_expression
        else:
            # Try to read from stdin (for piping)
            if not sys.stdin.isatty():
                # Data is being piped in
                query_code = sys.stdin.read().strip()
            else:
                # No input provided and stdin is a terminal - show concise help
                _show_concise_help("pl", None)
                return

        if not query_code or not query_code.strip():
            console.print("[red]Error:[/red] Empty query")
            sys.exit(1)

        # Execute the query
        with console.status("[bold green]Executing Polars expression..."):
            namespace = {
                "pl": pl,
                "polars": pl,
                "__builtins__": __builtins__,
            }

            try:
                # Check if this is a single expression or multiple statements
                if "\n" in query_code.strip() or any(
                    keyword in query_code
                    for keyword in [
                        " = ",
                        "import",
                        "for ",
                        "if ",
                        "def ",
                        "class ",
                        "with ",
                        "try:",
                    ]
                ):
                    # Multiple statements - use exec()
                    exec(query_code, namespace)

                    # Look for the result in the namespace
                    # Try common variable names first
                    result = None
                    for var_name in ["result", "df", "data", "table", "output"]:
                        if var_name in namespace:
                            result = namespace[var_name]
                            break

                    # If no common names found, look for any DataFrame/LazyFrame
                    if result is None:
                        for key, value in namespace.items():
                            if (
                                hasattr(value, "collect") or hasattr(value, "columns")
                            ) and not key.startswith("_"):
                                result = value
                                break

                    # If still no result, get the last assigned variable (excluding builtins)
                    if result is None:
                        # Get variables that were added to namespace (excluding our imports)
                        user_vars = {
                            k: v
                            for k, v in namespace.items()
                            if k not in ["pl", "polars", "__builtins__"] and not k.startswith("_")
                        }
                        if user_vars:
                            # Get the last variable (this is a heuristic)
                            last_var = list(user_vars.keys())[-1]
                            result = user_vars[last_var]

                    if result is None:
                        if pipe:
                            print(
                                "[red]Error:[/red] Could not find result variable", file=sys.stderr
                            )
                            print(
                                "[dim]Assign your final result to a variable like 'result', 'df', or 'data'[/dim]",
                                file=sys.stderr,
                            )
                            print(
                                "[dim]Or ensure your last line returns a DataFrame[/dim]",
                                file=sys.stderr,
                            )
                        else:
                            console.print("[red]Error:[/red] Could not find result variable")
                            console.print(
                                "[dim]Assign your final result to a variable like 'result', 'df', or 'data'[/dim]"
                            )
                            console.print("[dim]Or ensure your last line returns a DataFrame[/dim]")
                        sys.exit(1)

                else:
                    # Single expression - use eval()
                    result = eval(query_code, namespace)

                # Validate result
                if not hasattr(result, "collect") and not hasattr(result, "columns"):
                    if pipe:
                        print(
                            "[red]Error:[/red] Expression must return a Polars DataFrame or LazyFrame",
                            file=sys.stderr,
                        )
                        print(f"[dim]Got: {type(result)}[/dim]", file=sys.stderr)
                    else:
                        console.print(
                            "[red]Error:[/red] Expression must return a Polars DataFrame or LazyFrame"
                        )
                        console.print(f"[dim]Got: {type(result)}[/dim]")
                    sys.exit(1)

            except Exception as e:
                # When piping, send errors to stderr so they don't interfere with the pipe
                if pipe:
                    print(f"Error executing Polars expression: {e}", file=sys.stderr)
                    print(file=sys.stderr)

                    # Create a panel with the expression(s) for better readability
                    if "\n" in query_code.strip():
                        # Multi-line expression
                        print(f"Expression(s) provided:\n{query_code}", file=sys.stderr)
                    else:
                        # Single line expression
                        print(f"Expression provided: {query_code}", file=sys.stderr)
                else:
                    # Normal error handling when not piping
                    console.print(f"[red]Error executing Polars expression:[/red] {e}")
                    console.print()

                    # Create a panel with the expression(s) for better readability
                    if "\n" in query_code.strip():
                        # Multi-line expression
                        console.print(
                            Panel(
                                query_code,
                                title="Expression(s) provided",
                                border_style="red",
                                expand=False,
                                title_align="left",
                            )
                        )
                    else:
                        # Single line expression
                        console.print(
                            Panel(
                                query_code,
                                title="Expression provided",
                                border_style="red",
                                expand=False,
                                title_align="left",
                            )
                        )

                sys.exit(1)

        # Only print success message when not piping (so it doesn't interfere with pipe output)
        if not pipe:
            console.print("[green]✓[/green] Polars expression executed successfully")

        # Process output
        if pipe:
            # Output data for piping to other commands
            _handle_pl_pipe(result, pipe_format)
        elif output_format == "preview":
            _handle_pl_preview(result, preview_head, preview_tail, output_html)
        elif output_format == "scan":
            _handle_pl_scan(result, query_code, output_html)
        elif output_format == "missing":
            _handle_pl_missing(result, query_code, output_html)
        elif output_format == "info":
            _handle_pl_info(result, query_code, output_html)
        elif output_format == "validate":
            console.print("[yellow]Validation output format not yet implemented[/yellow]")
            console.print("Use 'pb validate' with a data file for now")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _handle_pl_preview(result: Any, head: int, tail: int, output_html: str | None) -> None:
    """Handle preview output for Polars results."""
    try:
        # Create preview using existing preview function
        gt_table = pb.preview(
            data=result,
            n_head=head,
            n_tail=tail,
            show_row_numbers=True,
        )

        if output_html:
            html_content = gt_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] HTML saved to: {output_html}")
        else:
            # Get metadata for enhanced preview
            try:
                total_rows = pb.get_row_count(result)
                total_columns = pb.get_column_count(result)
                table_type = _get_tbl_type(result)

                preview_info = {
                    "total_rows": total_rows,
                    "total_columns": total_columns,
                    "head_rows": head,
                    "tail_rows": tail,
                    "is_complete": total_rows <= (head + tail),
                    "source_type": "Polars expression",
                    "table_type": table_type,
                }

                _rich_print_gt_table(gt_table, preview_info)
            except Exception:
                # Fallback to basic display
                _rich_print_gt_table(gt_table)

    except Exception as e:
        console.print(f"[red]Error creating preview:[/red] {e}")
        sys.exit(1)


def _handle_pl_scan(result: Any, expression: str, output_html: str | None) -> None:
    """Handle scan output for Polars results."""
    try:
        scan_result = pb.col_summary_tbl(data=result)

        if output_html:
            html_content = scan_result.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] Data scan report saved to: {output_html}")
        else:
            # Get metadata for enhanced scan display
            try:
                total_rows = pb.get_row_count(result)
                total_columns = pb.get_column_count(result)
                table_type = _get_tbl_type(result)

                _rich_print_scan_table(
                    scan_result,
                    expression,
                    "Polars expression",
                    table_type,
                    total_rows,
                    total_columns,
                )
            except Exception as e:
                console.print(f"[yellow]Could not display scan summary: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error creating scan:[/red] {e}")
        sys.exit(1)


def _handle_pl_missing(result: Any, expression: str, output_html: str | None) -> None:
    """Handle missing values output for Polars results."""
    try:
        missing_table = pb.missing_vals_tbl(data=result)

        if output_html:
            html_content = missing_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] Missing values report saved to: {output_html}")
        else:
            _rich_print_missing_table(missing_table, result)

    except Exception as e:
        console.print(f"[red]Error creating missing values report:[/red] {e}")
        sys.exit(1)


def _handle_pl_info(result: Any, expression: str, output_html: str | None) -> None:
    """Handle info output for Polars results."""
    try:
        # Get basic info
        tbl_type = _get_tbl_type(result)
        row_count = pb.get_row_count(result)
        col_count = pb.get_column_count(result)

        # Get column names and types
        if hasattr(result, "columns"):
            columns = list(result.columns)
        elif hasattr(result, "schema"):
            columns = list(result.schema.names)
        else:
            columns = []

        dtypes_dict = _get_column_dtypes(result, columns)

        if output_html:
            # Create a simple HTML info page
            # TODO: Implement an improved version of this in the Python API and then
            # use that here
            html_content = f"""
            <html><body>
            <h2>Polars Expression Info</h2>
            <p><strong>Expression:</strong> {expression}</p>
            <p><strong>Table Type:</strong> {tbl_type}</p>
            <p><strong>Rows:</strong> {row_count:,}</p>
            <p><strong>Columns:</strong> {col_count:,}</p>
            <h3>Column Details</h3>
            <ul>
            {"".join(f"<li>{col}: {dtypes_dict.get(col, '?')}</li>" for col in columns)}
            </ul>
            </body></html>
            """
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] HTML info saved to: {output_html}")
        else:
            # Display info table
            from rich.box import SIMPLE_HEAD

            info_table = Table(
                title="Polars Expression Info",
                show_header=True,
                header_style="bold magenta",
                box=SIMPLE_HEAD,
                title_style="bold cyan",
                title_justify="left",
            )
            info_table.add_column("Property", style="cyan", no_wrap=True)
            info_table.add_column("Value", style="green")

            info_table.add_row("Expression", expression)
            # Capitalize "polars" to "Polars" for consistency with pb info command
            display_tbl_type = (
                tbl_type.replace("polars", "Polars") if "polars" in tbl_type.lower() else tbl_type
            )
            info_table.add_row("Table Type", display_tbl_type)
            info_table.add_row("Rows", f"{row_count:,}")
            info_table.add_row("Columns", f"{col_count:,}")

            console.print()
            console.print(info_table)

            # Show column details
            if columns:
                console.print("\n[bold cyan]Column Details:[/bold cyan]")
                for col in columns[:10]:  # Show first 10 columns
                    dtype = dtypes_dict.get(col, "?")
                    console.print(f"  • {col}: [yellow]{dtype}[/yellow]")

                if len(columns) > 10:
                    console.print(f"  ... and {len(columns) - 10} more columns")

    except Exception as e:
        console.print(f"[red]Error creating info:[/red] {e}")
        sys.exit(1)


def _handle_pl_pipe(result: Any, pipe_format: str) -> None:
    """Handle piped output from Polars results."""
    try:
        import sys
        import tempfile

        # Create a temporary file to store the data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{pipe_format}", prefix="pb_pipe_", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        # Write the data to the temporary file
        if pipe_format == "parquet":
            if hasattr(result, "write_parquet"):
                # Polars
                result.write_parquet(temp_path)
            elif hasattr(result, "to_parquet"):
                # Pandas
                result.to_parquet(temp_path)
            else:
                # Convert to pandas and write
                import pandas as pd

                pd_result = pd.DataFrame(result)
                pd_result.to_parquet(temp_path)
        else:  # CSV
            if hasattr(result, "write_csv"):
                # Polars
                result.write_csv(temp_path)
            elif hasattr(result, "to_csv"):
                # Pandas
                result.to_csv(temp_path, index=False)
            else:
                # Convert to pandas and write
                import pandas as pd

                pd_result = pd.DataFrame(result)
                pd_result.to_csv(temp_path, index=False)

        # Output the temporary file path to stdout for the next command
        print(temp_path)

    except Exception as e:
        print(f"[red]Error creating pipe output:[/red] {e}", file=sys.stderr)
        sys.exit(1)


def _get_best_editor() -> str:
    """Detect the best available editor on the system."""

    # Check environment variable first
    if "EDITOR" in os.environ:
        return os.environ["EDITOR"]

    # Check for common editors in order of preference
    editors = [
        "code",  # VS Code
        "micro",  # Modern terminal editor
        "nano",  # User-friendly terminal editor
        "vim",  # Vim
        "vi",  # Vi (fallback)
    ]

    for editor in editors:
        if shutil.which(editor):
            return editor

    # Ultimate fallback
    return "nano"


def _edit_with_vscode() -> str | None:
    """Edit Polars query using VS Code."""
    import subprocess
    import tempfile

    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="pb_pl_", delete=False) as f:
        f.write("import polars as pl\n")
        f.write("\n")
        f.write("# Enter your Polars query here\n")
        f.write("# Examples:\n")
        f.write("# \n")
        f.write("# Single expression:\n")
        f.write("# pl.read_csv('data.csv').select(['name', 'age'])\n")
        f.write("# \n")
        f.write("# Multiple statements:\n")
        f.write("# csv = pl.read_csv('data.csv')\n")
        f.write("# result = csv.select(['name', 'age']).filter(pl.col('age') > 25)\n")
        f.write("# \n")
        f.write("# For multi-statement code, assign your final result to a variable\n")
        f.write("# like 'result', 'df', 'data', or just ensure it's the last line\n")
        f.write("# \n")
        f.write("# Save and then close this file in VS Code to execute the query\n")
        f.write("\n")
        temp_file = f.name

    try:
        # Open in VS Code and wait for it to close
        result = subprocess.run(
            ["code", "--wait", temp_file], capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            console.print(f"[yellow]VS Code exited with code {result.returncode}[/yellow]")

        # Read the edited content
        with open(temp_file, "r") as f:
            content = f.read()

        # Remove comments, empty lines, and import statements for cleaner execution
        lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("import polars")
                and not stripped.startswith("import polars as pl")
            ):
                lines.append(line)

        return "\n".join(lines) if lines else None

    except subprocess.TimeoutExpired:
        console.print("[red]Timeout:[/red] VS Code took too long to respond")
        return None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error:[/red] Could not open VS Code: {e}")
        return None
    except FileNotFoundError:
        console.print("[red]Error:[/red] VS Code not found in PATH")
        return None
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


def _show_concise_help(command_name: str, ctx: click.Context) -> None:
    """Show concise help for a command when required arguments are missing."""

    if command_name == "info":
        console.print("[bold cyan]pb info[/bold cyan] - Display information about a data source")
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb info data.csv")
        console.print("  pb info small_table")
        console.print()
        console.print("[dim]Shows table type, dimensions, column names, and data types[/dim]")
        console.print()
        console.print(
            "[dim]Use [bold]pb info --help[/bold] for complete options and examples[/dim]"
        )

    elif command_name == "preview":
        console.print(
            "[bold cyan]pb preview[/bold cyan] - Preview a data table showing head and tail rows"
        )
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb preview data.csv")
        console.print("  pb preview data.parquet --head 10 --tail 5")
        console.print()
        console.print("[bold yellow]Key Options:[/bold yellow]")
        console.print("  --head N          Number of rows from the top (default: 5)")
        console.print("  --tail N          Number of rows from the bottom (default: 5)")
        console.print("  --columns LIST    Comma-separated list of columns to display")
        console.print("  --output-html     Save HTML output to file")
        console.print()
        console.print(
            "[dim]Use [bold]pb preview --help[/bold] for complete options and examples[/dim]"
        )

    elif command_name == "scan":
        console.print(
            "[bold cyan]pb scan[/bold cyan] - Generate a comprehensive data profile report"
        )
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb scan data.csv")
        console.print("  pb scan data.parquet --output-html report.html")
        console.print()
        console.print("[bold yellow]Key Options:[/bold yellow]")
        console.print("  --output-html     Save HTML scan report to file")
        console.print("  --columns LIST    Comma-separated list of columns to scan")
        console.print()
        console.print(
            "[dim]Use [bold]pb scan --help[/bold] for complete options and examples[/dim]"
        )

    elif command_name == "missing":
        console.print("[bold cyan]pb missing[/bold cyan] - Generate a missing values report")
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb missing data.csv")
        console.print("  pb missing data.parquet --output-html missing_report.html")
        console.print()
        console.print("[bold yellow]Key Options:[/bold yellow]")
        console.print("  --output-html     Save HTML output to file")
        console.print()
        console.print(
            "[dim]Use [bold]pb missing --help[/bold] for complete options and examples[/dim]"
        )

    elif command_name == "validate":
        console.print("[bold cyan]pb validate[/bold cyan] - Perform data validation checks")
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb validate data.csv")
        console.print("  pb validate data.csv --check col-vals-not-null --column email")
        console.print()
        console.print("[bold yellow]Key Options:[/bold yellow]")
        console.print("  --check TYPE      Validation check type (default: rows-distinct)")
        console.print("  --column COL      Column name for column-specific checks")
        console.print("  --show-extract    Show failing rows if validation fails")
        console.print("  --list-checks     List all available validation checks")
        console.print()
        console.print(
            "[dim]Use [bold]pb validate --help[/bold] for complete options and examples[/dim]"
        )

    elif command_name == "run":
        console.print("[bold cyan]pb run[/bold cyan] - Run a Pointblank validation script")
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb run validation_script.py")
        console.print("  pb run validation_script.py --data data.csv")
        console.print()
        console.print("[bold yellow]Key Options:[/bold yellow]")
        console.print("  --data SOURCE     Replace data source in validation objects")
        console.print("  --output-html     Save HTML validation report to file")
        console.print("  --show-extract    Show failing rows if validation fails")
        console.print("  --fail-on LEVEL   Exit with error on critical/error/warning/any")
        console.print()
        console.print("[dim]Use [bold]pb run --help[/bold] for complete options and examples[/dim]")

    elif command_name == "make-template":
        console.print(
            "[bold cyan]pb make-template[/bold cyan] - Create a validation script or YAML template"
        )
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb make-template my_validation.py    # Python script template")
        console.print("  pb make-template my_validation.yaml  # YAML config template")
        console.print()
        console.print("[dim]Creates sample templates with validation examples[/dim]")
        console.print("[dim]Edit the template and run with [bold]pb run[/bold][/dim]")
        console.print()
        console.print(
            "[dim]Use [bold]pb make-template --help[/bold] for complete options and examples[/dim]"
        )

    elif command_name == "pl":
        console.print(
            "[bold cyan]pb pl[/bold cyan] - Execute Polars expressions and display results"
        )
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  pb pl \"pl.read_csv('data.csv')\"")
        console.print("  pb pl --edit")
        console.print()
        console.print("[bold yellow]Key Options:[/bold yellow]")
        console.print("  --edit            Open editor for multi-line input")
        console.print("  --file FILE       Read query from file")
        console.print("  --output-format   Output format: preview, scan, missing, info")
        console.print("  --pipe            Output for piping to other pb commands")
        console.print()
        console.print("[dim]Use [bold]pb pl --help[/bold] for complete options and examples[/dim]")

    # Fix the exit call at the end
    if ctx is not None:
        ctx.exit(1)
    else:
        sys.exit(1)
