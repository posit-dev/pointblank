# get_data_path()

Get the file path to a dataset included with the Pointblank package.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L1525-L1702)

``` python
get_data_path(
    dataset="small_table",
    file_type="csv",
)
```

This function provides direct access to the file paths of datasets included with Pointblank. These paths can be used in examples and documentation to demonstrate file-based data loading without requiring the actual data files. The returned paths can be used with `Validate(data=path)` to demonstrate CSV and Parquet file loading capabilities.

Each call writes the dataset to a new temporary file and returns the path to that file. The temporary file isn't deleted automatically, so it stays available for as long as the operating system keeps its temporary files.

## Parameters

`dataset: Literal["small_table", "game_revenue", "nycflights", "global_sales"] = "small_table"`  
The name of the dataset to get the path for. Current options are `"small_table"`, `"game_revenue"`, `"nycflights"`, and `"global_sales"`.

`file_type: Literal["csv", "parquet", "duckdb"] = "csv"`  
The file format to get the path for. Options are `"csv"`, `"parquet"`, or `"duckdb"`.

## Returns

`str`  
The path to a temporary file holding the requested dataset in the requested format.

## Included Datasets

The available datasets are the same as those in [`load_dataset()`](../reference/load_dataset.md#pointblank.load_dataset):

- `"small_table"`: A small dataset with 13 rows and 8 columns. Ideal for testing and examples.
- `"game_revenue"`: A dataset with 2000 rows and 11 columns. Revenue data for a game company.
- `"nycflights"`: A dataset with 336,776 rows and 18 columns. Flight data from NYC airports.
- `"global_sales"`: A dataset with 50,000 rows and 20 columns. Global sales data across regions.

## File Types

Each dataset is available in multiple formats:

- `"csv"`: Comma-separated values file (`.csv`)
- `"parquet"`: Parquet file (`.parquet`)
- `"duckdb"`: DuckDB database file (`.ddb`)

## Examples

Get the path to a CSV file and use it with [Validate](../reference/Validate.md#pointblank.Validate):

``` python
import pointblank as pb

# Get path to the small_table CSV file
csv_path = pb.get_data_path("small_table", "csv")

# Use the path directly with Validate
validation = (
    pb.Validate(data=csv_path)
    .col_exists(["a", "b", "c"])
    .col_vals_gt(columns="d", value=0)
    .interrogate()
)

validation
```

[TABLE]

The CSV file is read into a table during interrogation. The three `col_exists()` steps and the `col_vals_gt()` step all pass.

We can also get a Parquet file path for validation examples:

``` python
# Get path to the game_revenue Parquet file
parquet_path = pb.get_data_path(dataset="game_revenue", file_type="parquet")

# Validate the Parquet file directly
validation = (
    pb.Validate(data=parquet_path, label="Game Revenue Data Validation")
    .col_vals_not_null(columns=["player_id", "session_id"])
    .col_vals_gt(columns="item_revenue", value=0)
    .interrogate()
)

validation
```

[TABLE]

All 2,000 rows have player and session IDs and positive revenue values, so every step passes.

This is particularly useful for documentation examples where you want to demonstrate file-based workflows without requiring users to have specific data files:

``` python
# Example showing CSV file validation
sales_csv = pb.get_data_path(dataset="global_sales", file_type="csv")

validation = (
    pb.Validate(data=sales_csv, label="Sales Data Validation")
    .col_exists(["customer_id", "product_id", "revenue"])
    .col_vals_regex(columns="customer_id", pattern=r"[0-9a-f]{8}")
    .interrogate()
)

validation
```

[TABLE]

All three columns exist, and every customer ID that's present is an eight-character hexadecimal string. The 334 failing test units in the last step are rows where `customer_id` is missing; setting `na_pass=True` would count those rows as passing.

## See Also

[load_dataset()](../reference/load_dataset.md#pointblank.load_dataset)  
Load a dataset directly as a table object.
