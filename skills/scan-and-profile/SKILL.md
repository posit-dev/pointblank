---
name: scan-and-profile
description: >
  Profile and scan datasets with Pointblank before writing validation
  rules. Covers DataScan for column-level statistics, Schema inference
  with schema_from_tbl(), missing values analysis with missing_vals_tbl(),
  and table previewing. Use when exploring a new dataset or understanding
  data distributions before validation.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-profiling
    - data-scan
    - schema
    - data-exploration
    - missing-values
---

# Scan and Profile

Skill for profiling datasets before writing validation rules.
Understanding your data's shape, types, distributions, and
missingness patterns helps you write targeted, effective
validation plans.

## Quick start

```python
import pointblank as pb

# Scan a dataset
scan = pb.DataScan(data=df, tbl_name="orders")
scan.get_tabular_report()  # rich HTML summary
```

## Skill directory structure

```
skills/scan-and-profile/
+-- SKILL.md                      <- This file
+-- references/
    +-- datascan-reference.md     <- DataScan details and output
    +-- schema-inference.md       <- Schema inference and construction
```

## When to use what

| I want to...                     | Use                                     |
| -------------------------------- | --------------------------------------- |
| Get a full column-level profile  | `DataScan`                              |
| See column types and basic stats | `DataScan.get_tabular_report()`         |
| Export profile as JSON           | `DataScan.to_json()`                    |
| Infer a schema from data         | `schema_from_tbl()`                     |
| Infer a schema with constraints  | `Schema.from_table()`                   |
| See a quick preview of the table | `preview()`                             |
| Analyze missing values           | `missing_vals_tbl()`                    |
| Get row/column counts            | `get_row_count()`, `get_column_count()` |

## Core concepts

### DataScan

`DataScan` produces a comprehensive profile of every column in a
dataset:

```python
scan = pb.DataScan(data=df, tbl_name="monthly_sales")

# View as HTML table
scan.get_tabular_report()

# Include sample data in the report
scan.get_tabular_report(show_sample_data=True)

# Access raw summary data
scan.summary_data

# Export to JSON
json_str = scan.to_json()
scan.save_to_json("profile_output.json")
```

The report includes per-column:

- Data type
- Count of non-null values
- Missingness (count and percentage)
- Distinct value count
- Negative / zero / positive value counts (numeric)
- Descriptive statistics (mean, median, std, min, max)
- Quantiles (Q1, Q3, IQR)

### Shortcut: col_summary_tbl

For a quick column summary without creating a DataScan object:

```python
pb.col_summary_tbl(data=df, tbl_name="orders")
```

### Table preview

Quick visual preview of the first and last rows:

```python
pb.preview(data=df, n_head=5, n_tail=5)

# Customize
pb.preview(
    data=df,
    columns_subset=["id", "name", "amount"],
    n_head=10,
    n_tail=3,
    limit=50,
    show_row_numbers=True,
    max_col_width=250,
)
```

### Missing values analysis

Dedicated analysis of missingness patterns:

```python
# Basic missing values table
pb.missing_vals_tbl(data=df)

# As a heatmap
pb.missing_vals_tbl(data=df, as_heatmap=True)

# With structured missingness definitions
missing_specs = {
    "measurement": pb.MissingSpec(
        reasons={-999: "not collected", -1: "redacted"},
    ),
    "notes": pb.MissingSpec(
        reasons={"N/A": "not applicable"},
    ),
}
pb.missing_vals_tbl(data=df, missing=missing_specs)
```

### Schema inference

Infer a schema from an existing table:

```python
# Basic inference (column names and types)
schema = pb.schema_from_tbl(df)
print(schema.get_column_list())
print(schema.get_dtype_list())

# With constraint inference
schema = pb.Schema.from_table(
    df,
    infer_constraints=True,       # infer value ranges, sets, etc.
    categorical_threshold=20,     # columns with <= 20 distinct values
    detect_presets=True,           # detect email, URL, etc. patterns
    sample_size=None,             # sample rows for inference (None=all)
)
```

### Constructing schemas manually

```python
# From keyword arguments
schema = pb.Schema(id="Int64", name="String", amount="Float64")

# From a dictionary
schema = pb.Schema({"id": "Int64", "name": "String"})

# From a list of tuples
schema = pb.Schema([("id", "Int64"), ("name", "String")])

# Column names only (type checking skipped)
schema = pb.Schema(["id", "name", "amount"])
```

### Schema inspection

```python
schema.get_column_list()   # ["id", "name", "amount"]
schema.get_dtype_list()    # ["Int64", "String", "Float64"]
```

### Quick counts

```python
pb.get_row_count(df)       # number of rows
pb.get_column_count(df)    # number of columns
```

## Workflows

### Profiling a new dataset

1. Load or connect to the data.
2. Run `pb.preview(data)` for a quick look.
3. Run `pb.DataScan(data=df).get_tabular_report()` for full stats.
4. Run `pb.missing_vals_tbl(data=df)` to understand missingness.
5. Infer a schema: `schema = pb.schema_from_tbl(df)`.
6. Use the profile to inform validation rules.

### From profile to validation plan

1. Profile the data with `DataScan`.
2. Note columns with high missingness -- add `col_pct_null` checks.
3. Note columns with few distinct values -- add `col_vals_in_set`.
4. Note numeric ranges -- add `col_vals_between` checks.
5. Infer schema and use in `col_schema_match`.
6. Build the validation plan with the `write-validation` skill.

### Comparing profiles over time

```python
scan_today = pb.DataScan(data=today_df)
scan_yesterday = pb.DataScan(data=yesterday_df)

# Compare by exporting to JSON
scan_today.save_to_json("profile_today.json")
scan_yesterday.save_to_json("profile_yesterday.json")
```

## Gotchas

1. **DataScan reads the full table.** For large datasets, consider
   sampling first.
2. **Schema type names are backend-specific.** Polars uses `"Int64"`,
   Pandas uses `"int64"`. Use `schema_from_tbl()` to get the right
   names automatically.
3. **`schema_from_tbl` infers from current data.** If the data has
   unexpected types (e.g., string column with numbers), the inferred
   schema reflects that.
4. **`missing_vals_tbl` only shows null by default.** Pass
   `MissingSpec` definitions to include sentinel values.
5. **`preview()` returns a GT table object.** In notebooks it renders
   automatically; in scripts, you may need to display it.

## Related skills

| Skill            | When to use it                               |
| ---------------- | -------------------------------------------- |
| pointblank       | Full Validate workflow overview              |
| write-validation | Build validation plans from profile insights |
| generate-data    | Create synthetic data matching a schema      |
