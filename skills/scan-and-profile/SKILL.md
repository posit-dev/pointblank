---
name: scan-and-profile
description: >
  Profile and scan datasets with Pointblank before writing validation
  rules. Covers DataScan for column-level statistics, comparing a scan
  with a saved baseline to find drift (DataScan.compare), schemas taken
  from a table, missing values analysis with missing_vals_tbl(), and
  table previewing. Use when exploring a new dataset, understanding
  data distributions before validation, or checking a refreshed table
  against an earlier profile.
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
    - drift
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

| I want to...                         | Use                                     |
| ------------------------------------ | --------------------------------------- |
| Get a full column-level profile      | `DataScan`                              |
| See column types and basic stats     | `DataScan.get_tabular_report()`         |
| Export a profile as JSON or a dict   | `to_json()`, `save_to_json()`, `to_dict()` |
| Bring back a saved profile           | `DataScan.load_from_json()`, `from_json()`, `from_dict()` |
| Compare a table's profile with a baseline | `DataScan.compare()`               |
| Find which rows of two tables differ | `pb.compare()` (rows, not profiles)     |
| Get a schema for `col_schema_match`  | `Schema(tbl=df)`                        |
| Infer constraints for generating data | `schema_from_tbl()`, `Schema.from_table()` |
| See a quick preview of the table     | `preview()`                             |
| Analyze missing values               | `missing_vals_tbl()`                    |
| Get row/column counts                | `get_row_count()`, `get_column_count()` |

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

# The profile as a Polars DataFrame, one row per column
scan.summary_data

# Export
json_str = scan.to_json()
scan.save_to_json("profile_output.json")
profile = scan.to_dict()      # {"metadata": ..., "columns": ...}
```

`summary_data` has these columns: `colname`, `coltype`, `icon`,
`n_missing`, `n_unique`, `mean`, `std`, `min`, `p05`, `q_1`, `median`,
`q_3`, `p95`, `max`, `iqr`, `freqs`, and `sample_data`. For a string
column the statistics are of the string lengths; `freqs` holds the
value counts of a Boolean column.

### Comparing with a baseline

A table that's refreshed regularly can change in ways no single row
shows: a column dropped, a type changed, values shifted. Save a
profile of a version known to be good, then compare later scans with
it. The scan you call `compare()` on is the current one:

```python
pb.DataScan(data=good_df).save_to_json("profile_baseline.json")

baseline = pb.DataScan.load_from_json("profile_baseline.json")
current = pb.DataScan(data=new_df)

diff = current.compare(baseline)       # a DataScanDiff

diff.has_changes                       # True if anything differs
diff.row_count_diff                    # (baseline rows, current rows)
diff.columns_added                     # only in the current scan
diff.columns_removed                   # only in the baseline
diff.columns_type_changed
diff.column_diffs                      # per column: status, stat_diffs, drift_scores
diff.to_dict()
diff.get_tabular_report()
```

Each statistic is compared exactly, so any change is reported with
both values. Drift scores show how far the distribution moved: numeric
columns get `psi` (population stability index), `ks_statistic`, and
`ks_p_value`, and other columns get a `psi` from their value counts. A
PSI below 0.1 is usually read as little shift, 0.1 to 0.25 as moderate,
and above 0.25 as large.

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

# With coded missing values, keyed by column name
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

### Schemas from a table

There are two kinds, for two jobs:

```python
# Column names and dtypes, for col_schema_match() and contracts
schema = pb.Schema(tbl=df)
schema.get_column_list()   # ["id", "name", "amount"]
schema.get_dtype_list()    # ["Int64", "String", "Float64"]

# Field objects with constraints inferred from the values, for
# generating synthetic data with schema.generate()
gen_schema = pb.schema_from_tbl(
    df,
    infer_constraints=True,       # ranges, null rates, uniqueness, allowed sets
    categorical_threshold=20,     # <= 20 distinct values becomes `allowed=`
    detect_presets=True,          # match email, URL, etc. by name and values
    sample_size=None,             # sample rows first (None = all)
)
synthetic = gen_schema.generate(n=1000, seed=23)
```

`pb.Schema.from_table()` is the same as `schema_from_tbl()`.

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
5. Take a schema: `schema = pb.Schema(tbl=df)`.
6. Use the profile to inform validation rules.

### From profile to validation plan

1. Profile the data with `DataScan`.
2. Note columns with high missingness -- add `col_pct_null` checks.
3. Note columns with few distinct values -- add `col_vals_in_set`.
4. Note numeric ranges -- add `col_vals_between` checks.
5. Use `pb.Schema(tbl=df)` in `col_schema_match`.
6. Build the validation plan with the `write-validation` skill.

### Watching a refreshed table for drift

1. Scan a version known to be good and `save_to_json()` it.
2. On each refresh, scan the new data and `compare()` it with
   `DataScan.load_from_json(...)`.
3. Check `diff.has_changes`, the added, removed, and type-changed
   columns, and the drift scores; show `diff.get_tabular_report()`.
4. Turn anything that matters into a validation step.

## Gotchas

1. **DataScan reads the full table.** For large datasets, consider
   sampling first.
2. **Schema type names are backend-specific.** Polars uses `"Int64"`,
   Pandas uses `"int64"`, and Ibis uses `"int64"`. `Schema(tbl=df)`
   gets the right names automatically.
3. **`schema_from_tbl()` is for generating data, not for
   `col_schema_match()`.** Its fields hold short dtypes (`"Datetime"`
   rather than `"Datetime(time_unit='us', time_zone=None)"`), so a
   schema match with the default `full_match_dtypes=True` can fail on
   the same table. Use `Schema(tbl=df)` to validate.
4. **A restored profile has no values.** After `load_from_json()`,
   `from_json()`, or `from_dict()`, `compare()` still compares the
   statistics, but only Boolean columns get drift scores. Compare two
   scans made from data to get them for every column.
5. **`compare()` direction matters.** Call it on the current scan and
   pass the baseline; "added" means in the current scan only.
6. **`missing_vals_tbl` only shows null by default.** Pass
   `MissingSpec` definitions to include sentinel values.
7. **`preview()` returns a GT table object.** In notebooks it renders
   automatically; in scripts, you may need to display it.

## Related skills

| Skill               | When to use it                                   |
| ------------------- | ------------------------------------------------ |
| pointblank          | Full Validate workflow overview                  |
| write-validation    | Build validation plans from profile insights     |
| generate-data       | Create synthetic data matching a schema          |
| draft-validation    | Have an LLM draft a plan from the profile        |
| check-training-data | Drift between train/test splits (`check_splits`) |
