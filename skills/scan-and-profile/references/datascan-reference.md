# DataScan reference

## Constructor

```python
pb.DataScan(
    data: Any,              # DataFrame, Ibis table, or file path
    tbl_name: str | None = None,
)
```

## Properties

| Property       | Type              | Description                        |
|----------------|-------------------|------------------------------------|
| `summary_data` | Polars DataFrame  | The profile, one row per column    |

## Methods

| Method                                        | Returns        | Description                       |
|-----------------------------------------------|----------------|-----------------------------------|
| `get_tabular_report(*, show_sample_data=False)` | `GT`         | HTML summary table                |
| `to_json()`                                   | `str`          | JSON string of the profile        |
| `save_to_json(output_file)`                   | `None`         | Write the JSON to a file          |
| `to_dict()`                                   | `dict`         | `{"metadata": ..., "columns": ...}` |
| `DataScan.from_json(json_string)`             | `DataScan`     | Restore from a JSON string        |
| `DataScan.load_from_json(input_file)`         | `DataScan`     | Restore from a JSON file          |
| `DataScan.from_dict(d)`                       | `DataScan`     | Restore from `to_dict()` output   |
| `compare(baseline)`                           | `DataScanDiff` | Differences from a baseline scan  |

The metadata has `table_name`, `row_count`, `columns`, and
`skipped_columns`.

## Columns of summary_data

| Column        | Description                                         |
|---------------|-----------------------------------------------------|
| `colname`     | Column name                                         |
| `coltype`     | Column data type                                    |
| `icon`        | Type icon used in the report                        |
| `n_missing`   | Count of missing values                             |
| `n_unique`    | Count of distinct values                            |
| `mean`        | Mean                                                |
| `std`         | Standard deviation                                  |
| `min`         | Minimum                                             |
| `p05`         | 5th percentile                                      |
| `q_1`         | First quartile                                      |
| `median`      | Median                                              |
| `q_3`         | Third quartile                                      |
| `p95`         | 95th percentile                                     |
| `max`         | Maximum                                             |
| `iqr`         | Interquartile range                                 |
| `freqs`       | Value counts (Boolean columns)                      |
| `sample_data` | A few sample values                                 |

For string columns the statistics describe the string lengths.

## DataScanDiff

Returned by `current_scan.compare(baseline_scan)`:

| Attribute / Method        | Description                                          |
|---------------------------|------------------------------------------------------|
| `has_changes`             | `True` if anything differs                           |
| `row_count_diff`          | `(baseline rows, current rows)`                      |
| `columns_added`           | Columns only in the current scan                     |
| `columns_removed`         | Columns only in the baseline                         |
| `columns_type_changed`    | Columns whose type differs                           |
| `column_diffs`            | One record per column: `colname`, `coltype_baseline`, `coltype_current`, `status` (`unchanged`, `changed`, `type_changed`, `added`, `removed`), `stat_diffs` (`{stat: (baseline, current)}`), `drift_scores` |
| `to_dict()`               | `row_count`, `columns_added`, `columns_removed`, `columns_type_changed`, `stat_diffs`, `drift_scores` |
| `get_tabular_report()`    | GT report of the differences                         |

Drift scores (zero scores are left out):

| Column kind | Scores                                         |
|-------------|------------------------------------------------|
| Numeric     | `psi` (ten bins at the baseline's deciles; needs at least ten values on each side), `ks_statistic`, `ks_p_value` (SciPy if installed, else an approximation) |
| Other       | `psi` from the frequency of each distinct value |

Drift scores need the values, so a scan restored from JSON or a dict
gives them only for Boolean columns.

## Shortcut function

```python
pb.col_summary_tbl(data=df, tbl_name="my_table")
```

Equivalent to `DataScan(data, tbl_name).get_tabular_report()`.

## missing_vals_tbl

```python
pb.missing_vals_tbl(
    data: Any,
    missing: dict[str, MissingSpec] | None = None,   # keyed by column name
    as_heatmap: bool = False,
) -> GT
```

Analyzes null and missing values across all columns. Returns a GT
table showing missingness counts and patterns.

With `as_heatmap=True`, renders a visual heatmap of missingness
across rows and columns.

## preview

```python
pb.preview(
    data: Any,
    columns_subset: str | list[str] | Column | None = None,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int = 50,
    show_row_numbers: bool = True,
    max_col_width: int = 250,
    min_tbl_width: int = 500,
    incl_header: bool | None = None,
) -> GT
```

## Utility functions

```python
pb.get_row_count(data: Any) -> int
pb.get_column_count(data: Any) -> int
```
