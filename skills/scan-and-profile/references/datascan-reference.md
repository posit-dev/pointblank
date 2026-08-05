# DataScan reference

## Constructor

```python
pb.DataScan(
    data: Any,              # DataFrame, Ibis table, or file path
    tbl_name: str | None = None,
)
```

## Properties

| Property       | Type   | Description                          |
|----------------|--------|--------------------------------------|
| `summary_data` | `dict` | Raw column-level statistics          |

## Methods

| Method                                   | Returns  | Description                     |
|------------------------------------------|----------|---------------------------------|
| `get_tabular_report(show_sample_data=False)` | `GT` | HTML summary table              |
| `to_json()`                              | `str`    | JSON string of profile          |
| `save_to_json(output_file)`              | `None`   | Write JSON to file              |

## Column statistics in summary_data

For each column, the summary includes:

| Statistic       | Description                              |
|-----------------|------------------------------------------|
| `dtype`          | Column data type                        |
| `n_non_null`     | Count of non-null values                |
| `n_null`         | Count of null values                    |
| `pct_null`       | Percentage null                         |
| `n_distinct`     | Count of distinct values                |
| `pct_distinct`   | Percentage distinct                     |
| `n_negative`     | Count of negative values (numeric)      |
| `n_zero`         | Count of zero values (numeric)          |
| `n_positive`     | Count of positive values (numeric)      |
| `mean`           | Arithmetic mean (numeric)               |
| `median`         | Median value (numeric)                  |
| `std`            | Standard deviation (numeric)            |
| `min`            | Minimum value                           |
| `max`            | Maximum value                           |
| `q1`             | First quartile (numeric)                |
| `q3`             | Third quartile (numeric)                |

## Shortcut function

```python
pb.col_summary_tbl(data=df, tbl_name="my_table")
```

Equivalent to `DataScan(data, tbl_name).get_tabular_report()`.

## missing_vals_tbl

```python
pb.missing_vals_tbl(
    data: Any,
    missing: dict[str, MissingSpec] | None = None,
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
    columns_subset: list[str] | None = None,
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
