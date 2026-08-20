# DataScanDiff


The result of comparing two [DataScan](DataScan.md#pointblank.DataScan) profiles.


Usage

``` python
DataScanDiff(
    current,
    baseline,
)
```


Created by calling [DataScan.compare()](DataScan.compare.md#pointblank.DataScan.compare). Provides programmatic access to schema changes and per-column statistical drift, plus a tabular report via [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report).


## Attributes


`columns_added: list[str]`  
Column names present in the current scan but not the baseline.

`columns_removed: list[str]`  
Column names present in the baseline but not the current scan.

`columns_type_changed: list[str]`  
Column names whose data type changed between baseline and current.

`column_diffs: list[_ColumnDiff]`  
Per-column diff details for all columns that appear in either scan.


## Attributes

| Name | Description |
|----|----|
| [has_changes](#has_changes) | Return `True` if any schema or statistical changes were detected. |
| [row_count_diff](#row_count_diff) | Return `(baseline_row_count, current_row_count)`. |

------------------------------------------------------------------------


### has_changes


Return `True` if any schema or statistical changes were detected.


`has_changes: bool`


------------------------------------------------------------------------


### row_count_diff


Return `(baseline_row_count, current_row_count)`.


`row_count_diff: tuple[int, int]`


## Methods

| Name | Description |
|----|----|
| [get_tabular_report()](#get_tabular_report) | Generate a GT table summarizing the differences between the two scans. |
| [to_dict()](#to_dict) | Export the comparison results as a dictionary. |

------------------------------------------------------------------------


### get_tabular_report()


Generate a GT table summarizing the differences between the two scans.


Usage

``` python
get_tabular_report()
```


#### Returns


`GT`  
A styled Great Tables report showing schema and statistical drift.


------------------------------------------------------------------------


### to_dict()


Export the comparison results as a dictionary.


Usage

``` python
to_dict()
```


#### Returns


`dict[str, Any]`  
A dictionary with schema changes, row count diff, and per-column stat diffs.
