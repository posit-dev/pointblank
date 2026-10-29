# DataScanDiff

The result of comparing two [DataScan](../reference/DataScan.md#pointblank.DataScan) profiles.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L1448-L1995)

``` python
DataScanDiff(
    current,
    baseline,
)
```

A [DataScanDiff](../reference/DataScanDiff.md#pointblank.DataScanDiff) is what [`DataScan.compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare) returns, and it isn't meant to be created directly. It records how a table's profile changed between a baseline scan and a current one: which columns were added, removed, or changed type, which statistics changed and by how much, and how far each column's distribution drifted (see [`DataScan.compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare) for how the drift scores are computed).

The results can be used in three ways. The attributes and the [`has_changes`](../reference/DataScanDiff.md#pointblank.DataScanDiff.has_changes) property suit checks in code, such as stopping a pipeline when a column disappears. [`to_dict()`](../reference/DataScanDiff.md#pointblank.DataScanDiff.to_dict) gives every difference as plain data, for logging or alerting. And [`get_tabular_report()`](../reference/DataScanDiff.md#pointblank.DataScanDiff.get_tabular_report) shows the differences as a table, for a person to review.

## Parameters

`current: DataScan`  
The newer [DataScan](../reference/DataScan.md#pointblank.DataScan) (the one whose `compare()` method was called).

`baseline: DataScan`  
The older [DataScan](../reference/DataScan.md#pointblank.DataScan) that `current` is compared against.

## Attributes

`columns_added: list[str]`  
Column names present in the current scan but not the baseline, sorted alphabetically.

`columns_removed: list[str]`  
Column names present in the baseline but not the current scan, sorted alphabetically.

`columns_type_changed: list[str]`  
Column names whose data type changed between baseline and current.

`column_diffs: list[_ColumnDiff]`  
Per-column diff details for all columns that appear in either scan, with the baseline's columns first. Each entry has the attributes `colname`, `coltype_baseline` and `coltype_current` (`None` for a column missing from that scan), `status` (one of `"added"`, `"removed"`, `"type_changed"`, `"changed"`, or `"unchanged"`), `stat_diffs` (a dictionary mapping each changed statistic to a `(baseline, current)` tuple), and `drift_scores` (a dictionary of drift scores, which might be empty).

## Examples

In this example, a table gains a row and a `note` column, and its `score` column changes from integers to strings.

``` python
import pointblank as pb
import polars as pl

old = pl.DataFrame({"id": [1, 2, 3], "score": [7, 9, 8]})

new = pl.DataFrame(
    {"id": [1, 2, 3, 4], "score": ["7", "9", "8", "6"], "note": ["a", None, "b", None]}
)

diff = pb.DataScan(new).compare(pb.DataScan(old))

diff.columns_added, diff.columns_removed, diff.columns_type_changed
```

    (['note'], [], ['score'])

The schema changes are available directly as lists. For a look at each column, go through `column_diffs`:

``` python
for col in diff.column_diffs:
    print(col.colname, col.status, col.coltype_baseline, "->", col.coltype_current)
```

    id changed Int64 -> Int64
    score type_changed Int64 -> String
    note added None -> String

The `id` column kept its type, but its statistics changed because of the new row. A pipeline could stop whenever `columns_removed` or `columns_type_changed` isn't empty, since those changes tend to break code that reads the table.

## See Also

[DataScan](../reference/DataScan.md#pointblank.DataScan)  
Profile a table, column by column.

[compare()](../reference/compare.md#pointblank.compare)  
Compare two tables row by row.

## Attributes

| Name | Description |
|----|----|
| [has_changes](#has_changes) | Return `True` if any schema or statistical changes were detected. |
| [row_count_diff](#row_count_diff) | Return `(baseline_row_count, current_row_count)`. |

### has_changes

Return `True` if any schema or statistical changes were detected.

`has_changes: bool`

This is the quickest check of whether a table changed. It's `True` if a column was added, removed, or changed type, or if any statistic or drift score differs between the two scans. A change in the row count alone doesn't count, since tables often grow without changing in character. Check [`row_count_diff`](../reference/DataScanDiff.md#pointblank.DataScanDiff.row_count_diff) for that.

#### Examples

Here, a log of job runs grows from three rows to five, but every status is still `"ok"`.

``` python
import pointblank as pb
import polars as pl

runs_before = pl.DataFrame({"status": ["ok"] * 3})
runs_after = pl.DataFrame({"status": ["ok"] * 5})

diff = pb.DataScan(runs_after).compare(pb.DataScan(runs_before))

diff.has_changes
```

    False

The table has more rows, but nothing about its columns changed (they have the same values in the same proportions), so [has_changes](../reference/DataScanDiff.md#pointblank.DataScanDiff.has_changes) is `False`. A single `"failed"` status in the new rows would have changed the number of distinct values and made it `True`.

### row_count_diff

Return `(baseline_row_count, current_row_count)`.

`row_count_diff: tuple[int, int]`

The row counts are kept apart from the other differences, since a change in size is often expected (a table that collects events grows every day). Having both counts lets you decide what change is acceptable, such as a minimum growth rate or a maximum drop.

#### Examples

Let's compare two versions of a log of job runs, where the newer one has two more rows.

``` python
import pointblank as pb
import polars as pl

runs_before = pl.DataFrame({"status": ["ok"] * 3})
runs_after = pl.DataFrame({"status": ["ok"] * 5})

diff = pb.DataScan(runs_after).compare(pb.DataScan(runs_before))

baseline_rows, current_rows = diff.row_count_diff

baseline_rows, current_rows
```

    (3, 5)

The baseline has three rows and the current scan five. The [`has_changes`](../reference/DataScanDiff.md#pointblank.DataScanDiff.has_changes) property is `False` here, so code that should also react to a table shrinking (for instance, when a load was incomplete) needs to check these counts.

## Methods

| Name | Description |
|----|----|
| [get_tabular_report()](#get_tabular_report) | Generate a GT table summarizing the differences between the two scans. |
| [to_dict()](#to_dict) | Export the comparison results as a dictionary. |

### get_tabular_report()

Generate a GT table summarizing the differences between the two scans.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L1814-L1985)

``` python
get_tabular_report()
```

The report is for a person reviewing the changes: it has one row per column of either scan, in the order of the baseline's columns followed by the added ones. The title names the two tables (baseline first) and the subtitle gives both row counts. The `Status` column says what happened to each column:

- `Added` or `Removed`: the column is in only one of the scans
- `Type Changed` or `Type + Stats Changed`: the data type changed (and maybe the statistics too)
- `Stats Changed`: the type is the same, but statistics or drift scores differ
- `OK`: nothing changed

The types in the baseline and the current scan come next, then every changed statistic with its baseline and current values. If any column has drift scores, a final column shows them (PSI, and for numeric columns the KS statistic with its p-value).

This method needs Polars, whatever the type of the scanned tables.

#### Returns

`GT`  
A styled Great Tables report showing schema and statistical drift.

#### Examples

Let's compare a scan of the `small_table` dataset with a scan of an older version, where the `f` column only had `"low"` and `"high"` values and there was no `e` column.

``` python
import pointblank as pb
import polars as pl

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

older = small_table.filter(pl.col("f") != "mid").drop("e")

diff = pb.DataScan(small_table, tbl_name="current").compare(
    pb.DataScan(older, tbl_name="older")
)

diff.get_tabular_report()
```

[TABLE]

The `e` column shows as added, and the other columns have changed statistics, since the older version had two fewer rows. The PSI of `f` (about 1.15) comes from the `"mid"` values that are only in the current version. The `date_time`, `date`, and `b` columns get a similar PSI, but for a different reason: nearly all of their values are distinct, so removing any rows changes their value frequencies a lot. The PSI of a non-numeric column means most for columns with a few distinct values, like `f`. Like other GT tables, the report can be saved as HTML with `write_raw_html()`, to share with others who want to review the changes.

### to_dict()

Export the comparison results as a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L1728-L1812)

``` python
to_dict()
```

The dictionary has every difference the comparison found, in plain Python data. That suits logging, alerting (by sending only the parts that matter), or storing the results next to the scans that produced them. Columns without changes are left out, so the dictionary only grows with the number of differences.

#### Returns

`dict[str, Any]`  
A dictionary with these keys: `"row_count"` (a dictionary with the `"baseline"` and `"current"` row counts), `"columns_added"` and `"columns_removed"` (lists of column names), `"columns_type_changed"` (a list of dictionaries with `"column"`, `"baseline_type"`, and `"current_type"`), `"stat_diffs"` (a dictionary that maps each column with changed statistics to a dictionary that maps each statistic to its `"baseline"` and `"current"` values), and `"drift_scores"` (a dictionary mapping each column that has drift scores to those scores).

#### Examples

Let's compare two versions of a table of orders. The newer version has one more order, a refund for -15.5.

``` python
import pointblank as pb
import polars as pl

orders = pl.DataFrame({"order_id": [1, 2, 3, 4], "amount": [25.0, 40.0, 15.5, 60.0]})

orders_now = pl.DataFrame(
    {"order_id": [1, 2, 3, 4, 5], "amount": [25.0, 40.0, 15.5, 60.0, -15.5]}
)

diff_dict = pb.DataScan(orders_now).compare(pb.DataScan(orders)).to_dict()

diff_dict["row_count"], diff_dict["columns_added"], diff_dict["columns_removed"]
```

    ({'baseline': 4, 'current': 5}, [], [])

There are no schema changes. The changed statistics of `amount` include its minimum:

``` python
diff_dict["stat_diffs"]["amount"]["min"]
```

    {'baseline': 15.5, 'current': -15.5}

Picking out an entry like this one makes it easy to raise an alert on specific changes (here, a negative amount) and ignore others, such as a mean that moves a little as new orders come in.
