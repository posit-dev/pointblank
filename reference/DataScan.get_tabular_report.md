# DataScan.get_tabular_report()

Get a tabular report of the scan.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L330-L642)

``` python
DataScan.get_tabular_report(
    *,
    show_sample_data=False,
)
```

The report presents the profile of each column as one row of a table: the column's name and type, the number and fraction of missing and distinct values, and its statistics. It's the quickest way to get an overview of a table you haven't seen before, and it's what [`col_summary_tbl()`](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) returns.

Each row starts with an icon for the column's type. Statistics that don't apply to any column are left out, and those that don't apply to a particular column are shown as `-`. Statistics for string columns describe the lengths of the strings (as the note below the table says). Boolean columns show the percentages of `True` and `False` values in place of the number of distinct values. Columns of unsupported types are listed with an *unsupported type* badge and no statistics.

## Parameters

`show_sample_data: bool = False`  
Should a column with the first five non-missing values of each column be added? This is off by default, since the values can be long, and the table may hold sensitive data.

## Returns

`GT`  
A Great Tables object.

## Examples

Let's scan the `game_revenue` dataset and look at the report.

``` python
import pointblank as pb

game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="polars")

scan = pb.DataScan(game_revenue, tbl_name="game_revenue")

scan.get_tabular_report()
```

[TABLE]

The header shows the table type and its dimensions. In the rows, `item_revenue` has a long right tail (its 95th percentile is far below its maximum), and the string columns `player_id` and `session_id` have fixed lengths. To see what the values look like, add the sample data:

``` python
scan.get_tabular_report(show_sample_data=True)
```

[TABLE]

The samples make it clear that `player_id` and `session_id` are codes, and they show the format of each column's values. The report is a Great Tables object, so it can be saved to a standalone HTML file with its `write_raw_html()` method.
