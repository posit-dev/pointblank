# Comparison.rows()

Get a sample of rows with a given status.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L1567-L1684)

``` python
Comparison.rows(
    status,
    limit=100,
)
```

The counts in [`status_counts`](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) say how many rows differ, and [rows()](../reference/Comparison.rows.md#pointblank.Comparison.rows) shows which ones. It's the step that turns "12 rows changed" into something that can be investigated: the keys to look up, the values on each side, and the rows that were dropped or added.

A comparison collects counts when it's made, but it doesn't collect rows until they're asked for. [rows()](../reference/Comparison.rows.md#pointblank.Comparison.rows) then fetches up to `limit=` rows with the requested status from the tables' own engine, so it's safe to use with very large tables and with database tables. The result is cached, so asking for the same status and limit again returns the same table without querying the tables a second time.

## Parameters

`status: Literal["changed", "missing", "extra", "dup_key", "match"]`  
The row status to retrieve:

- `"changed"`: the keys, then a `<column>_source` and a `<column>_target` column for each compared column (including columns whose values are equal in that row)
- `"missing"`: source rows not in the target (source column names)
- `"extra"`: target rows not in the source (target column names)
- `"dup_key"`: duplicated key values with their occurrence counts in each table (`n_source`, `n_target`), one row per key
- `"match"`: the keys of matching rows

`limit: int = 100`  
The maximum number of rows to return.

## Returns

`Any`  
A table of the same type as the (collected) input tables. In positional mode, rows are identified by a 1-based `_row_num_` column, and a `"changed"` sample has the `_source` and `_target` columns for every column. In multiset mode (`keys="*"`), `"missing"` and `"extra"` return distinct rows, with the compared values as strings, plus `n_source` and `n_target` columns giving how often each row occurs in each table.

## Raises

`ValueError`  
If `status=` isn't one of the five row statuses.

## Examples

Let's compare two versions of an employee table, where one salary changed, one employee left, and one joined.

``` python
import pointblank as pb
import polars as pl

staff_jan = pl.DataFrame(
    {
        "emp_id": [101, 102, 103, 104],
        "dept": ["Sales", "Sales", "Ops", "Ops"],
        "salary": [52000, 61000, 58000, 47000],
    }
)
staff_feb = pl.DataFrame(
    {
        "emp_id": [101, 102, 103, 105],
        "dept": ["Sales", "Sales", "Ops", "Ops"],
        "salary": [52000, 63500, 58000, 49000],
    }
)

comparison = pb.compare(staff_jan, staff_feb, keys="emp_id")

comparison.rows("changed")
```

shape: (1, 5)

| emp_id | dept_source | dept_target | salary_source | salary_target |
|--------|-------------|-------------|---------------|---------------|
| i64    | str         | str         | i64           | i64           |
| 102    | "Sales"     | "Sales"     | 61000         | 63500         |

Each changed row has both versions of every compared column side by side. Here, `dept` is the same on both sides and `salary` is the column that changed. The departed employee is a `"missing"` row, and it's shown as it was in the source table:

``` python
comparison.rows("missing")
```

shape: (1, 3)

| emp_id | dept  | salary |
|--------|-------|--------|
| i64    | str   | i64    |
| 104    | "Ops" | 47000  |

The new employee is an `"extra"` row, shown as it is in the target table:

``` python
comparison.rows("extra")
```

shape: (1, 3)

| emp_id | dept  | salary |
|--------|-------|--------|
| i64    | str   | i64    |
| 105    | "Ops" | 49000  |

In multiset mode, rows have no identity apart from their values. So `"missing"` and `"extra"` rows are listed once each, with the number of times they occur in each table. Here, the target table has one fewer copy of a row than the source:

``` python
scans_source = pl.DataFrame({"badge": [7, 7, 9], "door": ["A", "A", "B"]})
scans_target = pl.DataFrame({"badge": [7, 9], "door": ["A", "B"]})

pb.compare(scans_source, scans_target, keys="*").rows("missing")
```

shape: (1, 4)

| badge | door | n_source | n_target |
|-------|------|----------|----------|
| str   | str  | i64      | i64      |
| "7"   | "A"  | 2        | 1        |

The row with badge 7 at door A occurs twice in the source and once in the target, so one copy is missing. For a summary of all the samples in one place, see [`get_tabular_report()`](../reference/Comparison.get_tabular_report.md#pointblank.Comparison.get_tabular_report), which shows up to 10 rows for each status by default.
