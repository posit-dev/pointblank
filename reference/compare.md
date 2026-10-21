# compare()

Compare two tables row by row and report how they differ.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L116-L291)

``` python
compare(
    source,
    target,
    keys=None,
    columns=None,
    column_map=None,
    tolerance=None,
    normalize=None,
    null_equal=True,
    schema="strict",
    dup_keys="flag",
    order_by=None,
    partitions=None,
    source_name=None,
    target_name=None,
)
```

The `compare()` function aligns the rows of a *source* table (the reference, or expected data) with those of a *target* table (the data being checked) and classifies every aligned row: rows that match, rows whose values changed, rows missing from the target, and extra rows in the target. It also reports schema differences (columns added, removed, retyped, or reordered). The returned [Comparison](../reference/Comparison.md#pointblank.Comparison) object gives you the counts, samples of the differing rows, and a tabular report.

Rows are aligned in one of two ways:

- **by key** (`keys=` given): rows are matched on the values of one or more key columns, so row order doesn't matter. This is the most useful mode for comparing extracts of the same data (e.g., the source and target of a pipeline run).
- **by position** (`keys=None`, the default): the first row of the source is compared with the first row of the target, and so on.

The comparison is backend-agnostic and runs inside the table's own engine: tables can be Polars or Pandas DataFrames, Polars LazyFrames, Ibis tables (e.g., DuckDB, PostgreSQL), or paths to CSV/Parquet files. File paths are scanned lazily, and only counts and small samples of differing rows are ever collected, so tables that don't fit in memory can still be compared.

## Parameters

`source: Any`  
The reference table. Can be any table type supported by Pointblank, a path to a CSV or Parquet file, or a callable that returns a table.

`target: Any`  
The table to check against the source. Same input types as `source=`.

`keys: str | list[str] | None = None`  
One or more columns that uniquely identify a row, used to align rows between the tables. Key names refer to the source table (use `column_map=` if the target names them differently). If `None`, rows are aligned by position, which requires the tables to have a defined row order (eager DataFrames, Polars LazyFrames, or any table with `order_by=`).

`columns: str | list[str] | None = None`  
The source columns whose values should be compared. By default, all columns present in both tables (other than the keys) are compared. Giving `columns=` also limits the schema check to the keys and these columns.

`column_map: dict[str, str] | None = None`  
A mapping of source column names to target column names, for columns that were renamed.

`tolerance: Any = None`  
Tolerance for numeric and datetime comparisons. A number is an absolute tolerance applied to all numeric columns. A dict with `"atol"` and/or `"rtol"` keys applies to all numeric columns, where values are equal if `|source - target| <= atol + rtol * |source|`. A dict mapping column names to either of the above sets per-column tolerances. Datetime tolerances are given as a `datetime.timedelta` or a string like `"1s"`, `"500ms"`, `"2m"`, or `"1d"`, e.g., `tolerance={"updated_at": "1s"}`.

`normalize: dict[str, Any] | None = None`  
Per-column transformations applied to both sides before comparing, as a dict mapping a column name to a list of operations: `"strip"`, `"lower"`, `"upper"`, `"collapse_whitespace"`, or `{"round": n}`. The string operations only apply to string columns and `round` only to numeric ones (other columns are left as they are). Reported values are always the original ones.

`null_equal: bool = True`  
Should a null in the source and a null in the target count as equal? By default, they do. NaN values in floating-point columns are treated as null.

`schema: Literal["strict", "common"] = "strict"`  
How schema differences affect the result. With `"strict"` (the default), any schema difference (other than renames given in `column_map=`) means the tables don't match, and every row counts as failing. With `"common"`, only the shared columns are compared and schema differences are reported without failing rows.

`dup_keys: Literal["flag", "compare"] = "flag"`  
Rows whose key appears more than once in either table can't be aligned unambiguously, so they always get the `"dup_key"` status. With `"flag"` (the default), they are only flagged and counted. With `"compare"`, the report also compares the duplicated rows pairwise (every source row against every target row with the same key) to show which columns differ.

`order_by: str | list[str] | None = None`  
Column(s) defining row order for positional alignment on backends without an inherent row order (e.g., database tables). Ignored when `keys=` is given.

`partitions: int | None = None`  
Compare the tables in this many passes, each covering a disjoint subset of the key values (by key hash). This bounds peak memory for tables too large to compare in one pass (each pass joins only about `1/partitions` of the rows), at the cost of reading the tables once per pass. Requires `keys=`, and Polars, DuckDB, or Ibis tables (including CSV/Parquet paths). The results are identical to a single-pass comparison.

`source_name: str | None = None`  
Optional labels for the two tables, used in the report.

`target_name: str | None = None`  
Optional labels for the two tables, used in the report.

## Returns

`Comparison`  
The result of the comparison.

## Test Units And Row Statuses

Every aligned row is one *test unit* and gets exactly one status:

- `"match"`: the row is in both tables and all compared values are equal (within tolerance)
- `"changed"`: the row is in both tables but at least one compared value differs
- `"missing"`: the row is in the source but not in the target
- `"extra"`: the row is in the target but not in the source
- `"dup_key"`: the row's key appears more than once in either table. A duplicated key counts as many units as the larger of its two occurrence counts.

Rows with a null in a key column can't be matched: they count as `"missing"` (source) or `"extra"` (target).

## Large Tables

The comparison runs inside the tables' own engine and only counts and small samples of rows are collected, so file paths (scanned lazily) and database tables never have to be loaded into Python. The engine's join still needs memory for the rows it compares. If that's too much, use `partitions=` to compare the tables in several smaller passes. For the lowest memory use with very large files, read them as DuckDB relations (e.g., `duckdb.connect().read_csv(path)`) and use `partitions=`: in a benchmark with two 10-million-row CSV files (730 MB each), peak memory was about 0.8 GB with DuckDB and `partitions=8`, versus about 5.7 GB to read both files with Polars and join them. With 100-million-row files (7.4 GB each), DuckDB with `partitions=8` peaked at about 2 GB.

## Examples

``` python
import pointblank as pb
import polars as pl

source = pl.DataFrame(
    {"id": [1, 2, 3, 4], "amount": [10.0, 20.0, 30.0, 40.0], "region": ["N", "S", "E", "W"]}
)
target = pl.DataFrame(
    {"id": [1, 2, 4, 5], "amount": [10.0, 20.5, 40.0, 50.0], "region": ["N", "S", "W", "N"]}
)

cmp = pb.compare(source, target, keys="id")
cmp
```

[TABLE]

The counts are available as attributes, and the differing rows can be retrieved by status:

``` python
cmp.n_changed, cmp.n_missing, cmp.n_extra
```

    (1, 1, 1)

``` python
cmp.rows("changed")
```

shape: (1, 5)

| id  | amount_source | amount_target | region_source | region_target |
|-----|---------------|---------------|---------------|---------------|
| i64 | f64           | f64           | str           | str           |
| 2   | 20.0          | 20.5          | "S"           | "S"           |

A tolerance can absorb small numeric differences:

``` python
pb.compare(source, target, keys="id", tolerance={"amount": 1}).n_changed
```

    0
