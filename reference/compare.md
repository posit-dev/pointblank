# compare()

Compare two tables row by row and report how they differ.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L321-L586)

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

A common data question is *how do these two tables differ?* You may have refactored a pipeline and want to know whether its output changed, migrated a table from one database to another, or received a new extract of a data feed. Spot checks and row counts can tell you *that* something changed. The `compare()` function tells you *which* rows changed, how, and in which columns.

It aligns the rows of a *source* table (the reference, or expected data) with those of a *target* table (the data being checked) and classifies every aligned row: rows that match, rows whose values changed, rows missing from the target, and extra rows in the target. It also reports schema differences (columns added, removed, retyped, or reordered). The result is a [`Comparison`](../reference/Comparison.md#pointblank.Comparison) object. It holds the counts, gives samples of the differing rows on request, and displays as a tabular report in a notebook.

Rows are aligned in one of three ways:

- **by key** (`keys=` gives one or more columns): rows are matched on the key values, so row order doesn't matter. This is the most informative mode, and the one to use whenever the data has an identifier.
- **by position** (`keys=None`, the default): the first row of the source is compared with the first row of the target, and so on. This suits outputs that are supposed to be identical row for row.
- **as multisets** (`keys="*"`): whole rows are compared regardless of their order, with duplicates counted. This is for tables that have no key at all.

The comparison is backend-agnostic and runs inside the table's own engine: tables can be Polars or Pandas DataFrames, Polars LazyFrames, Ibis tables (e.g., DuckDB, PostgreSQL), or paths to CSV/Parquet files. File paths are scanned lazily, and only counts and small samples of differing rows are ever collected, so tables that don't fit in memory can still be compared.

The same comparison is available as a validation step with [`Validate.tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), where every aligned row is a test unit that thresholds and actions can respond to. On the command line, it's available as `pb compare`. To compare totals per group rather than individual rows, use [`Validate.tbl_reconcile()`](../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile).

## Parameters

`source: Any`  
The reference table. Can be any table type supported by Pointblank, a path to a CSV or Parquet file, or a callable that returns a table.

`target: Any`  
The table to check against the source. Same input types as `source=`.

`keys: str | list[str] | None = None`  
One or more columns that uniquely identify a row, used to align rows between the tables. Key names refer to the source table (use `column_map=` if the target names them differently). If `None`, rows are aligned by position, which requires the tables to have a defined row order (eager DataFrames, Polars LazyFrames, or any table with `order_by=`). Use `"*"` to compare the tables as multisets of whole rows (see the *Test Units and Row Statuses* section). `tolerance=` can't be used in that mode.

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

In multiset mode (`keys="*"`), a row is either present in the other table or not, so there are no `"changed"` or `"dup_key"` rows. Each distinct row matches as many times as it occurs in both tables, and any surplus copies count as `"missing"` or `"extra"`.

## Large Tables

The comparison runs inside the tables' own engine and only counts and small samples of rows are collected, so file paths (scanned lazily) and database tables never have to be loaded into Python. The engine's join still needs memory for the rows it compares. If that's too much, use `partitions=` to compare the tables in several smaller passes. For the lowest memory use with very large files, read them as DuckDB relations (e.g., `duckdb.connect().read_csv(path)`) and use `partitions=`: in a benchmark with two 10-million-row CSV files (730 MB each), peak memory was about 0.8 GB with DuckDB and `partitions=8`, versus about 5.7 GB to read both files with Polars and join them. With 100-million-row files (7.4 GB each), DuckDB with `partitions=8` peaked at about 2 GB.

## Examples

Let's compare two versions of a product catalog. In the newer version, one product was discontinued, one was added, one had its price lowered, and one went out of stock.

``` python
import pointblank as pb
import polars as pl

catalog_v1 = pl.DataFrame(
    {
        "sku": ["A-101", "A-102", "A-103", "A-104", "A-105"],
        "name": ["Desk lamp", "Office chair", "Monitor arm", "Keyboard", "Mouse pad"],
        "price": [24.99, 149.00, 59.50, 45.00, 9.99],
        "in_stock": [True, True, False, True, True],
    }
)

catalog_v2 = pl.DataFrame(
    {
        "sku": ["A-101", "A-102", "A-104", "A-105", "A-106"],
        "name": ["Desk lamp", "Office chair", "Keyboard", "Mouse pad", "USB hub"],
        "price": [24.99, 139.00, 45.00, 9.99, 19.99],
        "in_stock": [True, True, True, False, True],
    }
)
```

Every product has a unique `sku`, so we'll use it as the key that lines up the rows. Naming the two tables makes the report easier to read.

``` python
comparison = pb.compare(
    catalog_v1, catalog_v2, keys="sku", source_name="catalog_v1", target_name="catalog_v2"
)

comparison
```

[TABLE]

The report's header gives the verdict and a breakdown of the six aligned rows. Below it are the columns with mismatched values, then samples of the changed, missing, and extra rows. For changed rows, only the differing values are shown.

The same information is available programmatically. The status counts add up to the number of test units (aligned rows):

``` python
comparison.status_counts
```

    {'match': 2, 'changed': 2, 'missing': 1, 'extra': 1, 'dup_key': 0}

To work with the differing rows, get them by status with [`rows()`](../reference/Comparison.rows.md#pointblank.Comparison.rows). Changed rows have a `<column>_source` and a `<column>_target` column for each compared column.

``` python
comparison.rows("changed")
```

shape: (2, 7)

| sku | name_source | name_target | price_source | price_target | in_stock_source | in_stock_target |
|----|----|----|----|----|----|----|
| str | str | str | f64 | f64 | bool | bool |
| "A-102" | "Office chair" | "Office chair" | 149.0 | 139.0 | true | true |
| "A-105" | "Mouse pad" | "Mouse pad" | 9.99 | 9.99 | true | false |

`A-102` changed price and `A-105` changed its stock status. If only prices matter, list the columns to compare with `columns=`. The stock change is then ignored and `A-105` matches:

``` python
pb.compare(catalog_v1, catalog_v2, keys="sku", columns="price").status_counts
```

    {'match': 3, 'changed': 1, 'missing': 1, 'extra': 1, 'dup_key': 0}

Without `keys=`, rows are compared by position. Since a product was removed from the middle of the catalog, most rows no longer line up, and four of the five rows count as changed. This is why a key gives a much more useful answer when the data has one.

``` python
pb.compare(catalog_v1, catalog_v2).status_counts
```

    {'match': 1, 'changed': 4, 'missing': 0, 'extra': 0, 'dup_key': 0}

Schema changes are found too. Suppose the next version renames `price` to `unit_price`. With the default `schema="strict"`, the comparison fails every row until the schemas agree. `column_map=` pairs up the renamed column, after which the values are compared as before:

``` python
catalog_v3 = catalog_v2.rename({"price": "unit_price"})

pb.compare(catalog_v1, catalog_v3, keys="sku", column_map={"price": "unit_price"})
```

[TABLE]

The rename is listed as a schema difference, but it no longer causes a failure, and the result is the same as the comparison of `catalog_v1` with `catalog_v2`. To make a comparison like this part of a validation, with thresholds, actions, and a step in the validation report, use [`Validate.tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) with the same arguments.

## See Also

[Comparison](../reference/Comparison.md#pointblank.Comparison)  
The object returned by this function.

[SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff)  
The schema differences found by a comparison.

[tbl_match()](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match)  
Use a table comparison as a validation step.

[tbl_reconcile()](../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile)  
Compare aggregates per group instead of rows.
