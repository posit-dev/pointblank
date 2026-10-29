# Comparison

The result of comparing a source table with a target table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L785-L2540)

``` python
Comparison()
```

[Comparison](../reference/Comparison.md#pointblank.Comparison) objects are created by [`compare()`](../reference/compare.md#pointblank.compare); they can't be created directly. A comparison does its counting when it's created, inside the tables' own engine, and only fetches rows when you ask for them. This keeps it cheap to create and to pass around, even for very large tables.

What you do with a [Comparison](../reference/Comparison.md#pointblank.Comparison) depends on the question being asked:

- **Do the tables match?** A [Comparison](../reference/Comparison.md#pointblank.Comparison) is truthy only when the tables match (see [`identical`](../reference/Comparison.identical.md#pointblank.Comparison.identical)), so `if pb.compare(a, b):` and `assert pb.compare(a, b)` read naturally. [`n_failed`](../reference/Comparison.n_failed.md#pointblank.Comparison.n_failed) says by how many rows they don't.
- **How do they differ?** [`status_counts`](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) breaks the rows down by status, [`schema_diff`](../reference/SchemaDiff.md#pointblank.SchemaDiff) lists the column differences, and [`column_summary()`](../reference/Comparison.column_summary.md#pointblank.Comparison.column_summary) shows which columns have mismatched values.
- **Which rows differ?** [`rows()`](../reference/Comparison.rows.md#pointblank.Comparison.rows) returns a sample of the rows with a given status, as a table of the same type as the inputs.
- **How can I show or store the result?** In a notebook, a [Comparison](../reference/Comparison.md#pointblank.Comparison) displays as its [tabular report](../reference/Comparison.get_tabular_report.md#pointblank.Comparison.get_tabular_report). For logs, files, and other programs, there's [`to_dict()`](../reference/Comparison.to_dict.md#pointblank.Comparison.to_dict) and [`to_json()`](../reference/Comparison.to_json.md#pointblank.Comparison.to_json).

## Attributes

`source_name, target_name`  
Labels for the two tables.

`keys`  
The key columns used to align rows (empty for positional alignment).

`mode`  
`"keyed"`, `"positional"`, or `"multiset"` (for `keys="*"`).

`compared_columns`  
The source columns whose values were compared.

`column_map`  
The mapping of source to target column names given in `column_map=`.

`schema_mode`  
How schema differences affect the result: `"strict"` or `"common"`.

`schema_diff`  
A [`SchemaDiff`](../reference/SchemaDiff.md#pointblank.SchemaDiff) describing schema differences.

`n_source_rows, n_target_rows`  
Row counts of the two tables.

`n_match, n_changed, n_missing, n_extra, n_dup_key`  
Number of test units (aligned rows) with each status.

`n_dup_key_values`  
Number of distinct key values that are duplicated in either table.

`column_mismatches`  
A dict mapping each compared source column to the number of matched rows where its values differ.

## Examples

Let's compare the output of a reporting job with the output of a new version of the job. The new version should produce exactly the same daily totals.

``` python
import pointblank as pb
import polars as pl

old_totals = pl.DataFrame(
    {
        "day": ["2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04"],
        "orders": [118, 131, 97, 142],
        "revenue": [5902.10, 6611.45, 4875.00, 7120.80],
    }
)

new_totals = pl.DataFrame(
    {
        "day": ["2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04"],
        "orders": [118, 131, 97, 142],
        "revenue": [5902.10, 6611.45, 4857.00, 7120.80],
    }
)

comparison = pb.compare(old_totals, new_totals, keys="day")

print(repr(comparison))
```

    Comparison('source' vs 'target': match=3, changed=1, missing=0, extra=0, dup_key=0; schema ok; identical=False)

The printed form is a one-line summary: one day changed and the schemas agree, so the tables aren't identical. In a test or a deployment script, the comparison can be used as a condition directly. Here, when the tables differ, we look at the rows that changed:

``` python
if comparison:
    print("The new job reproduces the old totals.")
else:
    print(f"{comparison.n_failed} of {comparison.n} days differ:")
    print(comparison.rows("changed"))
```

    1 of 4 days differ:
    shape: (1, 5)
    ┌────────────┬───────────────┬───────────────┬────────────────┬────────────────┐
    │ day        ┆ orders_source ┆ orders_target ┆ revenue_source ┆ revenue_target │
    │ ---        ┆ ---           ┆ ---           ┆ ---            ┆ ---            │
    │ str        ┆ i64           ┆ i64           ┆ f64            ┆ f64            │
    ╞════════════╪═══════════════╪═══════════════╪════════════════╪════════════════╡
    │ 2025-06-03 ┆ 97            ┆ 97            ┆ 4875.0         ┆ 4857.0         │
    └────────────┴───────────────┴───────────────┴────────────────┴────────────────┘

The revenue for June 3 has two digits swapped. To see where the mismatches are concentrated, which matters more when there are many columns, use [column_summary()](../reference/Comparison.column_summary.md#pointblank.Comparison.column_summary):

``` python
comparison.column_summary()
```

shape: (2, 6)

| column    | target_column | dtype_source | dtype_target | n_mismatch | f_mismatch |
|-----------|---------------|--------------|--------------|------------|------------|
| str       | str           | str          | str          | i64        | f64        |
| "orders"  | "orders"      | "Int64"      | "Int64"      | 0          | 0.0        |
| "revenue" | "revenue"     | "Float64"    | "Float64"    | 1          | 0.25       |

Only `revenue` has a mismatch, in one of the four matched rows. From here, the full report is available with [`get_tabular_report()`](../reference/Comparison.get_tabular_report.md#pointblank.Comparison.get_tabular_report), and the result can be saved with `to_json()` to keep a record of the check.

## See Also

[compare()](../reference/compare.md#pointblank.compare)  
Create a [Comparison](../reference/Comparison.md#pointblank.Comparison) from two tables.

[SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff)  
The schema differences held in `schema_diff`.

[tbl_match()](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match)  
Run the same comparison as a validation step.
