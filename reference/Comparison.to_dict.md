# Comparison.to_dict()

Get the comparison results as a JSON-serializable dict.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L2265-L2377)

``` python
Comparison.to_dict(
    samples=True,
    limit=10,
)
```

A comparison is a live object that refers to the tables it compared. `to_dict()` turns its results into plain data that can be stored or sent elsewhere: written to a log or a database table, attached to a CI run, checked in a test, or passed to a dashboard. Everything in the dictionary is a string, number, boolean, list, dict, or `None` (dates and other values in the samples are converted to strings), so it can be serialized as it is. For a JSON string, use [`to_json()`](../reference/Comparison.to_json.md#pointblank.Comparison.to_json).

## Parameters

`samples: bool = True`  
Should samples of differing rows be included? Set to `False` when the data might contain sensitive values (e.g., PII) and only counts and schema details may be stored.

`limit: int = 10`  
The maximum number of sample rows per status.

## Returns

`dict[str, Any]`  
A dictionary with the comparison settings (`"source_name"`, `"target_name"`, `"mode"`, `"keys"`, `"compared_columns"`, `"column_map"`, `"schema_mode"`, `"null_equal"`, and `"partitions"`), the row counts of both tables, the results (`"n"`, `"n_passed"`, `"n_failed"`, `"identical"`, `"status_counts"`, and `"column_mismatches"`), and the schema differences (`"schema_diff"`, from [`SchemaDiff.to_dict()`](../reference/SchemaDiff.md#pointblank.SchemaDiff.to_dict)). If `samples=True`, there's also a `"samples"` entry, which maps each failing status that has rows to a list of row dictionaries, structured as in [`rows()`](../reference/Comparison.rows.md#pointblank.Comparison.rows).

## Examples

Let's compare two months of an employee table and turn the result into a dictionary.

``` python
import pointblank as pb
import polars as pl

staff_jan = pl.DataFrame(
    {"emp_id": [101, 102, 103, 104], "salary": [52000, 61000, 58000, 47000]}
)
staff_feb = pl.DataFrame(
    {"emp_id": [101, 102, 103, 105], "salary": [52000, 63500, 58000, 49000]}
)

comparison = pb.compare(
    staff_jan, staff_feb, keys="emp_id", source_name="staff_jan", target_name="staff_feb"
)

result = comparison.to_dict()

result["status_counts"]
```

    {'match': 2, 'changed': 1, 'missing': 1, 'extra': 1, 'dup_key': 0}

The samples hold the actual rows, keyed by status:

``` python
result["samples"]
```

    {'changed': [{'emp_id': 102, 'salary_source': 61000, 'salary_target': 63500}],
     'missing': [{'emp_id': 104, 'salary': 47000}],
     'extra': [{'emp_id': 105, 'salary': 49000}]}

Salaries are sensitive, so a record that's kept in a shared log might leave the samples out. With `samples=False`, only counts, settings, and schema details remain:

``` python
record = comparison.to_dict(samples=False)

{k: record[k] for k in ["source_name", "target_name", "n_failed", "identical"]}
```

    {'source_name': 'staff_jan',
     'target_name': 'staff_feb',
     'n_failed': 3,
     'identical': False}

Leaving out the samples also means the tables aren't queried again, because the counts were collected when the comparison was made. To limit the samples instead of dropping them, set `limit=` to the number of rows to keep for each status.
