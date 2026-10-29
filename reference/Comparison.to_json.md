# Comparison.to_json()

Get the comparison results as a JSON string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L2379-L2437)

``` python
Comparison.to_json(
    samples=True,
    limit=10,
    indent=2,
)
```

JSON is the format to use when a comparison result leaves Python: written to a file alongside a pipeline run, returned from an API, or read by another tool. The structure is the same as that of [`to_dict()`](../reference/Comparison.to_dict.md#pointblank.Comparison.to_dict). The `pb compare` command line tool produces the same JSON with `--format json` (to the terminal) or `--output-json` (to a file), and its `--no-samples` option corresponds to `samples=False`.

## Parameters

`samples: bool = True`  
Should samples of differing rows be included? Set to `False` when the data might contain sensitive values (e.g., PII) and only counts and schema details may be stored.

`limit: int = 10`  
The maximum number of sample rows per status.

`indent: int | None = 2`  
The number of spaces to indent nested levels by. Use `None` for the most compact output, on a single line (as for JSON Lines files, with one comparison per line).

## Returns

`str`  
The comparison results as a JSON string.

## Examples

Here, a new version of a price list has one changed price.

``` python
import pointblank as pb
import polars as pl

prices_v1 = pl.DataFrame({"sku": ["A-1", "A-2", "A-3"], "price": [9.99, 24.50, 5.00]})
prices_v2 = pl.DataFrame({"sku": ["A-1", "A-2", "A-3"], "price": [9.99, 26.00, 5.00]})

comparison = pb.compare(prices_v1, prices_v2, keys="sku")

print(comparison.to_json(limit=5))
```

    {
      "source_name": "source",
      "target_name": "target",
      "mode": "keyed",
      "keys": [
        "sku"
      ],
      "compared_columns": [
        "price"
      ],
      "column_map": {},
      "schema_mode": "strict",
      "null_equal": true,
      "partitions": 1,
      "n_source_rows": 3,
      "n_target_rows": 3,
      "n": 3,
      "n_passed": 2,
      "n_failed": 1,
      "identical": false,
      "status_counts": {
        "match": 2,
        "changed": 1,
        "missing": 0,
        "extra": 0,
        "dup_key": 0
      },
      "column_mismatches": {
        "price": 1
      },
      "schema_diff": {
        "matches": true,
        "only_in_source": [],
        "only_in_target": [],
        "dtype_changed": [],
        "renamed": [],
        "case_only": [],
        "reordered": false
      },
      "samples": {
        "changed": [
          {
            "sku": "A-2",
            "price_source": 24.5,
            "price_target": 26.0
          }
        ]
      }
    }

For a compact record without row samples, set `samples=False` and `indent=None`:

``` python
print(comparison.to_json(samples=False, indent=None))
```

    {"source_name": "source", "target_name": "target", "mode": "keyed", "keys": ["sku"], "compared_columns": ["price"], "column_map": {}, "schema_mode": "strict", "null_equal": true, "partitions": 1, "n_source_rows": 3, "n_target_rows": 3, "n": 3, "n_passed": 2, "n_failed": 1, "identical": false, "status_counts": {"match": 2, "changed": 1, "missing": 0, "extra": 0, "dup_key": 0}, "column_mismatches": {"price": 1}, "schema_diff": {"matches": true, "only_in_source": [], "only_in_target": [], "dtype_changed": [], "renamed": [], "case_only": [], "reordered": false}}

Each such record is one line, so a series of comparisons can be appended to a single file and read back later, one comparison per line.
