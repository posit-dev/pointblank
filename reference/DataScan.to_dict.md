# DataScan.to_dict()

Export the profile as a structured dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L708-L790)

``` python
DataScan.to_dict()
```

A profile is worth keeping when it serves as a baseline: a record of what a table looked like at one point, to compare later scans against, or to store alongside the table in a catalog. `to_dict()` returns the whole profile as plain Python data, which [`from_dict()`](../reference/DataScan.from_dict.md#pointblank.DataScan.from_dict) turns back into a [DataScan](../reference/DataScan.md#pointblank.DataScan). For a JSON string or file, use [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json) or [`save_to_json()`](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json).

The profile includes the first five non-missing values of each column (`sample_data`). If the table holds sensitive values, keep that in mind before storing or sharing the result.

## Returns

`dict[str, Any]`  
A dictionary with two keys. `"metadata"` holds the table name, the row count, the list of column names, and `"skipped_columns"` (a dictionary mapping each column of an unsupported type to its type). `"columns"` holds a list with one entry per summarized column, each a dictionary with `"colname"`, `"coltype"`, `"sample_data"`, and `"statistics"` (a dictionary mapping each statistic's name to its value).

## Examples

Let's scan a small table of sensor readings, where one reading is missing.

``` python
import pointblank as pb
import polars as pl

readings = pl.DataFrame(
    {"sensor": ["T1", "T2", "T3", "T4"], "temp_c": [21.4, 22.1, None, 20.8]}
)

profile = pb.DataScan(readings, tbl_name="readings").to_dict()

profile["metadata"]
```

    {'table_name': 'readings',
     'row_count': 4,
     'columns': ['sensor', 'temp_c'],
     'skipped_columns': {}}

Each column has its own entry, with the statistics in a nested dictionary:

``` python
profile["columns"][1]["statistics"]
```

    {'n_missing': 1,
     'n_unique': 4,
     'mean': 21.433333333333334,
     'median': 21.4,
     'std': 0.6506407098647716,
     'min': 20.8,
     'max': 22.1,
     'p05': 20.806,
     'q_1': 21.1,
     'q_3': 21.75,
     'p95': 22.03,
     'iqr': 0.6499999999999986}

The missing reading is counted in `n_missing` and left out of the other statistics. The statistics keep their Python types (the minimum and maximum of a date column, for example, are `datetime.date` values), and passing the dictionary to [`from_dict()`](../reference/DataScan.from_dict.md#pointblank.DataScan.from_dict) gives back the same profile.
