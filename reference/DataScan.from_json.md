# DataScan.from_json()

Restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a JSON string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L994-L1049)

``` python
DataScan.from_json(json_string)
```

The JSON string is the one produced by [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json). It's the counterpart to that method for profiles stored as text, such as in a database column or an API response. To read a JSON file, use [`load_from_json()`](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json).

As with [`from_dict()`](../reference/DataScan.from_dict.md#pointblank.DataScan.from_dict), the restored scan has the statistics but not the data. Values that were written as strings in the JSON (such as dates) stay strings.

## Parameters

`json_string: str`  
A JSON string as produced by [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json).

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.

## Examples

Suppose the profile of the `small_table` dataset was stored as JSON text some time ago.

``` python
import pointblank as pb
import polars as pl

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

stored_json = pb.DataScan(small_table, tbl_name="small_table").to_json()
```

Restoring it gives a [DataScan](../reference/DataScan.md#pointblank.DataScan) that can be queried like the original one:

``` python
restored = pb.DataScan.from_json(stored_json)

restored.summary_data.select("colname", "n_missing", "n_unique")
```

shape: (8, 3)

| colname     | n_missing | n_unique |
|-------------|-----------|----------|
| str         | i64       | i64      |
| "date_time" | 0         | 12       |
| "date"      | 0         | 11       |
| "a"         | 0         | 7        |
| "b"         | 0         | 12       |
| "c"         | 2         | 7        |
| "d"         | 0         | 12       |
| "e"         | 0         | 2        |
| "f"         | 0         | 3        |

The table name, row count, and every column's statistics come from the JSON, so the original table isn't needed to answer questions about it later.
