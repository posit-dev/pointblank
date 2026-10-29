# DataScan.from_dict()

Restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L892-L992)

``` python
DataScan.from_dict(d)
```

The dictionary is the one produced by [`to_dict()`](../reference/DataScan.to_dict.md#pointblank.DataScan.to_dict). This reconstructs the profile without needing the original data, so a profile that was stored earlier can be viewed again with [`get_tabular_report()`](../reference/DataScan.get_tabular_report.md#pointblank.DataScan.get_tabular_report), queried through [`summary_data`](../reference/DataScan.summary_data.md#pointblank.DataScan.summary_data), or used as the baseline in [`compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare).

A restored scan has the statistics but not the data. Its [summary_data](../reference/DataScan.summary_data.md#pointblank.DataScan.summary_data) is a Polars DataFrame, whatever the type of the original table. And when it's compared with another scan, the statistics are compared as usual, but drift scores (which need the values themselves) are only computed for Boolean columns.

## Parameters

`d: dict[str, Any]`  
A dictionary with `"metadata"` and `"columns"` keys, as produced by [`to_dict()`](../reference/DataScan.to_dict.md#pointblank.DataScan.to_dict).

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.

## Examples

Let's take a profile of a table of sensor readings as a dictionary, then restore it.

``` python
import pointblank as pb
import polars as pl

readings = pl.DataFrame(
    {"sensor": ["T1", "T2", "T3", "T4"], "temp_c": [21.4, 22.1, None, 20.8]}
)

profile = pb.DataScan(readings, tbl_name="readings").to_dict()

restored = pb.DataScan.from_dict(profile)

restored.get_tabular_report()
```

[TABLE]

The report is the same as the one for the original scan, even though `restored` was made from the dictionary alone. Converting the restored scan back to a dictionary gives the same dictionary:

``` python
restored.to_dict() == profile
```

    True

So a profile can go through any number of save-and-restore cycles without losing information.
