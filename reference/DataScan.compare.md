# DataScan.compare()

Compare this scan against a baseline and return the differences.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L619-L637)

``` python
DataScan.compare(baseline)
```

The comparison covers schema changes (columns added, removed, or with changed types) and statistical drift for columns present in both scans. The returned [DataScanDiff](../reference/DataScanDiff.md#pointblank.DataScanDiff) object provides programmatic access to the results and a tabular report via `get_tabular_report()`.

## Parameters

`baseline: DataScan`  
The baseline [DataScan](../reference/DataScan.md#pointblank.DataScan) to compare against (typically the older scan).

## Returns

`DataScanDiff`  
An object describing the differences between the two scans.
