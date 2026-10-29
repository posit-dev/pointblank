# DataScan.compare()

Compare this scan against a baseline and return the differences.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L1123-L1245)

``` python
DataScan.compare(baseline)
```

Tables that are refreshed regularly (a monthly extract, a nightly load, a model's input data) can change in ways that no single row reveals: a column is dropped, a type changes, or the values shift to a different range. Comparing a scan of the current table with a scan of an earlier version that's known to be good finds those changes. The baseline is often a profile saved earlier with [`save_to_json()`](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json) and brought back with [`load_from_json()`](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json).

The scan this method is called on is the *current* one, and `baseline` is the one it's compared against. The result is a [`DataScanDiff`](../reference/DataScanDiff.md#pointblank.DataScanDiff), which has the differences as attributes, as a dictionary through [`to_dict()`](../reference/DataScanDiff.md#pointblank.DataScanDiff.to_dict), and as a report through [`get_tabular_report()`](../reference/DataScanDiff.md#pointblank.DataScanDiff.get_tabular_report).

This compares profiles, not rows. To find out which rows of two tables differ, use [`compare()`](../reference/compare.md#pointblank.compare) instead.

## What Is Compared

Columns are matched by name. A column that's only in the current scan is *added*, one that's only in the baseline is *removed*, and one whose data type differs is *type-changed*. For every column in both scans, each statistic is compared for exact equality, so any difference at all (even in the last decimal place of a mean) is reported along with both values. When the type has changed, the statistics might no longer measure the same thing (a numeric column's mean is of its values, but a string column's mean is of its lengths).

## Drift Scores

Statistics show that something changed but not how much the distribution of values moved. For that, drift scores are computed for columns whose type hasn't changed:

- numeric columns get the population stability index (`psi`), with the values divided into ten bins at the baseline's deciles (at least ten values are needed on each side), and the Kolmogorov-Smirnov statistic (`ks_statistic`, the largest gap between the two cumulative distributions) with its p-value (`ks_p_value`). The p-value comes from SciPy if it's installed and from an approximation otherwise
- all other columns get a `psi` computed from how often each distinct value appears

A common rule of thumb reads a PSI below 0.1 as little or no shift, from 0.1 to 0.25 as a moderate shift, and above 0.25 as a large one. A score of zero is left out.

Drift scores need the values themselves, so they're only computed when both scans were made from data. A scan restored with [`from_dict()`](../reference/DataScan.from_dict.md#pointblank.DataScan.from_dict), [`from_json()`](../reference/DataScan.from_json.md#pointblank.DataScan.from_json), or [`load_from_json()`](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json) only has statistics, and then only Boolean columns (whose value counts are part of the profile) get a drift score.

## Parameters

`baseline: DataScan`  
The baseline [DataScan](../reference/DataScan.md#pointblank.DataScan) to compare against (typically the older scan).

## Returns

`DataScanDiff`  
An object describing the differences between the two scans.

## Examples

Let's compare two monthly extracts of an orders table. Between January and February, the `gift` column was dropped, a `coupon` column was added, a new `"app"` channel appeared, and order amounts went up.

``` python
import pointblank as pb
import polars as pl

jan = pl.DataFrame(
    {
        "order_id": list(range(1, 21)),
        "amount": [20.0 + i for i in range(20)],
        "channel": ["web"] * 14 + ["store"] * 6,
        "gift": [i % 4 == 0 for i in range(20)],
    }
)

feb = pl.DataFrame(
    {
        "order_id": list(range(21, 45)),
        "amount": [35.0 + 1.5 * i for i in range(24)],
        "channel": ["web"] * 10 + ["store"] * 8 + ["app"] * 6,
        "coupon": ["SPRING"] * 5 + [None] * 19,
    }
)

scan_jan = pb.DataScan(jan, tbl_name="orders_jan")
scan_feb = pb.DataScan(feb, tbl_name="orders_feb")

diff = scan_feb.compare(scan_jan)

diff
```

    DataScanDiff(added=1, removed=1, type_changed=0, stat_changed=3)

The printed summary counts one added column, one removed column, and three columns whose statistics changed. The report shows the details:

``` python
diff.get_tabular_report()
```

[TABLE]

The PSI of `amount` (about 6.7) and its KS statistic (0.875) are far beyond the rule of thumb, so the amounts really did move. The `channel` column has a PSI of about 2.1 because of the new `"app"` value. The shift in `order_id` is expected, since each month has its own IDs, which shows that drift scores still need to be read with the meaning of each column in mind.

## See Also

[DataScanDiff](../reference/DataScanDiff.md#pointblank.DataScanDiff)  
The result of comparing two scans.

[compare()](../reference/compare.md#pointblank.compare)  
Compare two tables row by row.
