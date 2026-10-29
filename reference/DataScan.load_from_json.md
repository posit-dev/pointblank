# DataScan.load_from_json()

Load a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a JSON file produced by [save_to_json()](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L1051-L1121)

``` python
DataScan.load_from_json(input_file)
```

This is how a saved baseline is brought back: load the profile from the file, then pass it to the [`compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare) method of a new scan to find out what has changed since. The file is the one written by [`save_to_json()`](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json) (or any file containing the output of [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json)).

As with [`from_dict()`](../reference/DataScan.from_dict.md#pointblank.DataScan.from_dict), the loaded scan has the statistics but not the data, so drift scores are only computed for its Boolean columns.

## Parameters

`input_file: str`  
The path to the JSON file.

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.

## Examples

Let's save a profile of a table of orders as a baseline.

``` python
import pointblank as pb
import polars as pl
import tempfile
from pathlib import Path

orders = pl.DataFrame(
    {"order_id": [1, 2, 3, 4], "amount": [25.0, 40.0, 15.5, 60.0]}
)

baseline_path = Path(tempfile.mkdtemp()) / "orders_profile.json"

pb.DataScan(orders, tbl_name="orders").save_to_json(baseline_path)
```

Later, the table has grown and a refund (a negative amount) has appeared. The baseline is loaded from the file and compared with a scan of the current table:

``` python
orders_now = pl.DataFrame(
    {"order_id": [1, 2, 3, 4, 5], "amount": [25.0, 40.0, 15.5, 60.0, -15.5]}
)

baseline = pb.DataScan.load_from_json(baseline_path)

diff = pb.DataScan(orders_now, tbl_name="orders").compare(baseline)

diff.to_dict()["stat_diffs"]["amount"]["min"]
```

    {'baseline': 15.5, 'current': -15.5}

The minimum `amount` went from 15.5 to -15.5. A change like this is easy to miss in the data but stands out in a comparison of profiles, and a check such as [`Validate.col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge) can then guard against it.
