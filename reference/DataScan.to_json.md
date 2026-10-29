# DataScan.to_json()

Export the profile as a JSON string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L792-L837)

``` python
DataScan.to_json()
```

JSON is the format to use when a profile is stored outside of Python or sent to another system: in a database column, an object store, or an API response. The structure is the same as that of [`to_dict()`](../reference/DataScan.to_dict.md#pointblank.DataScan.to_dict), and [`from_json()`](../reference/DataScan.from_json.md#pointblank.DataScan.from_json) restores a [DataScan](../reference/DataScan.md#pointblank.DataScan) from it. Values that JSON can't represent directly, such as dates, are written as strings. To write the JSON to a file, use [`save_to_json()`](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json).

## Returns

`str`  
A JSON string representing the profile, indented by four spaces.

## Examples

Here, a table of shipments with a date column is scanned and converted to JSON.

``` python
import pointblank as pb
import polars as pl
from datetime import date

shipments = pl.DataFrame(
    {"shipped": [date(2025, 3, 1), date(2025, 3, 4), date(2025, 3, 9)]}
)

profile_json = pb.DataScan(shipments, tbl_name="shipments").to_json()

print(profile_json)
```

    {
        "metadata": {
            "table_name": "shipments",
            "row_count": 3,
            "columns": [
                "shipped"
            ],
            "skipped_columns": {}
        },
        "columns": [
            {
                "colname": "shipped",
                "coltype": "Date",
                "sample_data": [
                    "2025-03-01",
                    "2025-03-04",
                    "2025-03-09"
                ],
                "statistics": {
                    "n_missing": 0,
                    "n_unique": 3,
                    "min": "2025-03-01",
                    "max": "2025-03-09"
                }
            }
        ]
    }

The earliest and latest dates are stored as strings. Because the result is text, it can be stored wherever text can, and [`from_json()`](../reference/DataScan.from_json.md#pointblank.DataScan.from_json) turns it back into a [DataScan](../reference/DataScan.md#pointblank.DataScan).
