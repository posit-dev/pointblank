# DataScan.save_to_json()

Save the profile to a JSON file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L839-L890)

``` python
DataScan.save_to_json(output_file)
```

Saving a scan to a file is the simplest way to keep a baseline: scan a table once, save the profile, and load it later with [`load_from_json()`](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json) to compare a new scan against it with [`compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare). The file holds the output of [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json), so it's small (it stores the statistics, not the data) and can be kept in version control next to the code that produces the table.

## Parameters

`output_file: str`  
The path to the output JSON file. An existing file is overwritten.

## Returns

`None`  
Nothing is returned. The profile is written to `output_file`.

## Examples

Let's save a profile of the `small_table` dataset to a file in a temporary directory.

``` python
import pointblank as pb
import tempfile
from pathlib import Path

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

profile_path = Path(tempfile.mkdtemp()) / "small_table_profile.json"

pb.DataScan(small_table, tbl_name="small_table").save_to_json(profile_path)

print(profile_path.read_text()[:180])
```

    {
        "metadata": {
            "table_name": "small_table",
            "row_count": 13,
            "columns": [
                "date_time",
                "date",
                "a",
                "b"

The file starts with the table's metadata, and the column profiles follow. Loading it with [`load_from_json()`](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json) gives back the same profile, without needing the table itself.
