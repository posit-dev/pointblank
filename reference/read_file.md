# read_file()

Read a Validate object from disk that was previously saved with [write_file()](../reference/write_file.md#pointblank.write_file).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L757-L922)

``` python
read_file(filepath)
```

This function loads a validation object that was previously serialized to disk using the [write_file()](../reference/write_file.md#pointblank.write_file) function. The validation object will be restored with all its validation results, metadata, and optionally the source data (if it was saved with `keep_tbl=True`).

> **Warning**
>
> The [read_file()](../reference/read_file.md#pointblank.read_file) function is currently experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).

## Parameters

`filepath: str | Path`  
The path to the saved validation file. Can be a string or Path object.

## Returns

`Validate`  
The restored validation object with all its original state, validation results, and metadata.

## Examples

Let's first save a validation object to disk with [write_file()](../reference/write_file.md#pointblank.write_file). We'll use a temporary directory here, and `keep_tbl=True` saves the target table along with the validation results:

``` python
import pointblank as pb
import os
import tempfile

validation = (
    pb.Validate(data=pb.load_dataset("small_table"), tbl_name="small_table")
    .col_vals_gt(columns="d", value=1000)
    .col_vals_not_null(columns="c")
    .interrogate()
)

save_dir = tempfile.mkdtemp()
pb.write_file(validation, "my_validation", path=save_dir, keep_tbl=True, quiet=True)
```

Now we can load the validation object from disk with [read_file()](../reference/read_file.md#pointblank.read_file). The filename can be given without its extension, in which case [read_file()](../reference/read_file.md#pointblank.read_file) looks for `"my_validation.pkl"`:

``` python
restored = pb.read_file(os.path.join(save_dir, "my_validation"))

restored
```

[TABLE]

The restored object has the same results as the original: the first step has 6 failing test units and the second has 2. The loaded validation object retains all its functionality, so we can, for example, get the rows that failed any step (since the target table was saved):

``` python
pb.preview(restored.get_sundered_data(type="fail"))
```

| PolarsRows8Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 2 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 3 | 2016-01-15 18:46:00 | 2016-01-15 | 7 | 1-knw-093 | 3 | 843.34 | True | high |
| 4 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 5 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 6 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 7 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 8 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

These are the 8 rows of `small_table` that failed at least one of the two steps. Without `keep_tbl=True` in [write_file()](../reference/write_file.md#pointblank.write_file), the table wouldn't be saved and methods that need the data, like [get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data), wouldn't be available after loading.

## See Also

[write_file()](../reference/write_file.md#pointblank.write_file)  
Save a validation object to disk for later retrieval with [read_file()](../reference/read_file.md#pointblank.read_file).
