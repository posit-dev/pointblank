## read_file()


Read a Validate object from disk that was previously saved with [write_file()](write_file.md#pointblank.write_file).


Usage

``` python
read_file(filepath)
```


This function loads a validation object that was previously serialized to disk using the [write_file()](write_file.md#pointblank.write_file) function. The validation object will be restored with all its validation results, metadata, and optionally the source data (if it was saved with `keep_tbl=True`).

> **Warning: Warning**
>
> The [read_file()](read_file.md#pointblank.read_file) function is currently experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).


## Parameters


`filepath: str | Path`  
The path to the saved validation file. Can be a string or Path object.


## Returns


`Validate`  
The restored validation object with all its original state, validation results, and metadata.


## Examples

Load a validation object that was previously saved:

``` python
import pointblank as pb

# Load a validation object from disk
validation = pb.read_file("my_validation.pkl")

# View the validation results
validation
```

You can also load using just the filename (without extension):

``` python
# This will automatically look for "my_validation.pkl"
validation = pb.read_file("my_validation")
```

The loaded validation object retains all its functionality:

``` python
# Get validation summary
summary = validation.get_json_report()

# Get sundered data (if original table was saved)
if validation.data is not None:
    failing_rows = validation.get_sundered_data(type="fail")
```


#### See Also

[Use](Use.md), [to](to.md)
