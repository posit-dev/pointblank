## write_file()


Write a Validate object to disk as a serialized file.


Usage

``` python
write_file(
    validation,
    filename,
    path=None,
    keep_tbl=False,
    keep_extracts=False,
    quiet=False
)
```


Writing a validation object to disk with [write_file()](write_file.md#pointblank.write_file) can be useful for keeping data validation results close at hand for later retrieval (with [read_file()](read_file.md#pointblank.read_file)). By default, any data table that the validation object holds will be removed before writing to disk (not applicable if no data table is present). This behavior can be changed by setting `keep_tbl=True`, but this only works when the table is not of a database type (e.g., DuckDB, PostgreSQL, etc.), as database connections cannot be serialized.

Extract data from failing validation steps can also be preserved by setting `keep_extracts=True`, which is useful for later analysis of data quality issues.

The serialized file uses Python's pickle format for storage of the validation object state, including all validation results, metadata, and optionally the source data.

**Important note.** If your validation uses custom preprocessing functions (via the `pre=` parameter), these functions must be defined at the module level (not interactively or as lambda functions) to ensure they can be properly restored when loading the validation in a different Python session. Read the *Creating Serializable Validations* section below for more information.

> **Warning: Warning**
>
> The [write_file()](write_file.md#pointblank.write_file) function is currently experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).


## Parameters


`validation: Validate`  
The [Validate](Validate.md#pointblank.Validate) object to write to disk.

`filename: str`  
The filename to create on disk for the validation object. Should not include the file extension as `.pkl` will be added automatically.

`path: str | None = None`  
An optional directory path where the file should be saved. If not provided, the file will be saved in the current working directory. The directory will be created if it doesn't exist.

`keep_tbl: bool = ``False`  
An option to keep the data table that is associated with the validation object. The default is `False` where the data table is removed before writing to disk. For database tables (e.g., Ibis tables with database backends), the table is always removed even if `keep_tbl=True`, as database connections cannot be serialized.

`keep_extracts: bool = ``False`  
An option to keep any collected extract data for failing rows from validation steps. By default, this is `False` (i.e., extract data is removed to save space).

`quiet: bool = ``False`  
Should the function not inform when the file is written? By default, this is `False`, so a message will be printed when the file is successfully written.


## Returns


`None`  
This function doesn't return anything but saves the validation object to disk.


## Creating Serializable Validations

To ensure your validations work reliably across different Python sessions, the recommended approach is to use module-Level functions. So, create a separate Python file for your preprocessing functions:

``` python
# preprocessing_functions.py
import polars as pl

def multiply_by_100(df):
    return df.with_columns(pl.col("value") * 100)

def add_computed_column(df):
    return df.with_columns(computed=pl.col("value") * 2 + 10)
```

Then import and use them in your validation:

``` python
# your_main_script.py
import pointblank as pb
from preprocessing_functions import multiply_by_100, add_computed_column

validation = (
    pb.Validate(data=my_data)
    .col_vals_gt(columns="value", value=500, pre=multiply_by_100)
    .col_vals_between(columns="computed", left=50, right=1000, pre=add_computed_column)
    .interrogate()
)

# Save validation and it will work reliably across sessions
pb.write_file(validation, "my_validation", keep_tbl=True)
```


#### Problematic Patterns to Avoid

Don't use lambda functions as they will cause immediate errors.

``` python
validation = pb.Validate(data).col_vals_gt(
    columns="value", value=100,
    pre=lambda df: df.with_columns(pl.col("value") * 2)
)
```

Don't use interactive function definitions (as they may fail when loading).

``` python
def my_function(df):  # Defined in notebook/REPL
    return df.with_columns(pl.col("value") * 2)

validation = pb.Validate(data).col_vals_gt(
    columns="value", value=100, pre=my_function
)
```

------------------------------------------------------------------------


#### Automatic Analysis and Guidance

When you call [write_file()](write_file.md#pointblank.write_file), it automatically analyzes your validation and provides:

- confirmation when all functions will work reliably
- warnings for functions that may cause cross-session issues
- clear errors for unsupported patterns (lambda functions)
- specific recommendations and code examples
- loading instructions tailored to your validation

------------------------------------------------------------------------


#### Loading Your Validation

To load a saved validation in a new Python session:

``` python
# In a new Python session
import pointblank as pb

# Import the same preprocessing functions used when creating the validation
from preprocessing_functions import multiply_by_100, add_computed_column

# Upon loading the validation, functions will be automatically restored
validation = pb.read_file("my_validation.pkl")
```

\*\* Testing Your Validation:\*\*

To verify your validation works across sessions:

1.  save your validation in one Python session
2.  start a fresh Python session (restart kernel/interpreter)
3.  import required preprocessing functions
4.  load the validation using [read_file()](read_file.md#pointblank.read_file)
5.  test that preprocessing functions work as expected

------------------------------------------------------------------------


#### Performance and Storage

- use `keep_tbl=False` (default) to reduce file size when you don't need the original data
- use `keep_extracts=False` (default) to save space by excluding extract data
- set `quiet=True` to suppress guidance messages in automated scripts
- files are saved using pickle's highest protocol for optimal performance


## Examples

Let's create a simple validation and save it to disk:


``` python
import pointblank as pb

# Create a validation
validation = (
    pb.Validate(data=pb.load_dataset("small_table"), label="My validation")
    .col_vals_gt(columns="d", value=100)
    .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
    .interrogate()
)

# Save to disk (without the original table data)
pb.write_file(validation, "my_validation")
```


      Serialization Analysis:
       ✓ No preprocessing functions detected
       ✓ This validation should serialize and load reliably across sessions
    ✅ Validation object written to: my_validation.pkl
       📖 To load: validation = pb.read_file('my_validation.pkl')


To keep the original table data for later analysis:


``` python
# Save with the original table data included
pb.write_file(validation, "my_validation_with_data", keep_tbl=True)
```


      Serialization Analysis:
       ✓ No preprocessing functions detected
       ✓ This validation should serialize and load reliably across sessions
    ✅ Validation object written to: my_validation_with_data.pkl
       📖 To load: validation = pb.read_file('my_validation_with_data.pkl')


You can also specify a custom directory and keep extract data:

``` python
pb.write_file(
    validation,
    filename="detailed_validation",
    path="/path/to/validations",
    keep_tbl=True,
    keep_extracts=True
)
```


#### Working with Preprocessing Functions

For validations that use preprocessing functions to be portable across sessions, define your functions in a separate `.py` file:

``` python
# In `preprocessing_functions.py`

import polars as pl

def multiply_by_100(df):
    return df.with_columns(pl.col("value") * 100)

def add_computed_column(df):
    return df.with_columns(computed=pl.col("value") * 2 + 10)
```

Then import and use them in your validation:

``` python
# In your main script

import pointblank as pb
from preprocessing_functions import multiply_by_100, add_computed_column

validation = (
    pb.Validate(data=my_data)
    .col_vals_gt(columns="value", value=500, pre=multiply_by_100)
    .col_vals_between(columns="computed", left=50, right=1000, pre=add_computed_column)
    .interrogate()
)

# This validation can now be saved and loaded reliably
pb.write_file(validation, "my_validation", keep_tbl=True)
```

When you load this validation in a new session, simply import the preprocessing functions again and they will be automatically restored.


#### See Also

[Use](Use.md), [previously](previously.md)
