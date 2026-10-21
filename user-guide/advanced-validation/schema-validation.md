# Schema Validation

Schema validation in Pointblank allows you to verify that your data conforms to an expected structure and type specification. This is particularly useful when ensuring data consistency across systems or validating incoming data against predefined requirements.

Let's first look at the dataset we'll use for the first example:

``` python
import pointblank as pb

# Preview the small_table dataset we'll use throughout this guide
pb.preview(pb.load_dataset(dataset="small_table", tbl_type="polars"))
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

## Schema Definition and Validation

A schema in Pointblank is created using the [Schema](../../reference/Schema.md#pointblank.Schema) class which defines the expected structure of a table. Once created, you apply schema validation through the [`Validate.col_schema_match()`](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) validation step.

``` python
# Create a schema definition matching small_table structure
schema = pb.Schema(
    columns=[
        ("date_time",),   # Only check column name
        ("date",),        # Only check column name
        ("a", "Int64"),   # Check name and type
        ("b", "String"),  # Check name and type
        ("c", "Int64"),   # Check name and type
        ("d", "Float64"), # Check name and type
        ("e", "Boolean"), # Check name and type
        ("f",),           # Only check column name
    ]
)

# Validate the small_table against the schema
small_table_validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="polars"),
        label="Schema validation of `small_table`.",
    )
    .col_schema_match(schema=schema)
    .interrogate()
)

small_table_validation
```

[TABLE]

The output shows the validation passed successfully. When all columns have the correct names and types as specified in the schema, the validation passes with a single passing test unit. If there were discrepancies, this would fail, but the basic output wouldn't show specific issues.

For detailed information about validation results, use [`Validate.get_step_report()`](../../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report):

``` python
small_table_validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | date_time | Datetime(time_unit='us', time_zone=None) | 1 | date_time | ✓ | -- |  |
| 2 | date | Date | 2 | date | ✓ | -- |  |
| 3 | a | Int64 | 3 | a | ✓ | Int64 | ✓ |
| 4 | b | String | 4 | b | ✓ | String | ✓ |
| 5 | c | Int64 | 5 | c | ✓ | Int64 | ✓ |
| 6 | d | Float64 | 6 | d | ✓ | Float64 | ✓ |
| 7 | e | Boolean | 7 | e | ✓ | Boolean | ✓ |
| 8 | f | String | 8 | f | ✓ | -- |  |
|  Supplied Column Schema: `[('date_time',), ('date',), ('a', 'Int64'), ('b', 'String'), ('c', 'Int64'), ('d', 'Float64'), ('e', 'Boolean'), ('f',)]` |  |  |  |  |  |  |  |

The step report provides specific details about which columns were checked and whether they matched the schema, helping diagnose issues when validation fails.

## Schema Components and Column Types

When defining a schema, you need to specify column names and optionally their data types. By default, Pointblank enforces strict validation where:

- all columns in your table must match the specified schema
- column order must match the schema
- column types are case-sensitive
- type names must match exactly

The schema definition accepts column types as string representations, which vary depending on your data source:

- `string`: Character data (may also be `"String"`, `"varchar"`, `"character"`, etc.)
- `integer`: Integer values (may also be `"Int64"`, `"int"`, `"bigint"`, etc.)
- `numeric`: Numeric values including integers and floating-point numbers (may also be `"Float64"`, `"double"`, `"decimal"`, etc.)
- `boolean`: Logical values (`True`/`False`) (may also be `"Boolean"`, `"bool"`, etc.)
- `datetime`: Date and time values (may also be `"Datetime"`, `"timestamp"`, etc.)
- `date`: Date values (may also be `"Date"`, etc.)
- `time`: Time values

For specific database engines or DataFrame libraries, you may need to use their exact type names (like `"VARCHAR(255)"` for SQL databases or `"Int64"` for Polars integers).

## Discovering Column Types

To easily determine the correct type string for columns in your data, Pointblank provides two helpful functions:

``` python
import polars as pl
from datetime import date

# Define a sample dataframe
sample_df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "join_date": [date(2020, 1, 1), date(2021, 3, 15), date(2022, 7, 10)]
})
```

``` python
# Method 1: Using `preview()` with `show_types=True` to see column types
pb.preview(sample_df)
```

| PolarsRows3Columns3 |            |               |                  |
|---------------------|------------|---------------|------------------|
|                     | id *Int64* | name *String* | join_date *Date* |
| 1                   | 1          | Alice         | 2020-01-01       |
| 2                   | 2          | Bob           | 2021-03-15       |
| 3                   | 3          | Charlie       | 2022-07-10       |

``` python
# Method 2: Using `col_summary_tbl()` which shows column types and other details
pb.col_summary_tbl(sample_df)
```

| PolarsRows3Columns3 |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | Column | NA | UQ | Mean | SD | Min | P5 | Q1 | Med | Q3 | P95 | Max | IQR |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>id\</div\>\<div style='font-size: 11px; color: gray;'\>Int64\</div\> | 0\<br\>0 | 3\<br\>1 | 2 | 1 | 1 | 1.01 | 1.5 | 2 | 2.5 | 2.9 | 3 | 1 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>string\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="string" transform="translate(4.000000, 4.500000)"\> \<rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>name\</div\>\<div style='font-size: 11px; color: gray;'\>String\</div\> | 0\<br\>0 | 3\<br\>1 | 5 | 2 | 3 | 3.02 | 4 | 5 | 6 | 6.8 | 7 | 2 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>date\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="date" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>join_date\</div\>\<div style='font-size: 11px; color: gray;'\>Date\</div\> | 0\<br\>0 | 3\<br\>1 | \- | \- | 2020\<br\>01\<br\>01 | \- | \- | \- | \- | \- | 2022\<br\>07\<br\>10 | \- |
| String columns statistics regard the string's length. |  |  |  |  |  |  |  |  |  |  |  |  |  |

These functions help you identify the exact type strings to use in your schema definitions, eliminating guesswork and ensuring compatibility with your data source.

## Creating a Schema

You can create a schema in four different ways, each with its own advantages. All schema objects can be printed to display their column names and data types.

### 1. Using a List of Tuples with `columns=`

This approach allows for mixed validation: some columns checked for both name and type, others only for name:

``` python
schema_tuples = pb.Schema(

    # List of tuples approach: flexible for mixed type/name checking ---
    columns=[
        ("name", "String"), # Check name and type
        ("age", "Int64"),   # Check name and type
        ("height",)         # Check name only
    ]
)

print(schema_tuples)
```

    Pointblank Schema
      name: String
      age: Int64
      height: <ANY>

This is the only method that allows checking just column names for some columns while checking both names and types for others.

### 2. Using a Dictionary with `columns=`

This approach is often the most readable when defining a schema manually, especially for larger schemas:

``` python
schema_dict = pb.Schema(

    # Dictionary approach (keys are column names, values are data types) ---
    columns={
        "name": "String",
        "age": "Int64",
        "height": "Float64",
        "created_at": "Datetime"
    }
)

print(schema_dict)
```

    Pointblank Schema
      name: String
      age: Int64
      height: Float64
      created_at: Datetime

With this method, you must always provide both column names (as keys) and their types (as values).

### 3. Using Keyword Arguments

For more readable code with a small number of columns:

``` python
schema_kwargs = pb.Schema(

    # Keyword arguments approach (more readable for simple schemas) ---
    name="String",
    age="Int64",
    height="Float64"
)

print(schema_kwargs)
```

    Pointblank Schema
      name: String
      age: Int64
      height: Float64

Like the dictionary method, this approach requires both column names and types.

### 4. Extracting from an Existing Table with `tbl=`

You can automatically extract a schema from an existing table:

``` python
import polars as pl

# Create a sample dataframe
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "height": [5.6, 6.0, 5.8]
})

# Extract schema from table
schema_from_table = pb.Schema(tbl=df)

print(schema_from_table)
```

    Pointblank Schema
      name: String
      age: Int64
      height: Float64

This is especially useful when you want to validate that future data matches the structure of a reference dataset.

## Multiple Data Types for a Column

You can specify multiple acceptable types for a column by providing a list of types:

``` python
# Schema with multiple possible types for a column
schema_multi_types = pb.Schema(
    columns={
        "name": "String",
        "age": ["Int64", "Float64"],  # Accept either integer or float
        "active": "Boolean"
    }
)

print(schema_multi_types)
```

    Pointblank Schema
      name: String
      age: ['Int64', 'Float64']
      active: Boolean

This is useful when working with data sources that might represent the same information in different ways (e.g., integers sometimes stored as floats).

## Schema Validation Options

When using [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), you can customize validation behavior with several important options:

| Option | Default | Description |
|----|----|----|
| `complete` | `True` | Require exact column presence (no extra columns allowed) |
| `in_order` | `True` | Enforce column order |
| `case_sensitive_colnames` | `True` | Make column name matching case-sensitive |
| `case_sensitive_dtypes` | `True` | Make data type matching case-sensitive |
| `full_match_dtypes` | `True` | Require exact (not partial) type name matches |

### Controlling Column Presence

By default, [`Validate.col_schema_match()`](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) requires a complete match between the schema's columns and the table's columns. You can make this more flexible:

``` python
# Create a sample table
users_table_extra = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "extra_col": ["a", "b", "c"]  # Extra column not in schema
})

# Create a schema
schema = pb.Schema(
    columns={"id": "Int64", "name": "String", "age": "Int64"}
)

# Validate without requiring all columns to be present
validation = (
    pb.Validate(data=users_table_extra)
    .col_schema_match(
        schema=schema,

        # Allow schema columns to be a subset ---
        complete=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 1 | id | ✓ | Int64 | ✓ |
| 2 | name | String | 2 | name | ✓ | String | ✓ |
| 3 | age | Int64 | 3 | age | ✓ | Int64 | ✓ |
| 4 | extra_col | String |  |  |  |  |  |
|  Supplied Column Schema: `[('id', 'Int64'), ('name', 'String'), ('age', 'Int64')]` |  |  |  |  |  |  |  |

### Column Order Enforcement

You can control whether column order matters in your validation:

``` python
# Create a sample table
users_table = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
})

# Create a schema
schema = pb.Schema(
    columns={"name": "String", "age": "Int64", "id": "Int64"}
)

# Validate without enforcing column order
validation = (
    pb.Validate(data=users_table)
    .col_schema_match(
        schema=schema,

        # Don't enforce column order ---
        in_order=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 3 | id | ✓ | Int64 | ✓ |
| 2 | name | String | 1 | name | ✓ | String | ✓ |
| 3 | age | Int64 | 2 | age | ✓ | Int64 | ✓ |
|  Supplied Column Schema: `[('name', 'String'), ('age', 'Int64'), ('id', 'Int64')]` |  |  |  |  |  |  |  |

### Case Sensitivity

Control whether column names and data types are case-sensitive:

``` python
# Create schema with different case charactistics
case_schema = pb.Schema(
    columns={"ID": "int64", "NAME": "string", "AGE": "int64"}
)

# Create validation with case-insensitive column names and types
validation = (
    pb.Validate(data=users_table)
    .col_schema_match(
        schema=case_schema,

        # Ignore case in column names ---
        case_sensitive_colnames=False,

        # Ignore case in data type names ---
        case_sensitive_dtypes=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN = column DTYPE = dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 1 | ID | ✓ | int64 | ✓ |
| 2 | name | String | 2 | NAME | ✓ | string | ✓ |
| 3 | age | Int64 | 3 | AGE | ✓ | int64 | ✓ |
|  Supplied Column Schema: `[('ID', 'int64'), ('NAME', 'string'), ('AGE', 'int64')]` |  |  |  |  |  |  |  |

### Type Matching Precision

Control how strictly data types must match:

``` python
# Create schema with simplified type names
type_schema = pb.Schema(

    # Using simplified type names ---
    columns={"id": "int", "name": "str", "age": "int"}
)

# Allow partial type matches
validation = (
    pb.Validate(data=users_table)
    .col_schema_match(
        schema=type_schema,

        # Ignore case in data type names ---
        case_sensitive_dtypes=False,

        # Allow partial type name matches ---
        full_match_dtypes=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE = dtype float = float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 1 | id | ✓ | int | ✓ |
| 2 | name | String | 2 | name | ✓ | str | ✓ |
| 3 | age | Int64 | 3 | age | ✓ | int | ✓ |
|  Supplied Column Schema: `[('id', 'int'), ('name', 'str'), ('age', 'int')]` |  |  |  |  |  |  |  |

## Common Schema Validation Patterns

This section explores common patterns for applying schema validation to different scenarios. Each pattern addresses specific validation needs you might encounter when working with real-world data. We'll examine the step reports ([`Validate.get_step_report()`](../../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report)) for these validations since they provide more detailed information about what was checked and how the validation performed, offering an intuitive way to understand the results beyond simple pass/fail indicators.

### Structural Validation Only

When you only care about column names but not their types:

``` python
# Create a schema with only column names
structure_schema = pb.Schema(
    columns=["id", "name", "age", "extra_col"]
)

# Validate structure only
validation = (
    pb.Validate(data=users_table_extra)
    .col_schema_match(schema=structure_schema)
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 1 | id | ✓ | -- |  |
| 2 | name | String | 2 | name | ✓ | -- |  |
| 3 | age | Int64 | 3 | age | ✓ | -- |  |
| 4 | extra_col | String | 4 | extra_col | ✓ | -- |  |
|  Supplied Column Schema: `[('id',), ('name',), ('age',), ('extra_col',)]` |  |  |  |  |  |  |  |

### Mixed Validation

Validate types for critical columns but just presence for others:

``` python
# Mixed validation for different columns
mixed_schema = pb.Schema(
    columns=[
        ("id", "Int64"),     # Check name and type
        ("name", "String"),  # Check name and type
        ("age",),            # Check name only
        ("extra_col",)       # Check name only
    ]
)

# Validate with mixed approach
validation = (
    pb.Validate(data=users_table_extra)
    .col_schema_match(schema=mixed_schema)
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 1 | id | ✓ | Int64 | ✓ |
| 2 | name | String | 2 | name | ✓ | String | ✓ |
| 3 | age | Int64 | 3 | age | ✓ | -- |  |
| 4 | extra_col | String | 4 | extra_col | ✓ | -- |  |
|  Supplied Column Schema: `[('id', 'Int64'), ('name', 'String'), ('age',), ('extra_col',)]` |  |  |  |  |  |  |  |

### Progressive Schema Evolution

As your data evolves, you might need to adapt your validation approach:

``` python
# Original schema
original_schema = pb.Schema(
    columns={
        "id": "Int64",
        "name": "String"
    }
)

# New data with additional columns
evolved_data = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],           # New column
    "active": [True, False, True]  # New column
})

# Validate with flexible parameters
validation = (
    pb.Validate(evolved_data)
    .col_schema_match(
        schema=original_schema,

        # Allow extra columns ---
        complete=False,

        # Don't enforce order ---
        in_order=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✓ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TARGET |  |  | EXPECTED |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | id | Int64 | 1 | id | ✓ | Int64 | ✓ |
| 2 | name | String | 2 | name | ✓ | String | ✓ |
| 3 | age | Int64 |  |  |  |  |  |
| 4 | active | Boolean |  |  |  |  |  |
|  Supplied Column Schema: `[('id', 'Int64'), ('name', 'String')]` |  |  |  |  |  |  |  |

## Integrating with Larger Validation Workflows

Schema validation is often just one part of a comprehensive data validation strategy. You can combine schema checks with other validation steps:

``` python
# Define a schema
schema = pb.Schema(
    columns={
        "id": "Int64",
        "name": "String",
        "age": "Int64"
    }
)

# Create a validation plan
validation = (
    pb.Validate(
        users_table,
        label="User data validation",
        thresholds=pb.Thresholds(warning=0.05, error=0.1)
    )

    # Add schema validation ---
    .col_schema_match(schema=schema)

    # Add other validation steps ---
    .col_vals_not_null(columns="id")
    .col_vals_gt(columns="age", value=26)
    .interrogate()
)

validation
```

[TABLE]

This approach allows you to first validate the structure of your data and then check specific business rules or constraints.

## Best Practices

1.  Define schemas early: document and define expected data structures early in your data workflow.

2.  Choose the right creation method:

    - use `columns=<dict>` for readability with many columns
    - use `columns=<list of tuples>` for mixed name/type validation
    - use `kwargs` for small schemas with simple column names
    - use `tbl=` to extract schemas from reference datasets

3.  Be deliberate about strictness: choose validation parameters based on your specific needs:

    - strict validation (`complete=True`) for critical data interfaces
    - flexible validation (`complete=False`, `in_order=False`) for evolving datasets

4.  Reuse schemas: create schema definitions that can be reused across multiple validation contexts.

5.  Version control schemas: as your data evolves, maintain versions of your schemas to track changes.

6.  Extract schemas from reference data: when you have a 'golden' dataset that represents your ideal structure, use `Schema(tbl=reference_data)` to extract its schema.

7.  Consider type flexibility: use multiple types per column (`["Int64", "Float64"]`) when working with data from diverse sources.

8.  Combine with targeted validation: use schema validation for structural checks and add specific validation steps for business rules.

## Conclusion

Schema validation provides a powerful mechanism for ensuring your data adheres to expected structural requirements. It serves as an excellent first line of defense in your data validation strategy, verifying that the data you're working with has the expected shape before applying more detailed business rule validations.

The [Schema](../../reference/Schema.md#pointblank.Schema) class offers multiple ways to define schemas, from manual specification with dictionaries or keyword arguments to automatic extraction from reference tables. When combined with the flexible options of [`Validate.col_schema_match()`](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), you can implement validation approaches ranging from strict structural enforcement to more flexible evolution-friendly checks.

By understanding the different schema creation methods and validation options, you can efficiently validate the structure of your data tables and ensure they meet your requirements before processing.
