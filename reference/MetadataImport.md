# MetadataImport

The metadata for a single dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L585-L1107)

``` python
MetadataImport(
    source_format,
    source_path=None,
    source_version=None,
    dataset_name=None,
    dataset_label=None,
    dataset_description=None,
    creation_date=None,
    study_id=None,
    domain=None,
    variables=list(),
    codelists=dict(),
    missing_value_codes=dict(),
)
```

A [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) is what [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) returns for a file that describes one dataset. It holds the dataset-level information (name, label, study and domain), a [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) for each variable, the codelists that the variables use, and any declared missing-value codes, all in the same form whatever the format of the file.

The methods turn the metadata into things Pointblank can use: [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate) builds a validation plan from the declared types and constraints, [`to_schema()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_schema) gives a [`Schema`](../reference/Schema.md#pointblank.Schema) of the column types, and [`missing_specs()`](../reference/MetadataImport.md#pointblank.MetadataImport.missing_specs) builds [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) objects from the missing-value codes. Variables and codelists can be looked up with [`get_variable()`](../reference/MetadataImport.md#pointblank.MetadataImport.get_variable) and [`get_codelist()`](../reference/MetadataImport.md#pointblank.MetadataImport.get_codelist), and [`summary()`](../reference/MetadataImport.md#pointblank.MetadataImport.summary) describes the whole dataset in text.

A [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) can also be created directly, which is a way to write down the rules for a dataset that doesn't come with a metadata file and still use the methods above. Calling `len()` on it gives the number of variables.

## Parameters

`source_format: str`  
The format the metadata was read from, such as `"frictionless"`, `"cdisc_define"` or `"spss"`. Any string can be used for metadata created directly.

`source_path: str | None = None`  
The path of the file that was read. The default is `None`.

`source_version: str | None = None`  
The version of the format or standard, such as `"Define-XML 2.0"`. The default is `None`.

`dataset_name: str | None = None`  
The name of the dataset, such as `"DM"`. The default is `None`.

`dataset_label: str | None = None`  
A human-readable label for the dataset, such as `"Demographics"`. The default is `None`.

`dataset_description: str | None = None`  
A longer description of the dataset. The default is `None`.

`creation_date: str | None = None`  
When the metadata was created, as given in the file. The default is `None`.

`study_id: str | None = None`  
The study identifier, for clinical data. The default is `None`.

`domain: str | None = None`  
The CDISC domain, such as `"DM"` or `"AE"`. The default is `None`.

`variables: list[VariableMetadata] = list()`  
The metadata of each variable, as a list of [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) objects. The default is an empty list.

`codelists: dict[str, Codelist] = dict()`  
The codelists used by the variables, as a dictionary from codelist name to [`Codelist`](../reference/Codelist.md#pointblank.Codelist). The default is an empty dictionary.

`missing_value_codes: dict[str, list[MissingValueCode]] = dict()`  
The declared missing-value codes, as a dictionary from column name to a list of [`MissingValueCode`](../reference/MissingValueCode.md#pointblank.MissingValueCode) objects. The default is an empty dictionary.

## Examples

Let's import the bundled Frictionless Table Schema, which describes a table of sensor readings:

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("table_schema.json"))

print(len(meta))
print(meta.variable_names)
print(list(meta.codelists))
```

    6
    ['sensor_id', 'reading_time', 'temperature', 'pressure_hpa', 'battery_pct', 'status']
    ['status_enum']

The schema describes six variables, and the allowed values of `status` were read into a codelist named `"status_enum"`. Metadata can also be written down directly, here for a small survey table:

``` python
survey = pb.MetadataImport(
    source_format="custom",
    dataset_name="survey",
    dataset_label="Household survey",
    variables=[
        pb.VariableMetadata(name="respondent_id", dtype="String", required=True, unique=True),
        pb.VariableMetadata(name="age", dtype="Int64", required=True, min_val=18, max_val=99),
    ],
)

survey
```

    MetadataImport(source_format='custom', variables=2, codelists=0)

Either way, the result can be turned into a validation plan with [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate), which has an example of checking a table against its metadata.

## See Also

[import_metadata()](../reference/import_metadata.md#pointblank.import_metadata)  
Read a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) from a file.

[MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage)  
The metadata for several datasets.

[VariableMetadata](../reference/VariableMetadata.md#pointblank.VariableMetadata)  
The metadata of one variable.

## Attributes

| Name                              | Description                           |
|-----------------------------------|---------------------------------------|
| [variable_names](#variable_names) | The names of all variables, in order. |

### variable_names

The names of all variables, in order.

`variable_names: list[str]`

#### Examples

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))

meta.variable_names
```

    ['station_id',
     'timestamp',
     'temperature_c',
     'humidity_pct',
     'wind_speed_kmh',
     'precipitation_mm',
     'condition']

These are the column names that a table described by this metadata should have. This is a property, so it's used without parentheses.

## Methods

| Name | Description |
|----|----|
| [to_validate()](#to_validate) | Build a validation plan for a table from the metadata. |
| [to_schema()](#to_schema) | Get a [Schema](../reference/Schema.md#pointblank.Schema) of the variables' names and types. |
| [get_variable()](#get_variable) | Get the metadata of a variable by name. |
| [get_codelist()](#get_codelist) | Get a codelist by name. |
| [missing_specs()](#missing_specs) | Build a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) for each variable with missing values. |
| [summary()](#summary) | Get a text summary of the metadata. |

### to_validate()

Build a validation plan for a table from the metadata.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L740-L833)

``` python
to_validate(
    data,
    **kwargs,
)
```

This turns the declared types and constraints into validation steps, so that a table can be checked against its own documentation. The first step is a [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) with the schema from [`to_schema()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_schema). Then, for each variable in turn:

- [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) if it's `required`
- [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) on the column if it's `unique`
- [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between) if it has both `min_val` and `max_val`, or [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge) or [`col_vals_le()`](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le) if it has only one of them
- [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set) if it has `allowed_values`
- [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) if it has a `pattern`
- [`col_vals_not_in_set()`](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set) with its `missing_values`, if it's a string variable that declares some

The last step checks that the declared missing-value markers don't appear in the data, as when a file was read without converting markers like `"NA"` to nulls. The `min_length` and `max_length` of a variable aren't checked. None of the steps pass null values, so a null in an optional variable fails its range, set and pattern checks. To change any of the steps, get the plan as code with [`to_code()`](../reference/Validate.to_code.md#pointblank.Validate.to_code) and edit it.

#### Parameters

`data: Any`  
The table to validate.

`**kwargs: Any`  
Arguments for [`Validate`](../reference/Validate.md#pointblank.Validate), such as `tbl_name=`, `thresholds=` or `actions=`. A `label=` replaces the default label, which is `"Validation: <dataset name> (<format>)"`, or `"Validation from <format> metadata"` if the metadata has no dataset name.

#### Returns

`Validate`  
A [Validate](../reference/Validate.md#pointblank.Validate) object with the steps added. It hasn't been interrogated yet.

#### Examples

The bundled Frictionless Data Package describes a table of sales transactions. Let's check a few transactions against it:

``` python
import datetime
import polars as pl
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

transactions = pl.DataFrame(
    {
        "transaction_id": ["TXN-00001", "TXN-00002", "TXN-00003", "TXN-00004"],
        "customer_id": ["CUST-101", "CUST-102", "CUST-101", "CUST-103"],
        "amount": [25.50, 1200.00, 0.00, 89.99],
        "quantity": [1, 3, 2, 1],
        "category": ["food", "electronics", "home", "toys"],
        "sale_date": [datetime.date(2024, 5, d) for d in (1, 1, 2, 3)],
        "discount_pct": [0.0, 10.0, 0.0, 5.0],
        "email": ["a@example.com", "b@example.com", "c@example.com", "not-an-email"],
    }
)

validation = meta.to_validate(data=transactions, tbl_name="transactions").interrogate()

validation
```

[TABLE]

The metadata gave 17 steps. The table matches the declared schema, and three of the constraint checks find a failing row: the `amount` of `0.00` is below the minimum of `0.01`, `"toys"` isn't an allowed `category`, and `"not-an-email"` doesn't match the pattern for `email`. To see how the plan was built, or to start from it and change some steps, print it as code:

``` python
print(meta.to_validate(data=transactions).to_code())
```

    import pointblank as pb

    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
            label="Validation: transactions (frictionless)",
        )
        .col_schema_match(schema=pb.Schema(columns=[("transaction_id", "String"), ("customer_id", "String"), ("amount", "Float64"), ("quantity", "Int64"), ("category", "String"), ("sale_date", "Date"), ("discount_pct", "Float64"), ("email", "String")]))
        .col_vals_not_null(columns="transaction_id")
        .rows_distinct(columns_subset="transaction_id")
        .col_vals_not_in_set(columns="transaction_id", set=["NA", "N/A"])
        .col_vals_not_null(columns="customer_id")
        .col_vals_not_in_set(columns="customer_id", set=["NA", "N/A"])
        .col_vals_not_null(columns="amount")
        .col_vals_between(columns="amount", left=0.01, right=99999.99)
        .col_vals_not_null(columns="quantity")
        .col_vals_between(columns="quantity", left=1.0, right=1000.0)
        .col_vals_not_null(columns="category")
        .col_vals_in_set(columns="category", set=["electronics", "clothing", "food", "home", "sports"])
        .col_vals_not_in_set(columns="category", set=["NA", "N/A"])
        .col_vals_not_null(columns="sale_date")
        .col_vals_between(columns="discount_pct", left=0.0, right=50.0)
        .col_vals_regex(columns="email", pattern="^[^@]+@[^@]+\\.[^@]+$")
        .col_vals_not_in_set(columns="email", set=["NA", "N/A"])
    )

    validation

Each step can be traced back to the metadata. For instance, the `rows_distinct()` step for `transaction_id` comes from its `unique` constraint, and the `col_vals_not_in_set()` steps check the `"NA"` and `"N/A"` markers declared for the string columns.

### to_schema()

Get a [Schema](../reference/Schema.md#pointblank.Schema) of the variables' names and types.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L706-L738)

``` python
to_schema()
```

The schema has a column for each variable, in order, with the variable's `dtype` (or `"String"` if the dtype isn't known). It can be used with [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) to check that a table has the expected columns and types. Only the names and types are included: constraints such as ranges and allowed values aren't part of a schema, and are checked by the steps that [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate) adds after its schema check.

#### Returns

`Schema`  
A [`Schema`](../reference/Schema.md#pointblank.Schema) with one column per variable.

#### Examples

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

meta.to_schema()
```

    Schema(columns=[('transaction_id', 'String'), ('customer_id', 'String'), ('amount', 'Float64'), ('quantity', 'Int64'), ('category', 'String'), ('sale_date', 'Date'), ('discount_pct', 'Float64'), ('email', 'String')])

The schema lists the eight columns of the transactions table with their types. The dtype names are those of Polars, so a schema check against a table from another backend fails on the type names (pandas, for example, has `object` and `int64` where Polars has `String` and `Int64`). Convert such a table to Polars before checking it.

### get_variable()

Get the metadata of a variable by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L835-L871)

``` python
get_variable(name)
```

#### Parameters

`name: str`  
The name of the variable. Names are matched exactly, including case.

#### Returns

`VariableMetadata`  
The [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) of the variable.

#### Raises

`KeyError`  
If there's no variable with that name.

#### Examples

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))
humidity = meta.get_variable("humidity_pct")

print(humidity.label, humidity.dtype, humidity.min_val, humidity.max_val)
```

    Relative Humidity (%) Float64 0.0 100.0

The CSVW file declares relative humidity as a number from `0` to `100`. Use [`variable_names`](../reference/MetadataImport.md#pointblank.MetadataImport.variable_names) to see which names are available.

### get_codelist()

Get a codelist by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L951-L988)

``` python
get_codelist(name)
```

The codelists are keyed by name in `codelists`, and a variable that uses one names it in its `codelist_ref`.

#### Parameters

`name: str`  
The name of the codelist, such as `"Sex"`.

#### Returns

`Codelist`  
The [`Codelist`](../reference/Codelist.md#pointblank.Codelist) with that name.

#### Raises

`KeyError`  
If there's no codelist with that name. The error message lists the available names.

#### Examples

``` python
import pointblank as pb

dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
race_ref = dm.get_variable("RACE").codelist_ref

dm.get_codelist(race_ref).to_set()
```

    ['WHITE',
     'BLACK OR AFRICAN AMERICAN',
     'ASIAN',
     'AMERICAN INDIAN OR ALASKA NATIVE',
     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']

Starting from the variable's `codelist_ref` finds the codelist that applies to `RACE`, and [`to_set()`](../reference/Codelist.md#pointblank.Codelist.to_set) gives its permitted values.

### missing_specs()

Build a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) for each variable with missing values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L873-L949)

``` python
missing_specs()
```

Each variable that declares missing values (such as the user-defined missing values of an SPSS variable, or the missing-value markers of a Frictionless schema) gets a spec made by its [`to_missing_spec()`](../reference/VariableMetadata.md#pointblank.VariableMetadata.to_missing_spec) method. The dictionary can be given to [`missing_vals_tbl()`](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) as its `missing=`, and a single spec from it can be given to the `missing=` parameter of a validation method such as [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex).

The specs keep the values as they were declared. Frictionless and CSVW markers are strings (like `"NA"`) and are declared for every field, but in a table read with typed columns they can only appear in the string columns. Keep only the specs of those columns before using them, since a string marker can't be compared with a numeric or date column.

#### Returns

`dict[str, MissingSpec]`  
A dictionary from column name to [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec). Variables that don't declare missing values aren't included.

#### Examples

The bundled Frictionless Data Package declares `"NA"` and `"N/A"` as missing-value markers, so every one of its variables gets a spec:

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))
specs = meta.missing_specs()

print(list(specs))
print(specs["customer_id"].reasons)
```

    ['transaction_id', 'customer_id', 'amount', 'quantity', 'category', 'sale_date', 'discount_pct', 'email']
    {'NA': 'missing_na', 'N/A': 'missing_n_a'}

The reasons were made from the markers, as no labels were declared for them. Let's keep the specs of the string columns and use one of them in a validation step:

``` python
import polars as pl

string_specs = {
    name: spec for name, spec in specs.items() if meta.get_variable(name).dtype == "String"
}

tbl = pl.DataFrame({"customer_id": ["CUST-101", "N/A", "CUST-103", "NA"]})

(
    pb.Validate(data=tbl)
    .col_vals_regex(columns="customer_id", pattern="^CUST-[0-9]+$")
    .col_vals_regex(
        columns="customer_id", pattern="^CUST-[0-9]+$", missing=string_specs["customer_id"]
    )
    .interrogate()
)
```

[TABLE]

Without the spec, the `"N/A"` and `"NA"` markers fail the pattern check in step 1. With it, step 2 treats them as missing data instead, so all four rows pass. The `string_specs` dictionary could be given to [missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) in the same way, to summarize the missing data in each string column of a table.

### summary()

Get a text summary of the metadata.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1009-L1094)

``` python
summary()
```

The summary gives the format, the source file, the dataset name, label and domain (when known), and the numbers of variables and codelists. It then lists each variable with its dtype, its label, and its constraints: `required`, `unique`, the range, the number of allowed values, and the codelist. Printing a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) with `print()` shows the same summary.

#### Returns

`str`  
The summary, as a string with one line per item.

#### Examples

``` python
import pointblank as pb

survey = pb.MetadataImport(
    source_format="custom",
    dataset_name="survey",
    dataset_label="Household survey",
    variables=[
        pb.VariableMetadata(
            name="respondent_id",
            label="Respondent ID",
            dtype="String",
            required=True,
            unique=True,
        ),
        pb.VariableMetadata(
            name="age", label="Age in years", dtype="Int64", min_val=18, max_val=99
        ),
        pb.VariableMetadata(
            name="region", dtype="String", allowed_values=["north", "south", "east", "west"]
        ),
    ],
)

print(survey.summary())
```

    Metadata Import (custom)
      Dataset: survey
      Label: Household survey
      Variables: 3
      Codelists: 0

      Variables:
        respondent_id (String) -- Respondent ID [required, unique]
        age (Int64) -- Age in years [range=[18, 99]]
        region (String) [4 allowed values]

The constraints in brackets are among those that [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate) turns into validation steps, so the summary is a quick way to see what a table will be checked for (a `pattern` isn't listed, though it's checked too). For metadata read from a file, the summary also has a `Source:` line with the path of the file.
