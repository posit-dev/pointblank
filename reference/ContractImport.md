# ContractImport

The result of importing a contract or schema from another format.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L103-L723)

``` python
ContractImport(
    source_format,
    source_path=None,
    source_version=None,
    columns=list(),
    constraints=list(),
    metadata=dict(),
    warnings=list(),
    coverage=1.0,
)
```

[`import_contract()`](../reference/import_contract.md#pointblank.import_contract) returns a [ContractImport](../reference/ContractImport.md#pointblank.ContractImport). It's the halfway point between an external definition (a JSON Schema, a Frictionless Table Schema, a dbt `schema.yml` file, and so on) and a Pointblank validation. The definition has been read and translated, but nothing has been run yet, so this is the place to review the translation and decide how to use it.

The object holds what the adapter found: the columns and their types, the rules translated into [`MappedConstraint`](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint) objects, the source's metadata, and a warning for each rule that couldn't be translated. Its methods turn this into something usable:

- [`to_validate()`](../reference/ContractImport.md#pointblank.ContractImport.to_validate) builds a [`Validate`](../reference/Validate.md#pointblank.Validate) object for a table, ready to interrogate or to extend with more steps
- [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract) builds a [`Contract`](../reference/Contract.md#pointblank.Contract), which can be used in a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline) or saved as YAML
- [`to_python()`](../reference/ContractImport.md#pointblank.ContractImport.to_python) writes equivalent Python code, for when the imported rules should become a starting point that's edited by hand
- [`to_yaml()`](../reference/ContractImport.md#pointblank.ContractImport.to_yaml) lists the steps as YAML
- [`summary()`](../reference/ContractImport.md#pointblank.ContractImport.summary) describes what was imported and what wasn't

[ContractImport](../reference/ContractImport.md#pointblank.ContractImport) is a dataclass, so its fields can be read and changed directly. For instance, an unwanted constraint can be removed from `constraints` before calling `to_validate()`. Custom adapters (see [`ContractAdapter`](../reference/ContractAdapter.md#pointblank.ContractAdapter)) create these objects by calling the constructor with the parameters below.

## Parameters

`source_format: str`  
The name of the adapter that produced the import, such as `"json_schema"` or `"dbt"`.

`source_path: str | None = None`  
The path of the imported file, or `None` if the definition was given as a dictionary.

`source_version: str | None = None`  
The version of the source format, when the document states one (e.g., the `$schema` URL of a JSON Schema). Otherwise `None` or an empty string.

`columns: list[tuple[str, str | None]] = list()`  
The columns found in the source, as a list of `(name, dtype)` tuples. The `dtype` is a Pointblank type name such as `"Int64"` or `"String"`, or `None` when the source doesn't give a type that maps to one.

`constraints: list[MappedConstraint] = list()`  
The translated rules, as a list of [MappedConstraint](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint) objects, in the order in which they become validation steps.

`metadata: dict[str, Any] = dict()`  
Other information from the source, such as a `"title"` and a `"description"`.

`warnings: list[str] = list()`  
A message for each rule that was recognized but couldn't be translated.

`coverage: float = 1.0`  
The share of recognized rules that were translated, from `0.0` to `1.0`.

## Examples

Let's import a Frictionless Data Package that describes two tables. The `resource=` option selects the `readings` table, which has a primary key and a foreign key to the `stations` table.

``` python
import pointblank as pb
import polars as pl

package = {
    "name": "weather",
    "resources": [
        {
            "name": "stations",
            "schema": {"fields": [{"name": "station_id", "type": "string"}]},
        },
        {
            "name": "readings",
            "schema": {
                "fields": [
                    {"name": "station_id", "type": "string"},
                    {"name": "reading_no", "type": "integer"},
                    {
                        "name": "temp_c",
                        "type": "number",
                        "constraints": {"minimum": -90, "maximum": 60},
                    },
                ],
                "primaryKey": ["station_id", "reading_no"],
                "foreignKeys": [
                    {
                        "fields": "station_id",
                        "reference": {"resource": "stations", "fields": "station_id"},
                    }
                ],
            },
        },
    ],
}

readings = pb.import_contract(package, resource="readings")

readings
```

    ContractImport(format='frictionless', columns=3, constraints=5, coverage=83%)

The fields hold the details. The columns come with their Frictionless types translated to Pointblank types:

``` python
readings.columns
```

    [('station_id', 'String'), ('reading_no', 'Int64'), ('temp_c', 'Float64')]

The constraints show how the temperature range became a pair of bounds, and how the primary key was expressed as not-null checks plus a `rows_distinct()` check over both key columns.

``` python
for c in readings.constraints:
    print(c)
```

    MappedConstraint('col_vals_ge', columns='temp_c', value=-90)
    MappedConstraint('col_vals_le', columns='temp_c', value=60)
    MappedConstraint('col_vals_not_null', columns='station_id')
    MappedConstraint('col_vals_not_null', columns='reading_no')
    MappedConstraint('rows_distinct', columns_subset=['station_id', 'reading_no'])

The foreign key couldn't be translated, since checking it would need the `stations` table. It shows up in `warnings`, and it's why the coverage is below 100%.

``` python
readings.warnings
```

    ['Foreign key constraint skipped (cross-table validation not supported): station_id → stations.station_id']

Because the fields can be changed, the import can be adjusted before it's used. Here the temperature range is dropped (perhaps the sensors are known to report out-of-range values that are dealt with later) and the remaining checks are run against a table.

``` python
readings.constraints = [c for c in readings.constraints if c.kwargs.get("columns") != "temp_c"]

readings_tbl = pl.DataFrame(
    {
        "station_id": ["S1", "S1", "S2", "S2"],
        "reading_no": [1, 2, 1, 1],
        "temp_c": [12.5, 13.1, 75.0, 8.4],
    }
)

readings.to_validate(readings_tbl, tbl_name="readings").interrogate()
```

[TABLE]

The duplicated `("S2", 1)` key is caught by the `rows_distinct()` step, while the `75.0` reading no longer fails any step because the range check was removed.

## See Also

[import_contract()](../reference/import_contract.md#pointblank.import_contract)  
Create a [ContractImport](../reference/ContractImport.md#pointblank.ContractImport) from an external definition.

[MappedConstraint](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint)  
The objects in the `constraints` list.

[ContractAdapter](../reference/ContractAdapter.md#pointblank.ContractAdapter)  
The base class for adapters, which create these objects.

[Contract](../reference/Contract.md#pointblank.Contract)  
The Pointblank contract that [to_contract()](../reference/ContractImport.md#pointblank.ContractImport.to_contract) produces.

## Methods

| Name | Description |
|----|----|
| [to_validate()](#to_validate) | Build a validation plan for a table from the imported rules. |
| [to_contract()](#to_contract) | Turn the imported rules into a reusable contract. |
| [to_python()](#to_python) | Write the imported rules as Pointblank Python code. |
| [to_yaml()](#to_yaml) | List the imported validation steps as YAML. |
| [summary()](#summary) | Describe what was imported and what was left out. |

### to_validate()

Build a validation plan for a table from the imported rules.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L271-L389)

``` python
to_validate(
    data,
    **kwargs,
)
```

This is the most direct way to put an imported definition to work: give it a table and get back a [`Validate`](../reference/Validate.md#pointblank.Validate) object that checks the table against the definition. The object isn't interrogated yet, so further validation steps can be chained on before calling [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). That makes it easy to treat the external definition as a baseline and add checks that it can't express.

The plan starts with a [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) step built from the columns that have a type, followed by one step for each entry in `constraints`. The schema check uses the default settings of `col_schema_match()`, so the table must have the same columns, in the same order, with the same types. Types are compared exactly: a JSON Schema `integer` becomes `Int64`, so an `Int32` column fails the check. If no column has a type, the schema step is left out.

#### Parameters

`data: Any`  
The table to validate. Any table type supported by [Validate](../reference/Validate.md#pointblank.Validate) can be used.

`**kwargs: Any`  
Arguments for the [Validate](../reference/Validate.md#pointblank.Validate) constructor, such as `tbl_name=`, `label=`, `thresholds=`, and `actions=`.

#### Returns

`Validate`  
A [Validate](../reference/Validate.md#pointblank.Validate) object with the imported validation steps, not yet interrogated.

#### Examples

Here's a Frictionless Table Schema for a table of products. Its constraints say that each `sku` is required and unique, that prices can't be negative, and that the category must be one of three values.

``` python
import pointblank as pb
import polars as pl

product_schema = {
    "fields": [
        {
            "name": "sku",
            "type": "string",
            "constraints": {"required": True, "unique": True},
        },
        {"name": "price", "type": "number", "constraints": {"minimum": 0}},
        {
            "name": "category",
            "type": "string",
            "constraints": {"enum": ["books", "games", "music"]},
        },
    ]
}

products = pl.DataFrame(
    {
        "sku": ["B-001", "G-002", "G-002", "M-004", "B-005"],
        "price": [12.99, 59.99, 59.99, -1.0, 8.5],
        "category": ["books", "games", "games", "music", "film"],
    }
)
```

The arguments after the table go to the [Validate](../reference/Validate.md#pointblank.Validate) constructor. Here we name the table and set a threshold so that any failing test unit counts as a warning.

``` python
imported = pb.import_contract(product_schema)

imported.to_validate(products, tbl_name="products", thresholds=(1,)).interrogate()
```

[TABLE]

The report shows the schema check followed by the four imported steps. The duplicated `G-002` row, the negative price, and the `"film"` category each cause a failure.

Since the [Validate](../reference/Validate.md#pointblank.Validate) object isn't interrogated yet, we can add a rule that the Table Schema doesn't have (here, that prices stay below `100`) before running everything together.

``` python
(
    imported.to_validate(products, tbl_name="products")
    .col_vals_lt(columns="price", value=100)
    .interrogate()
)
```

[TABLE]

The added rule appears as the last step, and every price passes it. The imported steps fail exactly as before. Mixing imported and hand-written steps like this lets the external definition stay the source of truth for its own rules, while Pointblank covers what the format can't express.

### to_contract()

Turn the imported rules into a reusable contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L391-L498)

``` python
to_contract(
    name="imported_contract",
    **kwargs,
)
```

Where [`to_validate()`](../reference/ContractImport.md#pointblank.ContractImport.to_validate) checks one table right away, [to_contract()](../reference/ContractImport.md#pointblank.ContractImport.to_contract) produces a [`Contract`](../reference/Contract.md#pointblank.Contract): a declarative specification that isn't tied to any table. A contract can be validated against many tables over time, used at either end of a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline), and saved with [`to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml) so that the translated rules can be reviewed and versioned like any other Pointblank contract.

The contract's schema is a [`Schema`](../reference/Schema.md#pointblank.Schema) built from the columns that have a type (no schema is set if none do), and its steps are [`Step`](../reference/Step.md#pointblank.Step) objects made from the constraints. If the source had a description, it becomes the contract's description unless `description=` is given. The source's title isn't used, so pass it as `name=` if it should be kept.

#### Parameters

`name: str = "imported_contract"`  
The name of the contract. The default is `"imported_contract"`.

`**kwargs: Any`  
Other arguments for the [Contract](../reference/Contract.md#pointblank.Contract) constructor, such as `version=`, `owner=`, `description=`, `direction=`, `on_violation=`, and `thresholds=`.

#### Returns

`Contract`  
A contract with the imported schema and steps.

#### Examples

Let's import a JSON Schema that has a title and a description, then turn it into a contract. We use the schema's title as the contract name and add some metadata that the JSON Schema doesn't have.

``` python
import pointblank as pb
import polars as pl

imported = pb.import_contract(
    {
        "title": "sensor_reading",
        "description": "One reading from a temperature sensor.",
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string", "pattern": "^T-[0-9]{3}$"},
            "temp_c": {"type": "number", "minimum": -40, "maximum": 85},
        },
        "required": ["sensor_id", "temp_c"],
    }
)

sensor_contract = imported.to_contract(
    name=imported.metadata["title"],
    version="1.0.0",
    owner="iot-team",
)

sensor_contract
```

    Contract(name='sensor_reading', direction='source', version='1.0.0', schema=<defined>, steps=5)

The contract can now validate tables with its [`validate()`](../reference/Contract.md#pointblank.Contract.validate) method.

``` python
readings = pl.DataFrame(
    {
        "sensor_id": ["T-001", "T-002", "T-03", None],
        "temp_c": [21.5, 19.0, 120.4, 22.1],
    }
)

sensor_contract.validate(readings)
```

[TABLE]

The short sensor ID `"T-03"`, the missing sensor ID, and the reading of `120.4` (above the maximum of `85`) are caught by the steps that came from the JSON Schema.

### to_python()

Write the imported rules as Pointblank Python code.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L500-L580)

``` python
to_python()
```

An import doesn't have to stay an import. Sometimes the external definition is just a starting point, and the plan is to maintain the checks in Python from then on: tightening some rules, adding others, and setting thresholds and actions. [to_python()](../reference/ContractImport.md#pointblank.ContractImport.to_python) writes the code for a validation plan equivalent to what [`to_validate()`](../reference/ContractImport.md#pointblank.ContractImport.to_validate) builds, so it can be pasted into a script or notebook and edited from there.

The code chains a [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) step (when any column has a type) and one validation method per constraint onto [`Validate`](../reference/Validate.md#pointblank.Validate), and ends by calling [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). The table is referred to as `data`, so either define a variable with that name or replace it with the table to validate.

#### Returns

`str`  
The Python code, as a string.

#### Examples

Let's import a small dbt model definition and look at the code for it.

``` python
import pointblank as pb

dbt_schema = {
    "version": 2,
    "models": [
        {
            "name": "customers",
            "columns": [
                {
                    "name": "customer_id",
                    "data_type": "integer",
                    "data_tests": ["not_null", "unique"],
                },
                {
                    "name": "tier",
                    "data_type": "text",
                    "data_tests": [
                        {"accepted_values": {"values": ["bronze", "silver", "gold"]}}
                    ],
                },
            ],
        }
    ],
}

print(pb.import_contract(dbt_schema).to_python())
```

    import pointblank as pb

    validation = (
        pb.Validate(data=data)
        .col_schema_match(schema=pb.Schema(customer_id="Int64", tier="String"))
        .col_vals_not_null(columns='customer_id')
        .rows_distinct(columns_subset='customer_id')
        .col_vals_in_set(columns='tier', set=['bronze', 'silver', 'gold'])
    )

    validation.interrogate()

The dbt types became Pointblank types in the schema check, and each dbt test became a validation method call. The code runs as it is once `data` refers to a table.

### to_yaml()

List the imported validation steps as YAML.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L582-L646)

``` python
to_yaml()
```

YAML is easier to scan than Python for people who don't write code, so this method is handy for sharing what an import produced, such as in a code review or a ticket. The output has a `validation:` key holding a `steps:` list. That list starts with a `col_schema_match` step (when any column has a type) and then has one entry per constraint, each named after its validation method and holding that method's arguments.

This is a listing of the steps rather than a complete YAML validation configuration: it has no `tbl:` entry and its steps are nested under `validation:`, so it can't be passed to [`yaml_interrogate()`](../reference/yaml_interrogate.md#pointblank.yaml_interrogate) as is. To keep the imported rules in a YAML file that Pointblank can read back, convert them to a contract with [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract) and write that with [`Contract.to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml).

#### Returns

`str`  
The YAML text.

#### Examples

Here's the YAML listing for a small Frictionless Table Schema.

``` python
import pointblank as pb

imported = pb.import_contract(
    {
        "fields": [
            {"name": "id", "type": "integer", "constraints": {"required": True}},
            {"name": "score", "type": "number", "constraints": {"maximum": 100}},
        ]
    }
)

print(imported.to_yaml())
```

    validation:
      steps:
      - col_schema_match:
          schema:
            id: Int64
            score: Float64
      - col_vals_not_null:
          columns: id
      - col_vals_le:
          columns: score
          value: 100

The column types appear under the schema step, and each constraint has become a step with its arguments.

### summary()

Describe what was imported and what was left out.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L648-L715)

``` python
summary()
```

Translating a definition from another format is rarely perfect, so it's worth checking the result before relying on it. `summary()` gives that check in a few lines: the format, the file (if the definition came from one), the version of the format (if the document states one), the number of columns and constraints found, the coverage, and every warning about a rule that couldn't be translated. The warnings name the column and the rule, which makes it clear what still needs to be checked some other way.

#### Returns

`str`  
The summary text. Use `print()` to display it.

#### Examples

This JSON Schema uses the `date` format for one of its properties, which has no Pointblank equivalent among the format checks.

``` python
import pointblank as pb

imported = pb.import_contract(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "signup_date": {"type": "string", "format": "date"},
        },
        "required": ["email"],
    }
)

print(imported.summary())
```

    Contract Import Summary
      Format: json_schema
      Format version: https://json-schema.org/draft/2020-12/schema
      Columns detected: 2
      Constraints mapped: 2
      Coverage: 67%
      Warnings (1):
        - Column 'signup_date': JSON Schema format 'date' has no Pointblank equivalent -- skipped.

Two of the three rules were translated. The warning explains that the `date` format of `signup_date` was skipped, so a check on that column's dates would have to be added by hand (e.g., with [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex)).
