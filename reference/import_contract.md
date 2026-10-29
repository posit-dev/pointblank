# import_contract()

Import a contract or schema written in another format.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_api.py#L9-L269)

``` python
import_contract(
    source,
    *,
    format=None,
    **kwargs,
)
```

Data is often already described somewhere outside of Pointblank: a JSON Schema for an API payload, a Frictionless Table Schema inside a Data Package, the `schema.yml` file of a dbt project, or an Open Data Contract Standard (ODCS) document. `import_contract()` reads such a definition and translates its rules into Pointblank validation steps. That way the existing definition can be checked against real data without rewriting it by hand, and the rules stay in the place where they're already maintained.

The result is a [`ContractImport`](../reference/ContractImport.md#pointblank.ContractImport) object. It holds the column names and types, the mapped constraints, any metadata from the source (such as a title and a description), and a warning for every rule that has no Pointblank equivalent. From there you can:

- get a [`Validate`](../reference/Validate.md#pointblank.Validate) object, ready to interrogate, with [`to_validate()`](../reference/ContractImport.md#pointblank.ContractImport.to_validate)
- get a reusable [`Contract`](../reference/Contract.md#pointblank.Contract) (for a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline), or to save as YAML) with [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract)
- generate equivalent Python code with [`to_python()`](../reference/ContractImport.md#pointblank.ContractImport.to_python)
- review what was and wasn't translated with [`summary()`](../reference/ContractImport.md#pointblank.ContractImport.summary)

The reverse direction is handled by [`export_contract()`](../reference/export_contract.md#pointblank.export_contract). Support for further formats can be added with [`register_adapter()`](../reference/register_adapter.md#pointblank.register_adapter).

## Parameters

`source: Any`  
The definition to import. This can be a path to a file, given as a string, or a dictionary holding content that has already been loaded (e.g., with `json.load()` or `yaml.safe_load()`). `pathlib.Path` objects aren't accepted, so pass `str(path)` instead.

`format: str | None = None`  
The name of the adapter to use: `"json_schema"`, `"frictionless"`, `"dbt"`, or `"odcs"` for the built-in formats, or the name of a custom adapter. If `None` (the default), the format is detected from the file name or the content (see the *Format Detection* section).

`**kwargs: Any`  
Options for the adapter. The Frictionless adapter takes `resource=` (the name or position of the resource to use in a Data Package), the dbt adapter takes `model=` (the name of a model or source table), and the ODCS adapter takes `table=` (the name of a dataset table). Each defaults to the first entry in the document. The JSON Schema adapter takes no options. Options that an adapter doesn't recognize are ignored.

## Returns

`ContractImport`  
The imported columns, constraints, metadata, and warnings, along with methods for turning them into a [Validate](../reference/Validate.md#pointblank.Validate) object, a [Contract](../reference/Contract.md#pointblank.Contract), Python code, or a text summary.

## Raises

`ValueError`  
If the format can't be detected, if no adapter is registered under `format=`, or if the adapter doesn't support importing. Adapters also raise `ValueError` when the document lacks the content they need (e.g., a dbt file with no models) or when a `resource=`, `model=`, or `table=` value doesn't match anything in it.

`FileNotFoundError`  
If `source` is a path to a file that doesn't exist.

## Supported Formats

Each built-in adapter reads the column names and types and the following rules:

- **JSON Schema** (`"json_schema"`): the `properties` of an object schema become the columns. `required` maps to `col_vals_not_null()`, `minimum`/`maximum` to `col_vals_ge()`/ `col_vals_le()`, `exclusiveMinimum`/`exclusiveMaximum` to `col_vals_gt()`/`col_vals_lt()`, `enum` to `col_vals_in_set()`, `pattern` to `col_vals_regex()`, and `const` to `col_vals_eq()`. The `format` values `email`, `uri`, `uri-reference`, `ipv4`, and `ipv6` map to `col_vals_within_spec()`; other formats (such as `date`) produce a warning. The `title` and `description` keywords are kept as metadata and `$schema` is recorded as the source version.
- **Frictionless Table Schema** (`"frictionless"`): a standalone Table Schema (with `fields`) or a Data Package (with `resources`). The field constraints `required`, `unique`, `minimum`, `maximum`, `enum`, and `pattern` are mapped, and a `primaryKey` becomes `col_vals_not_null()` on each key column plus `rows_distinct()` over all of them. Foreign keys produce a warning.
- **dbt** (`"dbt"`): the models and source tables of a `schema.yml` file. The column tests `not_null`, `unique`, and `accepted_values` are mapped (from either `data_tests:` or the older `tests:` key), as is a model-level `unique` test with `combination_of_columns`. `relationships` and any other tests produce a warning. The model name becomes the title.
- **Open Data Contract Standard** (`"odcs"`): documents with a `dataset:` list of tables and their columns. `isNullable: false`, `isUnique`, `isPrimaryKey`, `enum` (or [values](../reference/MetadataPackage.md#pointblank.MetadataPackage.values)), `pattern`, and `minimum`/`maximum` are mapped. Length limits and `checks`/`quality` entries produce a warning.

## Format Detection

When `format=` isn't given, a file path is first matched against the file extensions of the registered adapters: `.schema.json` (JSON Schema), `.resource.json` and `.datapackage.json` (Frictionless), `.yml` and `.yaml` (dbt), and `.odcs.yml`, `.odcs.yaml`, and `.odcs.json` (ODCS). If no extension matches, each adapter in turn inspects the content of the file or dictionary (e.g., a `fields` list identifies a Table Schema and a `models` list a dbt file). Since the dbt adapter claims every `.yml` and `.yaml` file, pass `format="odcs"` when importing an ODCS document stored as YAML. Use [`list_adapters()`](../reference/list_adapters.md#pointblank.list_adapters) to see the registered formats and their extensions.

## Coverage And Warnings

Not every rule in an external format can be expressed as a Pointblank validation step. When the adapter recognizes a rule but can't translate it, it adds a message to `warnings` and leaves the rule out. The `coverage` value is the share of recognized rules that were translated, so `1.0` means nothing recognized was dropped. Keywords that an adapter doesn't look at all (such as `minLength` in a JSON Schema) are skipped without a warning and don't count toward coverage, so it's worth checking [`summary()`](../reference/ContractImport.md#pointblank.ContractImport.summary) against the source when importing a definition for the first time.

## Examples

Suppose an API team publishes a JSON Schema for user profiles and we want to check a table of profiles against it. The schema can be passed as a dictionary (here it's written inline, but it would usually come from `json.load()`). Because it has a `$schema` key, the format is detected without needing `format=`.

``` python
import pointblank as pb
import polars as pl

user_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "user_profile",
    "description": "A registered user of the app.",
    "type": "object",
    "properties": {
        "user_id": {"type": "integer", "minimum": 1},
        "email": {"type": "string", "format": "email"},
        "country": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "plan": {"type": "string", "enum": ["free", "pro", "team"]},
    },
    "required": ["user_id", "email"],
}

imported = pb.import_contract(user_schema)

imported
```

    ContractImport(format='json_schema', columns=4, constraints=6, coverage=100%)

The printed form gives the counts at a glance: four columns, six constraints, and nothing left untranslated. The [`summary()`](../reference/ContractImport.md#pointblank.ContractImport.summary) method gives a little more detail, including the version of the source specification.

``` python
print(imported.summary())
```

    Contract Import Summary
      Format: json_schema
      Format version: https://json-schema.org/draft/2020-12/schema
      Columns detected: 4
      Constraints mapped: 6
      Coverage: 100%

Now let's validate some data. The [`to_validate()`](../reference/ContractImport.md#pointblank.ContractImport.to_validate) method builds a [Validate](../reference/Validate.md#pointblank.Validate) object with a schema check followed by one step per constraint, and it accepts the usual [Validate](../reference/Validate.md#pointblank.Validate) arguments such as `tbl_name=` and `label=`.

``` python
users = pl.DataFrame(
    {
        "user_id": [1, 2, 3, 0],
        "email": ["ana@example.com", "ben@example", "cy@example.org", None],
        "country": ["US", "CA", "gb", "DE"],
        "plan": ["free", "pro", "enterprise", "team"],
    }
)

imported.to_validate(users, tbl_name="users", label="Checks from user_profile").interrogate()
```

[TABLE]

The table has the right columns and types, so the schema check passes. The remaining failures come straight from the JSON Schema rules: a `user_id` of `0` (below the minimum of `1`), a missing email (which fails both the `required` check and the `email` format check), an email without a valid domain, a lowercase country code, and a plan that isn't in the `enum`.

Files work the same way. In this next example a dbt `schema.yml` file with two models is written to the working directory. The `.yml` extension identifies it as a dbt file, and `model=` picks the model to import (without it, the first model would be used).

``` python
from pathlib import Path

dbt_yml = '''
version: 2
models:
  - name: customers
    columns:
      - name: customer_id
        data_tests: [not_null, unique]
  - name: orders
    description: One row per order.
    columns:
      - name: order_id
        data_type: integer
        data_tests: [not_null, unique]
      - name: customer_id
        data_type: integer
        data_tests:
          - not_null
          - relationships:
              to: ref('customers')
              field: customer_id
      - name: status
        data_type: varchar(20)
        data_tests:
          - accepted_values:
              values: [placed, shipped, returned]
'''

Path("schema.yml").write_text(dbt_yml)

orders_import = pb.import_contract("schema.yml", model="orders")

print(orders_import.summary())
```

    Contract Import Summary
      Format: dbt
      Source: schema.yml
      Columns detected: 3
      Constraints mapped: 4
      Coverage: 80%
      Warnings (1):
        - Column 'customer_id': relationship test (customer_id → ref('customers').customer_id) skipped (cross-table validation not supported).

Four of the five tests were translated. The `relationships` test checks values against another model, which a single-table validation can't do, so it was reported as a warning and the coverage is 80%. Note also that the dbt type `varchar(20)` was read as Pointblank's `String` type.

The import can also be turned into a [Contract](../reference/Contract.md#pointblank.Contract) with [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract), which accepts any [Contract](../reference/Contract.md#pointblank.Contract) argument. Printing the contract as YAML shows that the schema, the steps, and the model's description were all carried over.

``` python
orders_contract = orders_import.to_contract(name="orders", owner="analytics")

print(orders_contract.to_yaml())
```

    contract:
      name: orders
      owner: analytics
      description: One row per order.
      schema:
        order_id: Int64
        customer_id: Int64
        status: String
      steps:
      - col_vals_not_null:
          columns: order_id
      - rows_distinct:
          columns_subset: order_id
      - col_vals_not_null:
          columns: customer_id
      - col_vals_in_set:
          columns: status
          set:
          - placed
          - shipped
          - returned

The contract now has a name and an owner, and the dbt model's four translated tests are its steps. From here it works like any other contract: it can be saved as YAML next to the dbt project, enforced with [`Contract.validate()`](../reference/Contract.md#pointblank.Contract.validate), or used as a boundary in a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline).

## See Also

[export_contract()](../reference/export_contract.md#pointblank.export_contract)  
Export a [Validate](../reference/Validate.md#pointblank.Validate) object or [Contract](../reference/Contract.md#pointblank.Contract) to an external format.

[ContractImport](../reference/ContractImport.md#pointblank.ContractImport)  
The object returned by this function.

[list_adapters()](../reference/list_adapters.md#pointblank.list_adapters)  
List the formats available for import and export.

[register_adapter()](../reference/register_adapter.md#pointblank.register_adapter)  
Add support for another format.

[Contract](../reference/Contract.md#pointblank.Contract)  
Pointblank's own declarative contract.
