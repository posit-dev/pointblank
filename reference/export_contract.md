# export_contract()

Export a Pointblank validation or contract to another format.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_api.py#L272-L439)

``` python
export_contract(
    validation_or_contract,
    destination=None,
    *,
    format,
    **kwargs,
)
```

Data quality rules written in Pointblank are often useful to tools that don't run Pointblank. An API team may want a JSON Schema, an analytics engineering team may want the tests in a dbt `schema.yml` file, and a data catalog may accept Open Data Contract Standard (ODCS) documents. `export_contract()` translates a [`Contract`](../reference/Contract.md#pointblank.Contract) or a [`Validate`](../reference/Validate.md#pointblank.Validate) object into one of these formats, so a single set of rules can be published wherever it's needed.

Contracts export most completely because they carry a name, a description, and a [`Schema`](../reference/Schema.md#pointblank.Schema) with column types. A [Validate](../reference/Validate.md#pointblank.Validate) object is exported from the steps in its plan; it doesn't need to be interrogated first. The translation is lossy by nature: only checks that have an equivalent in the target format are written, and the rest are left out without a warning (see the *What Gets Exported* section). To bring a definition back into Pointblank, use [`import_contract()`](../reference/import_contract.md#pointblank.import_contract).

## Parameters

`validation_or_contract: Any`  
The [Contract](../reference/Contract.md#pointblank.Contract) or [Validate](../reference/Validate.md#pointblank.Validate) object to export.

`destination: str | None = None`  
A file path to write the exported document to. Parent directories are created as needed. If `None` (the default), nothing is written and the document is only returned.

`format: str`  
The name of the adapter to use: `"json_schema"`, `"frictionless"`, `"dbt"`, or `"odcs"` for the built-in formats, or the name of a custom adapter. This argument is required and must be given by keyword.

`**kwargs: Any`  
Options for the adapter. None of the built-in adapters currently take any options.

## Returns

`str | dict`  
The exported document. The built-in adapters return it as a dictionary (whether or not it's also written to `destination=`), which can be serialized with `json.dumps()` or `yaml.dump()`.

## Raises

`ValueError`  
If no adapter is registered under `format=` (the message lists the available ones), or if the adapter doesn't support exporting.

`TypeError`  
If `validation_or_contract` is neither a [Contract](../reference/Contract.md#pointblank.Contract) nor a [Validate](../reference/Validate.md#pointblank.Validate) object.

## What Gets Exported

The column types come from the contract's schema, and each built-in adapter translates these checks:

- **JSON Schema** (`"json_schema"`): a draft 2020-12 object schema whose `title` and `description` come from the contract. `col_vals_not_null()` adds the column to `required`; `col_vals_ge()`/`col_vals_le()` become `minimum`/`maximum`, `col_vals_gt()`/`col_vals_lt()` become `exclusiveMinimum`/`exclusiveMaximum`, and `col_vals_between()` sets both `minimum` and `maximum`. `col_vals_in_set()` becomes `enum`, `col_vals_regex()` becomes `pattern`, `col_vals_eq()` becomes `const`, and `col_vals_within_spec()` becomes a `format` for the `email`, `url`, `ipv4`, and `ipv6` specs.
- **Frictionless Table Schema** (`"frictionless"`): a `fields` list with field constraints. `col_vals_not_null()` becomes `required`, `rows_distinct()` becomes `unique`, `col_vals_ge()`/`col_vals_le()` become `minimum`/`maximum`, `col_vals_in_set()` becomes `enum`, and `col_vals_regex()` becomes `pattern`.
- **dbt** (`"dbt"`): a `version: 2` document with one model named after the contract. `col_vals_not_null()`, `rows_distinct()`, and `col_vals_in_set()` become the generic tests `not_null`, `unique`, and `accepted_values`.
- **Open Data Contract Standard** (`"odcs"`): a `DataContract` document with an `info` block and one dataset table. `col_vals_not_null()` sets `isNullable: false`, `rows_distinct()` sets `isUnique`, `col_vals_in_set()` becomes `enum`, `col_vals_regex()` becomes `pattern`, and `col_vals_ge()`/`col_vals_le()` become `minimum`/`maximum`.

Any other step, along with thresholds, actions, and owner information, isn't exported.

When `destination=` is given, the JSON Schema and Frictionless documents are written as JSON and the dbt document as YAML. ODCS documents are written as JSON if the path ends in `.json` and as YAML otherwise.

## Examples

Let's start with a [Contract](../reference/Contract.md#pointblank.Contract) for a table of orders. It has a schema with column types, a description, and a mix of checks.

``` python
import pointblank as pb
import json

orders_contract = pb.Contract(
    name="orders",
    description="One row per customer order.",
    schema=pb.Schema(order_id="Int64", status="String", amount="Float64"),
    steps=[
        pb.Step("col_vals_not_null", columns=["order_id", "status"]),
        pb.Step("rows_distinct", columns_subset="order_id"),
        pb.Step("col_vals_in_set", columns="status", set=["placed", "shipped", "returned"]),
        pb.Step("col_vals_between", columns="amount", left=0, right=10000),
    ],
)
```

Exporting it as a JSON Schema gives us a dictionary, which we can print with `json.dumps()`.

``` python
orders_json_schema = pb.export_contract(orders_contract, format="json_schema")

print(json.dumps(orders_json_schema, indent=2))
```

    {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "type": "object",
      "title": "orders",
      "description": "One row per customer order.",
      "properties": {
        "order_id": {
          "type": "integer"
        },
        "status": {
          "type": "string",
          "enum": [
            "placed",
            "shipped",
            "returned"
          ]
        },
        "amount": {
          "type": "number",
          "minimum": 0,
          "maximum": 10000
        }
      },
      "required": [
        "order_id",
        "status"
      ]
    }

The Pointblank types became JSON types, the not-null check filled the `required` list, the set of statuses became an `enum`, and the range check became `minimum` and `maximum` values. The `rows_distinct()` step was left out because JSON Schema has no way to say that a value must be unique across records.

The same contract can be written out as a dbt `schema.yml` file. Here we write it to a `models` directory (which is created if it doesn't exist) and print the file's contents.

``` python
from pathlib import Path

pb.export_contract(orders_contract, "models/schema.yml", format="dbt")

print(Path("models/schema.yml").read_text())
```

    version: 2
    models:
    - name: orders
      description: One row per customer order.
      columns:
      - name: order_id
        data_type: integer
        data_tests:
        - not_null
        - unique
      - name: status
        data_type: string
        data_tests:
        - not_null
        - accepted_values:
            values:
            - placed
            - shipped
            - returned
      - name: amount
        data_type: float
        data_tests: []

This time the uniqueness check is kept (as dbt's `unique` test) but the range check is not, since dbt has no generic test for it. The `amount` column is still listed, with its type and an empty list of tests.

An exported document can be imported again with `import_contract()`. Doing so is a quick way to see which rules survive the round trip.

``` python
print(pb.import_contract(orders_json_schema).summary())
```

    Contract Import Summary
      Format: json_schema
      Format version: https://json-schema.org/draft/2020-12/schema
      Columns detected: 3
      Constraints mapped: 5
      Coverage: 100%

All five constraints in the JSON Schema were mapped back, so the coverage is 100%. The range check returns as two steps (one each for `minimum` and `maximum`), and the uniqueness check is gone because it was dropped on export. Coverage measures how much of the *document* was understood, not how much of the original contract survived. To find what an export loses, compare the steps before and after.

## See Also

[import_contract()](../reference/import_contract.md#pointblank.import_contract)  
Import an external definition into Pointblank.

[list_adapters()](../reference/list_adapters.md#pointblank.list_adapters)  
List the formats available for import and export.

[register_adapter()](../reference/register_adapter.md#pointblank.register_adapter)  
Add support for another format.

[Contract](../reference/Contract.md#pointblank.Contract)  
The declarative contract that exports most completely.
