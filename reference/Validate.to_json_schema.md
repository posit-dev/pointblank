# Validate.to_json_schema()

Export this validation plan as a JSON Schema document.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L19902-L20009)

``` python
Validate.to_json_schema(path=None)
```

The [to_json_schema()](../reference/Validate.to_json_schema.md#pointblank.Validate.to_json_schema) method walks the validation steps and maps each one to its closest JSON Schema equivalent. This lets you share your validation rules as a portable, language-neutral schema that tools across many ecosystems can consume.

When the [Validate](../reference/Validate.md#pointblank.Validate) object has data attached, the schema is enriched with column type information inferred from the data (e.g., `"type": "integer"` for Int64 columns).

## Parameters

`path: str | Path | None = None`  
An optional file path. If provided, the JSON Schema is written to this file (parent directories are created as needed) in addition to being returned.

## Returns

`dict[str, Any]`  
The JSON Schema document as a dictionary. This is a valid JSON Schema that can be serialized with `json.dumps()` or consumed by any JSON Schema validator.

## Supported Mappings

The following validation methods have direct JSON Schema equivalents:

| Pointblank method        | JSON Schema keyword   |
|--------------------------|-----------------------|
| `col_vals_not_null()`    | `required`            |
| `col_vals_gt()`          | `exclusiveMinimum`    |
| `col_vals_ge()`          | `minimum`             |
| `col_vals_lt()`          | `exclusiveMaximum`    |
| `col_vals_le()`          | `maximum`             |
| `col_vals_eq()`          | `const`               |
| `col_vals_in_set()`      | `enum`                |
| `col_vals_between()`     | `minimum` + `maximum` |
| `col_vals_regex()`       | `pattern`             |
| `col_vals_within_spec()` | `format`              |

Steps with no JSON Schema equivalent (e.g., `rows_distinct()`, `tbl_match()`, `col_vals_outside()`) are silently skipped.

## Examples

Let's write a validation plan for a table of user accounts and export it as JSON Schema.

``` python
import pointblank as pb
import polars as pl
import json

accounts = pl.DataFrame(
    {
        "name": ["Ana", "Ben", "Chloé"],
        "age": [34, 27, 45],
        "status": ["active", "inactive", "active"],
    }
)

validation = (
    pb.Validate(data=accounts)
    .col_vals_gt(columns="age", value=0)
    .col_vals_not_null(columns="name")
    .col_vals_in_set(columns="status", set=["active", "inactive"])
    .rows_distinct()
)

schema = validation.to_json_schema()

print(json.dumps(schema, indent=2))
```

    {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "type": "object",
      "properties": {
        "age": {
          "exclusiveMinimum": 0,
          "type": "integer"
        },
        "name": {
          "type": "string"
        },
        "status": {
          "enum": [
            "active",
            "inactive"
          ],
          "type": "string"
        }
      },
      "required": [
        "name"
      ]
    }

Each column has a property: `age > 0` became `exclusiveMinimum`, the set of statuses became `enum`, and the not-null check on `name` put it in `required`. The `type` of each property comes from the data attached to the plan. The [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step has no JSON Schema equivalent (a schema describes one record at a time), so it isn't in the output. Since the result is a standard schema, it can be given to any JSON Schema validator, for example to check records in an API before they're loaded into the table. With `path=`, the schema is also written to a file.
