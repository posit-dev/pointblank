# Validate.to_json_schema()


Export this validation plan as a JSON Schema document.


Usage

``` python
Validate.to_json_schema(path=None)
```


The [to_json_schema()](Validate.to_json_schema.md#pointblank.Validate.to_json_schema) method walks the validation steps and maps each one to its closest JSON Schema equivalent. This lets you share your validation rules as a portable, language-neutral schema that tools across many ecosystems can consume.

When the [Validate](Validate.md#pointblank.Validate) object has data attached, the schema is enriched with column type information inferred from the data (e.g., `"type": "integer"` for Int64 columns).


## Parameters


`path: str | Path | None = None`  
An optional file path. If provided, the JSON Schema is written to this file (parent directories are created as needed) in addition to being returned.


## Returns


`dict[str, Any]`  
The JSON Schema document as a dictionary. This is a valid JSON Schema that can be serialized with `json.dumps()` or consumed by any JSON Schema validator.


## Supported Mappings

The following validation methods have direct JSON Schema equivalents:

| Pointblank method | JSON Schema keyword |
|----|----|
| [col_vals_not_null()](Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) | `required` |
| [col_vals_gt()](Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) | `exclusiveMinimum` |
| [col_vals_ge()](Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge) | `minimum` |
| [col_vals_lt()](Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt) | `exclusiveMaximum` |
| [col_vals_le()](Validate.col_vals_le.md#pointblank.Validate.col_vals_le) | `maximum` |
| [col_vals_eq()](Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq) | `const` |
| [col_vals_in_set()](Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set) | `enum` |
| [col_vals_between()](Validate.col_vals_between.md#pointblank.Validate.col_vals_between) | `minimum` + `maximum` |
| [col_vals_regex()](Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) | `pattern` |
| [col_vals_within_spec()](Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec) | `format` |

Steps with no JSON Schema equivalent (e.g., [rows_distinct()](Validate.rows_distinct.md#pointblank.Validate.rows_distinct), [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match), [col_vals_outside()](Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)) are silently skipped.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=df)
    .col_vals_gt(columns="age", value=0)
    .col_vals_not_null(columns="name")
    .col_vals_in_set(columns="status", set=["active", "inactive"])
)

schema = validation.to_json_schema()
# {'$schema': 'https://json-schema.org/draft/2020-12/schema',
#  'type': 'object',
#  'properties': {
#      'age': {'exclusiveMinimum': 0, 'type': 'integer'},
#      'name': {'type': 'string'},
#      'status': {'enum': ['active', 'inactive'], 'type': 'string'}
#  },
#  'required': ['name']}

# Write to a file
validation.to_json_schema("output.schema.json")
```
