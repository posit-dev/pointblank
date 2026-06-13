## export_contract()


Export a Pointblank validation or contract to an external format.


Usage

``` python
export_contract(
    validation_or_contract,
    destination=None,
    *,
    format,
    **kwargs,
)
```


## Parameters


`validation_or_contract: Any`  
A [Validate](Validate.md#pointblank.Validate) or [Contract](Contract.md#pointblank.Contract) object to export.

`destination: str | None = None`  
Optional file path to write the output. If None, the result is returned without writing to disk.

`format: str`  
The target format identifier (e.g., `"json_schema"`, `"frictionless"`, `"dbt"`, etc.).

`**kwargs: Any`  
Format-specific options passed to the adapter.


## Returns


`str | dict`  
The exported content.


## Raises


`ValueError`  
If no adapter is registered for the format, or the adapter doesn't support export.


## Examples

``` python
import pointblank as pb

validation = pb.Validate(data=df).col_vals_gt(columns="age", value=0).interrogate()
pb.export_contract(validation, "output.schema.json", format="json_schema")
```
