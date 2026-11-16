# from_dict()

Create a contract from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L1021-L1149)

``` python
from_dict(data)
```

This is the reverse of [`to_dict()`](../reference/pointblank.Contract.to_dict.md). Use it when the contract comes from somewhere other than a YAML file: a JSON document, an API response, a database record, or a dictionary built in code. To read a YAML file directly, use [`from_yaml()`](../reference/pointblank.Contract.from_yaml.md).

The dictionary uses the contract's parameter names as keys. Only `"name"` is required. `"schema"` should map column names to type strings, `"steps"` should be a list of single-key mappings from method name to arguments, and `"thresholds"` should map any of `"warning"`, `"error"`, and `"critical"` to a value. The parsed contract is checked as usual, so an invalid `direction`, `on_violation`, or step method raises an error.

## Parameters

`data: dict[str, Any]`  
A dictionary describing a contract, in the form produced by [`to_dict()`](../reference/pointblank.Contract.to_dict.md).

## Returns

`Contract`  
A new [Contract](../reference/Contract.md#pointblank.Contract) object.

## Raises

`ValueError`  
If `"name"` is missing or empty, if `"direction"` or `"on_violation"` has an invalid value, or if a step names a method that contracts don't support.

`TypeError`  
If `"steps"` isn't a list, or if a step's arguments aren't a dictionary.

## Examples

Suppose a contract definition arrives as JSON from a configuration service. After parsing, it's a dictionary that `from_dict()` can turn into a [Contract](../reference/Contract.md#pointblank.Contract).

``` python
import json
import pointblank as pb

contract_json = '''
{
  "name": "shipments",
  "version": "0.3.0",
  "owner": "logistics",
  "schema": {"shipment_id": "String", "weight_kg": "Float64"},
  "steps": [
    {"col_vals_not_null": {"columns": "shipment_id"}},
    {"col_vals_between": {"columns": "weight_kg", "left": 0, "right": 1000}}
  ],
  "thresholds": {"warning": 1, "error": 0.05}
}
'''

shipments_contract = pb.Contract.from_dict(json.loads(contract_json))

shipments_contract
```

    Contract(name='shipments', direction='source', version='0.3.0', schema=<defined>, steps=2)

The result is a regular [Contract](../reference/Contract.md#pointblank.Contract), ready to use on data:

``` python
import polars as pl

shipments = pl.DataFrame(
    {"shipment_id": ["S1", "S2", "S3"], "weight_kg": [12.5, 980.0, 1250.0]}
)

shipments_contract.validate(shipments)
```

[TABLE]

The schema check and the not-null check pass, and the range check flags the shipment weighing 1250 kg.
