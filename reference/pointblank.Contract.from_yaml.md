# from_yaml()

Read a contract from a YAML file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L1151-L1277)

``` python
from_yaml(path)
```

Keeping contracts in YAML files separates the definition of "valid data" from the code that checks it. The files are easy to read, can be reviewed and versioned like any other change, and can be shared between the teams that produce and consume a dataset. This method loads such a file into a [Contract](../reference/Contract.md#pointblank.Contract), ready to use with [`validate()`](../reference/pointblank.Contract.validate.md) or a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline).

The file can hold the contract under a top-level `contract:` key, as [`to_yaml()`](../reference/pointblank.Contract.to_yaml.md) writes it, or have the contract fields at the top level. The fields are the contract's parameter names (see [`from_dict()`](../reference/pointblank.Contract.from_dict.md) for the expected structure). A hand-written contract file might look like this:

``` yaml
contract:
  name: payments
  version: 1.0.0
  owner: billing-team
  schema:
    payment_id: String
    amount: Float64
  steps:
  - col_vals_not_null:
      columns: payment_id
  - col_vals_gt:
      columns: amount
      value: 0
  thresholds:
    warning: 1
```

## Parameters

`path: str | Path`  
The path to the YAML file.

## Returns

`Contract`  
A new [Contract](../reference/Contract.md#pointblank.Contract) object.

## Raises

`FileNotFoundError`  
If no file exists at `path`.

`ValueError`  
If the file is empty, if it has no contract `name`, or if a field has an invalid value (see [`from_dict()`](../reference/pointblank.Contract.from_dict.md)).

## Examples

Let's write the contract file shown above to a temporary directory, then load it.

``` python
import pointblank as pb
import polars as pl
import tempfile
from pathlib import Path

contract_yaml = '''
contract:
  name: payments
  version: 1.0.0
  owner: billing-team
  schema:
    payment_id: String
    amount: Float64
  steps:
  - col_vals_not_null:
      columns: payment_id
  - col_vals_gt:
      columns: amount
      value: 0
  thresholds:
    warning: 1
'''

contract_path = Path(tempfile.mkdtemp()) / "payments.yaml"
contract_path.write_text(contract_yaml)

payments_contract = pb.Contract.from_yaml(contract_path)

payments_contract
```

    Contract(name='payments', direction='source', version='1.0.0', schema=<defined>, steps=2)

The loaded contract works like one defined in Python. Here it is applied to a small payments table:

``` python
payments = pl.DataFrame({"payment_id": ["T1", "T2", "T3"], "amount": [50.0, 0.0, 12.5]})

payments_contract.validate(payments)
```

[TABLE]

The report label, `Contract: payments v1.0.0`, comes from the file. The payment with an amount of `0.0` fails the [col_vals_gt](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step and triggers the warning threshold.
