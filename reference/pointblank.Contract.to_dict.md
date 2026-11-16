# to_dict()

Convert the contract to a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L932-L1019)

``` python
to_dict()
```

This method gives you the contract as plain Python data (strings, numbers, lists, and dictionaries), ready to be written as JSON, stored in a database or configuration system, or compared with another contract's definition. It's the same structure that [`to_yaml()`](../reference/pointblank.Contract.to_yaml.md) writes, but without the top-level `contract` key. [`from_dict()`](../reference/pointblank.Contract.from_dict.md) turns it back into a [Contract](../reference/Contract.md#pointblank.Contract).

The dictionary always has a `"name"` key. Other fields are only included when they are set. `direction=` is left out when it's `"source"` and `on_violation=` when it's `"warn"`, because those are the defaults. The schema becomes a mapping of column names to type strings, each step becomes a single-key mapping from method name to arguments (see [`Step.to_dict()`](../reference/Step.md#pointblank.Step.to_dict)), and the thresholds become a mapping with the levels that are set.

## Returns

`dict[str, Any]`  
A dictionary representation of the contract.

## Examples

Let's create a contract for a target table and convert it to a dictionary.

``` python
import pointblank as pb

accounts_contract = pb.Contract(
    name="accounts",
    direction="target",
    version="1.0.0",
    schema=pb.Schema(account_id="Int64", email="String"),
    steps=[
        pb.Step("rows_distinct", columns_subset="account_id"),
        pb.Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+$"),
    ],
    thresholds=pb.Thresholds(warning=0.01),
)

accounts_contract.to_dict()
```

    {'name': 'accounts',
     'direction': 'target',
     'version': '1.0.0',
     'schema': {'account_id': 'Int64', 'email': 'String'},
     'steps': [{'rows_distinct': {'columns_subset': 'account_id'}},
      {'col_vals_regex': {'columns': 'email', 'pattern': '^[^@]+@[^@]+$'}}],
     'thresholds': {'warning': 0.01}}

`direction` is included because `"target"` isn't the default, while `on_violation`, `owner`, and the other unset fields are left out. Only the warning level of the thresholds is set, so it's the only one listed.

A dictionary like this converts directly to JSON with the standard library:

``` python
import json

print(json.dumps(accounts_contract.to_dict(), indent=2))
```

    {
      "name": "accounts",
      "direction": "target",
      "version": "1.0.0",
      "schema": {
        "account_id": "Int64",
        "email": "String"
      },
      "steps": [
        {
          "rows_distinct": {
            "columns_subset": "account_id"
          }
        },
        {
          "col_vals_regex": {
            "columns": "email",
            "pattern": "^[^@]+@[^@]+$"
          }
        }
      ],
      "thresholds": {
        "warning": 0.01
      }
    }

The JSON text can be stored in a database column or sent to an API. After it's parsed with `json.loads()`, [`from_dict()`](../reference/pointblank.Contract.from_dict.md) rebuilds the contract.
