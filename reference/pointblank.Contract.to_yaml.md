# to_yaml()

Write the contract as YAML.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L1279-L1374)

``` python
to_yaml(path=None)
```

YAML is the natural way to store and share a contract. It can sit in version control next to the pipeline code, so changes to the rules go through review, and other teams can read it without running Python. This method writes the contract under a top-level `contract:` key, with the fields in a fixed order (name and metadata first, then the schema, steps, and thresholds). [`from_yaml()`](../reference/pointblank.Contract.from_yaml.md) reads the file back into an equivalent [Contract](../reference/Contract.md#pointblank.Contract).

Fields that aren't set, and the default `direction=` and `on_violation=` values, are left out of the YAML (see [`to_dict()`](../reference/pointblank.Contract.to_dict.md)). Step arguments must be plain data (strings, numbers, booleans, dates, lists, and dictionaries) so that [from_yaml()](../reference/pointblank.Contract.from_yaml.md) can read them back. Step-level thresholds are converted to dictionaries automatically, but column selectors, expressions, and functions can't be written.

## Parameters

`path: str | Path | None = None`  
An optional path to write the YAML to. Any missing parent directories are created, and an existing file is overwritten. If `None` (the default), nothing is written.

## Returns

`str`  
The YAML text. It's returned whether or not `path=` is given.

## Raises

`ValueError`  
If a step argument can't be represented in YAML (for example, a column selector such as `pb.starts_with("amt_")` or a Polars expression). The message names the step and argument. Nothing is written to `path=` in this case.

## Examples

Let's define a contract for a table of web sessions and get its YAML representation.

``` python
import pointblank as pb

sessions_contract = pb.Contract(
    name="web_sessions",
    version="3.1.0",
    owner="web-analytics",
    consumers=["marketing", "product"],
    on_violation="raise",
    schema=pb.Schema(session_id="String", duration_s="Int64", device="String"),
    steps=[
        pb.Step("col_vals_ge", columns="duration_s", value=0),
        pb.Step("col_vals_in_set", columns="device", set=["desktop", "mobile", "tablet"]),
    ],
)

print(sessions_contract.to_yaml())
```

    contract:
      name: web_sessions
      version: 3.1.0
      owner: web-analytics
      consumers:
      - marketing
      - product
      on_violation: raise
      schema:
        session_id: String
        duration_s: Int64
        device: String
      steps:
      - col_vals_ge:
          columns: duration_s
          value: 0
      - col_vals_in_set:
          columns: device
          set:
          - desktop
          - mobile
          - tablet

The output lists the metadata first, including `on_violation` because it isn't the default, then the schema and the steps. Each step is written as its method name with the arguments nested below it.

To save the contract, pass a file path. The parent directory is created if needed, and the YAML text is still returned. Reading the file back gives a contract equal to the original:

``` python
import tempfile
from pathlib import Path

contract_path = Path(tempfile.mkdtemp()) / "contracts" / "web_sessions.yaml"

_ = sessions_contract.to_yaml(contract_path)

pb.Contract.from_yaml(contract_path) == sessions_contract
```

    True

The file holds everything needed to rebuild the contract. A natural place for it is version control, next to the code that produces the data, so that changes to the rules are reviewed like any other change. Wherever the contract is enforced, it can be loaded with [`from_yaml()`](../reference/pointblank.Contract.from_yaml.md) (or as part of a pipeline with [`Pipeline.from_yaml()`](../reference/pointblank.Pipeline.from_yaml.md)).
