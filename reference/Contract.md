## Contract


A declarative boundary contract for pipeline data.


Usage

``` python
Contract(
    name,
    direction="source",
    schema=None,
    steps=list(),
    version=None,
    owner=None,
    consumers=None,
    description=None,
    thresholds=None,
    on_violation="warn"
)
```


A Contract defines what data must look like at a specific point in a pipeline. It combines a Schema (structural expectations) with validation Steps (semantic expectations) and metadata (ownership, versioning, directionality).


## Parameters


`name: str`  
A human-readable name for this contract (e.g., "raw_clickstream_feed").

`direction: Literal[``"source", `<span class="st">`"target"``]`</span>` = ``"source"`  
Either "source" (inbound data) or "target" (outbound data product). This is metadata for reporting and does not change validation behavior.

`schema: Schema | None = None`  
A Schema object defining expected column names and types.

`steps: list[Step] = list()`\  
A list of Step objects defining validation rules beyond schema checks.

`version: str | None = None`  
Semantic version string for tracking contract evolution.

`owner: str | None = None`  
Who is responsible for maintaining this contract.

`consumers: str | list[str] | None = None`  
Who depends on data conforming to this contract.

`description: str | None = None`  
Optional longer description of the contract's purpose.

`thresholds: Thresholds | None = None`  
Default thresholds for this contract's validations.

`on_violation: Literal[``"warn", `<span class="st">`"raise"``, ``"log"``]`</span>` = ``"warn"`  
What to do when the contract is violated: "warn", "raise", or "log". Default is "warn".


## Examples

``` python
import pointblank as pb

source_contract = pb.Contract(
    name="raw_sales_feed",
    direction="source",
    schema=pb.Schema(
        order_id="String",
        amount_cents="Int64",
        currency="String",
    ),
    steps=[
        pb.Step("col_vals_not_null", columns=["order_id", "amount_cents"]),
        pb.Step("rows_distinct", columns=["order_id"]),
    ],
    version="1.0.0",
    owner="data-platform-team",
)
```


## Methods

| Name | Description |
|----|----|
| [from_dict()](#from_dict) | Construct a Contract from a dictionary (e.g., parsed from YAML). |
| [from_yaml()](#from_yaml) | Load a Contract from a YAML file. |
| [to_dict()](#to_dict) | Serialize the Contract to a dictionary for YAML/JSON export. |
| [to_validate()](#to_validate) | Compile this Contract into a Validate object ready for interrogation. |
| [to_yaml()](#to_yaml) | Serialize this Contract to YAML. |
| [validate()](#validate) | Compile and interrogate this Contract against the provided data. |

------------------------------------------------------------------------


#### from_dict()


Construct a Contract from a dictionary (e.g., parsed from YAML).


Usage

``` python
from_dict(data)
```


##### Parameters


`data: dict[str, Any]`  
A dictionary representation of a contract.


##### Returns


`Contract`  
A new Contract instance.


------------------------------------------------------------------------


#### from_yaml()


Load a Contract from a YAML file.


Usage

``` python
from_yaml(path)
```


##### Parameters


`path: str`  
Path to the YAML file.


##### Returns


`Contract`  
A new Contract instance.


------------------------------------------------------------------------


#### to_dict()


Serialize the Contract to a dictionary for YAML/JSON export.


Usage

``` python
to_dict()
```


##### Returns


`dict`  
A dictionary representation of this contract.


------------------------------------------------------------------------


#### to_validate()


Compile this Contract into a Validate object ready for interrogation.


Usage

``` python
to_validate(data)
```


This creates a Validate object with all schema checks and validation steps from this contract applied. The resulting Validate object has NOT been interrogated yet -- call `.interrogate()` on it to execute the validation.


##### Parameters


`data: IntoDataFrame`  
The data table to validate against this contract.


##### Returns


`Validate`  
A Validate object with all contract checks applied (not yet interrogated).


##### Examples

``` python
import pointblank as pb

contract = pb.Contract(
    name="my_data",
    schema=pb.Schema(id="Int64", name="String"),
    steps=[pb.Step("col_vals_not_null", columns=["id"])],
)

# Create and interrogate
validation = contract.to_validate(my_data).interrogate()
```

------------------------------------------------------------------------


#### to_yaml()


Serialize this Contract to YAML.


Usage

``` python
to_yaml(path=None)
```


##### Parameters


`path: str | None = None`  
Optional file path. If provided, the YAML is written to this file. If None, the YAML string is returned.


##### Returns


`str`  
The YAML representation of this contract.


------------------------------------------------------------------------


#### validate()


Compile and interrogate this Contract against the provided data.


Usage

``` python
validate(data)
```


This is a convenience method that calls `to_validate(data).interrogate()`.


##### Parameters


`data: IntoDataFrame`  
The data table to validate against this contract.


##### Returns


`Validate`  
An interrogated Validate object with results.
