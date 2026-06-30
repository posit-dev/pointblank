## Pipeline


Binds source and target contracts into a pipeline boundary enforcement unit.


Usage

``` python
Pipeline(
    source=None,
    target=None,
    thresholds=None,
    actions=None,
    final_actions=None,
    label=None,
    short_circuit=True
)
```


A Pipeline enforces data quality at both the ingestion point ("boundary in") and the output point ("boundary out") of a data transformation. It validates that data entering a pipeline meets source contract requirements, and that data leaving meets target contract requirements.


## Parameters


`source: Contract | None = None`  
The source (inbound) Contract.

`target: Contract | None = None`  
The target (outbound) Contract, or None if only validating inbound data.

`thresholds: Thresholds | None = None`  
Global thresholds applied to both boundary validations (overrides contract-level thresholds).

`actions: Actions | None = None`  
Actions triggered on threshold exceedance at either boundary.

`final_actions: FinalActions | None = None`  
Actions triggered after both validations complete.

`label: str | None = None`  
A label for this pipeline (used in reports).

`short_circuit: bool = ``True`  
If True (default), skip the transform and target validation when source validation fails critically. Set to False to always run both validations.


## Examples

``` python
import pointblank as pb

source_contract = pb.Contract(
    name="raw_data",
    direction="source",
    steps=[pb.Step("col_vals_not_null", columns=["id"])],
)

target_contract = pb.Contract(
    name="clean_data",
    direction="target",
    steps=[pb.Step("col_vals_not_null", columns=pb.everything())],
)

pipeline = pb.Pipeline(
    source=source_contract,
    target=target_contract,
)

# Validate source data
source_result = pipeline.validate_source(raw_data)

# Validate target data
target_result = pipeline.validate_target(clean_data)

# Or do it all in one shot
result = pipeline.run(data=raw_data, transform=my_transform)
```


## Methods

| Name | Description |
|----|----|
| [from_dict()](#from_dict) | Construct a Pipeline from a dictionary (e.g., parsed from YAML). |
| [from_yaml()](#from_yaml) | Load a Pipeline from a YAML file. |
| [run()](#run) | Run the full pipeline: validate source, transform, validate target. |
| [to_dict()](#to_dict) | Serialize the Pipeline to a dictionary for YAML/JSON export. |
| [to_yaml()](#to_yaml) | Serialize this Pipeline to YAML. |
| [validate_source()](#validate_source) | Validate data against the source (inbound) contract. |
| [validate_target()](#validate_target) | Validate data against the target (outbound) contract. |

------------------------------------------------------------------------


#### from_dict()


Construct a Pipeline from a dictionary (e.g., parsed from YAML).


Usage

``` python
from_dict(data)
```


##### Parameters


`data: dict[str, Any]`  
A dictionary representation of a pipeline.


##### Returns


`Pipeline`  
A new Pipeline instance.


------------------------------------------------------------------------


#### from_yaml()


Load a Pipeline from a YAML file.


Usage

``` python
from_yaml(path)
```


##### Parameters


`path: str`  
Path to the YAML file.


##### Returns


`Pipeline`  
A new Pipeline instance.


------------------------------------------------------------------------


#### run()


Run the full pipeline: validate source, transform, validate target.


Usage

``` python
run(data, transform)
```


##### Parameters


`data: IntoDataFrame`  
The input data for the pipeline.

`transform: Callable[[Any], Any]`  
A callable that transforms the source data into target data. Must accept the data and return transformed data.


##### Returns


`PipelineResult`  
A result object containing both validations and the transform output.


------------------------------------------------------------------------


#### to_dict()


Serialize the Pipeline to a dictionary for YAML/JSON export.


Usage

``` python
to_dict()
```


##### Returns


`dict`  
A dictionary representation of this pipeline.


------------------------------------------------------------------------


#### to_yaml()


Serialize this Pipeline to YAML.


Usage

``` python
to_yaml(path=None)
```


##### Parameters


`path: str | None = None`  
Optional file path. If provided, the YAML is written to this file. If None, the YAML string is returned.


##### Returns


`str`  
The YAML representation of this pipeline.


------------------------------------------------------------------------


#### validate_source()


Validate data against the source (inbound) contract.


Usage

``` python
validate_source(data)
```


##### Parameters


`data: IntoDataFrame`  
The incoming data to validate.


##### Returns


`Validate`  
An interrogated Validate object with results.


##### Raises


`ValueError`  
If no source contract is defined.

`RuntimeError`  
If on_violation="raise" and validation fails.


------------------------------------------------------------------------


#### validate_target()


Validate data against the target (outbound) contract.


Usage

``` python
validate_target(data)
```


##### Parameters


`data: IntoDataFrame`  
The outgoing data to validate.


##### Returns


`Validate`  
An interrogated Validate object with results.


##### Raises


`ValueError`  
If no target contract is defined.

`RuntimeError`  
If on_violation="raise" and validation fails.
