# from_yaml()

Read a pipeline from a YAML file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1344-L1472)

``` python
from_yaml(path)
```

Storing a pipeline in YAML keeps the definition of what valid input and output look like separate from the transformation code, where it can be reviewed and versioned on its own. This method loads a file written by [`to_yaml()`](../reference/pointblank.Pipeline.to_yaml.md), or written by hand in the same structure (see [`from_dict()`](../reference/pointblank.Pipeline.from_dict.md) for the details). A hand-written file might look like this:

``` yaml
pipeline:
  label: Shipments
source:
  name: raw_shipments
  steps:
  - col_vals_not_null:
      columns: shipment_id
target:
  name: shipments
  steps:
  - col_vals_between:
      columns: weight_kg
      left: 0
      right: 1000
```

The transformation, actions, and final actions aren't part of the file. Pass the transformation to [`run()`](../reference/pointblank.Pipeline.run.md) as usual.

## Parameters

`path: str`  
The path to the YAML file.

## Returns

`Pipeline`  
A new [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) object.

## Raises

`FileNotFoundError`  
If no file exists at `path`.

`ValueError`  
If the file is empty, if it has neither a `source` nor a `target` contract, or if a contract is invalid (see [`Contract.from_dict()`](../reference/pointblank.Contract.from_dict.md)).

## Examples

Let's write the pipeline file shown above to a temporary directory, then load it.

``` python
import pointblank as pb
import polars as pl
import tempfile
from pathlib import Path

pipeline_yaml = '''
pipeline:
  label: Shipments
source:
  name: raw_shipments
  steps:
  - col_vals_not_null:
      columns: shipment_id
target:
  name: shipments
  steps:
  - col_vals_between:
      columns: weight_kg
      left: 0
      right: 1000
'''

pipeline_path = Path(tempfile.mkdtemp()) / "shipments.yaml"
pipeline_path.write_text(pipeline_yaml)

shipments_pipeline = pb.Pipeline.from_yaml(pipeline_path)

shipments_pipeline
```

    Pipeline(
      label='Shipments',
      source='raw_shipments',
      target='shipments',
      short_circuit=True
    )

The loaded pipeline works like one defined in Python. Here it runs a transformation that converts weights from pounds to kilograms:

``` python
raw_shipments = pl.DataFrame({"shipment_id": ["S1", "S2"], "weight_lb": [26.5, 410.0]})

result = shipments_pipeline.run(
    data=raw_shipments,
    transform=lambda df: df.select(
        "shipment_id", weight_kg=pl.col("weight_lb") * 0.4536
    ),
)

result.passed
```

    True

Both boundaries pass, since the heavier shipment (410 lb, or about 186 kg) is well within the target's range of 0 to 1000 kg. The transformation stays in Python while the contracts live in the file, so the rules can be changed without touching the job's code.
