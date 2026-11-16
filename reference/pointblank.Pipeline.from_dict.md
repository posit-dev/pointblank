# from_dict()

Create a pipeline from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1222-L1342)

``` python
from_dict(data)
```

This is the reverse of [`to_dict()`](../reference/pointblank.Pipeline.to_dict.md). Use it when the pipeline definition comes from somewhere other than a YAML file, such as a JSON document or a configuration service. To read a YAML file directly, use [`from_yaml()`](../reference/pointblank.Pipeline.from_yaml.md).

The dictionary can have a `"pipeline"` entry with any of `"label"`, `"thresholds"` (a mapping of `"warning"`, `"error"`, and `"critical"` to values), and `"short_circuit"`. Settings that are missing get their defaults. The contracts go under `"source"` and `"target"` in the form read by [`Contract.from_dict()`](../reference/pointblank.Contract.from_dict.md). A contract without a `"direction"` key is given the direction that matches its position, so `direction:` doesn't need to be repeated in hand-written definitions. Actions and final actions can't be given in the dictionary. To use them, create a new pipeline with the parsed contracts and settings plus your actions.

## Parameters

`data: dict[str, Any]`  
A dictionary describing a pipeline, in the form produced by [`to_dict()`](../reference/pointblank.Pipeline.to_dict.md).

## Returns

`Pipeline`  
A new [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) object.

## Raises

`ValueError`  
If the dictionary has neither a `"source"` nor a `"target"` contract, or if a contract is invalid (see [`Contract.from_dict()`](../reference/pointblank.Contract.from_dict.md)).

## Examples

Here's a pipeline definition written as JSON. It only has a target contract, so it acts as an output guard.

``` python
import json
import pointblank as pb
import polars as pl

pipeline_json = '''
{
  "pipeline": {"label": "Invoice export", "thresholds": {"warning": 1}},
  "target": {
    "name": "invoices",
    "steps": [
      {"col_vals_not_null": {"columns": "invoice_id"}},
      {"col_vals_gt": {"columns": "total", "value": 0}}
    ]
  }
}
'''

export_pipeline = pb.Pipeline.from_dict(json.loads(pipeline_json))

export_pipeline
```

    Pipeline(
      label='Invoice export',
      target='invoices',
      short_circuit=True
    )

The target contract didn't specify a direction, so it was given `"target"`. The pipeline can now check an export:

``` python
invoices = pl.DataFrame({"invoice_id": ["I-1", "I-2", "I-3"], "total": [120.0, 0.0, 75.5]})

export_pipeline.validate_target(invoices)
```

    UserWarning: Contract 'invoices' (target) violated: validation has failing steps.

[TABLE]

The invoice with a total of `0.0` fails the [col_vals_gt](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step and reaches the warning threshold set in the pipeline definition.
