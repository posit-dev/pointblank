# to_dict()

Convert the pipeline to a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1150-L1220)

``` python
to_dict()
```

This method gives you the pipeline's definition as plain Python data, ready to be written as JSON or stored in a configuration system. It's the same structure that [`to_yaml()`](../reference/pointblank.Pipeline.to_yaml.md) writes, and [`from_dict()`](../reference/pointblank.Pipeline.from_dict.md) turns it back into a [Pipeline](../reference/Pipeline.md#pointblank.Pipeline).

The dictionary has up to three keys. `"pipeline"` holds the pipeline's own settings: the `"label"` and `"thresholds"` (when set) and `"short_circuit"` (only when it's `False`, since `True` is the default). This entry is always present, even when it's empty. `"source"` and `"target"` hold the contracts, in the form produced by [`Contract.to_dict()`](../reference/pointblank.Contract.to_dict.md), and each is left out if the pipeline doesn't have that contract. The pipeline's actions and final actions aren't included.

## Returns

`dict[str, Any]`  
A dictionary representation of the pipeline.

## Examples

Let's create a pipeline with a label, thresholds, and short-circuiting turned off, and convert it to a dictionary.

``` python
import pointblank as pb

pipeline = pb.Pipeline(
    source=pb.Contract(
        name="raw_events", steps=[pb.Step("col_vals_not_null", columns="event_id")]
    ),
    target=pb.Contract(
        name="events",
        direction="target",
        steps=[pb.Step("rows_distinct", columns_subset="event_id")],
    ),
    thresholds=pb.Thresholds(warning=0.01, error=0.05),
    label="Event stream",
    short_circuit=False,
)

pipeline.to_dict()
```

    {'pipeline': {'label': 'Event stream',
      'thresholds': {'warning': 0.01, 'error': 0.05},
      'short_circuit': False},
     'source': {'name': 'raw_events',
      'steps': [{'col_vals_not_null': {'columns': 'event_id'}}]},
     'target': {'name': 'events',
      'direction': 'target',
      'steps': [{'rows_distinct': {'columns_subset': 'event_id'}}]}}

The pipeline settings come first, followed by each contract. The source contract has no `direction` key because `"source"` is the default direction.
