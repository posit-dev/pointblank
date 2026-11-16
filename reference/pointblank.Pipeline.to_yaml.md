# to_yaml()

Write the pipeline as YAML.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1474-L1551)

``` python
to_yaml(path=None)
```

YAML is a convenient way to keep a pipeline's contracts and settings under version control next to the transformation code, so changes to them go through review. This method writes the pipeline settings under a `pipeline:` key, followed by the contracts under `source:` and `target:`, in the structure described in [`to_dict()`](../reference/pointblank.Pipeline.to_dict.md). [`from_yaml()`](../reference/pointblank.Pipeline.from_yaml.md) reads the file back into an equivalent [Pipeline](../reference/Pipeline.md#pointblank.Pipeline).

The pipeline's actions and final actions aren't written. Contract step arguments must be plain data (strings, numbers, booleans, dates, lists, and dictionaries), as for [`Contract.to_yaml()`](../reference/pointblank.Contract.to_yaml.md).

## Parameters

`path: str | None = None`  
An optional path to write the YAML to. Any missing parent directories are created, and an existing file is overwritten. If `None` (the default), nothing is written.

## Returns

`str`  
The YAML text. It's returned whether or not `path=` is given.

## Raises

`ValueError`  
If a contract step argument can't be represented in YAML (for example, a column selector or a Polars expression). The message names the step and argument. Nothing is written to `path=` in this case.

## Examples

Let's create a pipeline for a product catalog and get its YAML representation.

``` python
import pointblank as pb

catalog_pipeline = pb.Pipeline(
    source=pb.Contract(
        name="supplier_feed",
        steps=[pb.Step("col_vals_not_null", columns=["sku", "price"])],
    ),
    target=pb.Contract(
        name="catalog",
        direction="target",
        schema=pb.Schema(sku="String", price="Float64"),
        steps=[pb.Step("rows_distinct", columns_subset="sku")],
        on_violation="raise",
    ),
    thresholds=pb.Thresholds(warning=1, error=0.05),
    label="Catalog refresh",
)

print(catalog_pipeline.to_yaml())
```

    pipeline:
      label: Catalog refresh
      thresholds:
        warning: 1
        error: 0.05
    source:
      name: supplier_feed
      steps:
      - col_vals_not_null:
          columns:
          - sku
          - price
    target:
      name: catalog
      direction: target
      on_violation: raise
      schema:
        sku: String
        price: Float64
      steps:
      - rows_distinct:
          columns_subset: sku

To save the file, pass a path, for example `catalog_pipeline.to_yaml("contracts/catalog_pipeline.yaml")`. The YAML text is still returned.
