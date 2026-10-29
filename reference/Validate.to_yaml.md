# Validate.to_yaml()

Serialize this validation plan to a [yaml_interrogate()](../reference/yaml_interrogate.md#pointblank.yaml_interrogate)-compatible YAML config.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L19748-L19890)

``` python
Validate.to_yaml(path=None)
```

The `to_yaml()` method renders the validation plan as a YAML document using the same schema consumed by [`yaml_interrogate()`](../reference/yaml_interrogate.md#pointblank.yaml_interrogate). This enables storing plans as configuration, sharing them across projects, and round-tripping a plan through YAML.

## Parameters

`path: str | Path | None = None`  
An optional file path. If provided, the YAML is written to this file (parent directories are created as needed) in addition to being returned.

## Returns

`str`  
The validation plan as a YAML string. The `tbl` field is set from `tbl_name` when available, otherwise to the placeholder `your_data`; set it to a loadable data source before passing the YAML to [yaml_interrogate()](../reference/yaml_interrogate.md#pointblank.yaml_interrogate). The `reference` field is written the same way as in [`to_code()`](../reference/Validate.to_code.md#pointblank.Validate.to_code): a file path or connection string as-is, and the placeholder `your_reference_data` (with a warning) for an in-memory table.

## Notes On Fidelity

As with [`to_code()`](../reference/Validate.to_code.md#pointblank.Validate.to_code), steps carrying non-serializable Python objects (`pre=` callables, `actions=`, callable `active=`, and the expressions of `col_vals_expr()`/`conjointly()`/[specially()](../reference/Validate.specially.md#pointblank.Validate.specially)) cannot be represented in YAML; a placeholder is emitted and a warning is raised.

## Examples

Let's create a validation plan for the `small_table` dataset and write it as YAML.

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

validation = (
    pb.Validate(data=small_table, tbl_name="small_table", thresholds=(0.1, 0.2))
    .col_vals_gt(columns="d", value=100)
    .col_vals_not_null(columns="c")
    .col_vals_in_set(columns="f", set=["low", "mid", "high"])
)

yaml_config = validation.to_yaml()

print(yaml_config)
```

    tbl: small_table
    tbl_name: small_table
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - col_vals_gt:
        columns: d
        value: 100
    - col_vals_not_null:
        columns: c
    - col_vals_in_set:
        columns: f
        set:
        - low
        - mid
        - high

Each step becomes an entry under `steps`, named after its method and with its arguments beneath. The `tbl` field was set from `tbl_name=`. Here, that name happens to be a dataset that [`yaml_interrogate()`](../reference/yaml_interrogate.md#pointblank.yaml_interrogate) can load, so the YAML runs as is:

``` python
pb.yaml_interrogate(yaml_config)
```

[TABLE]

For a table of your own, set `tbl` to a file path or connection string before running the YAML. To keep the configuration (for instance, in version control next to the pipeline it checks), give `to_yaml()` a `path=`, which writes the file and still returns the string.
