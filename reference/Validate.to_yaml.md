## Validate.to_yaml()


Serialize this validation plan to a [yaml_interrogate()](yaml_interrogate.md#pointblank.yaml_interrogate)-compatible YAML config.


Usage

``` python
Validate.to_yaml(path=None)
```


The [to_yaml()](Contract.md#pointblank.Contract.to_yaml) method renders the validation plan as a YAML document using the same schema consumed by <a href="yaml_interrogate.html#pointblank.yaml_interrogate" class="gdls-link"><code>yaml_interrogate()</code></a>. This enables storing plans as configuration, sharing them across projects, and round-tripping a plan through YAML.


## Parameters


`path: str | Path | None = None`  
An optional file path. If provided, the YAML is written to this file (parent directories are created as needed) in addition to being returned.


## Returns


`str`  
The validation plan as a YAML string. The `tbl` field is set from `tbl_name` when available, otherwise to the placeholder `your_data`; set it to a loadable data source before passing the YAML to [yaml_interrogate()](yaml_interrogate.md#pointblank.yaml_interrogate).


## Notes On Fidelity

As with <a href="Validate.to_code.html#pointblank.Validate.to_code" class="gdls-link"><code>to_code()</code></a>, steps carrying non-serializable Python objects (`pre=` callables, `actions=`, callable `active=`, and the expressions of [col_vals_expr()](Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)/[conjointly()](Validate.conjointly.md#pointblank.Validate.conjointly)/[specially()](Validate.specially.md#pointblank.Validate.specially)) cannot be represented in YAML; a placeholder is emitted and a warning is raised.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"), tbl_name="small_table")
    .col_vals_gt(columns="d", value=100)
    .col_vals_not_null(columns="a")
)

print(validation.to_yaml())
```
