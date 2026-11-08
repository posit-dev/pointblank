# yaml_ablation()

Set up an ablation from a YAML configuration.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/yaml.py#L1394-L1568)

``` python
yaml_ablation(
    yaml,
    evaluate=None,
    set_tbl=None,
    holdout=None,
    namespaces=None,
    allow_python=False,
)
```

A YAML configuration for [`yaml_interrogate()`](../reference/yaml_interrogate.md#pointblank.yaml_interrogate) can have an `ablation:` block, which describes an [`Ablation`](../reference/Ablation.md#pointblank.Ablation) of its validation: the `evaluate=` function, the settings, and the strategies that add variants. This function interrogates the validation and sets up the ablation, ready to [`run()`](../reference/Ablation.run.md#pointblank.Ablation.run) or to [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export). The validation and its ablation can then be kept together in one file, alongside a model's training code.

## Parameters

`yaml: Union[str, Path]`  
YAML configuration as a string or a file path, with an `ablation:` block.

`evaluate: Any = None`  
The `evaluate=` function of the ablation. If given, it's used in place of the block's `evaluate:` (which is then not imported), so that the configuration can be used without `allow_python=True`.

`set_tbl: Any = None`  
A table to use in place of the configuration's `tbl` (as in [yaml_interrogate()](../reference/yaml_interrogate.md#pointblank.yaml_interrogate)).

`holdout: Any = None`  
A table to use in place of the block's `holdout:` (the holdout set that the validation's steps are also applied to, as in the `holdout=` of [Ablation](../reference/Ablation.md#pointblank.Ablation)).

`namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None`  
Modules to make available to Python code in the configuration (as in [yaml_interrogate()](../reference/yaml_interrogate.md#pointblank.yaml_interrogate)). This only has an effect with `allow_python=True`.

`allow_python: bool = False`  
Whether to run Python code in the configuration. The `evaluate:` of the `ablation:` block is Python code (it imports a function and runs it on the training data), so it's only used with `allow_python=True`. Without it, either pass `evaluate=` or leave `evaluate:` out (e.g., to `export()` the versions of the table for evaluation elsewhere).

## Returns

`Ablation`  
The ablation, with its variants added but not yet run.

## Raises

`YAMLValidationError`  
If the configuration is invalid, has no `ablation:` block, or has Python code without `allow_python=True`.

## The `ablation:` Block

The block has the parameters of [Ablation](../reference/Ablation.md#pointblank.Ablation) (other than `validation=`), along with a list of `strategies:`:

``` yaml
ablation:
  evaluate: my_models.training:evaluate
  evaluate_version: "2024-06"
  metric: mae
  higher_is_better: false
  repeats: 5
  control:
    stratified: sensor
  cache: .ablation_cache
  holdout: data/holdout.parquet
  strategies:
    - drop_each
    - drop_all_but_each
    - drop_steps:
        steps: [1, 2]
        name: Without step 3
    - drop_groups:
        column: sensor
```

The `evaluate:` is the import path of a function (the module, a colon, and the name of the function), or a `python:` block whose result is the function. The `control:` is `size_matched` (the default), `null` (for no control), or `stratified:` with one or more columns (for a [`Stratified`](../reference/ablation.Stratified.md#pointblank.ablation.Stratified) control). The `holdout:` is a data source like the `tbl:` of the configuration (a dataset name, a file path, or a `python:` block), loaded with its `df_library:`. Each strategy is the name of an [Ablation](../reference/Ablation.md#pointblank.Ablation) method that adds variants (including the sweeps [sweep_score](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score) and [sweep_step](../reference/Ablation.sweep_step.md#pointblank.Ablation.sweep_step), the injected defects of [inject](../reference/Ablation.inject.md#pointblank.Ablation.inject), and the Shapley values of [shapley](../reference/Ablation.shapley.md#pointblank.Ablation.shapley)), with its parameters as a mapping. The strategies are added in order. In YAML, the `corruption:` of [inject](../reference/Ablation.inject.md#pointblank.Ablation.inject) is the name of a defect (`null`, `shuffle`, `truncate`, or `duplicate`), not a function.

The block is ignored by [yaml_interrogate()](../reference/yaml_interrogate.md#pointblank.yaml_interrogate), so the same configuration can be used both to validate the data and to measure what its steps do for a model.

## Examples

Here's a table of temperature readings with some out-of-range values, and a YAML configuration with a validation step and an `ablation:` block. The `evaluate()` function is passed from Python, so the configuration doesn't need `allow_python=True`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

yaml_config = '''
tbl: readings.csv
tbl_name: readings
steps:
- col_vals_between:
    columns: temp
    left: -10
    right: 45
ablation:
  metric: error
  higher_is_better: false
  label: Temperature model
  strategies:
    - drop_each
'''

ablation = pb.yaml_ablation(yaml_config, evaluate=evaluate, set_tbl=readings)

ablation.run()
```

| Ablation report Temperature model |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.392 ± 2.207** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The `tbl:` of the configuration is replaced with the table in `set_tbl=`, and the ablation was set up from the block, ready to run.
