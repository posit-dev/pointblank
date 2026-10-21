# Steps

A reusable collection of validation step definitions.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L21-L1069)

``` python
Steps(steps=None)
```

A [Steps](../reference/Steps.md#pointblank.Steps) object records validation steps without binding them to data, thresholds, or metadata. It mirrors the validation method API on [`Validate`](../reference/Validate.md#pointblank.Validate) so that defining steps feels identical, but the result is a portable recipe that can be imported into any number of pipelines via `Validate.add_steps()`.

## Parameters

`steps: list[Step] | None = None`  
An optional list of [`Step`](../reference/Step.md#pointblank.Step) objects to initialize with.

## Examples

``` python
import pointblank as pb

completeness = (
    pb.Steps()
    .col_vals_not_null(columns=pb.ends_with("_id"))
    .col_vals_not_null(columns="email")
)

positive_amounts = (
    pb.Steps()
    .col_vals_ge(columns=pb.starts_with("amt_"), value=0)
    .col_vals_gt(columns="amt_total", value=0)
)

validation = (
    pb.Validate(data=orders, label="Order quality")
    .add_steps(completeness, positive_amounts)
    .interrogate()
)
```
