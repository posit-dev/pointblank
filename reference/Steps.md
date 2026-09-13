# Steps


A reusable collection of validation step definitions.


Usage

``` python
Steps(steps=None)
```


A [Steps](Steps.md#pointblank.Steps) object records validation steps without binding them to data, thresholds, or metadata. It mirrors the validation method API on <a href="../reference/Validate.html#pointblank.Validate" class="gdls-link"><code>Validate</code></a> so that defining steps feels identical, but the result is a portable recipe that can be imported into any number of pipelines via `Validate.add_steps()`.


## Parameters


`steps: list[Step] | None = None`  
An optional list of <a href="../reference/Step.html#pointblank.Step" class="gdls-link"><code>Step</code></a> objects to initialize with.


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
