# Validate.to_code()

Render this validation plan as canonical Pointblank Python code.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L19594-L19746)

``` python
Validate.to_code()
```

The `to_code()` method walks the validation plan and reconstructs the equivalent `pb.Validate(...).col_vals_*()...` method chain as a string. This is the inverse of writing the plan by hand: it takes an in-memory [Validate](../reference/Validate.md#pointblank.Validate) object and produces source code that, when executed, recreates the same plan.

This is useful for sharing a plan, reviewing it in a diff, persisting it alongside results, and it is the foundation for the AI edit/iterate flow (see [`EditValidation`](../reference/EditValidation.md#pointblank.EditValidation)), which sends the current plan to a model as editable code.

## Returns

`str`  
The validation plan as a block of Python code. The data source is rendered as the placeholder `your_data`, since the original data variable name is not known to the plan. Reference data given as a file path or connection string is written out as-is; an in-memory reference table becomes the placeholder `your_reference_data` (with a warning).

## Notes On Fidelity

Most validation steps round-trip exactly. Steps that carry non-serializable Python objects, such as `pre=` preprocessing callables, `actions=`, callable `active=` conditions, or the expressions used by [`col_vals_expr()`](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr), [`conjointly()`](../reference/Validate.conjointly.md#pointblank.Validate.conjointly), and [`specially()`](../reference/Validate.specially.md#pointblank.Validate.specially), cannot be reproduced from memory. For those, a syntactically valid placeholder is emitted and a warning is raised so the code still parses and the loss is visible.

## Examples

Let's create a validation plan for the `small_table` dataset, with thresholds and four steps, and render it as code.

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

validation = (
    pb.Validate(data=small_table, tbl_name="small_table", thresholds=(0.1, 0.2))
    .col_vals_gt(columns="d", value=100)
    .col_vals_not_null(columns="c")
    .col_vals_in_set(columns="f", set=["low", "mid", "high"])
    .rows_distinct()
)

print(validation.to_code())
```

    import pointblank as pb

    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
            tbl_name="small_table",
            thresholds=pb.Thresholds(warning=0.1, error=0.2),
        )
        .col_vals_gt(columns="d", value=100)
        .col_vals_not_null(columns="c")
        .col_vals_in_set(columns="f", set=["low", "mid", "high"])
        .rows_distinct()
    )

    validation

The code has every step with its arguments, and the thresholds written out as a [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) object. The table is the placeholder `your_data`, and the code doesn't call [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate), since it describes the plan rather than a run of it. Running the code with `your_data` defined recreates the plan, which can then be interrogated:

``` python
namespace = {"your_data": small_table}

exec(validation.to_code(), namespace)

namespace["validation"].interrogate()
```

[TABLE]

The recreated plan finds the same failures as the original would (the two missing values in `c` and the duplicated rows). When a step has something that can't be written as code, such as a `pre=` function, the code still parses but a warning says what was lost:

``` python
validation_pre = pb.Validate(data=small_table).col_vals_gt(
    columns="d", value=100, pre=lambda df: df.head(5)
)

print(validation_pre.to_code())
```

    import pointblank as pb

    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
        )
        .col_vals_gt(columns="d", value=100, pre=None)
    )

    validation

    /var/folders/s8/bj_jsx3d7jqd2btw7bwm6yx80000gp/T/ipykernel_40571/975139424.py:5: UserWarning: Some parts of the validation plan could not be fully serialized to code:
    - Step 'col_vals_gt' has a `pre=` preprocessing callable that cannot be serialized; it was dropped from the generated plan.
      print(validation_pre.to_code())

The `pre=` function is written as `pre=None`, so the generated step would check every row rather than the first five. Fix up placeholders like this one before using the code.
