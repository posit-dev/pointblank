# accept()

Execute the revised plan and return the resulting [Validate](../reference/Validate.md#pointblank.Validate) object.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L700-L798)

``` python
accept(data=None)
```

The returned [Validate](../reference/Validate.md#pointblank.Validate) has NOT been interrogated; call `.interrogate()` on it to run the validation. Before running the code, [accept()](../reference/pointblank.EditValidation.accept.md) applies the same check as [`validate_syntax()`](../reference/pointblank.EditValidation.validate_syntax.md) and raises a `ValueError` if it fails. It then runs the code with the table bound to `your_data` and returns the [Validate](../reference/Validate.md#pointblank.Validate) object that the code assigns to `validation` (or the last [Validate](../reference/Validate.md#pointblank.Validate) object it assigns). Any `.interrogate()` call in the revised code is removed first.

> **Warning**
>
> [accept()](../reference/pointblank.EditValidation.accept.md) runs the revised code with Python's `exec()`. Read the code (with [`to_code()`](../reference/pointblank.EditValidation.to_code.md) or [`review()`](../reference/pointblank.EditValidation.review.md)) before accepting an edit, as you would any other code you didn't write.

## Parameters

`data: Any = None`  
The data table to bind to the plan's `your_data` placeholder. If omitted, the `data=` provided to [EditValidation](../reference/EditValidation.md#pointblank.EditValidation) is used. One of the two must be present.

## Returns

`Validate`  
The revised validation plan as a [Validate](../reference/Validate.md#pointblank.Validate) object.

## Examples

``` python
import pointblank as pb

small_table = pb.load_dataset("small_table")

edit = pb.EditValidation.from_plans(
    pb.Validate(data=small_table).col_vals_gt(columns="d", value=100),
    (
        pb.Validate(data=small_table)
        .col_vals_gt(columns="d", value=50)
        .col_vals_not_null(columns="a")
    ),
)

validation = edit.accept(data=small_table).interrogate()

validation
```

[TABLE]

The revised plan has both steps, and it ran against `small_table` because that's the table passed to [accept()](../reference/pointblank.EditValidation.accept.md). An [EditValidation](../reference/EditValidation.md#pointblank.EditValidation) built by [`from_plans()`](../reference/pointblank.EditValidation.from_plans.md) has no `data=` of its own, so the table has to be given here. Passing a different table is also how to run the same revised plan on new data.
