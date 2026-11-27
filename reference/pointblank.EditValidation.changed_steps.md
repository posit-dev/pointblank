# changed_steps()

Return a structured, step-level list of the changes between the plans.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L573-L623)

``` python
changed_steps()
```

Unlike [`diff()`](../reference/pointblank.EditValidation.diff.md), which is a textual diff sensitive to formatting, this compares the two plans at the level of validation steps and is robust to reformatting. Each entry is a dict with an `"action"` of `"add"`, `"remove"`, or `"modify"`, a `"method"` name, and the relevant `"old"`/`"new"` step text. A step whose method stays the same but whose arguments change is reported as `"modify"`, and a step replaced by one with a different method is reported as a `"remove"` and an `"add"`. Unchanged steps aren't listed.

## Returns

`list[dict]`  
One record per changed step, in plan order. The list is empty when the plans have the same steps.

## Examples

``` python
import pointblank as pb

small_table = pb.load_dataset("small_table")

edit = pb.EditValidation.from_plans(
    (
        pb.Validate(data=small_table)
        .col_vals_gt(columns="d", value=100)
        .rows_distinct()
    ),
    (
        pb.Validate(data=small_table)
        .col_vals_gt(columns="d", value=50)
        .col_vals_not_null(columns="a")
    ),
)

for change in edit.changed_steps():
    print(change)
```

    {'action': 'modify', 'method': 'col_vals_gt', 'old': "col_vals_gt(columns='d', value=100)", 'new': "col_vals_gt(columns='d', value=50)"}
    {'action': 'remove', 'method': 'rows_distinct', 'old': 'rows_distinct()'}
    {'action': 'add', 'method': 'col_vals_not_null', 'new': "col_vals_not_null(columns='a')"}

The [col_vals_gt()](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step is modified, so its record has both the old and the new step text. The [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step was removed and has only `"old"`, and the new [col_vals_not_null()](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) step has only `"new"`. The same records are behind the table that [`review()`](../reference/pointblank.EditValidation.review.md) shows.
