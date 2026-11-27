# review()

Return a `great_tables` GT object summarizing the step-level changes.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L625-L698)

``` python
review()
```

This renders the added, removed, and modified steps side by side for quick human review before accepting the edit. It pairs with [`accept()`](../reference/pointblank.EditValidation.accept.md): review the change, then accept it. Each row is one record from [`changed_steps()`](../reference/pointblank.EditValidation.changed_steps.md), colored by the kind of change, and the instruction is shown as the subtitle. When nothing changed, the table has a single "(no changes)" row.

## Returns

`GT`  
A `great_tables` table with one row per changed step.

## Examples

``` python
import pointblank as pb

small_table = pb.load_dataset("small_table")

edit = pb.EditValidation.from_plans(
    pb.Validate(data=small_table).col_vals_gt(columns="d", value=100).rows_distinct(),
    pb.Validate(data=small_table).col_vals_gt(columns="d", value=50),
    instruction="Loosen the d check and drop the duplicate-row check",
)

edit.review()
```

| Proposed plan changes |  |  |  |
|----|----|----|----|
| Loosen the d check and drop the duplicate-row check |  |  |  |
| Change | Method | Before | After |
| ~ modified | col_vals_gt | col_vals_gt(columns='d', value=100) | col_vals_gt(columns='d', value=50) |
| − removed | rows_distinct | rows_distinct() |  |

The modified [col_vals_gt()](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step shows its old and new text in the *Before* and *After* columns, and the removed [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step has only a *Before* entry. Because the result is a GT object, it can be styled further or saved with the `great_tables` API.
