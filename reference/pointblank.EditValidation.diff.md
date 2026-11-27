# diff()

Return a unified textual diff of the original plan versus the revised plan.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L520-L571)

``` python
diff()
```

The diff compares the two plans' code line by line, in the format used by `git diff` and code review tools (lines starting with `-` were removed and lines starting with `+` were added). It's the quickest way to see an edit in context, but it's sensitive to formatting: a step that was only reformatted shows up as changed. For a list of the changed steps that ignores formatting, use [`changed_steps()`](../reference/pointblank.EditValidation.changed_steps.md).

## Returns

`str`  
The diff as a single string, which is empty when the two plans have the same code.

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

print(edit.diff())
```

    --- original_plan.py
    +++ edited_plan.py
    @@ -4,7 +4,8 @@
         pb.Validate(
             data=your_data,  # Replace your_data with the actual data variable
         )
    -    .col_vals_gt(columns="d", value=100)
    +    .col_vals_gt(columns="d", value=50)
    +    .col_vals_not_null(columns="a")
     )
     
     validation

The old [col_vals_gt()](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) line is replaced by one with `value=50`, and a [col_vals_not_null()](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) line is added. The two files in the header (`original_plan.py` and `edited_plan.py`) are only labels, and nothing is written to disk.
