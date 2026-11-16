# to_code()

Return the revised validation plan as Python code.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L447-L480)

``` python
to_code()
```

The code is in the same form as [`Validate.to_code()`](../reference/Validate.to_code.md#pointblank.Validate.to_code) produces, with the table given as the placeholder `your_data`. Use it to save the revised plan, to read it in full rather than as a diff, or to edit it by hand when the revision isn't quite right. Printing the [EditValidation](../reference/EditValidation.md#pointblank.EditValidation) object shows the same code.

## Returns

`str`  
The revised plan as a string of Python code.

## Examples

``` python
import pointblank as pb

small_table = pb.load_dataset("small_table")

edit = pb.EditValidation.from_plans(
    pb.Validate(data=small_table).col_vals_gt(columns="d", value=100),
    pb.Validate(data=small_table).col_vals_gt(columns="d", value=50),
)

print(edit.to_code())
```

    import pointblank as pb

    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
        )
        .col_vals_gt(columns="d", value=50)
    )

    validation

The revised plan reads its table from `your_data`, so replace that with the table to use before running the code yourself, or let [`accept()`](../reference/pointblank.EditValidation.accept.md) fill it in.
