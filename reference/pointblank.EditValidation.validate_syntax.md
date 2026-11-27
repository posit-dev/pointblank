# validate_syntax()

Check whether the revised plan parses and uses only known validation methods.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L485-L518)

``` python
validate_syntax()
```

The check parses the revised code and looks at each method whose name starts with `col_`, `row_`, or `rows_`. It fails if the code isn't valid Python or if one of those methods isn't a Pointblank validation method, which catches a model inventing a method or misspelling one. The code isn't run, so a wrong argument name or value isn't caught until [`accept()`](../reference/pointblank.EditValidation.accept.md) or interrogation. When [EditValidation](../reference/EditValidation.md#pointblank.EditValidation) calls a model, the same check decides whether to re-prompt it (see `max_reprompts=`), and [accept()](../reference/pointblank.EditValidation.accept.md) runs it too.

## Returns

`bool`  
`True` if the revised plan passes the check, `False` otherwise.

## Examples

``` python
import pointblank as pb

original = "validation = pb.Validate(data=your_data).col_vals_gt(columns='d', value=100)"
misspelled = "validation = pb.Validate(data=your_data).col_vals_gtt(columns='d', value=50)"

print(pb.EditValidation.from_plans(original, original).validate_syntax())
print(pb.EditValidation.from_plans(original, misspelled).validate_syntax())
```

    True
    False

The second revision calls `col_vals_gtt()`, which doesn't exist, so it fails the check. Calling [accept()](../reference/pointblank.EditValidation.accept.md) on it would raise a `ValueError` naming the unknown method.
