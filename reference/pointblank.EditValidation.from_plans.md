# from_plans()

Build an [EditValidation](../reference/EditValidation.md#pointblank.EditValidation) from two known plans, without calling a model.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/edit.py#L351-L448)

``` python
from_plans(
    original,
    revised,
    instruction="Manual plan comparison",
)
```

This is a lightweight way to compare an original plan against a revised one and review the difference with [`diff()`](../reference/pointblank.EditValidation.diff.md), [`changed_steps()`](../reference/pointblank.EditValidation.changed_steps.md), and [`review()`](../reference/pointblank.EditValidation.review.md). It is useful for reviewing the change between two saved versions of a plan (for example, in code review) and does not require an LLM provider, an API key, or the `chatlas` package.

## Parameters

`original: Any`  
The original plan. Accepts anything the `validation=` argument accepts: a [`Validate`](../reference/Validate.md#pointblank.Validate) object, a code string, a YAML string, or a file path.

`revised: Any`  
The revised plan, in any of the same forms as `original`.

`instruction: str = "Manual plan comparison"`  
An optional label describing the comparison (shown as the subtitle in [`review()`](../reference/pointblank.EditValidation.review.md)).

## Returns

`EditValidation`  
An [EditValidation](../reference/EditValidation.md#pointblank.EditValidation) holding the code of the two plans, ready for [`diff()`](../reference/pointblank.EditValidation.diff.md), [`changed_steps()`](../reference/pointblank.EditValidation.changed_steps.md), [`review()`](../reference/pointblank.EditValidation.review.md), and [`accept()`](../reference/pointblank.EditValidation.accept.md). It has no `data=`, so pass the table to [accept()](../reference/pointblank.EditValidation.accept.md).

## Examples

Two versions of a plan can be compared directly, here as [Validate](../reference/Validate.md#pointblank.Validate) objects:

``` python
import pointblank as pb

small_table = pb.load_dataset("small_table")

v1 = pb.Validate(data=small_table).col_vals_gt(columns="d", value=100)
v2 = pb.Validate(data=small_table).col_vals_gt(columns="d", value=50)

comparison = pb.EditValidation.from_plans(v1, v2)

comparison.changed_steps()
```

    [{'action': 'modify',
      'method': 'col_vals_gt',
      'old': "col_vals_gt(columns='d', value=100)",
      'new': "col_vals_gt(columns='d', value=50)"}]

The only change is the `value=` of the [col_vals_gt()](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step, reported as a single modification. The plans can also be YAML configurations, which are converted to Python code before they're compared:

``` python
v1_yaml = '''
tbl: small_table
steps:
  - col_vals_gt:
      columns: d
      value: 100
'''
v2_yaml = '''
tbl: small_table
steps:
  - col_vals_gt:
      columns: d
      value: 100
  - rows_distinct
'''

comparison = pb.EditValidation.from_plans(v1_yaml, v2_yaml)

comparison.changed_steps()
```

    [{'action': 'add', 'method': 'rows_distinct', 'new': 'rows_distinct()'}]

The second version adds a [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step. Since the inputs can be file paths too, this works well for checking a change to a saved plan before it's merged.
