## Validate.to_code()


Render this validation plan as canonical Pointblank Python code.


Usage

``` python
Validate.to_code()
```


The [to_code()](Validate.to_code.md#pointblank.Validate.to_code) method walks the validation plan and reconstructs the equivalent `pb.Validate(...).col_vals_*()...` method chain as a string. This is the inverse of writing the plan by hand: it takes an in-memory [Validate](Validate.md#pointblank.Validate) object and produces source code that, when executed, recreates the same plan.

This is useful for sharing a plan, reviewing it in a diff, persisting it alongside results, and it is the foundation for the AI edit/iterate flow (see <a href="EditValidation.html#pointblank.EditValidation" class="gdls-link"><code>EditValidation</code></a>), which sends the current plan to a model as editable code.


## Returns


`str`  
The validation plan as a block of Python code. The data source is rendered as the placeholder `your_data`, since the original data variable name is not known to the plan.


## Notes On Fidelity

Most validation steps round-trip exactly. Steps that carry non-serializable Python objects, such as `pre=` preprocessing callables, `actions=`, callable `active=` conditions, or the expressions used by <a href="Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr" class="gdls-link"><code>col_vals_expr()</code></a>, <a href="Validate.conjointly.html#pointblank.Validate.conjointly" class="gdls-link"><code>conjointly()</code></a>, and <a href="Validate.specially.html#pointblank.Validate.specially" class="gdls-link"><code>specially()</code></a>, cannot be reproduced from memory. For those, a syntactically valid placeholder is emitted and a warning is raised so the code still parses and the loss is visible.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_gt(columns="d", value=100)
    .col_vals_not_null(columns="a")
    .rows_distinct()
)

print(validation.to_code())
```
