## Validate.suggest_improvements()


Propose AI-generated improvements to this validation plan.


Usage

``` python
Validate.suggest_improvements(
    model, api_key=None, verify_ssl=True, max_reprompts=1
)
```


This is a thin, convenience wrapper over <a href="EditValidation.html#pointblank.EditValidation" class="gdls-link"><code>EditValidation</code></a>: it profiles the table with <a href="DataScan.html#pointblank.DataScan" class="gdls-link"><code>DataScan</code></a>, derives an instruction that targets gaps in the current plan (columns with no coverage, missing thresholds), and asks the model to extend the plan accordingly. As with [EditValidation](EditValidation.md#pointblank.EditValidation), you review the proposed change as a diff and explicitly accept it.


## Parameters


`model: str`  
The model to use, in `provider:model` form (e.g., `"anthropic:claude-opus-4-8"`).

`api_key: str | None = None`  
The API key for the model provider.

`verify_ssl: bool = ``True`  
Whether to verify SSL certificates for provider requests. Defaults to `True`.

`max_reprompts: int = ``1`  
Maximum automatic re-prompts if the returned plan fails the syntax check.


## Returns


`EditValidation`  
An <a href="EditValidation.html#pointblank.EditValidation" class="gdls-link"><code>EditValidation</code></a> with the proposed improvements, ready to inspect via `.diff()`/`.changed_steps()` and finalize via `.accept()`.


## Examples

``` python
import pointblank as pb

validation = pb.Validate(data=pb.load_dataset("small_table")).col_vals_gt(
    columns="d", value=100
)

proposal = validation.suggest_improvements(model="anthropic:claude-opus-4-8")
print(proposal.diff())
improved = proposal.accept()
```
