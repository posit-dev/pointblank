## Validate.from_prompt()


Build a validation plan for this table from a natural-language prompt.


Usage

``` python
Validate.from_prompt(
    prompt, model, api_key=None, verify_ssl=True, max_reprompts=1
)
```


This is the same AI edit flow as <a href="EditValidation.html#pointblank.EditValidation" class="gdls-link"><code>EditValidation</code></a> but starting from an *empty* plan: the model is given a bare `pb.Validate(...)` (carrying this object's table name, label, and thresholds) plus a [DataScan](DataScan.md#pointblank.DataScan) profile of the data, and is asked to author steps that satisfy the prompt.


## Parameters


`prompt: str`  
A plain-English description of the checks the plan should perform (e.g., "ensure no nulls in any id column and that ids are unique").

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
An <a href="EditValidation.html#pointblank.EditValidation" class="gdls-link"><code>EditValidation</code></a> whose revised plan realizes the prompt; inspect it with `.diff()`/`.to_code()` and finalize with `.accept()`.


## Examples

``` python
import pointblank as pb

base = pb.Validate(data=pb.load_dataset("small_table"), tbl_name="small_table")
proposal = base.from_prompt(
    "ensure column a has no nulls and column d is always positive",
    model="anthropic:claude-opus-4-8",
)
plan = proposal.accept()
```
