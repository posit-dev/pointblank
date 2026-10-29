# Validate.suggest_improvements()

Propose AI-generated improvements to this validation plan.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20286-L20353)

``` python
Validate.suggest_improvements(
    model,
    api_key=None,
    verify_ssl=True,
    max_reprompts=1,
)
```

This is a thin, convenience wrapper over [`EditValidation`](../reference/EditValidation.md#pointblank.EditValidation): it profiles the table with [`DataScan`](../reference/DataScan.md#pointblank.DataScan), derives an instruction that targets gaps in the current plan (columns with no coverage, missing thresholds), and asks the model to extend the plan accordingly. As with [EditValidation](../reference/EditValidation.md#pointblank.EditValidation), you review the proposed change as a diff and explicitly accept it.

## Parameters

`model: str`  
The model to use, in `provider:model` form (e.g., `"anthropic:claude-opus-4-8"`).

`api_key: str | None = None`  
The API key for the model provider.

`verify_ssl: bool = True`  
Whether to verify SSL certificates for provider requests. Defaults to `True`.

`max_reprompts: int = 1`  
Maximum automatic re-prompts if the returned plan fails the syntax check.

## Returns

`EditValidation`  
An [`EditValidation`](../reference/EditValidation.md#pointblank.EditValidation) with the proposed improvements, ready to inspect via `.diff()`/`.changed_steps()` and finalize via `.accept()`.

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

This example isn't executed here because it needs an API key and a call to the model provider. The plan above only checks column `d`, so the instruction sent to the model asks it to cover the other columns and to add thresholds. Printing `proposal.diff()` shows the proposed steps as a diff against the current plan, so you can review them before anything runs. If they look right, [accept()](../reference/EditValidation.md#pointblank.EditValidation.accept) returns the revised plan as a new [Validate](../reference/Validate.md#pointblank.Validate) object. That object hasn't been interrogated yet, so call [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) on it to run the checks. The original `validation` object is left unchanged.
