# text.token_length()

Make a validation step that checks the length of text in tokens.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/text.py#L428-L528)

``` python
text.token_length(
    column,
    tokenizer,
    min_val=None,
    max_val=None,
    na_pass=False,
    cache=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
)
```

A language model reads text as tokens, and most have a limit on how many they can take in, so the documents of a corpus for training (or a set of prompts) need to be neither too short nor too long, in the tokens of the model's tokenizer. The step is a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`token_count()`](../reference/scorers.token_count.md#pointblank.scorers.token_count) scorer, and a row fails if its number of tokens is below `min_val=` or above `max_val=`.

## Parameters

`column: str`  
The column with the text.

`tokenizer: str | Any`  
The tokenizer: the name of a Hugging Face tokenizer, `"tiktoken:"` and the name of an encoding or a model, `"whitespace"`, a tokenizer object, or a function (see [`token_count()`](../reference/scorers.token_count.md#pointblank.scorers.token_count)).

`min_val: int | None = None`  
The fewest and the most tokens that a text can have (at least one of them is needed).

`max_val: int | None = None`  
The fewest and the most tokens that a text can have (at least one of them is needed).

`na_pass: bool = False`  
Whether a row with a missing value passes.

`cache: bool | str = True`  
Where the counts are kept: in memory (`True`), in a directory, or not at all (`False`).

`pre: Callable | None = None`  
A function that transforms the table before the step, as in the validation methods.

`segments: SegmentSpec | None = None`  
The segments of the step, as in the validation methods.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
The thresholds of the step (by default, those of the validation).

`actions: Actions | None = None`  
The actions of the step.

`brief: str | bool | None = None`  
A brief for the step.

`active: bool | Callable = True`  
Whether the step is active.

## Returns

`Steps`  
The step, to be added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Examples

A real check would use the tokenizer of a model (e.g., `"tiktoken:cl100k_base"`). Here, the tokens are the words, split on whitespace.

``` python
import pointblank as pb
import polars as pl

prompts = pl.DataFrame(
    {"prompt": ["Summarize the article in two sentences.", "Hi", "List three uses of salt."]}
)

(
    pb.Validate(data=prompts)
    .add_steps(pb.text.token_length("prompt", tokenizer="whitespace", min_val=3))
    .interrogate()
)
```

[TABLE]

The second prompt has a single token, so it fails.
