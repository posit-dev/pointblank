# Validate.col_vals_token_len()

Validate whether the length of text in tokens is within bounds.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L16297-L16455)

``` python
Validate.col_vals_token_len(
    columns,
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
    dimension=None,
)
```

A language model reads text as tokens, and most have a limit on how many they can take in, so the documents of a corpus for training (or a set of prompts) need to be neither too short nor too long, in the tokens of the model's tokenizer. The `col_vals_token_len()` validation method counts the tokens of the text of each row, and a row fails if its count is below `min_val=` or above `max_val=`. Each row is a test unit. To check the length in characters instead, use [col_vals_str_len()](../reference/Validate.col_vals_str_len.md#pointblank.Validate.col_vals_str_len).

The step is a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`token_count()`](../reference/scorers.token_count.md#pointblank.scorers.token_count) scorer (it's listed in the report as `col_vals_scored()`, with a brief that says what it checks), so the counts are in the step report and in the extracts of failing rows.

## Parameters

`columns: str | list[str]`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column.

`tokenizer: str | Any`  
The tokenizer: the name of a Hugging Face tokenizer, `"tiktoken:"` and the name of an encoding or a model, `"whitespace"`, a tokenizer object, or a function (see [`token_count()`](../reference/scorers.token_count.md#pointblank.scorers.token_count)).

`min_val: int | None = None`  
The fewest and the most tokens that a text can have (at least one of them is needed).

`max_val: int | None = None`  
The fewest and the most tokens that a text can have (at least one of them is needed).

`na_pass: bool = False`  
Should a missing value pass? The default is `False`.

`cache: bool | str | Path = True`  
Where the counts are kept, so that a validation that's run again doesn't compute them again: in memory (`True`, the default), in a directory, or not at all (`False`).

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
The brief of the validation step, displayed in the reporting table. By default (`None` or `True`), the brief says what the step checks (e.g., "Between 3 and 4,096 tokens in `text`."). A string is used instead, with `"{auto}"` replaced by that description (and `"{step}"` by the step number). Use `False` for no brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Raises

`TypeError`  
If `min_val=` or `max_val=` isn't a number, or `tokenizer=` isn't a tokenizer.

`ValueError`  
If neither `min_val=` nor `max_val=` is given, or `min_val=` is greater than `max_val=`.

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
    .col_vals_token_len(columns="prompt", tokenizer="whitespace", min_val=3)
    .interrogate()
)
```

[TABLE]

The second prompt has a single token, so it fails.
