# Validate.col_vals_in_language()

Validate whether text is in the expected language.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L16070-L16266)

``` python
Validate.col_vals_in_language(
    columns,
    expected=None,
    expected_column=None,
    min_confidence=0.5,
    backend="langid",
    model=None,
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

The `col_vals_in_language()` validation method checks that the text of each row is in one of the expected languages. A language identification model gives the probability that the text is in each language, and a row fails if the probability of the expected ones is below `min_confidence=`. Each row is a test unit. The expected languages are the same for every row (`expected=`) or come from a column (`expected_column=`), for a corpus with documents in several languages.

The step is a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`langid()`](../reference/scorers.langid.md#pointblank.scorers.langid) scorer (it's listed in the report as `col_vals_scored()`, with a brief that says what it checks), so the probabilities are in the step report and in the extracts of failing rows.

## Parameters

`columns: str | list[str]`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column.

`expected: str | list[str] | None = None`  
The expected language (an ISO 639 code like `"en"`), or a list of them.

`expected_column: str | None = None`  
A column with the expected language of each row, instead of `expected=`.

`min_confidence: float = 0.5`  
The smallest probability that a text is in the expected languages for it to pass, from 0 to 1. The default is `0.5`.

`backend: str | Callable[[str], Any] = "langid"`  
The language identification model: `"langid"` (the default, which needs the [langid](../reference/scorers.langid.md#pointblank.scorers.langid) package), `"fasttext"` (with the path of the model as `model=`), or a function that gives the probabilities of a text by language (see [`langid()`](../reference/scorers.langid.md#pointblank.scorers.langid)).

`model: str | None = None`  
The path of the fastText model, with `backend="fasttext"`.

`na_pass: bool = False`  
Should a missing value pass? The default is `False`.

`cache: bool | str | Path = True`  
Where the probabilities are kept, so that a validation that's run again doesn't compute them again: in memory (`True`, the default), in a directory, or not at all (`False`).

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
The brief of the validation step, displayed in the reporting table. By default (`None` or `True`), the brief says what the step checks (e.g., "The text of `text` is in `en` (probability of at least 0.5)."). A string is used instead, with `"{auto}"` replaced by that description (and `"{step}"` by the step number). Use `False` for no brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Raises

`TypeError`  
If `min_confidence=` isn't a number.

`ValueError`  
If `min_confidence=` isn't between 0 and 1, if neither (or both) of `expected=` and `expected_column=` is given, or if `expected_column=` is one of `columns=`.

## Examples

A real check would use the [langid](../reference/scorers.langid.md#pointblank.scorers.langid) package or a fastText model. To keep this example self-contained, the backend is a toy function that guesses the language from a few words.

``` python
import pointblank as pb
import polars as pl

def guess(text):
    words = set(text.lower().split())
    if words & {"the", "and", "is"}:
        return {"en": 0.9, "fr": 0.1}
    if words & {"le", "et", "est"}:
        return {"en": 0.1, "fr": 0.9}
    return {"en": 0.5, "fr": 0.5}

docs = pl.DataFrame(
    {
        "text": ["the soup is good", "le pain est bon", "the bread is fresh"],
        "lang": ["en", "fr", "fr"],
    }
)

(
    pb.Validate(data=docs)
    .col_vals_in_language(columns="text", expected="en", backend=guess)
    .col_vals_in_language(columns="text", expected_column="lang", backend=guess)
    .interrogate()
)
```

[TABLE]

The document in French fails step 1, which expects English for every row. In step 2, each row is checked against its own label, and the last document fails: it's labelled as French, but it's in English.
