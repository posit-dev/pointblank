# text.in_language()

Make a validation step that checks that text is in the expected language.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/text.py#L297-L425)

``` python
text.in_language(
    column,
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
)
```

The step is a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`langid()`](../reference/scorers.langid.md#pointblank.scorers.langid) scorer: a language identification model gives the probability that the text of each row is in one of the expected languages, and a row fails if the probability is below `min_confidence=`. The expected languages are the same for every row (`expected=`) or come from a column (`expected_column=`), for a corpus with documents in several languages.

## Parameters

`column: str`  
The column with the text.

`expected: str | list[str] | None = None`  
The expected language (an ISO 639 code like `"en"`), or a list of them.

`expected_column: str | None = None`  
A column with the expected language of each row, instead of `expected=`.

`min_confidence: float = 0.5`  
The smallest probability that a text is in the expected languages for it to pass.

`backend: str | Callable[[str], Any] = "langid"`  
The language identification model: `"langid"`, `"fasttext"` (with the path of the model as `model=`), or a function that gives the probabilities of a text by language (see [`langid()`](../reference/scorers.langid.md#pointblank.scorers.langid)).

`model: str | None = None`  
The path of the fastText model, with `backend="fasttext"`.

`na_pass: bool = False`  
Whether a row with a missing value passes.

`cache: bool | str = True`  
Where the scores are kept: in memory (`True`), in a directory, or not at all (`False`).

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
    .add_steps(pb.text.in_language("text", expected="en", backend=guess))
    .add_steps(pb.text.in_language("text", expected_column="lang", backend=guess))
    .interrogate()
)
```

[TABLE]

The document in French fails step 1, which expects English for every row. In step 2, each row is checked against its own label, and the last document fails: it's labelled as French, but it's in English.
