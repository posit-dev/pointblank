# scorers.langid()

Score text by how likely it is to be in the expected languages.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L1079-L1184)

``` python
scorers.langid(
    expected=None,
    backend="langid",
    model=None,
    column=None,
    expected_column=None,
    name=None,
    version=None,
    batch_size=None,
)
```

The score of a row is the probability, according to a language identification model, that its text is in one of the expected languages. It's a check that a corpus or a column of free text is in the languages it should be, with `min_val=` as the confidence needed (e.g., `0.5`).

## Parameters

`expected: str | list[str] | None = None`  
The expected language (an ISO 639 code like `"en"`), or a list of them. A code matches the model's languages with the same code or with a region or script after it (`"en"` matches `"en_US"`). Give this or `expected_column=`.

`backend: str | Callable[[str], Any] = "langid"`  
The language identification model: `"langid"` (the default, which needs the [langid](../reference/scorers.langid.md#pointblank.scorers.langid) package), `"fasttext"` (a fastText model, which needs the `fasttext` package and the path of the model as `model=`), or a function that takes a text and returns its probabilities by language (a dictionary or a list of `(language, probability)` pairs).

`model: str | None = None`  
The path of the fastText model (e.g., `lid.176.bin`), with `backend="fasttext"`.

`column: str | None = None`  
The column with the text to score. It can be left out if the step has one column (or two, with one of them as `expected_column=`).

`expected_column: str | None = None`  
A column with the expected language of each row, instead of `expected=` (for a multilingual corpus, where each document is labelled with its language). The column must be one of the columns of the step.

`name: str | None = None`  
A name for the scorer, for the score cache and the report. By default, it's made from the expected languages (or `expected_column=`), the backend, and the model.

`version: str | None = None`  
A version, which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of texts scored at a time (`256` by default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

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

tbl = pl.DataFrame(
    {"review": ["the soup is good", "le pain est bon", "and the bread", "ok"]}
)

pb.Validate(data=tbl).col_vals_scored(
    columns="review", scorer=pb.scorers.langid(expected="en", backend=guess), min_val=0.6
).interrogate()
```

[TABLE]

The review in French fails, and so does `"ok"`, which can't be told apart. In a corpus with documents in several languages, the expected language of each row can come from a column, with `expected_column=`:

``` python
tbl = pl.DataFrame(
    {
        "review": ["the soup is good", "le pain est bon", "le vin est bon"],
        "lang": ["en", "fr", "en"],
    }
)

pb.Validate(data=tbl).col_vals_scored(
    columns=["review", "lang"],
    scorer=pb.scorers.langid(expected_column="lang", backend=guess),
    min_val=0.6,
).interrogate()
```

[TABLE]

The third review is labelled as English, but it's in French, so it fails.
