# scorers.perplexity()

Score text by its perplexity under a language model.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L1149-L1224)

``` python
scorers.perplexity(
    fn=None,
    kenlm_model=None,
    column=None,
    per_row=True,
    name=None,
    version=None,
    batch_size=None,
)
```

Perplexity measures how surprising a text is to a language model: it's low for fluent text like the text the model was trained on, and high for garbled text, boilerplate in another language, or encoding errors. A check of perplexity keeps the text that's fluent, with `max_val=` as the highest perplexity allowed. The model is a function given as `fn=` (e.g., one that runs a small causal language model) or a [KenLM](https://github.com/kpu/kenlm) model, which needs the `kenlm` package.

## Parameters

`fn: Callable[…, Any] | None = None`  
A function that gives the perplexity of a text (or with `per_row=False`, of each of a list of texts).

`kenlm_model: str | None = None`  
The path of a KenLM model (an `.arpa` or `.bin` file), instead of `fn=`.

`column: str | None = None`  
The column with the text to score. It can be left out if the step has one column.

`per_row: bool = True`  
Whether `fn=` takes one text (`True`, the default) or a list of texts (`False`).

`name: str | None = None`  
A name for the scorer, for the score cache and the report. By default, it's made from the function (if it's defined at the top level of a module) or the path of the KenLM model.

`version: str | None = None`  
A version (e.g., of the model), which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of texts scored at a time (`256` by default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Examples

A real model would be a language model. To keep this example self-contained, the "perplexity" of a text here is higher when it has fewer vowels, as garbled text often does.

``` python
import pointblank as pb
import polars as pl

def toy_perplexity(text):
    letters = [c for c in text.lower() if c.isalpha()]
    vowels = sum(c in "aeiou" for c in letters)
    return len(letters) / max(vowels, 1)

tbl = pl.DataFrame(
    {"text": ["a plain sentence", "xkcd qwrt zzpf", "another fine line", "ok then"]}
)

pb.Validate(data=tbl).col_vals_scored(
    columns="text", scorer=pb.scorers.perplexity(fn=toy_perplexity), max_val=4
).interrogate()
```

[TABLE]

The garbled text has the highest score, above `max_val=4`, so it's the row that fails.
