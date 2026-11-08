# scorers.token_count()

Score text by its number of tokens.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L1504-L1589)

``` python
scorers.token_count(
    tokenizer,
    column=None,
    name=None,
    version=None,
    batch_size=None,
)
```

A language model reads text as tokens, and most have a limit on how many they can take, so a corpus for training (or a set of prompts) usually needs texts with neither too few tokens nor too many. The score of a row is the number of tokens of its text, with the tokenizer of a model, which makes `min_val=` and `max_val=` bounds on the length of each text in tokens.

## Parameters

`tokenizer: str | Any`  
The tokenizer. It can be:

- the name of a Hugging Face tokenizer (e.g., `"gpt2"` or `"bert-base-uncased"`), which needs the `transformers` package (special tokens aren't counted)
- `"tiktoken:"` and the name of a `tiktoken` encoding or model (e.g., `"tiktoken:cl100k_base"` or `"tiktoken:gpt-4o"`), which needs the `tiktoken` package
- `"whitespace"`, to count the words of the text, split on whitespace
- a tokenizer object with an `encode()` method (from `tiktoken`, `tokenizers`, or `transformers`)
- a function that takes a text and returns its tokens, or the number of them

`column: str | None = None`  
The column with the text to score. It can be left out if the step has one column.

`name: str | None = None`  
A name for the scorer, for the score cache and the report. A tokenizer given by name names the scorer; a tokenizer object or a function needs a `name=` for the scores to be kept on disk.

`version: str | None = None`  
A version, which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of texts scored at a time (`512` by default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Examples

A real check would use the tokenizer of a model, such as `"tiktoken:cl100k_base"`. To keep this example self-contained, the tokenizer here splits text into words and punctuation.

``` python
import re

import pointblank as pb
import polars as pl

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

tbl = pl.DataFrame(
    {
        "prompt": [
            "Summarize the article in two sentences.",
            "Hi",
            "Translate this paragraph into French, keeping the names as they are.",
        ]
    }
)

pb.Validate(data=tbl).col_vals_scored(
    columns="prompt",
    scorer=pb.scorers.token_count(tokenize),
    min_val=3,
    max_val=12,
).interrogate()
```

[TABLE]

The second prompt has one token and the third has thirteen, so both fail.
