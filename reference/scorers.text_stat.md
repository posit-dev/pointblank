# scorers.text_stat()

Score text by a simple statistic, like its number of words or its share of bullet lines.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L1353-L1447)

``` python
scorers.text_stat(
    stat,
    column=None,
    stop_words=None,
    name=None,
    version=None,
    batch_size=None,
)
```

These are the statistics of the quality filters that are used to clean web text for language models (those of the Gopher models, by Rae et al., 2021). They need no model, so they're fast, and they catch much of the text that isn't prose: lists of links, menus, tables of numbers, and boilerplate. [col_vals_text_quality()](../reference/Validate.col_vals_text_quality.md#pointblank.Validate.col_vals_text_quality) adds a step for each of them, with the bounds used for Gopher.

Words are the text split on whitespace, and lines are the text split on newlines (blank lines are left out). The statistics are:

- `"char_count"`: the number of characters
- `"word_count"`: the number of words
- `"mean_word_length"`: the mean number of characters of the words
- `"symbol_ratio"`: the number of hash symbols (`#`) and ellipses for each word
- `"bullet_lines"`: the share of lines that start with a bullet point
- `"ellipsis_lines"`: the share of lines that end with an ellipsis
- `"alpha_words"`: the share of words with at least one letter
- `"stop_words"`: the number of distinct stop words in the text
- `"repeated_lines"`: the share of lines that are the same as an earlier line

A text without words has no score for the statistics that are shares of its words (and without lines, for those that are shares of its lines).

## Parameters

`stat: str`  
The statistic, one of those above.

`column: str | None = None`  
The column with the text to score. It can be left out if the step has one column.

`stop_words: list[str] | None = None`  
The stop words, with `stat="stop_words"`. By default, they're the eight English words of the Gopher filters (`the`, `be`, `to`, `of`, `and`, `that`, `have`, and `with`).

`name: str | None = None`  
A name for the scorer, for the score cache and the report. By default, it's made from the statistic and its settings.

`version: str | None = None`  
A version, which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of texts scored at a time (`1024` by default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Examples

Let's check that each document of a small corpus has at least eight words, and that no more than half of its lines are bullet points.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "text": [
            "The museum opens at nine and closes at five on weekdays.",
            "Home | About | Contact",
            "- Bread\n- Milk\n- Eggs\nA short list of what to buy for the weekend.",
        ]
    }
)

(
    pb.Validate(data=tbl)
    .col_vals_scored(columns="text", scorer=pb.scorers.text_stat("word_count"), min_val=8)
    .col_vals_scored(
        columns="text", scorer=pb.scorers.text_stat("bullet_lines"), max_val=0.5
    )
    .interrogate()
)
```

[TABLE]

The menu in the second row has only five words, so it fails step 1, and three of the four lines of the shopping list are bullet points, so it fails step 2.
