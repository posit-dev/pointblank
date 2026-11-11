# Validate.col_vals_text_quality()

Validate whether text passes the heuristic quality filters of web text.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15873-L16090)

``` python
Validate.col_vals_text_quality(
    columns,
    min_words=50,
    max_words=100000,
    min_mean_word_length=3,
    max_mean_word_length=10,
    max_symbol_ratio=0.1,
    max_bullet_lines=0.9,
    max_ellipsis_lines=0.3,
    min_alpha_words=0.8,
    min_stop_words=2,
    max_repeated_lines=0.3,
    stop_words=None,
    na_pass=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

Text collected from the web for training language models has a lot that isn't prose: menus, lists of links, tables of numbers, placeholder text, and pages that repeat themselves. The quality filters of the Gopher models (Rae et al., 2021) catch much of it with a few simple statistics of each document, and the `col_vals_text_quality()` validation method adds a validation step for each of them, with the bounds used for Gopher as the defaults:

- the number of words is between `min_words=` and `max_words=` (50 and 100,000)
- the mean length of the words is between `min_mean_word_length=` and `max_mean_word_length=` (3 and 10 characters)
- there are no more than `max_symbol_ratio=` hash symbols and ellipses for each word (0.1)
- no more than `max_bullet_lines=` of the lines start with a bullet point (90%)
- no more than `max_ellipsis_lines=` of the lines end with an ellipsis (30%)
- at least `min_alpha_words=` of the words have a letter (80%)
- the text has at least `min_stop_words=` distinct stop words (2 of the, be, to, of, and, that, have, and with)
- no more than `max_repeated_lines=` of the lines repeat an earlier line (30%)

Each check is a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`text_stat()`](../reference/scorers.text_stat.md#pointblank.scorers.text_stat) scorer, so the report shows each one, with its own counts of the documents that fail, and a brief that says what it checks. Each row is a test unit. Set a bound to `None` to leave its check out (a check with two bounds is left out when both are `None`). Documents without the words or lines that a check needs pass it (the check of the number of words catches them).

The defaults are for English web text. Text in other languages needs other stop words (and perhaps other bounds), and short texts (like titles or chat messages) need a lower `min_words=`.

## Parameters

`columns: str | list[str]`  
A single column or a list of columns to validate. There will be a separate set of validation steps for each column.

`min_words: int | None = 50`  
The fewest and the most words a document can have.

`max_words: int | None = 50`  
The fewest and the most words a document can have.

`min_mean_word_length: float | None = 3`  
The bounds of the mean length of the words of a document, in characters.

`max_mean_word_length: float | None = 3`  
The bounds of the mean length of the words of a document, in characters.

`max_symbol_ratio: float | None = 0.1`  
The most hash symbols and ellipses that a document can have for each word.

`max_bullet_lines: float | None = 0.9`  
The largest share of the lines of a document that can start with a bullet point.

`max_ellipsis_lines: float | None = 0.3`  
The largest share of the lines of a document that can end with an ellipsis.

`min_alpha_words: float | None = 0.8`  
The smallest share of the words of a document that must have a letter.

`min_stop_words: int | None = 2`  
The fewest distinct stop words that a document must have.

`max_repeated_lines: float | None = 0.3`  
The largest share of the lines of a document that can be the same as an earlier line.

`stop_words: Sequence[str] | None = None`  
The stop words, for `min_stop_words=`. By default, they're the eight English words of the Gopher filters.

`na_pass: bool = True`  
Should a missing value pass? The default is `True`, so that missing documents are left to a check like [col_vals_not_null()](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null).

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
The brief of each validation step, displayed in the reporting table. By default (`None` or `True`), each step's brief says what it checks (e.g., "Between 50 and 100,000 words in `text`."). A string is used for every step, with `"{auto}"` replaced by that description (and `"{step}"` by the step number). Use `False` for no brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation steps should be active. Using `False` will make the validation steps inactive (still reporting their presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing.

`dimension: str | None = None`  
An optional data quality dimension to categorize the validation steps for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation steps.

## Raises

`TypeError`  
If a bound isn't a number or `None`.

`ValueError`  
If `min_words=` is greater than `max_words=`, or `min_mean_word_length=` is greater than `max_mean_word_length=`.

## Examples

Let's check a small corpus of web documents. The documents are short, so we'll use a lower `min_words=` than the default.

``` python
import pointblank as pb
import polars as pl

corpus = pl.DataFrame(
    {
        "text": [
            "The museum is open from nine to five, and the entry is free for children "
            "under the age of twelve, so it's a good place to spend the day with family.",
            "Home\nAbout\nHome\nAbout\nHome\nContact us",
            "#sale #deals #shoes #free #today ... click to see more of the best deals",
            "Our library lends books, films, and music to members of the town for up to "
            "three weeks, and the books can be returned to any of its branches.",
        ]
    }
)

(
    pb.Validate(data=corpus)
    .col_vals_text_quality(columns="text", min_words=10)
    .interrogate()
)
```

[TABLE]

The menu in the second row has too few words and too many repeated lines, and few stop words. The third row, a list of hashtags, has too many symbols for its words. The two documents of prose pass every check.
