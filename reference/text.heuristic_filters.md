# text.heuristic_filters()

Make the heuristic quality filters of web text, as validation steps.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/text.py#L61-L294)

``` python
text.heuristic_filters(
    column,
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
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    active=True,
)
```

Text collected from the web for training language models has a lot that isn't prose: menus, lists of links, tables of numbers, placeholder text, and pages that repeat themselves. The quality filters of the Gopher models (Rae et al., 2021) catch much of it with a few simple statistics of each document, and this function makes a validation step for each of them, with the bounds used for Gopher as the defaults:

- the number of words is between `min_words=` and `max_words=` (50 and 100,000)
- the mean length of the words is between `min_mean_word_length=` and `max_mean_word_length=` (3 and 10 characters)
- there are no more than `max_symbol_ratio=` hash symbols and ellipses for each word (0.1)
- no more than `max_bullet_lines=` of the lines start with a bullet point (90%)
- no more than `max_ellipsis_lines=` of the lines end with an ellipsis (30%)
- at least `min_alpha_words=` of the words have a letter (80%)
- the text has at least `min_stop_words=` distinct stop words (2 of the, be, to, of, and, that, have, and with)
- no more than `max_repeated_lines=` of the lines repeat an earlier line (30%)

Each check is a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`text_stat()`](../reference/scorers.text_stat.md#pointblank.scorers.text_stat) scorer, so the report shows each one, with its own counts of the documents that fail, and a brief that says what it checks. Set a bound to `None` to leave its check out (a check with two bounds is left out when both are `None`). A document with a missing value passes, as do documents without the words or lines that a check needs (the check of the number of words catches them).

The defaults are for English web text. Text in other languages needs other stop words (and perhaps other bounds), and short texts (like titles or chat messages) need a lower `min_words=`.

## Parameters

`column: str`  
The column with the text.

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

`pre: Callable | None = None`  
A function that transforms the table before each step, as in the validation methods.

`segments: SegmentSpec | None = None`  
The segments of the steps, as in the validation methods.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
The thresholds of the steps (by default, those of the validation).

`actions: Actions | None = None`  
The actions of the steps.

`active: bool | Callable = True`  
Whether the steps are active.

## Returns

`Steps`  
The steps, to be added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

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
    .add_steps(pb.text.heuristic_filters("text", min_words=10))
    .interrogate()
)
```

[TABLE]

The menu in the second row has too few words and too many repeated lines, and few stop words. The third row, a list of hashtags, has too many symbols for its words. The two documents of prose pass every check.
