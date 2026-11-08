# Validate.col_vals_scored()

Validate whether rows have a score from a model within bounds.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L14791-L15092)

``` python
Validate.col_vals_scored(
    columns,
    scorer,
    min_val=None,
    max_val=None,
    na_pass=False,
    cache=True,
    keep_scores=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The `col_vals_scored()` validation method gives each row a score with a model (a *scorer*) and checks that the score is at least `min_val=`, at most `max_val=`, or both. It's for the checks that are about meaning rather than form: whether a passage answers its query (a reranker), whether a document is of good quality or toxic (a classifier), whether text is in the expected language, or whether it's fluent (perplexity). Each row is a test unit, and the scorer gets the values of `columns=` for the row.

The scorer can be any function of a batch of rows (see the [`Scorer`](../reference/scorers.Scorer.md#pointblank.scorers.Scorer) protocol), or one of the adapters of `pb.scorers`, which use models from common packages: [`embedding_similarity()`](../reference/scorers.embedding_similarity.md#pointblank.scorers.embedding_similarity), [`reranker()`](../reference/scorers.reranker.md#pointblank.scorers.reranker), [`classifier()`](../reference/scorers.classifier.md#pointblank.scorers.classifier), [`langid()`](../reference/scorers.langid.md#pointblank.scorers.langid), and [`perplexity()`](../reference/scorers.perplexity.md#pointblank.scorers.perplexity). Pointblank never trains a model: the scorers use models that are given to them.

## Parameters

`columns: str | list[str]`  
The column (or a list of columns) whose values the scorer gets for each row. A reranker, for example, needs two columns: the query and the document.

`scorer: Callable[…, Any]`  
The scorer: a function that takes a list of rows (dictionaries of the values of `columns=`) and returns a score for each, or an adapter of `pb.scorers`. A score is a number, or `None` for a row that can't be scored.

`min_val: float | None = None`  
The lowest score that passes (inclusive).

`max_val: float | None = None`  
The highest score that passes (inclusive). At least one of `min_val=` and `max_val=` is needed.

`na_pass: bool = False`  
Should a row with a missing value in any of `columns=` (which isn't scored), or a row that the scorer gives no score, pass? The default is `False`, so such rows fail.

`cache: bool | str | Path = True`  
Where to keep the scores, so that rows aren't scored again: `True` (the default) keeps them in memory for the session, a path is the directory of a cache on disk (an SQLite file, `scores.sqlite`) that's kept between sessions, and `False` doesn't keep them. Scores are kept by the scorer's name and version, and by the values of the row. A cache on disk needs a scorer with a stable name (see the *Score Cache* section).

`keep_scores: bool = True`  
Should the results of the step keep the scores, in a column called `pb_score_`? With `True` (the default), the scores are in the extracts of failing rows, in the step report, and in [`get_row_flags()`](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags) (as `pb_score_<step>_`), and they can be used by [`Ablation.sweep_score()`](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score).

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Raises

`TypeError`  
If `columns=` isn't a column name or a list of them, if `scorer=` isn't callable, or if `min_val=` or `max_val=` isn't a number.

`ValueError`  
If neither `min_val=` nor `max_val=` is given, if `min_val=` is greater than `max_val=`, or if `cache=` is a directory and the scorer has no stable name.

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied, such as taking a sample of the rows to keep the cost of scoring down.

The preprocessing function can be any callable that takes a table as input and returns a modified table. Regardless of which is used, the function should return a table with the columns of `columns=`.

## How Rows Are Scored

The scorer gets the rows in batches (of the scorer's `batch_size`, or 256 rows), as dictionaries of the values of `columns=`. Rows with the same values are scored once, and a row with a missing value in any of `columns=` isn't scored at all (it passes or fails by `na_pass=`). A scorer that fails, or that returns something other than a number (or `None`) for each row, stops the step with an evaluation error and a note that says why, as does a scorer whose package isn't installed (the note says how to install it).

Scoring runs in Python, on the rows' values. A Polars or pandas table is scored as it is (a Polars LazyFrame is collected first). For a database table (through Ibis) or a PySpark DataFrame, the distinct rows of `columns=` are collected and scored, and the scores are joined back to the table on its own backend; other kinds of table give an evaluation error. The step's note says which scorer was used, how many rows it scored, how many scores came from the cache, and the range of the scores.

## Score Cache

Scores are kept in a cache by the scorer's name and version (its `name` and `version` attributes) and by the values of the row, so that interrogating the same data again (as an [`Ablation`](../reference/Ablation.md#pointblank.Ablation) does, or after a change to another step) doesn't run the model again. The scores are written after each batch, so a run that's interrupted picks up where it left off.

With `cache=True`, the scores are kept in memory until the end of the session (or until [`pb.scorers.clear_cache()`](../reference/scorers.clear_cache.md#pointblank.scorers.clear_cache)). With a directory as `cache=`, they're kept on disk and can be shared between sessions and projects; this needs a scorer with a stable name, which the adapters of `pb.scorers` have (unless they're given a model object), as does a function defined at the top level of a module. A lambda can be given a name with [`pb.scorers.from_function()`](../reference/scorers.from_function.md#pointblank.scorers.from_function). Change the version of a scorer when its model changes, so that older scores aren't used.

## Examples

For the examples here, we'll use a table of product reviews and a scorer that stands in for a model: it scores a review by the share of its words that are on a list of positive words. A real scorer would be a classifier, such as `pb.scorers.classifier(hf_model=..., label="POSITIVE")`.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "product": ["kettle", "kettle", "toaster", "toaster", "lamp", "lamp"],
        "review": [
            "great kettle and fast",
            "broke after a week",
            "good toast every time",
            "great value",
            "too dim",
            None,
        ],
    }
)

POSITIVE = {"great", "good", "fast", "value", "every"}

def positivity(batch):
    return [
        sum(w in POSITIVE for w in row["review"].split()) / len(row["review"].split())
        for row in batch
    ]

pb.preview(tbl)
```

|     | product *String* | review *String*       |
|-----|------------------|-----------------------|
| 1   | kettle           | great kettle and fast |
| 2   | kettle           | broke after a week    |
| 3   | toaster          | good toast every time |
| 4   | toaster          | great value           |
| 5   | lamp             | too dim               |
| 6   | lamp             | None                  |

Let's validate that each review scores at least `0.25`.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_scored(columns="review", scorer=positivity, min_val=0.25)
    .interrogate()
)

validation
```

[TABLE]

The step has six test units, one for each row, and three fail: two reviews have no positive words, and the missing review isn't scored (use `na_pass=True` to let such rows pass). The step's note gives the range of the scores. The failing rows can be seen with their scores, in the `pb_score_` column, with [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report).

``` python
validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `positivity(review) ≥ 0.25` **3** / **6** TEST UNIT FAILURES EXTRACT OF ALL **3** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | product *String* | review *String* | pb_score\_ *Float64* |
| 2 | kettle | broke after a week | 0.0 |
| 5 | lamp | too dim | 0.0 |
| 6 | lamp | None | None |

The scores of a scored step can be used to see how the threshold affects a model trained on the data, by removing more or fewer rows; see [`Ablation.sweep_score()`](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score).
