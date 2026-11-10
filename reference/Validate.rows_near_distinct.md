# Validate.rows_near_distinct()

Validate whether the text of rows has no near duplicates.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15115-L15345)

``` python
Validate.rows_near_distinct(
    columns,
    method="minhash",
    threshold=0.85,
    ngram=5,
    num_perm=128,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The `rows_near_distinct()` validation method finds rows whose text is the same as, or nearly the same as, another row's: the same document crawled twice, a template filled in with different names, or a passage with a changed footer. Such near duplicates make up a large share of many text corpora, and a model trained on them gives too much weight to the repeated text. Each row is a test unit, and every row in a group of near duplicates fails. Unlike [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct), which compares values exactly, the text is compared once normalized (in lower case, without punctuation or extra whitespace), and with `method="minhash"` (the default), texts that share most of their words in the same order are near duplicates too.

## Parameters

`columns: str | list[str]`  
The column (or a list of columns) with the text of each row. The values of several columns are joined by newlines.

`method: Literal["minhash", "exact"] = "minhash"`  
How texts are compared: `"minhash"` (the default) finds texts whose word n-grams are similar (see the *How Texts Are Compared* section) and `"exact"` finds texts that are the same once normalized.

`threshold: float = 0.85`  
With `method="minhash"`, the lowest Jaccard similarity of the word n-grams of two texts at which they are near duplicates, between `0` and `1`. The default, `0.85`, finds texts that differ by a few words.

`ngram: int = 5`  
With `method="minhash"`, the number of words in each n-gram (the default is `5`). A text shorter than this is one n-gram of all its words.

`num_perm: int = 128`  
With `method="minhash"`, the number of hash functions of the MinHash signatures (the default is `128`). More find near duplicates more reliably, but take longer.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

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
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"uniqueness"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Raises

`TypeError`  
If `columns=` isn't a column name or a list of them, or if `threshold=`, `ngram=`, or `num_perm=` isn't a number.

`ValueError`  
If `method=` isn't `"minhash"` or `"exact"`, if `threshold=` isn't between `0` and `1`, or if `ngram=` or `num_perm=` is less than `1`.

## How Texts Are Compared

Texts are first normalized: in lower case (with Unicode compatibility forms folded), with punctuation replaced by spaces, and with runs of whitespace made one space. Texts that are the same once normalized are always near duplicates. With `method="minhash"`, each text is also split into its overlapping n-grams of `ngram=` words, and two texts whose sets of n-grams have a Jaccard similarity of at least `threshold=` (the share of their n-grams that they have in common) are near duplicates. Comparing every pair of texts would take too long for a large table, so candidate pairs are found with MinHash signatures and locality-sensitive hashing (the way that corpora for language models are deduplicated), and each candidate pair is confirmed with its exact Jaccard similarity. No pair below the threshold is ever grouped, and a pair just above it is missed only rarely.

Groups of near duplicates are connected: if text A is a near duplicate of text B, and B of C, then A, B, and C are one group, even if A and C are less alike. A row with a missing value in any of `columns=` passes.

The group of each failing row (numbered from 1, in the order of the first row of each group) is in a column of the results called `pb_cluster_`, which is in the extracts of failing rows and in the step report, where the rows of each group are shown together. The step's note says how many groups were found and how many rows are in them.

Every row of a group fails, as with [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct). When the rows of a validation are used to clean a table (with [`get_row_flags()`](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags), or in an [`Ablation`](../reference/Ablation.md#pointblank.Ablation)), `duplicates=` decides which rows of a group are kept: `"keep_first"` keeps the first row of each group, `"keep_last"` the last, and `"flag_all"` keeps none.

Comparing texts runs in Python. A Polars or pandas table is compared as it is (a Polars LazyFrame is collected first). For a database table (through Ibis) or a PySpark DataFrame, the distinct rows of `columns=` are collected and compared, and the groups are joined back to the table on its own backend.

## Examples

For the examples here, we'll use a small table of documents, some of which are near duplicates of others.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "doc_id": [1, 2, 3, 4, 5, 6],
        "text": [
            "The library opens at nine in the morning and closes at six in the evening.",
            "The trail to the summit is steep, so bring water and good walking shoes.",
            "The library opens at nine in the morning and closes at six in the evening!",
            "Tickets for the museum are free for children under the age of twelve.",
            "the LIBRARY opens at nine in the morning, and closes at six in the evening",
            "The trail to the summit is steep, so bring water and sturdy walking shoes.",
        ],
    }
)

pb.preview(tbl)
```

|  | doc_id *Int64* | text *String* |
|----|----|----|
| 1 | 1 | The library opens at nine in the morning and closes at six in the evening. |
| 2 | 2 | The trail to the summit is steep, so bring water and good walking shoes. |
| 3 | 3 | The library opens at nine in the morning and closes at six in the evening! |
| 4 | 4 | Tickets for the museum are free for children under the age of twelve. |
| 5 | 5 | the LIBRARY opens at nine in the morning, and closes at six in the evening |
| 6 | 6 | The trail to the summit is steep, so bring water and sturdy walking shoes. |

Let's validate that the documents have no near duplicates.

``` python
validation = (
    pb.Validate(data=tbl)
    .rows_near_distinct(columns="text")
    .interrogate()
)

validation
```

[TABLE]

Three of the six rows fail: documents 1, 3, and 5 differ only in their punctuation and letter case, so they're one group. Documents 2 and 6 differ by a word, which leaves fewer than 85% of their 5-word n-grams in common, so they pass. A lower `threshold=` (or a smaller `ngram=`) finds them too:

``` python
validation = (
    pb.Validate(data=tbl)
    .rows_near_distinct(columns="text", threshold=0.5)
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `max jaccard(text) < 0.5` **5** / **6** TEST UNIT FAILURES EXTRACT OF ALL **5** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | doc_id *Int64* | text *String* | pb_cluster\_ *Int64* |
| 1 | 1 | The library opens at nine in the morning and closes at six in the evening. | 1 |
| 3 | 3 | The library opens at nine in the morning and closes at six in the evening! | 1 |
| 5 | 5 | the LIBRARY opens at nine in the morning, and closes at six in the evening | 1 |
| 2 | 2 | The trail to the summit is steep, so bring water and good walking shoes. | 2 |
| 6 | 6 | The trail to the summit is steep, so bring water and sturdy walking shoes. | 2 |

The step report shows the rows of each group together, with the group in the `pb_cluster_` column. Now five rows fail, in two groups: the library documents and the trail documents.
