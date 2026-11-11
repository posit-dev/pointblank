# Validate.rows_not_contaminated()

Validate whether the text of rows doesn't overlap with a reference, such as a benchmark.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15376-L15625)

``` python
Validate.rows_not_contaminated(
    columns,
    reference,
    reference_columns=None,
    method="ngram",
    ngram=None,
    threshold=0.8,
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

The `rows_not_contaminated()` validation method checks that the text of each row doesn't overlap with the texts of a reference: the questions of a benchmark that a model will be evaluated on, the rows of a held-out test set, or any text that shouldn't be in the data. Training data that contains the questions of a benchmark *contaminates* it, and the model's score on the benchmark no longer says how well it does on questions it hasn't seen. Each row is a test unit, and a row whose text overlaps with any text of the reference fails.

## Parameters

`columns: str | list[str]`  
The column (or a list of columns) with the text of each row. The values of several columns are joined by newlines.

`reference: Any`  
The texts to check against: a table (any table that [Validate](../reference/Validate.md#pointblank.Validate) can take, including the path of a CSV or Parquet file), a list of tables or a dictionary of them by name (so that the note can say which reference each overlap is with), or a list of texts.

`reference_columns: str | list[str] | None = None`  
The column (or a list of columns) of a reference table with its texts. The default is the columns of `columns=`.

`method: Literal["ngram", "exact", "minhash"] = "ngram"`  
How texts overlap: `"ngram"` (the default) if they share an n-gram of `ngram=` words, `"exact"` if they're the same once normalized, and `"minhash"` if the Jaccard similarity of their n-grams is at least `threshold=`. See the *How Overlap Is Found* section.

`ngram: int | None = None`  
The number of words in each n-gram. The default is `13` with `method="ngram"` (the length used to check training data for benchmark contamination) and `5` with `method="minhash"`.

`threshold: float = 0.8`  
With `method="minhash"`, the lowest Jaccard similarity of the n-grams of a row's text and a reference text at which they overlap, between `0` and `1` (the default is `0.8`).

`num_perm: int = 128`  
With `method="minhash"`, the number of hash functions of the MinHash signatures (the default is `128`).

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
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Raises

`TypeError`  
If `columns=` or `reference_columns=` isn't a column name or a list of them, or if `ngram=`, `threshold=`, or `num_perm=` isn't a number.

`ValueError`  
If `method=` isn't `"ngram"`, `"exact"`, or `"minhash"`, if `reference=` is `None` or empty, if `threshold=` isn't between `0` and `1`, or if `ngram=` or `num_perm=` is less than `1`.

## How Overlap Is Found

The texts of the rows and of the reference are normalized first: in lower case (with Unicode compatibility forms folded), with punctuation replaced by spaces, and with runs of whitespace made one space. With `method="ngram"`, a row's text overlaps with a reference text if they share any n-gram of `ngram=` words, which finds a benchmark question that's been pasted into a longer document. A reference text that's shorter than `ngram=` words overlaps with a row's text that contains it whole, and a row's text that's shorter than `ngram=` words overlaps only with a reference text that's the same. With `method="exact"`, a row's text overlaps only with a reference text that's the same, and with `method="minhash"`, it overlaps with a reference text whose n-grams are similar enough (found with MinHash signatures and locality-sensitive hashing, and confirmed with the exact Jaccard similarity).

A row with a missing value in any of `columns=` passes. The first reference text that a failing row overlaps with is in a column of the results called `pb_match_`, which is in the extracts of failing rows and in the step report, and the step's note says how many rows overlap with each reference.

The reference is read when the validation is interrogated, and its texts are kept in memory, so a very large reference takes a lot of memory. The texts of the rows are compared in Python: for a database table (through Ibis) or a PySpark DataFrame, the distinct rows of `columns=` are collected and compared, and the matches are joined back to the table on its own backend.

## Examples

For the examples here, we'll use a small table of training documents and a benchmark of two questions. One document contains the first question of the benchmark, word for word.

``` python
import pointblank as pb
import polars as pl

train = pl.DataFrame(
    {
        "doc_id": [1, 2, 3, 4],
        "text": [
            "Rivers carry water from the hills down to the sea over many years.",
            "Quiz: What is the name of the largest moon of the planet Saturn in our solar "
            "system? Answer: Titan.",
            "Bread is made of flour, water, salt, and yeast, and baked in an oven.",
            "The train leaves the station at eight and arrives in the city at ten.",
        ],
    }
)

benchmark = pl.DataFrame(
    {
        "question": [
            "What is the name of the largest moon of the planet Saturn in our solar "
            "system?",
            "How many bones are there in the body of an adult human being?",
        ]
    }
)
```

Let's validate that no training document overlaps with the benchmark's questions.

``` python
validation = (
    pb.Validate(data=train)
    .rows_not_contaminated(
        columns="text", reference=benchmark, reference_columns="question"
    )
    .interrogate()
)

validation
```

[TABLE]

The second document fails: it shares a run of 13 words (in fact, the whole question) with the benchmark. The step report shows the benchmark question that it matched, in the `pb_match_` column.

``` python
validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `13-grams(text) ∩ reference = ∅` **1** / **4** TEST UNIT FAILURES EXTRACT OF **1** ROW (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | doc_id *Int64* | text *String* | pb_match\_ *String* |
| 2 | 2 | Quiz: What is the name of the largest moon of the planet Saturn in our solar system? Answer: Titan. | What is the name of the largest moon of the planet Saturn in our solar system? |

The reference can also be a list of texts, or a dictionary of tables by name (such as `{"mmlu": ..., "gsm8k": ...}`), and the step's note then says how many rows overlap with each.
