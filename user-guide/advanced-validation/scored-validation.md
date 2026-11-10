# Scored Validation

Most validation checks are about the form of the data: whether a value is in a range, matches a pattern, or isn't missing. Some of the checks that matter most for text are about its meaning instead. Does a passage answer the query it was retrieved for? Is a document of good quality, or toxic? Is it in the expected language? Is it fluent, or garbled? These checks are made with a model that gives each row a score, and the [col_vals_scored()](../../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) validation method is for them: it scores each row with a *scorer* and checks that the score is within bounds.

This page covers how to write a scorer, the adapters of `pb.scorers` for common kinds of models, how scores are cached so that rows aren't scored twice, and how to use the scores once the validation is done.

## A First Scored Step

Let's make a small corpus of short documents, of the kind that might be collected to train a language model. Some of them aren't what we'd want: one is spam, one is in French, one is keyboard-mashing, and one is missing.

``` python
import polars as pl

corpus = pl.DataFrame(
    {
        "doc_id": list(range(1, 9)),
        "text": [
            "The museum opens at nine and closes at five on weekdays.",
            "Click here click here click here buy now buy now",
            "Le musée ouvre à neuf heures et ferme à dix-sept heures.",
            "Our library lends books, films, and music for three weeks.",
            "asdf qwer zxcv uiop hjkl",
            "The trail is steep, so bring water and good shoes.",
            None,
            "Tickets are free for children under twelve.",
        ],
    }
)

pb.preview(corpus)
```

| PolarsRows8Columns2 |  |  |
|----|----|----|
|  | doc_id *Int64* | text *String* |
| 1 | 1 | The museum opens at nine and closes at five on weekdays. |
| 2 | 2 | Click here click here click here buy now buy now |
| 3 | 3 | Le musée ouvre à neuf heures et ferme à dix-sept heures. |
| 4 | 4 | Our library lends books, films, and music for three weeks. |
| 5 | 5 | asdf qwer zxcv uiop hjkl |
| 6 | 6 | The trail is steep, so bring water and good shoes. |
| 7 | 7 | None |
| 8 | 8 | Tickets are free for children under twelve. |

A scorer is a function that takes a batch of rows and returns a score for each. Here's one that scores a document by the share of its words that are distinct, which is low for text that repeats itself.

``` python
def distinct_share(batch):
    scores = []
    for row in batch:
        words = row["text"].lower().split()
        scores.append(len(set(words)) / len(words))
    return scores
```

We'll validate that at least 60% of the words of each document are distinct.

``` python
validation = (
    pb.Validate(data=corpus, tbl_name="corpus")
    .col_vals_scored(columns="text", scorer=distinct_share, min_val=0.6)
    .interrogate()
)

validation
```

[TABLE]

Each row is a test unit, and two of the eight fail: the spam document, which has a score of `0.4`, and the missing document, which isn't scored at all. The note under the report says which scorer was used, how many rows it scored, and the range of the scores.

## Writing a Scorer

A scorer gets the rows as a list of dictionaries, one for each row, with the values of the columns of the step. With `columns="text"`, each row is a dictionary like `{"text": "..."}`; with `columns=["query", "passage"]`, it has both values. The scorer returns a list of scores, in the order of the rows. A score is a number, or `None` for a row that can't be scored.

Rows come in batches, which is what most models want: a model scores a batch of texts much faster than it scores them one at a time. Rows with the same values are scored only once, and a row with a missing value in any of the columns isn't given to the scorer at all.

A scorer can also have a `name`, a `version`, and a `batch_size`. The name identifies the scorer in the report and in the score cache (a function defined at the top level of a module is named after the function), and the version is part of the key of the cache. The `pb.scorers.from_function()` function gives these to any function, and it can also wrap a function that scores one row at a time, with `per_row=True`:

``` python
word_count = pb.scorers.from_function(
    lambda row: len(row["text"].split()), name="word_count", version="1", per_row=True
)

(
    pb.Validate(data=corpus, tbl_name="corpus")
    .col_vals_scored(columns="text", scorer=word_count, min_val=6, na_pass=True)
    .interrogate()
)
```

[TABLE]

The report shows the scorer by its name, `word_count`. The keyboard-mashing document has only five words, so it fails, while the missing document passes this time because of `na_pass=True` (more on that below).

A scorer that raises an error, or that returns something other than a number (or `None`) for each row, stops its step with an evaluation error, and the step's note says why. The other steps of the validation are still interrogated.

## The Adapters

The `pb.scorers` module has adapters for common kinds of models. Each one uses a package that's only imported when the scorer is first used, and if the package isn't installed, the step's note says how to install it. Pointblank never trains a model: the adapters use models that are given to them.

| Adapter | What it scores | Model |
|----|----|----|
| [embedding_similarity()](../../reference/scorers.embedding_similarity.md#pointblank.scorers.embedding_similarity) | Similarity to reference texts | sentence-transformers, or a function |
| [reranker()](../../reference/scorers.reranker.md#pointblank.scorers.reranker) | How well a document answers a query | Cohere Rerank, or a cross-encoder |
| [classifier()](../../reference/scorers.classifier.md#pointblank.scorers.classifier) | A label of a classifier (e.g., quality) | A Hugging Face model or pipeline |
| [langid()](../../reference/scorers.langid.md#pointblank.scorers.langid) | Being in the expected languages | [langid](../../reference/scorers.langid.md#pointblank.scorers.langid), fastText, or a function |
| [perplexity()](../../reference/scorers.perplexity.md#pointblank.scorers.perplexity) | How surprising text is to a language model | KenLM, or a function |
| [text_stat()](../../reference/scorers.text_stat.md#pointblank.scorers.text_stat) | A statistic of text, like its number of words | None |
| [token_count()](../../reference/scorers.token_count.md#pointblank.scorers.token_count) | The number of tokens of text | A Hugging Face tokenizer, `tiktoken`, or a function |

The last two need no model to score with (other than a tokenizer), and they're the scorers of the [col_vals_text_quality()](../../reference/Validate.col_vals_text_quality.md#pointblank.Validate.col_vals_text_quality) and [col_vals_token_len()](../../reference/Validate.col_vals_token_len.md#pointblank.Validate.col_vals_token_len) validation methods, for the heuristic quality filters and the length of text in tokens (see the [Checking Text Data](text-checks.md) article).

With real models, the adapters look like this:

``` python
# Is each document of educational quality (on a scale from 0 to 5)?
quality = pb.scorers.classifier(
    hf_model="HuggingFaceFW/fineweb-edu-classifier", function_to_apply="none"
)

# Does each passage answer its query?
relevance = pb.scorers.reranker(
    provider="sentence-transformers", query="query", document="passage"
)

# Is each title about data validation?
on_topic = pb.scorers.embedding_similarity(
    reference=["data validation", "data quality"], embedder="all-MiniLM-L6-v2"
)

# Is each document in English?
english = pb.scorers.langid(expected="en")

# Is each document in the language of its `lang` column?
own_language = pb.scorers.langid(expected_column="lang")

# Does each document fit in a model's context?
tokens = pb.scorers.token_count(tokenizer="tiktoken:cl100k_base")
```

The adapters that take a model can also take a function in its place, which is how we'll keep the examples on this page self-contained. Here, `guess_language()` stands in for a language identification model (it counts common English and French words), and `vowel_ratio()` stands in for a language model's perplexity (garbled text has few vowels for its letters).

``` python
def guess_language(text):
    words = set(text.lower().split())
    english = len(words & {"the", "and", "at", "is", "for", "are", "so", "our", "on"})
    french = len(words & {"le", "la", "et", "à", "les", "est", "un", "une"})
    if english + french == 0:
        return {"en": 0.5, "fr": 0.5}
    return {"en": english / (english + french), "fr": french / (english + french)}


def vowel_ratio(text):
    letters = [c for c in text.lower() if c.isalpha()]
    vowels = sum(c in "aeiouéà" for c in letters)
    return len(letters) / max(vowels, 1)
```

Now we can check the corpus in three ways at once: that it doesn't repeat itself, that it's in English, and that it's fluent (with `max_val=` as the highest perplexity allowed).

``` python
validation = (
    pb.Validate(data=corpus, tbl_name="corpus")
    .col_vals_scored(columns="text", scorer=distinct_share, min_val=0.6)
    .col_vals_scored(
        columns="text",
        scorer=pb.scorers.langid(expected="en", backend=guess_language),
        min_val=0.5,
    )
    .col_vals_scored(columns="text", scorer=pb.scorers.perplexity(fn=vowel_ratio), max_val=3)
    .interrogate()
)

validation
```

[TABLE]

Each step catches a different problem document: the spam in step 1, the French document in step 2, and the keyboard-mashing in step 3. The adapters are shown in the report by their kind, and the notes give their settings.

## Rows Without a Score

A row with a missing value in any of the step's columns isn't scored, and neither is a row that the scorer gives `None`. By default, such rows fail, which is why the missing document failed every step above. That's usually what we want: a document without text isn't a good document. When it isn't, use `na_pass=True` to let such rows pass, as we did with `word_count`. The note of the step says how many rows had no score, and whether they passed.

## The Score Cache

Models are slow and can be costly, so scores are kept in a cache: by the scorer's name and version, and by the values of the row. Interrogating the same data again (after a change to another step, say, or in an [`Ablation`](../post-interrogation/data-ablation.md), which interrogates many variants of a table) doesn't run the model again for rows that it has already scored. In fact, the note of step 1 in the last report says that every score came from the cache, since the first validation on this page had already scored those rows with `distinct_share`.

``` python
validation.get_note(1, "scores", format="text")
```

    'Scored by __main__.distinct_share: 7 distinct rows, with every score from the cache. Scores from 0.4 to 1 (median 1). 1 row had no score (a missing value, or none from the scorer) and failed (`na_pass=True` lets it pass).'

With `cache=True` (the default), the scores are kept in memory for the rest of the session. With a directory as `cache=`, they're kept on disk, in an SQLite file, and can be shared between sessions and projects:

``` python
pb.Validate(data=corpus).col_vals_scored(
    columns="text",
    scorer=pb.scorers.langid(expected="en"),
    min_val=0.5,
    cache="scores",
).interrogate()
```

The scores are written after each batch, so a run that's interrupted picks up where it left off. A cache on disk needs a scorer with a stable name: the adapters have one (unless they're given a model object), as does a function defined at the top level of a module, and [from_function()](../../reference/scorers.from_function.md#pointblank.scorers.from_function) can give one to a lambda. Use `cache=False` to not keep the scores at all.

When a scorer's model changes, give it a new version (`version=` in the adapters and in [from_function()](../../reference/scorers.from_function.md#pointblank.scorers.from_function)), so that the scores of the older model aren't used. To clear the scores kept in memory, use `pb.scorers.clear_cache()`; to clear a cache on disk, delete its directory.

## Using the Scores

The scores of a step are kept with its results, in a column called `pb_score_`. They're in the extracts of the failing rows and in the step report, which makes it easy to see how far a failing row is from the bounds:

``` python
validation.get_step_report(i=2)
```

| Report for Validation Step 2 ✗ ASSERTION `LANGID(text) ≥ 0.5` **2** / **8** TEST UNIT FAILURES EXTRACT OF ALL **2** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | doc_id *Int64* | text *String* | pb_score\_ *Float64* |
| 3 | 3 | Le musée ouvre à neuf heures et ferme à dix-sept heures. | 0.0 |
| 7 | 7 | None | None |

They're also in the table of [get_row_flags()](../../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags), as a column for each scored step (`pb_score_<step>_`), next to the flags of the steps:

``` python
validation.get_row_flags()
```

shape: (8, 10)

| doc_id | text | pb_failed_1\_ | pb_failed_2\_ | pb_failed_3\_ | pb_score_1\_ | pb_score_2\_ | pb_score_3\_ | pb_n_failed\_ | pb_row_score\_ |
|----|----|----|----|----|----|----|----|----|----|
| i64 | str | bool | bool | bool | f64 | f64 | f64 | i64 | f64 |
| 1 | "The museum opens at nine and c… | false | false | false | 0.909091 | 1.0 | 2.368421 | 0 | 1.0 |
| 2 | "Click here click here click he… | true | false | false | 0.4 | 0.5 | 3.0 | 1 | 0.666667 |
| 3 | "Le musée ouvre à neuf heures e… | false | true | false | 0.909091 | 0.0 | 2.0 | 1 | 0.666667 |
| 4 | "Our library lends books, films… | false | false | false | 1.0 | 1.0 | 2.875 | 0 | 1.0 |
| 5 | "asdf qwer zxcv uiop hjkl" | false | false | true | 1.0 | 0.5 | 4.0 | 1 | 0.666667 |
| 6 | "The trail is steep, so bring w… | false | false | false | 1.0 | 1.0 | 2.6 | 0 | 1.0 |
| 7 | null | true | true | true | null | null | null | 3 | 0.0 |
| 8 | "Tickets are free for children … | false | false | false | 1.0 | 1.0 | 2.769231 | 0 | 1.0 |

Use `keep_scores=False` to not keep the scores (the results then only say whether each row passed).

A bound like `min_val=0.5` is a choice, and it isn't always clear where to set it. The scores of a step can show how the bound affects a model trained on the data: [`Ablation.sweep_score()`](../post-interrogation/data-ablation.md#sweeping-a-threshold) removes more or fewer of the lowest-scoring rows and measures the effect of each on a model.

## Databases and PySpark

Scoring runs in Python, on the values of the rows. A Polars or pandas table is scored as it is. For a database table (through Ibis) or a PySpark DataFrame, the distinct rows of the step's columns are collected and scored, and the scores are joined back to the table on its own backend. With a large table, a `pre=` function that takes a sample of the rows keeps the cost of scoring down.

## Scored Steps in YAML

In a YAML workflow, a scorer is written as one of the adapters with its settings, as `{<adapter>: {<settings>}}`:

``` python
config = """
tbl: corpus.parquet
steps:
- col_vals_scored:
    columns: text
    scorer:
      langid:
        expected: en
    min_val: 0.5
    cache: .scores
"""

print(pb.yaml_to_python(config))
```

    ```python
    import pointblank as pb

    (
        pb.Validate(data="corpus.parquet")
        .col_vals_scored(columns="text", scorer=pb.scorers.langid(expected="en"), min_val=0.5, cache=".scores")
        .interrogate()
    )
    ```

A scorer that's a function of your own can be written as a `python:` block, which needs `allow_python=True` (see the [YAML Reference](../yaml/yaml-reference.md)). A validation with an adapter is written in this form by [to_yaml()](../../reference/Validate.to_yaml.md#pointblank.Validate.to_yaml), so it can be saved and run again.

## Conclusion

The [col_vals_scored()](../../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) validation method brings checks of meaning into a validation plan, with the same reports, thresholds, and actions as any other step. A scorer is any function of a batch of rows, and the adapters of `pb.scorers` use rerankers, classifiers, language identification, embeddings, perplexity, statistics of text, and tokenizers. Scores are cached, so that rows aren't scored again, and they're kept with the results, where they can be seen in the step report and used to choose the bounds with an [`Ablation`](../post-interrogation/data-ablation.md).
