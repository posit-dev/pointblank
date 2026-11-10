# Scorers reference

## col_vals_scored

```python
.col_vals_scored(
    columns,              # a column, or a list (each row dict gets all of them)
    scorer,               # a function of a batch of rows, or a pb.scorers adapter
    min_val=None,         # lowest passing score (inclusive)
    max_val=None,         # highest passing score (inclusive); give at least one
    na_pass=False,        # rows with a missing value, or with no score
    cache=True,           # True (memory), a directory (scores.sqlite), or False
    keep_scores=True,     # pb_score_ in extracts, get_row_flags(), sweep_score()
    pre=None, segments=None, thresholds=None, actions=None,
    brief=None, active=True,
)
```

A scorer takes a list of row dicts (`{"review": ..., "stars": ...}`)
and returns one number (or `None`) per row, in order. Rows with a
missing value in any of `columns=` aren't passed to it.

```python
def stars_match(batch):
    return [
        1.0 if (row["stars"] >= 4) == ("great" in row["review"].lower()) else 0.0
        for row in batch
    ]

pb.Validate(data=reviews).col_vals_scored(
    columns=["review", "stars"], scorer=stars_match, min_val=1, na_pass=True
)
```

## The Scorer protocol

Any callable is a scorer. Three optional attributes are read:

| Attribute    | Use                                                       |
|--------------|-----------------------------------------------------------|
| `name`       | Identifies the scorer in the cache and the report          |
| `version`    | Part of the cache key (change it when the model changes)  |
| `batch_size` | Rows per call (256 if absent)                             |

`pb.scorers.from_function(fn, name=None, version=None, batch_size=None,
per_row=False)` sets them. With `per_row=True`, `fn` takes one row
dict and returns its score:

```python
n_chars = pb.scorers.from_function(lambda row: len(row["text"]), name="n_chars", per_row=True)
```

## The score cache

Scores are cached by the scorer's name and version and the row's
values, so re-interrogating (as an `Ablation` does) doesn't call the
model again. They're written after each batch, so an interrupted run
resumes.

- `cache=True`: in memory for the session; `pb.scorers.clear_cache()`
  empties it.
- `cache="dir"`: an SQLite `dir/scores.sqlite`, shared between
  sessions. Needs a stable name: the adapters have one (unless given a
  model object), and so does a function defined at the top level of a
  module. A lambda or nested function needs `from_function(name=...)`.
- `cache=False`: score every time.

## Adapters (pb.scorers)

All take `name=`, `version=`, and `batch_size=`, and a `column=` that
can be left out when the step has one column.

| Adapter | Scores | Needs |
|---------|--------|-------|
| `text_stat(stat, stop_words=None)` | A text statistic (below) | nothing |
| `token_count(tokenizer)` | Number of tokens | depends on the tokenizer (see `col_vals_token_len()`) |
| `langid(expected=, expected_column=, backend="langid", model=None)` | Probability of the expected language(s) | `langid`, or `fasttext` + `model=`, or a function |
| `embedding_similarity(reference, embedder)` | Cosine similarity to the closest reference text | `sentence-transformers` for a model name; or an object with `encode()`; or a function of a list of texts returning vectors |
| `classifier(hf_model, label=None, function_to_apply=None, device=None)` | A text classifier's score for `label=` (no label for one-output models) | `transformers` + a backend for a model name; or a pipeline |
| `reranker(provider="cohere", model=None, query=None, document=None, api_key=None)` | Relevance of a document to a query (two columns) | `cohere` + `CO_API_KEY`, or `provider="sentence-transformers"` |
| `perplexity(fn=None, kenlm_model=None, per_row=True)` | Perplexity under a language model | a function, or `kenlm` + a model path |

Notes:

- `classifier()` with a function instead of a pipeline calls it as
  `fn(texts, top_k=None, truncation=True)` (plus `function_to_apply=`
  if set), so the function must accept those keywords; it returns, for
  each text, a list of `{"label": ..., "score": ...}` dicts.
- `langid()` with a function backend: `fn(text)` returns
  `{lang: prob}` or `[(lang, prob), ...]`.
- `reranker()` scores from different models aren't on one scale; pick
  `min_val=` for the model.

```python
def letter_counts(texts):          # stands in for an embedding model
    return [[t.count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]

pb.Validate(data=docs).col_vals_scored(
    columns="text",
    scorer=pb.scorers.embedding_similarity("checking data in tables", embedder=letter_counts),
    min_val=0.8,
)
```

## text_stat statistics

Words are split on whitespace; lines on newlines (blank lines left
out).

| `stat`             | Score                                          |
|--------------------|------------------------------------------------|
| `char_count`       | Characters                                     |
| `word_count`       | Words                                          |
| `mean_word_length` | Mean characters per word                       |
| `symbol_ratio`     | `#` symbols and ellipses per word              |
| `bullet_lines`     | Share of lines starting with a bullet          |
| `ellipsis_lines`   | Share of lines ending with an ellipsis         |
| `alpha_words`      | Share of words with a letter                   |
| `stop_words`       | Distinct stop words (`stop_words=` to change them) |
| `repeated_lines`   | Share of lines repeating an earlier line       |

A text without words (or lines) has no score for the share statistics.

## Scores after interrogation

- `validation.get_row_flags()` has `pb_score_<i>_` per scored step.
- The extracts of failing rows have a `pb_score_` column.
- `Ablation.sweep_score(i, quantiles=[...])` removes growing shares of
  the scored rows: the highest-scoring ones for a step with only
  `max_val=`, the lowest otherwise (or set `remove=`). `values=` gives
  score cutoffs instead, and a column name in place of `i` uses that
  column as the score.

## YAML

```yaml
steps:
- col_vals_scored:
    columns: text
    scorer:
      text_stat:
        stat: word_count
    min_val: 50
```

The adapter's settings must be a mapping (`text_stat: word_count`
raises a `YAMLValidationError`). A scorer of your own needs
`allow_python=True`; see the `validate-yaml` skill.
