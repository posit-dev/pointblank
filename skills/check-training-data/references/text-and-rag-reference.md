# Text and RAG checks reference

The `pb.text` and `pb.rag` functions return a `pb.Steps` bundle, added
to a validation with `add_steps()`. All of them also take `pre=`,
`thresholds=`, `actions=`, and `active=` (and most take `segments=` and
`brief=`), applied to each step they make.

## pb.text.heuristic_filters

```python
pb.text.heuristic_filters(
    column,
    min_words=50, max_words=100_000,
    min_mean_word_length=3, max_mean_word_length=10,
    max_symbol_ratio=0.1,
    max_bullet_lines=0.9,
    max_ellipsis_lines=0.3,
    min_alpha_words=0.8,
    min_stop_words=2,
    max_repeated_lines=0.3,
    stop_words=None,
)
```

The quality filters of the Gopher models (Rae et al., 2021), as eight
`col_vals_scored()` steps with `text_stat()` scorers:

| Step | Statistic          | Passes when                                   |
|------|--------------------|-----------------------------------------------|
| 1    | `word_count`       | between `min_words` and `max_words`           |
| 2    | `mean_word_length` | between the two bounds (characters)           |
| 3    | `symbol_ratio`     | `#` and ellipses per word <= `max_symbol_ratio` |
| 4    | `bullet_lines`     | share of lines starting with a bullet <= bound |
| 5    | `ellipsis_lines`   | share of lines ending with an ellipsis <= bound |
| 6    | `alpha_words`      | share of words with a letter >= bound         |
| 7    | `stop_words`       | distinct stop words >= `min_stop_words`       |
| 8    | `repeated_lines`   | share of lines repeating an earlier line <= bound |

Set a bound to `None` to drop its check (a two-bound check is dropped
when both are `None`). Missing values pass, as do texts without the
words or lines a check needs (step 1 catches those). The default stop
words are `the be to of and that have with`; set `stop_words=` for
other languages.

## pb.text.no_pii

```python
pb.text.no_pii(columns, entities=("email", "phone", "credit_card"), na_pass=True)
```

One `col_vals_regex(inverse=True)` step per column and entity. The
patterns use only portable regex features, so they run in databases
and PySpark.

| Entity        | Finds                                              |
|---------------|----------------------------------------------------|
| `email`       | Email addresses                                    |
| `phone`       | North American phone numbers (`(555) 123-4567`, `+1 555.123.4567`) |
| `credit_card` | Visa, Mastercard, Amex, Discover numbers (no checksum) |
| `ip_address`  | IPv4 addresses                                     |
| `us_ssn`      | US Social Security numbers (`123-45-6789`)         |
| `iban`        | IBANs                                              |

`entities=` can also be a dict of your own patterns by name. The
patterns find likely matches; for names and addresses, use a named
entity model through `col_vals_scored()`.

## pb.text.in_language

```python
pb.text.in_language(
    column,
    expected=None,            # "en" or ["en", "fr"]
    expected_column=None,     # or a column with each row's language
    min_confidence=0.5,
    backend="langid",         # "langid", "fasttext", or a function
    model=None,               # fastText model path (e.g., "lid.176.bin")
    na_pass=False,
    cache=True,
)
```

A `col_vals_scored()` step with the `pb.scorers.langid()` scorer. The
`"langid"` backend needs the `langid` package and `"fasttext"` needs
`fasttext`. A function backend takes a text and returns
`{language: probability}` or `[(language, probability), ...]`. `"en"`
matches `"en_US"`.

## pb.text.token_length

```python
pb.text.token_length(column, tokenizer, min_val=None, max_val=None,
                     na_pass=False, cache=True)
```

| Tokenizer                    | Needs          |
|------------------------------|----------------|
| `"gpt2"`, `"bert-base-uncased"`, ... (Hugging Face name) | `transformers` |
| `"tiktoken:cl100k_base"`, `"tiktoken:gpt-4o"` | `tiktoken`   |
| `"whitespace"`               | nothing (counts words) |
| An object with `encode()`    | its library    |
| A function returning tokens or a count | nothing |

## rows_near_distinct

```python
.rows_near_distinct(columns, method="minhash", threshold=0.85, ngram=5, num_perm=128)
```

Texts are normalized (lower case, no punctuation, single spaces).
`method="exact"` groups texts that are then the same; `"minhash"`
groups texts whose word n-gram Jaccard similarity is at least
`threshold=`. Several columns are joined by newlines. Every row of a
group fails; `get_row_flags()` leaves one copy unflagged.

## rows_not_contaminated

```python
.rows_not_contaminated(
    columns,
    reference,                 # table, path, list of tables, dict of tables, or list of texts
    reference_columns=None,    # default: the same names as columns=
    method="ngram",            # "ngram", "exact", or "minhash"
    ngram=None,                # 13 for "ngram", 5 for "minhash"
    threshold=0.8,             # "minhash" only
    num_perm=128,
)
```

- `"ngram"`: a row fails if it shares any `ngram`-word n-gram with a
  reference text. A reference text shorter than `ngram` words matches a
  row that contains it whole.
- `"exact"`: the normalized texts are the same.
- `"minhash"`: Jaccard similarity of n-grams >= `threshold`.

Missing values pass. The first matching reference text is in a
`pb_match_` column of the extracts, and the note counts overlaps per
reference (name them with a dict). The reference is held in memory.

## col_vals_chat_format

```python
.col_vals_chat_format(
    columns,
    roles=("system", "user", "assistant"),
    alternate=True,       # no role follows itself (after a first system message)
    first_role=None,      # e.g., "user"
    last_role=None,       # e.g., "assistant"
    allow_empty=False,
    role_key="role",
    content_key="content",
    na_pass=False,
)
```

Values are lists of messages (Polars list of structs, pandas list of
dicts) or JSON text, either a list or an object with `"messages"`. A
system message may only be first.

## pb.rag.corpus_checks

```python
pb.rag.corpus_checks(
    text,
    chunk_id=None,
    metadata=None,             # column or list, e.g. ["source", "acl"]
    min_tokens=None, max_tokens=None, tokenizer="whitespace",
    near_duplicates=True,      # or a threshold
    updated_at=None, max_age=None,   # "365 days", "2 weeks", or timedelta
    reference_time=None,       # default: when the steps are made (UTC)
)
```

| Step               | With                         | Method                 |
|--------------------|------------------------------|------------------------|
| Chunk has text     | always                       | `col_vals_regex`       |
| Ids are unique     | `chunk_id=`                  | `rows_distinct`        |
| Metadata present   | `metadata=` (one step each)  | `col_vals_not_null`    |
| Token length       | `min_tokens=` or `max_tokens=` | `pb.text.token_length` |
| No near duplicates | `near_duplicates=True`       | `rows_near_distinct`   |
| Fresh enough       | `updated_at=` and `max_age=` | `col_vals_ge`          |

The freshness cutoff is fixed when the steps are made, so pass
`reference_time=` for reproducible results.

## pb.rag.eval_checks

```python
pb.rag.eval_checks(
    query,
    gold_doc_id=None,
    corpus=None, corpus_id=None,     # corpus_id defaults to gold_doc_id's name
    passage=None, scorer=None, min_relevance=None,
    near_duplicates=True,
)
```

| Step                      | With                                  | Method              |
|---------------------------|---------------------------------------|---------------------|
| Query has text            | always                                | `col_vals_regex`    |
| No near-duplicate queries | `near_duplicates=True`                | `rows_near_distinct` |
| Has a gold document       | `gold_doc_id=`                        | `col_vals_not_null` |
| Gold document in corpus   | `corpus=`                             | `col_vals_in_table` |
| Passage is relevant       | `passage=`, `scorer=`, `min_relevance=` | `col_vals_scored` |

A relevance scorer gets the query and the passage, e.g.
`pb.scorers.reranker(provider="cohere")`.
