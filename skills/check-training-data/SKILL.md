---
name: check-training-data
description: >
  Check data for training and evaluating models with Pointblank: web
  text (pb.text heuristic filters, language, PII, token length), scored
  steps (col_vals_scored and pb.scorers), near duplicates, benchmark
  contamination, chat transcripts, RAG corpora and evaluation sets
  (pb.rag), LLM-judged rows with calibration, per-row flags and
  weights, data ablation (which checks help a model), train/test split
  checks, synthetic data checks, and dataset cards. Use when cleaning
  or auditing a corpus or a training table, or deciding which checks
  are worth enforcing.
license: MIT
compatibility: >
  Requires Python >=3.10, pointblank installed. Some scorers need extra
  packages (langid, tiktoken, transformers, sentence-transformers,
  cohere); sklearn_evaluator needs pointblank[ml].
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - training-data
    - llm
    - text
    - rag
    - ablation
    - data-cards
---

# Check Training Data

Skill for checking the data that a model is trained or evaluated on,
and for measuring whether the checks matter to the model. The checks
are ordinary validation steps, so they get the usual report, thresholds,
actions, and YAML form; what this skill adds is the steps made for
text and ML data and the tools that use their results.

## Quick start

```python
import pointblank as pb
import polars as pl

docs = pl.DataFrame({"doc_id": [...], "text": [...]})

validation = (
    pb.Validate(data=docs, tbl_name="web_docs")
    .add_steps(pb.text.heuristic_filters("text"))   # 8 Gopher quality steps
    .add_steps(pb.text.no_pii("text"))              # email, phone, credit card
    .rows_near_distinct(columns="text")             # MinHash near duplicates
    .interrogate()
)

flags = validation.get_row_flags()                  # pb_failed_<i>_ per step
clean = flags.filter(pl.col("pb_n_failed_") == 0)   # keeps one copy of each duplicate
```

## Skill directory structure

```
skills/check-training-data/
+-- SKILL.md                         <- This file
+-- references/
    +-- text-and-rag-reference.md    <- pb.text, pb.rag, text row checks
    +-- scorers-reference.md         <- col_vals_scored and pb.scorers
    +-- ablation-reference.md        <- Ablation, strategies, results, YAML
```

## When to use what

| I want to...                                       | Use                                         |
| -------------------------------------------------- | ------------------------------------------- |
| Filter web text like the Gopher models             | `add_steps(pb.text.heuristic_filters(col))` |
| Check text is in the expected language             | `add_steps(pb.text.in_language(col, expected="en"))` |
| Find emails, phones, card numbers, IPs, SSNs, IBANs | `add_steps(pb.text.no_pii(col, entities=...))` |
| Bound text length in model tokens                  | `add_steps(pb.text.token_length(col, tokenizer=..., max_val=...))` |
| Bound a score from any function or model           | `col_vals_scored(columns, scorer, min_val/max_val)` |
| Use a ready-made scorer (embeddings, reranker, classifier, ...) | `pb.scorers.*`                  |
| Find near-duplicate documents                      | `rows_near_distinct(columns)`               |
| Find benchmark or test text in training data       | `rows_not_contaminated(columns, reference)` |
| Check chat transcripts for fine-tuning             | `col_vals_chat_format(columns)`             |
| Check a RAG corpus of chunks                       | `add_steps(pb.rag.corpus_checks(...))`      |
| Check a RAG evaluation set                         | `add_steps(pb.rag.eval_checks(...))`        |
| Judge rows with an LLM                             | `prompt(prompt, model)`                     |
| Measure how far to trust the LLM judge             | `calibrate_prompt(i, labels)`               |
| See which checks each row failed, or weight rows   | `get_row_flags()`                           |
| Use a filtering `pre=` step in flags and ablation  | `pre=pb.row_preserving(fn)`                 |
| Measure whether removing failing rows helps a model | `pb.Ablation(validation, evaluate, metric)` |
| Write an ablation in YAML                          | `pb.yaml_ablation(yaml, evaluate=...)`      |
| Check splits for leaks, label coverage, drift      | `pb.check_splits(train=, valid=, test=)`    |
| Check synthetic data against its seed              | `pb.check_synthetic(synthetic, seed)`       |
| Write a Hugging Face card or a datasheet           | `validation.to_data_card()`                 |

## Core concepts

### Text checks (pb.text)

The `pb.text` functions return a `pb.Steps` bundle, added with
`add_steps()`. Each check is its own step in the report:

```python
validation = (
    pb.Validate(data=corpus)
    .add_steps(pb.text.heuristic_filters("text", min_words=10))
    .add_steps(pb.text.no_pii("text", entities=["email", "phone", "ip_address"]))
    .add_steps(pb.text.token_length("text", tokenizer="whitespace", max_val=4096))
    .interrogate()
)
```

- `heuristic_filters()` makes eight steps (word count, mean word length,
  symbol ratio, bullet lines, ellipsis lines, alphabetic words, stop
  words, repeated lines) with the Gopher bounds as defaults. Set a bound
  to `None` to leave its check out.
- `no_pii()` makes one regex step per entity (`email`, `phone`,
  `credit_card` by default; also `ip_address`, `us_ssn`, `iban`).
- `in_language()` scores each text's probability of being in
  `expected=` (or the language in `expected_column=`) and needs the
  `langid` package (or `backend="fasttext"` with `model=`).
- `token_length()` takes a Hugging Face tokenizer name,
  `"tiktoken:cl100k_base"`, `"whitespace"`, a tokenizer object, or a
  function.

See `references/text-and-rag-reference.md` for every parameter.

### Scored steps

`col_vals_scored()` passes rows to a scorer in batches and checks each
score against `min_val=` and/or `max_val=` (inclusive). A scorer is a
function that takes a list of row dicts (the values of `columns=`) and
returns a number (or `None`) for each:

```python
def n_words(batch):
    return [len(row["review"].split()) for row in batch]

# a stable name and a version for the score cache (a lambda or a
# nested function needs a name to be cached on disk)
n_words_scorer = pb.scorers.from_function(n_words, name="n_words", version="1")

validation = (
    pb.Validate(data=reviews)
    .col_vals_scored(columns="review", scorer=n_words, min_val=3)
    .col_vals_scored(columns="review", scorer=n_words_scorer, min_val=3,
                     cache=".scores", na_pass=True)
    .col_vals_scored(columns="review",
                     scorer=pb.scorers.text_stat("mean_word_length"), max_val=10)
    .interrogate()
)
```

- Rows with a missing value aren't scored; they fail unless
  `na_pass=True`.
- `cache=True` (default) keeps scores in memory for the session; a
  directory keeps them in `scores.sqlite` between sessions (the scorer
  needs a stable name). `pb.scorers.clear_cache()` empties the memory.
- `keep_scores=True` (default) puts the scores in the extracts and in
  `get_row_flags()` as `pb_score_<i>_`, and lets
  `Ablation.sweep_score()` use them.
- Several columns: `columns=["query", "passage"]` gives each row dict
  both values (a reranker needs this).

The adapters in `pb.scorers` (`embedding_similarity`, `reranker`,
`classifier`, `langid`, `perplexity`, `token_count`, `text_stat`) wrap
common models. See `references/scorers-reference.md`.

### Near duplicates and contamination

```python
validation = (
    pb.Validate(data=train)
    .rows_near_distinct(columns="text", threshold=0.85)
    .rows_not_contaminated(columns="text", reference=benchmark,
                           reference_columns="question")
    .interrogate()
)
```

- `rows_near_distinct()` normalizes text (case, punctuation,
  whitespace) and, with `method="minhash"` (default), groups texts whose
  word 5-gram Jaccard similarity is at least `threshold=`. Every row of
  a group fails the step.
- `rows_not_contaminated()` fails rows that share a 13-word n-gram with
  any reference text (`method="ngram"`), are the same once normalized
  (`"exact"`), or are similar by MinHash (`"minhash"`). A reference text
  shorter than `ngram=` words matches a row that contains it whole. The
  reference can be a table, a file path, a list of texts, or a dict of
  tables by name.

### Chat transcripts

```python
validation = (
    pb.Validate(data=chats)
    .col_vals_chat_format(columns="messages", last_role="assistant")
    .interrogate()
)
```

Each value is a list of `{"role", "content"}` messages (a Polars list of
structs, a pandas list of dicts, or JSON text). A row fails for an
unknown role, a system message that isn't first, turns that don't
alternate (`alternate=True`), the wrong `first_role=`/`last_role=`, or
empty content (unless `allow_empty=True`).

### RAG corpora and evaluation sets (pb.rag)

```python
import datetime

corpus_validation = (
    pb.Validate(data=chunks, tbl_name="chunks")
    .add_steps(
        pb.rag.corpus_checks(
            "text",
            chunk_id="chunk_id",
            metadata="source",
            updated_at="updated_at",
            max_age="365 days",
            reference_time=datetime.date(2026, 11, 1),
        )
    )
    .interrogate()
)

eval_validation = (
    pb.Validate(data=queries)
    .add_steps(pb.rag.eval_checks("query", gold_doc_id="doc_id", corpus=corpus))
    .interrogate()
)
```

`corpus_checks()` checks for empty chunks, repeated chunk ids, missing
metadata, near-duplicate chunks, stale chunks, and (with
`min_tokens=`/`max_tokens=`) chunk length. `eval_checks()` checks for
empty and repeated queries, missing gold document ids, and gold
documents that aren't in the corpus (`corpus_id=` names its id column
if it differs); with `passage=`, `scorer=`, and `min_relevance=`, it also
checks that each passage answers its query.

### LLM-judged rows

`prompt()` sends rows to a model (or a list of models, combined by
`aggregate="majority"`) and passes or fails each one.
`calibrate_prompt()` then compares the step with labels given by a
person (`True` should pass, `False` should fail, `None` unlabeled):

```python
validation = (
    pb.Validate(data=reviews)
    .prompt(
        "The review describes the product, not the delivery.",
        model="anthropic:claude-sonnet-4-6",
        columns_subset="review",
    )
    .interrogate()
)

validation.calibrate_prompt(1, labels="human_label")
# {'n_labeled': ..., 'agreement': ..., 'kappa': ..., 'true_pass': ...,
#  'false_pass': ..., 'false_fail': ..., 'true_fail': ...}
```

A kappa above 0.8 is strong agreement and below 0.4 is weak. The label
column must not be in `columns_subset=` (the model would see the
answers). Label a few dozen rows, including some that should fail.

### Using the results: flags and weights

`get_row_flags()` returns the whole table with `pb_failed_<i>_` for
each row-level step, `pb_score_<i>_` for scored steps, `pb_n_failed_`,
and `pb_row_score_` (the weighted share of the row's checks that
passed, from 0 to 1):

```python
flags = validation.get_row_flags()

clean = flags.filter(pl.col("pb_n_failed_") == 0)          # drop failing rows
weights = flags["pb_row_score_"]                            # or weight them
no_email = flags.filter(~pl.col("pb_failed_9_"))           # step 9 (emails) only
```

Duplicate groups from `rows_distinct()`/`rows_near_distinct()` keep
one copy unflagged (`duplicates="keep_first"`, or `"keep_last"` with
`duplicates_order_by=`, or `"flag_all"`).

Steps with `pre=` are left out of the flags, sundering, and ablation,
because a `pre=` function can change the rows. Mark a function that
only filters, reorders, or changes values with `pb.row_preserving()`:

```python
.col_vals_gt(
    columns="amount", value=0,
    pre=pb.row_preserving(lambda df: df.filter(pl.col("status") == "paid")),
)
```

### Data ablation

An `Ablation` trains the model on versions of the training table with
the failing rows of some steps removed, compares each version with a
control that removes as many random rows, and gives a verdict:
*helps*, *hurts*, *inconclusive*, or (for `drop_all_but_each()`)
*redundant*.

```python
temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10        # readings in the wrong units
for k in range(7, 300, 40):
    temps[k] = None       # missing readings
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
    return {"error": abs(sample["temp"].mean() - 20)}

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .drop_all()
)
ablation.plan()      # what will be evaluated, without evaluating
ablation.run()       # shows the report
ablation.get_effects().select("name", "delta_vs_control", "p_adj_vs_control", "verdict")
```

Here, removing the out-of-range readings (step 2) helps, and removing
the missing ones (step 1) is inconclusive, since `evaluate` drops nulls
anyway.

- `evaluate(train, *, seed)` trains on `train` (same table type as the
  validation's data), scores on fixed held-out data, and returns a
  number or a dict of numbers. It can also take `variant=`.
- `pb.ablation.sklearn_evaluator(holdout, target=..., scoring=...)`
  makes one for tabular scikit-learn models (needs `pointblank[ml]`).
- The other strategies: `drop_steps()`, `drop_all_but_each()`,
  `drop_groups(column)`, `sweep_step(step, param, values)`,
  `sweep_score(step, quantiles=...)`, `inject(corruption, step=,
  rates=)`, and `shapley()`. Then `suggest_thresholds(tolerance=)`,
  `annotate()`, `export()`/`load_results()`, and `to_json()`.

See `references/ablation-reference.md`.

### Split checks

```python
check = pb.check_splits(train=train, test=test, columns="text",
                        label="topic", near=True)

check.ok                       # False if there are issues
check.issues                   # e.g. "2 rows of 'test' (16.7%) are also in 'train' ..."
check.get_overlap("train", "test")   # the leaked rows (1-based row numbers)
check.get_tabular_report()
```

It checks overlap (exact, or near with `near=True`), label coverage and
shares (with `label=`), and drift of `drift_columns=` (PSI and KS).
Use `splits={"train": ..., "valid": ..., "test": ...}` for other names.

### Synthetic data

`pb.check_synthetic(synthetic, seed, text=...)` checks fidelity,
diversity, and copies of seed rows (see the `generate-data` skill).

### Data cards

```python
card = validation.to_data_card(
    ablation=ablation,                 # adds an Ablation Evidence section
    description="Hourly temperature readings from three sites.",
    license="cc-by-4.0",
    language="en",
    path="README.md",                  # optional; the card is returned either way
)
```

`format="huggingface"` (default) gives YAML front matter and the
Hugging Face template sections plus *Data Quality*;
`format="datasheet"` answers the *Datasheets for Datasets* questions.
The curator comes from `owner=`, uses from `consumers=`, and version
from `version=` of the validation. Unknown parts say
"[More Information Needed]".

## Workflows

### Cleaning a text corpus

1. Add `pb.text.heuristic_filters()` (adjust `min_words=` for short
   texts), `no_pii()`, and `rows_near_distinct()`.
2. Add `in_language()` or scorers (`classifier`, `perplexity`) if the
   packages are available.
3. Interrogate and read the report: which checks fail the most rows?
4. Use `get_row_flags()` to drop rows, or `pb_row_score_` to weight
   them.
5. Check the training split against benchmarks with
   `rows_not_contaminated()` and against the test split with
   `check_splits()`.

### Deciding which checks to enforce

1. Interrogate the training table with the candidate checks.
2. Write `evaluate` (or use `sklearn_evaluator`) on fixed held-out
   data, and run `ablation.check_determinism()`.
3. `Ablation(...).drop_each().drop_all_but_each().run()`.
4. Enforce the checks that help, drop those that hurt (they remove
   good data), and drop those that are redundant.
5. Tune limits with `sweep_step()`/`sweep_score()`; set thresholds
   with `inject()` and `suggest_thresholds()`.
6. `ablation.annotate()` and `to_data_card(ablation=ablation)` to
   record the evidence.

### Releasing a dataset

1. `pb.check_splits()` for leaks and label coverage.
2. Validate each split; give the validation `owner=`, `consumers=`,
   and `version=`.
3. `to_data_card(path="README.md", license=..., source=...,
   column_descriptions=...)`.

## Gotchas

1. **`get_sundered_data()` leaves out duplicate steps.** It skips
   `rows_distinct()` and `rows_near_distinct()`, so its passing rows
   keep every copy. Filter `get_row_flags()` instead, which keeps one
   copy of each group.
2. **The step counts every copy, the flags keep one.** A pair of near
   duplicates is 2 failing test units in the report, but only the
   second copy is flagged by `get_row_flags()`.
3. **Missing values fail scored steps.** They aren't scored, and fail
   unless `na_pass=True`. In `rows_not_contaminated()` and the regex
   steps of `no_pii()`, they pass.
4. **Lambda scorers cache only in memory.** A disk `cache=` needs a
   stable scorer name: use `pb.scorers.from_function(fn, name=...)`,
   and change `version=` when the model changes.
5. **`pre=` steps are left out of flags and ablation** unless the
   function is wrapped in `pb.row_preserving()` (it must keep the
   `pb_index_` column and not aggregate rows).
6. **The heuristic filters assume English web documents.** The default
   `min_words=50` fails short texts, and `min_stop_words=2` uses
   English stop words (set `stop_words=` for other languages).
7. **Ablation verdicts need at least 3 repeats** and are given only for
   `metric=` (other metrics in the dict are evaluated but get no
   verdict). Pick the metric before running.
8. **`evaluate` must use its seed** for all randomness; check with
   `ablation.check_determinism()`. Adding a variant clears the
   results, so call `run()` again.
9. **`yaml_ablation()` imports `evaluate:` only with
   `allow_python=True`.** Otherwise pass `evaluate=` directly.
10. **`to_data_card()` needs an interrogated validation.**

## Related skills

| Skill            | When to use it                                       |
| ---------------- | ---------------------------------------------------- |
| pointblank       | The general API, reports, and post-interrogation     |
| write-validation | Choosing and combining the steps                     |
| validate-yaml    | Scored steps and ablations in YAML                   |
| scan-and-profile | Profiling a table and comparing it with a baseline   |
| generate-data    | Synthetic data and `check_synthetic()`               |
| draft-validation | Drafting a plan with an LLM                          |
