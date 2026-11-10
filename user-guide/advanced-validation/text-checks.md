# Checking Text Data

A corpus of text for training a language model has problems that a table of numbers doesn't. Web pages are often menus, lists of hashtags, or the same page crawled twice. Some documents are in the wrong language, are too long for the model, or have someone's email address in them. Chat data for fine-tuning can have conversations that are out of order, and any training data can contain the questions of the benchmarks that the model will be scored on.

Pointblank has validation methods and step bundles for each of these problems. They give reports, thresholds, and actions like any other step, so a corpus can be checked with the same tools as any other table, and each check can be measured with an [`Ablation`](../post-interrogation/data-ablation.md) to see how much it matters to the model.

## A Corpus to Check

Let's make a small corpus of web documents. Most are fine, but there's a near duplicate, a menu, a list of hashtags, a document in French, and one with an email address.

``` python
import polars as pl

corpus = pl.DataFrame(
    {
        "doc_id": list(range(1, 10)),
        "text": [
            "The museum opens at nine and closes at five on weekdays, and entry is free for "
            "children under twelve.",
            "The museum opens at nine and closes at five on weekdays, and entry is free for "
            "children under 12!",
            "Home\nAbout us\nHome\nAbout us\nHome\nContact",
            "Our library lends books, films, and music to members of the town for up to three "
            "weeks at a time.",
            "#sale #deals #shoes #free #today #now #best #cheap ... click for more",
            "Le musée ouvre à neuf heures et ferme à dix-sept heures, et l'entrée est gratuite "
            "pour les enfants.",
            "Questions about your order? Write to help@shop-example.com and we will reply within "
            "a day or two.",
            "The trail to the lake is steep in places, so bring water, good shoes, and a jacket "
            "for the wind.",
            "Which planet in the solar system has the most moons, and how many of them have been "
            "named so far?",
        ],
    }
)

pb.preview(corpus)
```

| PolarsRows9Columns2 |  |  |
|----|----|----|
|  | doc_id *Int64* | text *String* |
| 1 | 1 | The museum opens at nine and closes at five on weekdays, and entry is free for children under twelve. |
| 2 | 2 | The museum opens at nine and closes at five on weekdays, and entry is free for children under 12! |
| 3 | 3 | Home About us Home About us Home Contact |
| 4 | 4 | Our library lends books, films, and music to members of the town for up to three weeks at a time. |
| 5 | 5 | \#sale \#deals \#shoes \#free \#today \#now \#best \#cheap ... click for more |
| 6 | 6 | Le musée ouvre à neuf heures et ferme à dix-sept heures, et l'entrée est gratuite pour les enfants. |
| 7 | 7 | Questions about your order? Write to help@shop-example.com and we will reply within a day or two. |
| 8 | 8 | The trail to the lake is steep in places, so bring water, good shoes, and a jacket for the wind. |
| 9 | 9 | Which planet in the solar system has the most moons, and how many of them have been named so far? |

## Heuristic Quality Filters

The corpora used to train language models are cleaned with simple rules about each document: how many words it has, how long they are, how many of its lines are bullet points or repeat an earlier line, and whether it has common words like *the* and *of*. The [`pb.text.heuristic_filters()`](../../reference/text.heuristic_filters.md#pointblank.text.heuristic_filters) function makes a validation step for each of the quality filters of the Gopher models, with their bounds as the defaults. It returns a [`Steps`](../../reference/Steps.md#pointblank.Steps) collection, which is added to a validation with [add_steps()](../../reference/Validate.add_steps.md#pointblank.Validate.add_steps). Our documents are short, so we'll lower the number of words that a document needs.

``` python
validation = (
    pb.Validate(data=corpus, tbl_name="corpus")
    .add_steps(pb.text.heuristic_filters("text", min_words=10))
    .interrogate()
)

validation
```

[TABLE]

Each rule is a [col_vals_scored()](../../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step with a [`text_stat()`](../../reference/scorers.text_stat.md#pointblank.scorers.text_stat) scorer, so the report shows what each one checks and how many documents fail it. The menu has too few words, repeats its lines, and has too few common words, and the list of hashtags has too many symbols for its words. The document in French fails the check of common words too, since the default stop words are English (use `stop_words=` for other languages). Any rule can be turned off by setting its bound to `None`, and since each is an ordinary step, it can be changed like any other.

## Near Duplicates

A document that's in a corpus many times gets too much weight in training. The [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) method finds rows that are exactly the same, but most duplicates in text differ a little: a changed date, a different footer, or a name filled into a template. The [rows_near_distinct()](../../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct) method finds those too. It normalizes the text (in lower case, without punctuation), and with the default `method="minhash"`, it finds texts whose sequences of words are mostly the same.

``` python
validation = (
    pb.Validate(data=corpus, tbl_name="corpus")
    .rows_near_distinct(columns="text", threshold=0.8)
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `max jaccard(text) < 0.8` **2** / **9** TEST UNIT FAILURES EXTRACT OF ALL **2** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | doc_id *Int64* | text *String* | pb_cluster\_ *Int64* |
| 1 | 1 | The museum opens at nine and closes at five on weekdays, and entry is free for children under twelve. | 1 |
| 2 | 2 | The museum opens at nine and closes at five on weekdays, and entry is free for children under 12! | 1 |

The first two documents differ only in *twelve* and *12!*, so both fail, and the step report shows them together as one group (in the `pb_cluster_` column). Every row of a group fails, as with [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct), since the validation can't know which copy we'd want to keep. That's decided when the results are used to clean the table: [get_row_flags()](../../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags) keeps the first row of each group by default (`duplicates="keep_first"`), and [get_sundered_data()](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) does so when given `duplicates="keep_first"` (without it, the step isn't used for sundering).

## Language, Length, and Personal Information

Three more bundles cover checks that most text corpora need:

- [`pb.text.in_language()`](../../reference/text.in_language.md#pointblank.text.in_language) checks that each document is in the expected language, with a language identification model (from the [langid](../../reference/scorers.langid.md#pointblank.scorers.langid) package, a fastText model, or a function of your own). The expected language can also come from a column, for a corpus with documents in several languages.
- [`pb.text.token_length()`](../../reference/text.token_length.md#pointblank.text.token_length) checks the length of each document in the tokens of a model's tokenizer (a Hugging Face tokenizer, a `tiktoken` encoding, or any function that splits text into tokens).
- [`pb.text.no_pii()`](../../reference/text.no_pii.md#pointblank.text.no_pii) checks that the text has no email addresses, phone numbers, or credit card numbers (or other kinds of personal information, or patterns of your own).

With real models, the first two look like this:

``` python
pb.text.in_language("text", expected="en", min_confidence=0.8)
pb.text.token_length("text", tokenizer="tiktoken:cl100k_base", max_val=4096)
```

To keep this page self-contained, we'll use a function that guesses the language from a few common words, and a tokenizer that splits text into words and punctuation.

``` python
import re


def guess_language(text):
    words = set(text.lower().split())
    english = len(words & {"the", "and", "at", "is", "for", "so", "our", "to", "in", "of"})
    french = len(words & {"le", "la", "et", "à", "les", "est", "un", "une", "pour"})
    if english + french == 0:
        return {"en": 0.5, "fr": 0.5}
    return {"en": english / (english + french), "fr": french / (english + french)}


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


validation = (
    pb.Validate(data=corpus, tbl_name="corpus")
    .add_steps(pb.text.in_language("text", expected="en", backend=guess_language))
    .add_steps(pb.text.token_length("text", tokenizer=tokenize, min_val=10, max_val=25))
    .add_steps(pb.text.no_pii("text"))
    .interrogate()
)

validation
```

[TABLE]

The document in French fails step 1, the menu is too short for step 2, and the email address fails step 3. The checks of personal information are [col_vals_regex()](../../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) steps, so they run on any backend, including in a database. Their patterns find likely matches rather than certain ones (a number with the shape of a credit card number matches, with no checksum), so for a thorough check, a model of named entities can be used with [col_vals_scored()](../../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Chat Conversations

Data for fine-tuning a chat model is a list of messages for each example, each with a role and some content. Data converted from other formats often has conversations with roles out of order, an unknown role, or an empty message, and a training run can fail on them (or quietly learn from them). The [col_vals_chat_format()](../../reference/Validate.col_vals_chat_format.md#pointblank.Validate.col_vals_chat_format) method checks each conversation, given as a list of messages or as JSON text.

``` python
chats = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "messages": [
            '[{"role": "user", "content": "Hi!"}, '
            '{"role": "assistant", "content": "Hello! How can I help?"}]',
            '[{"role": "user", "content": "Hi!"}, {"role": "user", "content": "Hello?"}]',
            '[{"role": "system", "content": "Be brief."}, '
            '{"role": "user", "content": "Capital of France?"}, '
            '{"role": "assistant", "content": "Paris."}]',
            '[{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": ""}]',
        ],
    }
)

validation = (
    pb.Validate(data=chats, tbl_name="chats")
    .col_vals_chat_format(columns="messages", first_role="user", last_role="assistant")
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `messages is a chat of {system, user, assistant}` **2** / **4** TEST UNIT FAILURES IN COLUMN **2** EXTRACT OF ALL **2** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | id *Int64* | messages *String* | pb_reason\_ *String* |
| 2 | 2 | \[{"role": "user", "content": "Hi!"}, {"role": "user", "content": "Hello?"}\] | not_alternating |
| 4 | 4 | \[{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": ""}\] | empty_content |

The second conversation has two user messages in a row, and the last one ends with an empty reply. The step report gives the problem with each failing row in the `pb_reason_` column, and the step's note counts the rows with each problem. The third conversation passes: a system message can come first, before the turns of the user and the assistant.

## Benchmark Contamination

A model is often judged by its score on a benchmark, a set of questions with known answers. If the training data contains those questions, the score no longer says how well the model does on questions it hasn't seen: the benchmark is *contaminated*. The [rows_not_contaminated()](../../reference/Validate.rows_not_contaminated.md#pointblank.Validate.rows_not_contaminated) method checks the text of each row against a reference, such as the questions of a benchmark or the rows of a test set.

``` python
benchmark = pl.DataFrame(
    {
        "question": [
            "Which planet in the solar system has the most moons?",
            "What is the boiling point of water at sea level in degrees Celsius?",
        ]
    }
)

validation = (
    pb.Validate(data=corpus, tbl_name="corpus")
    .rows_not_contaminated(
        columns="text", reference=benchmark, reference_columns="question", ngram=8
    )
    .interrogate()
)

validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `8-grams(text) ∩ reference = ∅` **1** / **9** TEST UNIT FAILURES EXTRACT OF **1** ROW (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | doc_id *Int64* | text *String* | pb_match\_ *String* |
| 9 | 9 | Which planet in the solar system has the most moons, and how many of them have been named so far? | Which planet in the solar system has the most moons? |

The last document of the corpus contains the first question of the benchmark, so it fails, and the step report shows the question it overlaps with (in the `pb_match_` column). By default, a row overlaps with a reference text if they share a sequence of 13 words, the length used to check the training data of large language models for contamination. We used a shorter one here, as the question is short. With `method="exact"`, a row overlaps only with a text that's the same, and with `method="minhash"`, with one that's nearly the same. The reference can also be a list of tables, or a dictionary of them by name, so that the note says which benchmark each overlap is with.

To check a dataset's own splits (that the test split doesn't overlap with the training split, that they have the same labels, and that their columns are distributed alike), see the [`check_splits()`](../table-comparison/checking-splits.md) article.

## Which Checks Matter?

Each of these checks removes some documents, and it isn't always clear which of them make a model better. Since they're ordinary validation steps, an [`Ablation`](../post-interrogation/data-ablation.md) can measure that: it trains and evaluates a model on the corpus with the failing documents of each step removed, and compares each version with the full corpus. Its [sweep_score()](../../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score) method can also find the bound of a scored step (like the number of words) that works best.

## Databases and PySpark

The checks of `pb.text.no_pii()` are regular expressions, so they run on any backend. The other checks run in Python: a Polars or pandas table is checked as it is, and for a database table (through Ibis) or a PySpark DataFrame, the distinct values of the step's columns are collected, checked, and the results are joined back to the table on its own backend. With a large table, a `pre=` function that takes a sample of the rows keeps the cost down. For [col_vals_chat_format()](../../reference/Validate.col_vals_chat_format.md#pointblank.Validate.col_vals_chat_format), the conversations in a database table must be JSON text.

## Conclusion

Text data has its own kinds of problems, and Pointblank checks them with the same reports, thresholds, and actions as any other validation. The `pb.text` functions make bundles of steps for the heuristic quality filters, the language and length of the text, and personal information; [rows_near_distinct()](../../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct) finds near duplicates; [col_vals_chat_format()](../../reference/Validate.col_vals_chat_format.md#pointblank.Validate.col_vals_chat_format) checks conversations for fine-tuning; and [rows_not_contaminated()](../../reference/Validate.rows_not_contaminated.md#pointblank.Validate.rows_not_contaminated) keeps benchmark questions out of the training data.
