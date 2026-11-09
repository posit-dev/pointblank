"""
Checks of text data, for corpora used to train or evaluate language models.

Each function here makes a [`Steps`](`pointblank.Steps`) collection of ordinary validation steps
(`col_vals_scored()` steps with the adapters of `pb.scorers`, and `col_vals_regex()` steps), to
be added to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`). The
steps have the same reports, thresholds, and actions as any other, and they can be used in an
[`Ablation`](`pointblank.Ablation`) to measure how much each check matters to a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

from pointblank import scorers
from pointblank._text import PII_DESCRIPTIONS, PII_PATTERNS
from pointblank.steps import Steps

if TYPE_CHECKING:
    from pointblank._typing import SegmentSpec
    from pointblank.actions import Actions
    from pointblank.thresholds import Thresholds

__all__ = ["heuristic_filters", "in_language", "token_length", "no_pii"]

# The entities that `no_pii()` looks for by default
DEFAULT_PII = ("email", "phone", "credit_card")


def _check_column_name(column: Any, param: str = "column") -> None:
    if not isinstance(column, str) or not column:
        raise TypeError(f"`{param}=` must be the name of a column.")


def _check_bound(value: Any, param: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise TypeError(f"`{param}=` must be a number or `None`, got {type(value).__name__}.")


def _common(
    pre: Callable | None,
    segments: SegmentSpec | None,
    thresholds: Any,
    actions: Actions | None,
    active: bool | Callable,
) -> dict[str, Any]:
    return {
        "pre": pre,
        "segments": segments,
        "thresholds": thresholds,
        "actions": actions,
        "active": active,
    }


def _number(value: float) -> str:
    return f"{value:g}" if isinstance(value, float) else f"{value:,}"


def heuristic_filters(
    column: str,
    min_words: int | None = 50,
    max_words: int | None = 100_000,
    min_mean_word_length: float | None = 3,
    max_mean_word_length: float | None = 10,
    max_symbol_ratio: float | None = 0.1,
    max_bullet_lines: float | None = 0.9,
    max_ellipsis_lines: float | None = 0.3,
    min_alpha_words: float | None = 0.8,
    min_stop_words: int | None = 2,
    max_repeated_lines: float | None = 0.3,
    stop_words: Sequence[str] | None = None,
    pre: Callable | None = None,
    segments: SegmentSpec | None = None,
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
    actions: Actions | None = None,
    active: bool | Callable = True,
) -> Steps:
    """
    Make the heuristic quality filters of web text, as validation steps.

    Text collected from the web for training language models has a lot that isn't prose: menus,
    lists of links, tables of numbers, placeholder text, and pages that repeat themselves. The
    quality filters of the Gopher models (Rae et al., 2021) catch much of it with a few simple
    statistics of each document, and this function makes a validation step for each of them,
    with the bounds used for Gopher as the defaults:

    - the number of words is between `min_words=` and `max_words=` (50 and 100,000)
    - the mean length of the words is between `min_mean_word_length=` and
      `max_mean_word_length=` (3 and 10 characters)
    - there are no more than `max_symbol_ratio=` hash symbols and ellipses for each word (0.1)
    - no more than `max_bullet_lines=` of the lines start with a bullet point (90%)
    - no more than `max_ellipsis_lines=` of the lines end with an ellipsis (30%)
    - at least `min_alpha_words=` of the words have a letter (80%)
    - the text has at least `min_stop_words=` distinct stop words (2 of the, be, to, of, and,
      that, have, and with)
    - no more than `max_repeated_lines=` of the lines repeat an earlier line (30%)

    Each check is a `~~pointblank.Validate.col_vals_scored()` step with a
    [`text_stat()`](`pointblank.scorers.text_stat`) scorer, so the report shows each one, with
    its own counts of the documents that fail, and a brief that says what it checks. Set a
    bound to `None` to leave its check out (a check with two bounds is left out when both are
    `None`). A document with a missing value passes, as do documents without the words or lines
    that a check needs (the check of the number of words catches them).

    The defaults are for English web text. Text in other languages needs other stop words (and
    perhaps other bounds), and short texts (like titles or chat messages) need a lower
    `min_words=`.

    Parameters
    ----------
    column
        The column with the text.
    min_words, max_words
        The fewest and the most words a document can have.
    min_mean_word_length, max_mean_word_length
        The bounds of the mean length of the words of a document, in characters.
    max_symbol_ratio
        The most hash symbols and ellipses that a document can have for each word.
    max_bullet_lines
        The largest share of the lines of a document that can start with a bullet point.
    max_ellipsis_lines
        The largest share of the lines of a document that can end with an ellipsis.
    min_alpha_words
        The smallest share of the words of a document that must have a letter.
    min_stop_words
        The fewest distinct stop words that a document must have.
    max_repeated_lines
        The largest share of the lines of a document that can be the same as an earlier line.
    stop_words
        The stop words, for `min_stop_words=`. By default, they're the eight English words of the
        Gopher filters.
    pre
        A function that transforms the table before each step, as in the validation methods.
    segments
        The segments of the steps, as in the validation methods.
    thresholds
        The thresholds of the steps (by default, those of the validation).
    actions
        The actions of the steps.
    active
        Whether the steps are active.

    Returns
    -------
    Steps
        The steps, to be added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's check a small corpus of web documents. The documents are short, so we'll use a lower
    `min_words=` than the default.

    ```{python}
    import pointblank as pb
    import polars as pl

    corpus = pl.DataFrame(
        {
            "text": [
                "The museum is open from nine to five, and the entry is free for children "
                "under the age of twelve, so it's a good place to spend the day with family.",
                "Home\\nAbout\\nHome\\nAbout\\nHome\\nContact us",
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

    The menu in the second row has too few words and too many repeated lines, and few stop
    words. The third row, a list of hashtags, has too many symbols for its words. The two
    documents of prose pass every check.
    """
    _check_column_name(column)
    bounds = {
        "min_words": min_words,
        "max_words": max_words,
        "min_mean_word_length": min_mean_word_length,
        "max_mean_word_length": max_mean_word_length,
        "max_symbol_ratio": max_symbol_ratio,
        "max_bullet_lines": max_bullet_lines,
        "max_ellipsis_lines": max_ellipsis_lines,
        "min_alpha_words": min_alpha_words,
        "min_stop_words": min_stop_words,
        "max_repeated_lines": max_repeated_lines,
    }
    for param, value in bounds.items():
        _check_bound(value, param)
    for low, high in (("min_words", "max_words"), ("min_mean_word_length", "max_mean_word_length")):
        if bounds[low] is not None and bounds[high] is not None and bounds[low] > bounds[high]:
            raise ValueError(f"`{low}=` must not be greater than `{high}=`.")
    if stop_words is not None:
        stop_words = list(stop_words)

    def between(low: Any, high: Any, what: str, prefix: str = "") -> str:
        if low is None and high is None:
            return ""  # (the check is left out)
        if low is not None and high is not None:
            bound = f"between {_number(low)} and {_number(high)}"
        elif low is not None:
            bound = f"at least {_number(low)}"
        else:
            bound = f"at most {_number(high)}"
        text = f"{prefix}{bound} {what}"
        return text[0].upper() + text[1:]

    def share(value: float) -> str:
        return f"{value:.0%}"

    rules = [
        ("word_count", min_words, max_words, between(min_words, max_words, "words")),
        (
            "mean_word_length",
            min_mean_word_length,
            max_mean_word_length,
            between(
                min_mean_word_length,
                max_mean_word_length,
                "characters",
                prefix="A mean word length of ",
            ),
        ),
        (
            "symbol_ratio",
            None,
            max_symbol_ratio,
            f"At most {_number(max_symbol_ratio or 0)} hash symbols and ellipses for each word",
        ),
        (
            "bullet_lines",
            None,
            max_bullet_lines,
            f"At most {share(max_bullet_lines or 0)} of the lines start with a bullet point",
        ),
        (
            "ellipsis_lines",
            None,
            max_ellipsis_lines,
            f"At most {share(max_ellipsis_lines or 0)} of the lines end with an ellipsis",
        ),
        (
            "alpha_words",
            min_alpha_words,
            None,
            f"At least {share(min_alpha_words or 0)} of the words have a letter",
        ),
        (
            "stop_words",
            min_stop_words,
            None,
            f"At least {_number(min_stop_words or 0)} distinct stop "
            f"{'word' if min_stop_words == 1 else 'words'}",
        ),
        (
            "repeated_lines",
            None,
            max_repeated_lines,
            f"At most {share(max_repeated_lines or 0)} of the lines repeat an earlier line",
        ),
    ]

    steps = Steps()
    for stat, low, high, brief in rules:
        if low is None and high is None:
            continue
        scorer = scorers.text_stat(stat, stop_words=stop_words if stat == "stop_words" else None)
        steps.col_vals_scored(
            columns=column,
            scorer=scorer,
            min_val=low,
            max_val=high,
            na_pass=True,
            cache=False,
            brief=f"{brief} in `{column}`.",
            **_common(pre, segments, thresholds, actions, active),
        )
    return steps


def in_language(
    column: str,
    expected: str | list[str] | None = None,
    expected_column: str | None = None,
    min_confidence: float = 0.5,
    backend: str | Callable[[str], Any] = "langid",
    model: str | None = None,
    na_pass: bool = False,
    cache: bool | str = True,
    pre: Callable | None = None,
    segments: SegmentSpec | None = None,
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
    actions: Actions | None = None,
    brief: str | bool | None = None,
    active: bool | Callable = True,
) -> Steps:
    """
    Make a validation step that checks that text is in the expected language.

    The step is a `~~pointblank.Validate.col_vals_scored()` step with a
    [`langid()`](`pointblank.scorers.langid`) scorer: a language identification model gives the
    probability that the text of each row is in one of the expected languages, and a row fails if
    the probability is below `min_confidence=`. The expected languages are the same for every row
    (`expected=`) or come from a column (`expected_column=`), for a corpus with documents in several
    languages.

    Parameters
    ----------
    column
        The column with the text.
    expected
        The expected language (an ISO 639 code like `"en"`), or a list of them.
    expected_column
        A column with the expected language of each row, instead of `expected=`.
    min_confidence
        The smallest probability that a text is in the expected languages for it to pass.
    backend
        The language identification model: `"langid"`, `"fasttext"` (with the path of the model
        as `model=`), or a function that gives the probabilities of a text by language (see
        [`langid()`](`pointblank.scorers.langid`)).
    model
        The path of the fastText model, with `backend="fasttext"`.
    na_pass
        Whether a row with a missing value passes.
    cache
        Where the scores are kept: in memory (`True`), in a directory, or not at all (`False`).
    pre
        A function that transforms the table before the step, as in the validation methods.
    segments
        The segments of the step, as in the validation methods.
    thresholds
        The thresholds of the step (by default, those of the validation).
    actions
        The actions of the step.
    brief
        A brief for the step.
    active
        Whether the step is active.

    Returns
    -------
    Steps
        The step, to be added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    A real check would use the `langid` package or a fastText model. To keep this example
    self-contained, the backend is a toy function that guesses the language from a few words.

    ```{python}
    import pointblank as pb
    import polars as pl

    def guess(text):
        words = set(text.lower().split())
        if words & {"the", "and", "is"}:
            return {"en": 0.9, "fr": 0.1}
        if words & {"le", "et", "est"}:
            return {"en": 0.1, "fr": 0.9}
        return {"en": 0.5, "fr": 0.5}

    docs = pl.DataFrame(
        {
            "text": ["the soup is good", "le pain est bon", "the bread is fresh"],
            "lang": ["en", "fr", "fr"],
        }
    )

    (
        pb.Validate(data=docs)
        .add_steps(pb.text.in_language("text", expected="en", backend=guess))
        .add_steps(pb.text.in_language("text", expected_column="lang", backend=guess))
        .interrogate()
    )
    ```

    The document in French fails step 1, which expects English for every row. In step 2, each
    row is checked against its own label, and the last document fails: it's labelled as French,
    but it's in English.
    """
    _check_column_name(column)
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
        raise TypeError("`min_confidence=` must be a number.")
    if not 0 <= min_confidence <= 1:
        raise ValueError(f"`min_confidence=` must be between 0 and 1, got {min_confidence}.")
    scorer = scorers.langid(
        expected=expected,
        backend=backend,
        model=model,
        column=column if expected_column is not None else None,
        expected_column=expected_column,
    )
    return Steps().col_vals_scored(
        columns=column if expected_column is None else [column, expected_column],
        scorer=scorer,
        min_val=min_confidence,
        na_pass=na_pass,
        cache=cache,
        brief=brief,
        **_common(pre, segments, thresholds, actions, active),
    )


def token_length(
    column: str,
    tokenizer: str | Any,
    min_val: int | None = None,
    max_val: int | None = None,
    na_pass: bool = False,
    cache: bool | str = True,
    pre: Callable | None = None,
    segments: SegmentSpec | None = None,
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
    actions: Actions | None = None,
    brief: str | bool | None = None,
    active: bool | Callable = True,
) -> Steps:
    """
    Make a validation step that checks the length of text in tokens.

    A language model reads text as tokens, and most have a limit on how many they can take in, so
    the documents of a corpus for training (or a set of prompts) need to be neither too short nor
    too long, in the tokens of the model's tokenizer. The step is a
    `~~pointblank.Validate.col_vals_scored()` step with a
    [`token_count()`](`pointblank.scorers.token_count`) scorer, and a row fails if its number of
    tokens is below `min_val=` or above `max_val=`.

    Parameters
    ----------
    column
        The column with the text.
    tokenizer
        The tokenizer: the name of a Hugging Face tokenizer, `"tiktoken:"` and the name of an
        encoding or a model, `"whitespace"`, a tokenizer object, or a function (see
        [`token_count()`](`pointblank.scorers.token_count`)).
    min_val, max_val
        The fewest and the most tokens that a text can have (at least one of them is needed).
    na_pass
        Whether a row with a missing value passes.
    cache
        Where the counts are kept: in memory (`True`), in a directory, or not at all (`False`).
    pre
        A function that transforms the table before the step, as in the validation methods.
    segments
        The segments of the step, as in the validation methods.
    thresholds
        The thresholds of the step (by default, those of the validation).
    actions
        The actions of the step.
    brief
        A brief for the step.
    active
        Whether the step is active.

    Returns
    -------
    Steps
        The step, to be added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    A real check would use the tokenizer of a model (e.g., `"tiktoken:cl100k_base"`). Here, the
    tokens are the words, split on whitespace.

    ```{python}
    import pointblank as pb
    import polars as pl

    prompts = pl.DataFrame(
        {"prompt": ["Summarize the article in two sentences.", "Hi", "List three uses of salt."]}
    )

    (
        pb.Validate(data=prompts)
        .add_steps(pb.text.token_length("prompt", tokenizer="whitespace", min_val=3))
        .interrogate()
    )
    ```

    The second prompt has a single token, so it fails.
    """
    _check_column_name(column)
    _check_bound(min_val, "min_val")
    _check_bound(max_val, "max_val")
    if min_val is None and max_val is None:
        raise ValueError("token_length() needs `min_val=`, `max_val=`, or both.")
    return Steps().col_vals_scored(
        columns=column,
        scorer=scorers.token_count(tokenizer),
        min_val=min_val,
        max_val=max_val,
        na_pass=na_pass,
        cache=cache,
        brief=brief,
        **_common(pre, segments, thresholds, actions, active),
    )


def no_pii(
    columns: str | list[str],
    entities: Sequence[str] | Mapping[str, str] = DEFAULT_PII,
    na_pass: bool = True,
    pre: Callable | None = None,
    segments: SegmentSpec | None = None,
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
    actions: Actions | None = None,
    active: bool | Callable = True,
) -> Steps:
    """
    Make validation steps that check that text has no personal information.

    Text collected for training or evaluating a model shouldn't have personal information in it,
    such as email addresses, phone numbers, or credit card numbers. This function makes a
    `~~pointblank.Validate.col_vals_regex()` step (with `inverse=True`) for each column and each
    kind of information: a row fails if its text has something that looks like one. The patterns use
    only the features of regular expressions that every backend has, so the checks run in the
    database (or in PySpark) like any other `~~pointblank.Validate.col_vals_regex()` step.

    The patterns find likely matches, not certain ones: a number with the shape of a credit card
    number matches, with no checksum, and an unusual format of a phone number can be missed. For
    a thorough check, use a model for named entities (e.g., with
    `~~pointblank.Validate.col_vals_scored()`).

    The entities are:

    - `"email"`: email addresses
    - `"phone"`: phone numbers of North America, with or without a country code (like
      `(555) 123-4567` or `+1 555.123.4567`)
    - `"credit_card"`: credit card numbers (Visa, Mastercard, American Express, and Discover),
      with or without spaces or dashes
    - `"ip_address"`: IPv4 addresses
    - `"us_ssn"`: US Social Security numbers (like `123-45-6789`)
    - `"iban"`: International Bank Account Numbers

    Parameters
    ----------
    columns
        The column with the text, or a list of them.
    entities
        The kinds of information to look for (by default, `"email"`, `"phone"`, and
        `"credit_card"`), or a dictionary of patterns of your own, by name (which can be mixed
        with the names above, as in `{"email": None, "employee_id": r"EMP-[0-9]{6}"}`, where
        `None` is the pattern of an entity above).
    na_pass
        Whether a row with a missing value passes (it does by default, as a missing text has no
        personal information).
    pre
        A function that transforms the table before each step, as in the validation methods.
    segments
        The segments of the steps, as in the validation methods.
    thresholds
        The thresholds of the steps (by default, those of the validation).
    actions
        The actions of the steps.
    active
        Whether the steps are active.

    Returns
    -------
    Steps
        The steps, to be added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's check a table of support tickets for personal information.

    ```{python}
    import pointblank as pb
    import polars as pl

    tickets = pl.DataFrame(
        {
            "text": [
                "The app crashes when I open the settings page.",
                "Please email me at jo.smith@example.com when it's fixed.",
                "Call me back on (555) 123-4567 after five.",
                "I was charged twice on card 4111 1111 1111 1111.",
            ]
        }
    )

    pb.Validate(data=tickets).add_steps(pb.text.no_pii("text")).interrogate()
    ```

    Each of the last three tickets fails one of the steps: the email address in step 1, the
    phone number in step 2, and the card number in step 3.
    """
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, (list, tuple)) or not columns:
        raise TypeError("`columns=` must be the name of a column or a list of them.")
    for column in columns:
        _check_column_name(column, "columns")

    if isinstance(entities, str):
        entities = [entities]
    if isinstance(entities, Mapping):
        patterns = {}
        for name, pattern in entities.items():
            if pattern is None:
                if name not in PII_PATTERNS:
                    raise ValueError(_unknown_entity(name))
                pattern = PII_PATTERNS[name]
            elif not isinstance(pattern, str) or not pattern:
                raise TypeError(f"The pattern of the entity {name!r} must be a string.")
            patterns[str(name)] = pattern
    elif isinstance(entities, (list, tuple)):
        for name in entities:
            if name not in PII_PATTERNS:
                raise ValueError(_unknown_entity(name))
        patterns = {name: PII_PATTERNS[name] for name in entities}
    else:
        raise TypeError("`entities=` must be a list of entities or a dictionary of patterns.")
    if not patterns:
        raise ValueError("`entities=` is empty.")

    steps = Steps()
    for column in columns:
        for name, pattern in patterns.items():
            what = PII_DESCRIPTIONS.get(name, f"values that look like `{name}`")
            steps.col_vals_regex(
                columns=column,
                pattern=pattern,
                inverse=True,
                na_pass=na_pass,
                brief=f"No {what} in `{column}`.",
                **_common(pre, segments, thresholds, actions, active),
            )
    return steps


def _unknown_entity(name: Any) -> str:
    return f"Unknown entity {name!r}: it must be one of {', '.join(map(repr, PII_PATTERNS))}."
