"""
Tests for the table comparison report (`Comparison.get_tabular_report()` and `tbl_match()` step
reports): translations, locale-aware numbers, right-to-left layout, and HTML snapshots.
"""

from __future__ import annotations

import datetime as dt
import decimal
import html as html_lib
import re
import string

import polars as pl
import pytest

import pointblank as pb
from pointblank._constants import REPORTING_LANGUAGES, RTL_LANGUAGES
from pointblank._constants_translations import COMPARISON_REPORT_TEXT


# A comparison that exercises every report section: a schema difference, a column mismatch,
# changed/missing/extra rows, null keys, and a duplicated key (compared pairwise)
SOURCE = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 6, 6, None],
        "amount": [10.0, 20.0, 30.0, float("nan"), 1.0, 2.0, 9.0],
        "region": ["N", "S", "E", "W", "x", "y", "z"],
        "legacy": [0] * 7,
    }
)
TARGET = pl.DataFrame(
    {
        "id": [1, 2, 4, 5, 6, None],
        "amount": [10.0, 20.5, None, 50.0, 1.0, 8.0],
        "region": ["N", "S", "W", "N", "x", "q"],
    }
)


@pytest.fixture(scope="module")
def comparison():
    return pb.compare(SOURCE, TARGET, keys="id", dup_keys="compare", tolerance={"amount": 0.1})


def _text(gt_tbl) -> str:
    html_str = gt_tbl.as_raw_html()
    html_str = html_str[html_str.find("<table") :]
    text = re.sub(r"<[^>]+>", " ", html_str)
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


def _html(gt_tbl) -> str:
    """Raw report HTML without great-tables' default stylesheet, which changes between releases."""
    return re.sub(r"<style>.*?</style>", "", gt_tbl.as_raw_html(), count=1, flags=re.DOTALL)


# ----------------------------------------------------------------------------------------------
# The translation table
# ----------------------------------------------------------------------------------------------


def _fields(text: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def test_translations_cover_all_languages_with_matching_placeholders():
    for key, entry in COMPARISON_REPORT_TEXT.items():
        if key.endswith("_one"):
            # Singular forms are optional per language but must be valid languages
            assert "en" in entry, key
            assert set(entry) <= set(REPORTING_LANGUAGES), key
        else:
            assert set(entry) == set(REPORTING_LANGUAGES), key
        expected = _fields(entry["en"])
        for lang, text in entry.items():
            assert _fields(text) == expected, (key, lang)


# ----------------------------------------------------------------------------------------------
# Rendering in every language
# ----------------------------------------------------------------------------------------------

_ENGLISH_SECTION_TITLES = [
    "Schema differences",
    "Column mismatches",
    "Changed rows",
    "Missing from target",
    "Extra in target",
    "Duplicate keys",
]


@pytest.mark.parametrize("lang", REPORTING_LANGUAGES)
def test_report_renders_in_every_language(comparison, lang):
    text = _text(comparison.get_tabular_report(lang=lang))

    assert COMPARISON_REPORT_TEXT["title"][lang] in text
    for key in ("group_schema", "group_changed", "group_missing", "group_extra", "group_dup"):
        assert COMPARISON_REPORT_TEXT[key][lang] in text, key
    if lang != "en":
        for english in _ENGLISH_SECTION_TITLES:
            if english not in COMPARISON_REPORT_TEXT["title"][lang]:
                assert english not in text, (lang, english)

    html_str = comparison.get_tabular_report(lang=lang).as_raw_html()
    assert ("direction: rtl" in html_str or "direction:rtl" in html_str) == (lang in RTL_LANGUAGES)


def test_language_codes_are_case_insensitive_and_validated(comparison):
    assert "Tabellenvergleich" in _text(comparison.get_tabular_report(lang="DE"))
    assert "表比较" in _text(comparison.get_tabular_report(lang="zh-hans"))
    with pytest.raises(ValueError, match="reporting language"):
        comparison.get_tabular_report(lang="xx")


def test_locale_number_formatting():
    s = pl.DataFrame({"id": range(12345), "v": 0})
    cmp = pb.compare(s, s.with_columns(v=pl.lit(1)), keys="id")

    assert "12,345" in _text(cmp.get_tabular_report())
    german = _text(cmp.get_tabular_report(lang="de"))
    assert "12.345" in german
    assert "100,0%" in german
    # The locale can differ from the language
    assert "12,345" in _text(cmp.get_tabular_report(lang="de", locale="en"))


def test_singular_forms(comparison):
    assert "Changed rows (1 row)" in _text(comparison.get_tabular_report())
    assert "Lignes modifiées (1 ligne)" in _text(comparison.get_tabular_report(lang="fr"))
    assert "Geänderte Zeilen (1 Zeile)" in _text(comparison.get_tabular_report(lang="de"))

    one = pl.DataFrame({"id": [1]})
    assert "The 1 row matches." in _text(pb.compare(one, one, keys="id").get_tabular_report())


# ----------------------------------------------------------------------------------------------
# Values and the header
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new, expected",
    [
        # Floats get as many digits as it takes to tell them apart, in positional notation
        (1234567.891, 1234568.0, ("1234567.9", "1234568")),
        (40.5, 45.5, ("40.5", "45.5")),
        (0.1 + 0.2, 0.3, ("0.30000000000000004", "0.3")),
        (2.5e9, 2.5e9 + 0.5, ("2500000000", "2500000000.5")),
        (1e20, 1.0000001e20, ("1e+20", "1.0000001e+20")),
        # Long strings are shown around a difference that would be cut off
        ("x" * 60 + "1", "x" * 60 + "2", ("…" + "x" * 38 + "1", "…" + "x" * 38 + "2")),
        ("short", "s" * 50, ("short", "s" * 39 + "…")),
        # Differences in whitespace alone are made visible
        ("abc", "abc ", ("'abc'", "'abc␣'")),
        ("a\tb", "a b", ("'a\\tb'", "'a␣b'")),
        (None, 1.5, ("NULL", "1.5")),
    ],
)
def test_changed_values_show_the_difference(old, new, expected):
    from pointblank._compare_report import _pair_texts

    assert _pair_texts(old, new) == expected


@pytest.mark.parametrize(
    "old, new, expected",
    [
        (40.5, 45.5, "+5"),
        (1234567.891, 1234568.0, "+0.109"),
        (3, 300, "+297"),
        (10, 4, "-6"),
        (dt.datetime(2024, 6, 1, 12), dt.datetime(2024, 6, 1, 15), "+3h"),
        (dt.datetime(2024, 6, 1, 12), dt.datetime(2024, 6, 1, 10, 30), "-1h 30m"),
        (dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 2, 2, 3), "+1d 2h"),
        (dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1, 0, 0, 0, 250_000), "+250ms"),
        (dt.date(2024, 1, 1), dt.date(2024, 3, 1), "+60d"),
        (dt.timedelta(hours=1), dt.timedelta(hours=1, seconds=90), "+1m 30s"),
        (decimal.Decimal("1.50"), decimal.Decimal("2.25"), "+0.75"),
        # No delta for other values, or values that can't be subtracted
        (True, False, None),
        ("a", "b", None),
        (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), None),
    ],
)
def test_changed_value_deltas(old, new, expected):
    from pointblank._compare_report import _delta_text

    assert _delta_text(old, new) == expected


def test_large_float_changes_are_visible_in_the_report():
    s = pl.DataFrame({"id": [1], "v": [1234567.891]})
    t = pl.DataFrame({"id": [1], "v": [1234568.0]})
    text = _text(pb.compare(s, t, keys="id").get_tabular_report())

    assert "1234567.9 → 1234568 (+0.109)" in text
    assert "e+06" not in text


def test_header_dims_absent_statuses_and_is_compact_when_identical(comparison):
    df = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    identical = pb.compare(df, df, keys="id").get_tabular_report().as_raw_html()
    assert "IDENTICAL" in identical
    # No status bar or legend: the verdict already says that every row matches
    assert "match <strong>" not in identical
    assert "<br>" not in identical

    changed = pb.compare(df, df.with_columns(v=pl.lit("z")), keys="id")
    html_str = changed.get_tabular_report().as_raw_html()
    legend = re.findall(r"<span style='white-space:nowrap; margin-right:12px;([^']*)'>", html_str)
    # match, changed, missing, extra: only `changed` has rows
    assert ["opacity" in style for style in legend] == [True, False, True, True]


def test_multiset_rows_show_surplus_copies_only_when_more_than_one():
    s = pl.DataFrame({"a": [1, 1, 1, 2]})
    t = pl.DataFrame({"a": [1, 3]})
    html_str = pb.compare(s, t, keys="*").get_tabular_report().as_raw_html()

    assert "&times;2" in html_str
    assert "&times;1" not in html_str


# ----------------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------------


def _grid_lines(html_str: str) -> list[list[tuple[str | None, str]]]:
    """The cells (width, text) of each grid line in a report."""
    lines = []
    for line in re.findall(
        r"<div style='white-space:nowrap; line-height:1.7;.*?'>(.*?)</div>", html_str
    ):
        cells = re.findall(
            r"<span(?: title='[^']*')? style='display:inline-block;(?: width:(\d+)ch;)?[^']*'>(.*?)</span>(?=<span(?: title='[^']*')? style='display:inline-block|$)",
            line,
        )
        lines.append([(w or None, html_lib.unescape(re.sub(r"<[^>]+>", "", t))) for w, t in cells])
    return lines


def test_sample_sections_are_aligned_grids_under_column_names():
    s = pl.DataFrame(
        {"id": [0, 1, 22, 333], "name": ["q", "a", "a much longer name", "c"], "v": [0, 1, 2, 3]}
    )
    t = pl.DataFrame({"id": [0, 9999], "name": ["q", "z"], "v": [0, 4]})
    html_str = pb.compare(s, t, keys="id").get_tabular_report().as_raw_html()
    lines = _grid_lines(html_str)

    # Column names head the section (and label the key column), then one line per row, with
    # each column as wide as its widest value
    assert ["id"] in [[text for _, text in line] for line in lines]
    header = next(line for line in lines if [text for _, text in line] == ["name", "v"])
    rows = [
        line for line in lines if len(line) == 2 and line[0][1] in {"a", "a much longer name", "c"}
    ]
    assert len(rows) == 3
    assert {line[0][0] for line in rows + [header]} == {str(len("a much longer name"))}
    # The key column is labeled, and only as wide as it needs to be
    assert '<col style="width:30%;"/>' not in html_str
    assert "white-space: nowrap; padding-inline-end: 16px;" in html_str


def test_changed_rows_are_one_aligned_line_per_changed_value():
    s = pl.DataFrame({"id": [1, 2], "a": [1, 2], "b": ["x", "y"]})
    t = pl.DataFrame({"id": [1, 2], "a": [10, 2], "b": ["xx", "z"]})
    html_str = (
        pb.compare(s, t, keys="id", source_name="old", target_name="new")
        .get_tabular_report()
        .as_raw_html()
    )
    lines = [[text for _, text in line] for line in _grid_lines(html_str)]

    # The table names label the values before and after
    assert ["", "old", "", "new"] in lines
    assert ["a", "1", "→", "10", "(+9)"] in lines
    assert ["b", "x", "→", "xx"] in lines
    assert ["b", "y", "→", "z"] in lines


def test_changed_rows_list_at_most_eight_changed_values():
    cols = {f"c{i}": [0] for i in range(10)}
    s = pl.DataFrame({"id": [1], **cols})
    t = s.with_columns(pl.exclude("id") + 1)
    text = _text(pb.compare(s, t, keys="id").get_tabular_report())
    changed = text[text.index("Changed rows") :]

    assert "c7 0 → 1" in changed
    assert "c8" not in changed
    assert changed.endswith("+2 more")


def test_sample_grids_show_as_many_columns_as_fit():
    def header(s, t):
        lines = _grid_lines(pb.compare(s, t, keys="id").get_tabular_report().as_raw_html())
        return next(texts for texts in ([t for _, t in line] for line in lines) if "c0" in texts)

    # Up to eight narrow columns, but fewer wide ones, and fewer still beside a long key
    narrow = pl.DataFrame({"id": [1, 2], **{f"c{i}": [i, i] for i in range(10)}})
    assert header(narrow, narrow.head(1)) == [f"c{i}" for i in range(8)] + ["+2 more"]

    wide = narrow.with_columns(pl.exclude("id").cast(pl.String).str.pad_end(30, "x"))
    assert header(wide, wide.head(1)) == ["c0", "c1", "c2", "c3", "+6 more"]

    long_key = wide.with_columns(id=pl.Series(["k" * 40, "j" * 40]))
    assert header(long_key, long_key.head(1)) == ["c0", "c1", "+8 more"]


def test_composite_keys_are_aligned_columns():
    s = pl.DataFrame({"region": ["W", "N", "South"], "id": [7, 1, 200], "v": [0, 1, 2]})
    t = pl.DataFrame({"region": ["W", "E"], "id": [7, 3], "v": [0, 3]})
    lines = _grid_lines(pb.compare(s, t, keys=["region", "id"]).get_tabular_report().as_raw_html())

    key_lines = [
        line
        for line in lines
        if [text for _, text in line] in (["region", "id"], ["N", "1"], ["South", "200"])
    ]
    # (the key columns head both the missing and the extra rows)
    assert len(key_lines) == 4
    assert {line[0][0] for line in key_lines} == {"6"}


def test_section_labels_are_colored_by_status(comparison):
    from pointblank._compare_report import STATUS_COLORS

    html_str = comparison.get_tabular_report().as_raw_html()
    for title, status in [
        ("Changed rows", "changed"),
        ("Missing from target", "missing"),
        ("Extra in target", "extra"),
        ("Duplicate keys", "dup_key"),
    ]:
        heading = re.search(rf"<th class=\"gt_group_heading\"[^>]*>{title}", html_str).group(0)
        assert f"border-inline-start: 4px solid {STATUS_COLORS[status]};" in heading, title


def test_right_to_left_layout(comparison):
    html_str = comparison.get_tabular_report(lang="ar").as_raw_html()
    # The columns run right to left, arrows point from source to target, and data stays left to
    # right inside its cell
    assert "#pb_comparison_tbl .gt_table { direction: rtl; }" in html_str
    assert "&larr;" in html_str and "&rarr;" not in html_str
    assert "<bdi dir='ltr'>" in html_str
    assert "&larr;" not in comparison.get_tabular_report().as_raw_html()


def test_verdict_flags_comparisons_without_common_columns():
    src = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    tgt = pl.DataFrame({"sku": ["A", "B", "C"], "price": [1.0, 2.0, 3.0]})

    common = _text(pb.compare(src, tgt, schema="common").get_tabular_report())
    assert "DIFFERENT" in common
    assert "NO COLUMNS COMPARED" in common
    assert "no shared columns to compare" in common
    assert "SCHEMA DIFFERS" not in common

    # Under the strict check, both badges show, with the more specific note
    strict = _text(pb.compare(src, tgt, keys="*").get_tabular_report())
    assert "SCHEMA DIFFERS" in strict
    assert "NO COLUMNS COMPARED" in strict
    assert "strict schema check" not in strict

    french = _text(pb.compare(src, tgt, schema="common").get_tabular_report(lang="fr"))
    assert COMPARISON_REPORT_TEXT["no_columns_compared"]["fr"] in french

    # A keyed comparison of the keys alone is still meaningful
    keyed = pl.DataFrame({"id": [1, 2, 3], "sku": ["A", "B", "C"]})
    text = _text(pb.compare(src, keyed, keys="id", schema="common").get_tabular_report())
    assert "IDENTICAL" in text
    assert "NO COLUMNS COMPARED" not in text


def test_step_report_follows_validation_language():
    validation = (
        pb.Validate(data=TARGET, lang="fr", locale="fr")
        .tbl_match(tbl_compare=SOURCE, keys="id", schema="common")
        .interrogate()
    )
    text = _text(validation.get_step_report(i=1))
    assert "Rapport pour l'étape de validation 1" in text
    assert "Lignes modifiées" in text
    assert "Changed rows" not in text


# ----------------------------------------------------------------------------------------------
# Degrees of difference
# ----------------------------------------------------------------------------------------------

BASE = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "b": ["x", "y", "z"]})
TWO_CHANGED = BASE.with_columns(a=pl.when(pl.col("id") < 3).then(pl.col("a") + 10).otherwise("a"))


@pytest.mark.parametrize(
    "source, target, kwargs, expected",
    [
        (BASE, BASE, {"keys": "id"}, None),
        (BASE, TWO_CHANGED, {"keys": "id"}, "2 of the 6 compared values differ."),
        (
            BASE,
            BASE.with_columns(a=pl.when(pl.col("id") == 1).then(9).otherwise("a")),
            {"keys": "id"},
            "1 of the 6 compared values differs.",
        ),
        (
            BASE,
            BASE.with_columns(a=pl.col("a") + 1, b=pl.col("b") + "!"),
            {"keys": "id"},
            "Every compared value differs.",
        ),
        # Missing and extra rows alone don't need a summary
        (BASE, BASE.head(2), {"keys": "id"}, None),
        (
            BASE,
            BASE.with_columns(id=pl.col("id") + 100),
            {"keys": "id"},
            "The tables have no keys in common.",
        ),
        (
            BASE,
            BASE.with_columns(a=pl.col("a") + 1),
            {"keys": "*"},
            "The tables have no rows in common.",
        ),
        (
            BASE,
            pl.DataFrame({"id": [1, 2], "sku": ["A", "B"]}),
            {"keys": "id", "schema": "common"},
            "The tables have no columns in common besides the keys.",
        ),
        (BASE, BASE.clear(), {"keys": "id"}, "The target table is empty."),
        (BASE.clear(), BASE, {"keys": "id"}, "The source table is empty."),
        (BASE.clear(), BASE.clear().drop("b"), {"keys": "id"}, "Both tables are empty."),
        # (the verdict already says that no columns were compared)
        (BASE, pl.DataFrame({"sku": ["A"]}), {}, None),
    ],
)
def test_summary_says_how_different_the_tables_are(source, target, kwargs, expected):
    from pointblank._compare_report import _Ctx, _summary_text

    cmp = pb.compare(source, target, **kwargs)
    assert _summary_text(cmp, _Ctx(unit=cmp.unit)) == expected
    if expected is not None:
        assert expected in _text(cmp.get_tabular_report())


def test_summary_is_translated():
    cmp = pb.compare(BASE, TWO_CHANGED, keys="id")
    assert "2 des 6 valeurs comparées diffèrent." in _text(cmp.get_tabular_report(lang="fr"))
    assert "2 der 6 verglichenen Werte weichen ab." in _text(cmp.get_tabular_report(lang="de"))


def test_empty_tables():
    # With no rows at all, there's no status bar or legend
    cmp = pb.compare(BASE.clear(), BASE.clear().drop("b"), keys="id")
    html_str = cmp.get_tabular_report().as_raw_html()
    assert "display:flex" not in html_str
    assert "match <strong>" not in html_str
    assert COMPARISON_REPORT_TEXT["no_rows"]["en"] not in html_str

    text = _text(pb.compare(BASE.clear(), BASE.clear(), keys="id").get_tabular_report())
    assert "Result ✓ Both tables are empty." in text
    assert "All 0 rows match" not in text


def test_rows_of_a_table_compared_with_an_empty_one_are_not_sampled():
    text = _text(pb.compare(BASE, BASE.clear(), keys="id").get_tabular_report())
    assert "Missing from target (3 rows) The target table is empty." in text
    assert "x" not in text.split("Missing from target")[1]

    text = _text(pb.compare(BASE.clear(), BASE, keys="id").get_tabular_report())
    assert "Extra in target (3 rows) The source table is empty." in text


def test_keys_are_listed_inline_when_no_keys_are_in_common():
    s = pl.DataFrame({"id": range(1, 13), "v": 0})
    text = _text(
        pb.compare(s, s.with_columns(id=pl.col("id") + 100), keys="id").get_tabular_report()
    )
    # One row per section, so that an offset (or a difference in format) is easy to see
    assert "Missing from target (showing 10 of 12 rows) id 1 · 2 · 3 · 4 · 5" in text
    assert "Extra in target (showing 10 of 12 rows) id 101 · 102 · 103" in text
    assert " v " not in text

    s = pl.DataFrame({"region": ["N", "S"], "id": [1, 2]})
    t = pl.DataFrame({"region": ["n", "s"], "id": [1, 2]})
    text = _text(pb.compare(s, t, keys=["region", "id"]).get_tabular_report())
    assert "region, id (N, 1) · (S, 2)" in text
    assert "region, id (n, 1) · (s, 2)" in text

    s = pl.DataFrame({"id": [1, None]})
    text = _text(pb.compare(s, pl.DataFrame({"id": [5]}), keys="id").get_tabular_report())
    assert "id 1 · NULL (null key: can't be matched)" in text


def test_columns_are_listed_together_when_no_columns_are_in_common():
    s = pl.DataFrame({"id": [1], "name": ["a"], "v": [1.0]})
    t = pl.DataFrame({"sku": ["A"], "price": [1.0]})
    report = pb.compare(s, t, keys="*").get_tabular_report()
    text = _text(report)
    assert "Schema differences only in source id , name , v only in target sku , price" in text

    # With a column in common, each column has its own row
    text = _text(pb.compare(s, t.with_columns(id=pl.lit(1)), keys="*").get_tabular_report())
    assert "name only in source" in text


def test_multiset_without_common_columns_notes_that_there_are_no_values():
    s = pl.DataFrame({"a": [1, 2, 3]})
    t = pl.DataFrame({"b": [1]})
    text = _text(pb.compare(s, t, keys="*").get_tabular_report())
    assert (
        "Missing from target (2 rows) (no columns in common, so there are no values to show)"
        in text
    )


# ----------------------------------------------------------------------------------------------
# HTML snapshots
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "fr", "ar"])
def test_report_html_snapshot(comparison, lang, snapshot):
    snapshot.assert_match(
        _html(comparison.get_tabular_report(lang=lang)), f"comparison_report_{lang}.html"
    )


def test_report_html_snapshot_positional_and_multiset(snapshot):
    s = pl.DataFrame({"a": [1, 2, 3, 3], "b": ["w", "x", "y", "y"]})
    t = pl.DataFrame({"a": [1, 2, 9], "b": ["w", "X", "y"]})
    snapshot.assert_match(
        _html(pb.compare(s, t).get_tabular_report()), "comparison_report_positional.html"
    )
    snapshot.assert_match(
        _html(pb.compare(s, t, keys="*").get_tabular_report()),
        "comparison_report_multiset.html",
    )


def test_report_html_snapshot_identical(snapshot):
    df = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    snapshot.assert_match(
        _html(pb.compare(df, df, keys="id").get_tabular_report()),
        "comparison_report_identical.html",
    )


def test_step_report_html_snapshot(snapshot):
    validation = (
        pb.Validate(data=TARGET, tbl_name="orders")
        .tbl_match(tbl_compare=SOURCE, keys="id", tolerance={"amount": 0.1}, schema="common")
        .interrogate()
    )
    snapshot.assert_match(_html(validation.get_step_report(i=1)), "tbl_match_step_report.html")
