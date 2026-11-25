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
