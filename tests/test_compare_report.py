"""
Tests for the table comparison report (`Comparison.get_tabular_report()` and `tbl_match()` step
reports): translations, locale-aware numbers, right-to-left layout, and HTML snapshots.
"""

from __future__ import annotations

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
        comparison.get_tabular_report(lang=lang).as_raw_html(), f"comparison_report_{lang}.html"
    )


def test_report_html_snapshot_positional_and_multiset(snapshot):
    s = pl.DataFrame({"a": [1, 2, 3, 3], "b": ["w", "x", "y", "y"]})
    t = pl.DataFrame({"a": [1, 2, 9], "b": ["w", "X", "y"]})
    snapshot.assert_match(
        pb.compare(s, t).get_tabular_report().as_raw_html(), "comparison_report_positional.html"
    )
    snapshot.assert_match(
        pb.compare(s, t, keys="*").get_tabular_report().as_raw_html(),
        "comparison_report_multiset.html",
    )


def test_report_html_snapshot_identical(snapshot):
    df = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    snapshot.assert_match(
        pb.compare(df, df, keys="id").get_tabular_report().as_raw_html(),
        "comparison_report_identical.html",
    )


def test_step_report_html_snapshot(snapshot):
    validation = (
        pb.Validate(data=TARGET, tbl_name="orders")
        .tbl_match(tbl_compare=SOURCE, keys="id", tolerance={"amount": 0.1}, schema="common")
        .interrogate()
    )
    snapshot.assert_match(
        validation.get_step_report(i=1).as_raw_html(), "tbl_match_step_report.html"
    )
