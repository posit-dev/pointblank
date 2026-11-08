"""
The text of the reports of ablations, data cards, and checks of splits and synthetic data, in each
of the reporting languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pointblank._constants import RTL_LANGUAGES
from pointblank._constants_translations import MODEL_DATA_TEXT

# Languages that don't put spaces between sentences
_NO_SENTENCE_SPACE = ("zh-Hans", "zh-Hant", "ja")


def _lookup(entry: dict[str, str], lang: str) -> str | None:
    """Find a language's text in a translation entry (language codes match case-insensitively)."""
    if lang in entry:
        return entry[lang]
    lang_lower = lang.lower()
    for code, text in entry.items():
        if code.lower() == lang_lower:
            return text
    return None


def _locale_marks(locale: str) -> tuple[str, str]:
    """The thousands separator and decimal mark for a locale (falling back to `,` and `.`)."""
    try:
        from great_tables._formats import _get_locale_dec_mark, _get_locale_sep_mark

        return (
            _get_locale_sep_mark(default=",", use_seps=True, locale=locale),
            _get_locale_dec_mark(default=".", locale=locale),
        )
    except Exception:  # pragma: no cover
        return ",", "."


@dataclass(frozen=True)
class Text:
    """
    A text that's translated when it's shown: a key of `MODEL_DATA_TEXT`, and the values of its
    fields (which can be texts themselves). An integer field is written with the locale's
    thousands separator, and a field `n` of 1 picks the singular form (`<key>_one`), if the
    language has one.
    """

    key: str
    fields: tuple[tuple[str, Any], ...] = ()
    capitalize: bool = False


def msg(key: str, /, **fields: Any) -> Text:
    return Text(key, tuple(fields.items()))


def capitalized(text: Text | str) -> Text | str:
    if isinstance(text, Text):
        return Text(text.key, text.fields, capitalize=True)
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class Num:
    """
    A number for a field of a text, written with a format spec and the locale's marks (without
    a spec, a whole number is written without decimals, e.g., `5` for `5.0`).
    """

    value: float
    spec: str = ""
    suffix: str = ""


class Localizer:
    """Write the texts of a report in a language, with the numbers formatted for a locale."""

    def __init__(self, lang: str = "en", locale: str | None = None) -> None:
        self.lang = lang
        self.locale = locale or lang
        self.sep, self.dec = _locale_marks(self.locale)

    @property
    def rtl(self) -> bool:
        return self.lang in RTL_LANGUAGES

    def t(self, key: str, /, **fields: Any) -> str:
        """The text of `key` (in English if it has no translation), with its fields filled."""
        n = fields.get("n")
        text = None
        if isinstance(n, int) and not isinstance(n, bool) and n == 1:
            text = _lookup(MODEL_DATA_TEXT.get(f"{key}_one", {}), self.lang)
        if text is None:
            entry = MODEL_DATA_TEXT[key]
            text = _lookup(entry, self.lang) or entry["en"]
        if not fields:
            return text
        return text.format(**{name: self.value(value) for name, value in fields.items()})

    def r(self, text: Text | str) -> str:
        """Write a text (a string is written as it is)."""
        if isinstance(text, str):
            return text
        out = self.t(text.key, **dict(text.fields))
        return out[:1].upper() + out[1:] if text.capitalize else out

    def value(self, value: Any) -> Any:
        if isinstance(value, Text):
            return self.r(value)
        if isinstance(value, Num):
            return self.num(value.value, value.spec) + value.suffix
        if isinstance(value, int) and not isinstance(value, bool):
            return self.int(value)
        if isinstance(value, (list, tuple)):
            # A list of step numbers (e.g., `1, 2, 3`)
            return self.list_sep.join(str(item) for item in value)
        return value

    # Numbers -----------------------------------------------------------------------------------

    def marks(self, text: str) -> str:
        """Swap the `,` and `.` of a number written in Python for the locale's marks."""
        return text.replace(",", "\0").replace(".", self.dec).replace("\0", self.sep)

    def int(self, n: int) -> str:
        return self.marks(f"{n:,}")

    def num(self, x: float, spec: str = "") -> str:
        if spec:
            text = format(x, spec)
        else:
            x = float(x)
            text = str(int(x)) if x.is_integer() and abs(x) < 1e15 else repr(x)
        return self.marks(text)

    def pct(self, share: float, digits: int = 1) -> str:
        """A share as a percentage (e.g., `"12.5%"`)."""
        return self.marks(f"{share:.{digits}%}")

    def pct_bounded(self, share: float) -> str:
        """A share as a percentage, without rounding to 100% or 0% unless it's exact."""
        if 0 < share < 0.001:
            return "<" + self.marks("0.1%")
        if 0.999 < share < 1:
            return ">" + self.marks("99.9%")
        return self.pct(share)

    def interval(self, lower: str, upper: str) -> str:
        """An interval, as `[lower, upper]` (or `[lower; upper]`, if the decimal mark is `,`)."""
        return f"[{lower}{';' if self.dec == ',' else ','} {upper}]"

    # Lists and sentences -----------------------------------------------------------------------

    def sentences(self, parts: list[str]) -> str:
        return ("" if self.lang in _NO_SENTENCE_SPACE else " ").join(part for part in parts if part)

    def and_list(self, items: list[str]) -> str:
        """A list of items, with "and" before the last (e.g., `"a, b, and c"`)."""
        if len(items) < 2:
            return "".join(items)
        if len(items) == 2:
            return self.t("and_two", first=items[0], last=items[1])
        return self.t("and_many", first=self.list_sep.join(items[:-1]), last=items[-1])

    @property
    def list_sep(self) -> str:
        if self.lang in _NO_SENTENCE_SPACE:
            return "、"
        return "، " if self.lang in ("ar", "fa") else ", "

    @property
    def clause_sep(self) -> str:
        """The separator of the clauses of a sentence (a semicolon)."""
        if self.lang in _NO_SENTENCE_SPACE:
            return "；"
        if self.lang in ("ar", "fa"):
            return "؛ "
        return " ; " if self.lang == "fr" else "; "


ENGLISH = Localizer()


def report_localizer(lang: str | None, locale: str | None, default: Any = None) -> Localizer:
    """
    A `Localizer` for a report's `lang=` and `locale=`. Without `lang=`, the report is in the
    language (and locale) of the `default` validation, if one is given, or in English.
    """
    from pointblank.validate import _normalize_reporting_language

    if lang is None and default is not None:
        lang = getattr(default, "lang", None) or "en"
        if locale is None:
            locale = getattr(default, "locale", None)
    return Localizer(lang=_normalize_reporting_language(lang=lang), locale=locale)
