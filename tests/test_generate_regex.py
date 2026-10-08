from __future__ import annotations

import random
import re

from pointblank.generate.regex import (
    _build_categories,
    _parse_in,
    _generate_one,
    generate_from_regex,
    CATEGORIES,
)

try:
    import re._parser as sre_parse
except ImportError:
    from re import sre_parse


def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ──────────────────────────────────────────────────────────────────────
# _build_categories
# ──────────────────────────────────────────────────────────────────────


def test_build_categories_returns_dict():
    cats = _build_categories()
    assert isinstance(cats, dict)


def test_build_categories_has_all_keys():
    cats = _build_categories()
    assert sre_parse.CATEGORY_SPACE in cats
    assert sre_parse.CATEGORY_NOT_SPACE in cats
    assert sre_parse.CATEGORY_DIGIT in cats
    assert sre_parse.CATEGORY_NOT_DIGIT in cats
    assert sre_parse.CATEGORY_WORD in cats
    assert sre_parse.CATEGORY_NOT_WORD in cats
    assert "category_any" in cats


def test_build_categories_digit_chars():
    cats = _build_categories()
    digits = cats[sre_parse.CATEGORY_DIGIT]
    assert set(digits) == set("0123456789")


def test_categories_module_level_constant():
    assert isinstance(CATEGORIES, dict)
    assert "category_any" in CATEGORIES


# ──────────────────────────────────────────────────────────────────────
# generate_from_regex - high level
# ──────────────────────────────────────────────────────────────────────


def test_generate_from_regex_literal():
    rng = make_rng(1)
    result = generate_from_regex(r"hello", rng)
    assert result == "hello"


def test_generate_from_regex_digit_range():
    rng = make_rng(10)
    for _ in range(20):
        result = generate_from_regex(r"\d", rng)
        assert result.isdigit()


def test_generate_from_regex_word_chars():
    rng = make_rng(7)
    for _ in range(30):
        result = generate_from_regex(r"\w", rng)
        assert re.fullmatch(r"\w", result)


def test_generate_from_regex_space():
    rng = make_rng(3)
    for _ in range(10):
        result = generate_from_regex(r"\s", rng)
        assert result in " \t\n\r\f\v"


def test_generate_from_regex_non_space():
    rng = make_rng(3)
    for _ in range(10):
        result = generate_from_regex(r"\S", rng)
        assert result not in " \t\n\r\f\v"


def test_generate_from_regex_non_digit():
    rng = make_rng(5)
    for _ in range(20):
        result = generate_from_regex(r"\D", rng)
        assert not result.isdigit()


def test_generate_from_regex_any():
    rng = make_rng(9)
    for _ in range(10):
        result = generate_from_regex(r".", rng)
        assert len(result) == 1


def test_generate_from_regex_character_class():
    rng = make_rng(2)
    for _ in range(20):
        result = generate_from_regex(r"[abc]", rng)
        assert result in "abc"


def test_generate_from_regex_character_class_range():
    rng = make_rng(2)
    for _ in range(20):
        result = generate_from_regex(r"[a-z]", rng)
        assert result.islower()


def test_generate_from_regex_negated_class():
    rng = make_rng(4)
    for _ in range(20):
        result = generate_from_regex(r"[^abc]", rng)
        assert result not in "abc"


def test_generate_from_regex_negated_class_range():
    rng = make_rng(4)
    for _ in range(20):
        result = generate_from_regex(r"[^a-z]", rng)
        assert not result.islower()


def test_generate_from_regex_quantifier_exact():
    rng = make_rng(1)
    result = generate_from_regex(r"[A-Z]{3}", rng)
    assert len(result) == 3
    assert result.isupper()


def test_generate_from_regex_quantifier_range():
    rng = make_rng(1)
    for _ in range(10):
        result = generate_from_regex(r"\d{2,4}", rng)
        assert 2 <= len(result) <= 4
        assert result.isdigit()


def test_generate_from_regex_quantifier_star():
    rng = make_rng(1)
    result = generate_from_regex(r"\d*", rng, limit=5)
    assert all(c.isdigit() for c in result)


def test_generate_from_regex_quantifier_plus():
    rng = make_rng(1)
    result = generate_from_regex(r"\d+", rng, limit=5)
    assert len(result) >= 1
    assert result.isdigit()


def test_generate_from_regex_optional():
    rng = make_rng(1)
    for _ in range(20):
        result = generate_from_regex(r"\d?", rng)
        assert len(result) in (0, 1)
        assert result == "" or result.isdigit()


def test_generate_from_regex_alternation():
    rng = make_rng(7)
    for _ in range(20):
        result = generate_from_regex(r"(foo|bar|baz)", rng)
        assert result in ("foo", "bar", "baz")


def test_generate_from_regex_group():
    rng = make_rng(1)
    result = generate_from_regex(r"(abc)", rng)
    assert result == "abc"


def test_generate_from_regex_lookahead():
    rng = make_rng(1)
    result = generate_from_regex(r"(?=abc)abc", rng)
    assert "abc" in result


def test_generate_from_regex_anchors_ignored():
    rng = make_rng(1)
    result = generate_from_regex(r"^\d+$", rng, limit=5)
    assert result.isdigit()


def test_generate_from_regex_not_literal():
    rng = make_rng(4)
    for _ in range(20):
        result = generate_from_regex(r"[^x]", rng)
        assert result != "x"
        assert len(result) == 1


def test_generate_from_regex_backreference():
    rng = make_rng(1)
    result = generate_from_regex(r"(ab)\1", rng)
    assert result == "abab"


def test_generate_from_regex_negative_lookahead():
    rng = make_rng(1)
    result = generate_from_regex(r"(?!abc)\w+", rng, limit=5)
    assert isinstance(result, str)


def test_generate_from_regex_complex_pattern():
    rng = make_rng(23)
    result = generate_from_regex(r"[A-Z]{3}-\d{4}", rng)
    assert re.fullmatch(r"[A-Z]{3}-\d{4}", result)


def test_generate_from_regex_email_like():
    rng = make_rng(99)
    result = generate_from_regex(r"\w+@\w+\.com", rng)
    assert "@" in result
    assert result.endswith(".com")


def test_generate_from_regex_reproducible():
    result1 = generate_from_regex(r"\d{5}", make_rng(42))
    result2 = generate_from_regex(r"\d{5}", make_rng(42))
    assert result1 == result2


def test_generate_from_regex_category_in_class():
    rng = make_rng(5)
    for _ in range(20):
        result = generate_from_regex(r"[\d]", rng)
        assert result.isdigit()


def test_generate_from_regex_negated_category_in_class():
    rng = make_rng(5)
    for _ in range(20):
        result = generate_from_regex(r"[^\d]", rng)
        assert not result.isdigit()


def test_generate_from_regex_word_class_in_bracket():
    rng = make_rng(5)
    for _ in range(20):
        result = generate_from_regex(r"[\w]", rng)
        assert re.fullmatch(r"\w", result)


def test_generate_from_regex_limit_parameter():
    rng = make_rng(1)
    result = generate_from_regex(r"\d+", rng, limit=3)
    assert len(result) <= 3


def test_generate_from_regex_min_repeat_zero_min():
    rng = make_rng(42)
    for _ in range(10):
        result = generate_from_regex(r"a{0,3}", rng)
        assert len(result) <= 3
        assert all(c == "a" for c in result)


def test_generate_one_category_opcode_directly():
    rng = make_rng(42)
    cat_item = (sre_parse.CATEGORY, sre_parse.CATEGORY_DIGIT)
    result = _generate_one([cat_item], rng)
    assert result.isdigit()


def test_generate_one_category_word_directly():
    rng = make_rng(7)
    cat_item = (sre_parse.CATEGORY, sre_parse.CATEGORY_WORD)
    result = _generate_one([cat_item], rng)
    assert re.fullmatch(r"\w", result)


def test_generate_one_category_space_directly():
    rng = make_rng(3)
    cat_item = (sre_parse.CATEGORY, sre_parse.CATEGORY_SPACE)
    result = _generate_one([cat_item], rng)
    assert result in " \t\n\r\f\v"


def test_generate_one_unknown_opcode_skipped():
    rng = make_rng(1)
    unknown_op = 9999
    unknown_item = (unknown_op, None)
    result = _generate_one([unknown_item], rng)
    assert result == ""


def test_generate_one_unknown_opcode_with_literals():
    rng = make_rng(1)
    items = [
        (sre_parse.LITERAL, ord("a")),
        (9999, None),
        (sre_parse.LITERAL, ord("b")),
    ]
    result = _generate_one(items, rng)
    assert result == "ab"
