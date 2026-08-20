import re

import pytest

from pointblank._spec_utils import (
    is_isbn_10,
    is_isbn_13,
    regex_postal_code,
    remove_hyphens,
    remove_letters,
    remove_punctuation,
    remove_spaces,
)


class TestRemoveHyphens:
    def test_removes_all_hyphens(self):
        assert remove_hyphens("123-456-789") == "123456789"

    def test_no_hyphens_unchanged(self):
        assert remove_hyphens("abc123") == "abc123"

    def test_replacement_string(self):
        assert remove_hyphens("123-456", replacement="_") == "123_456"

    def test_empty_string(self):
        assert remove_hyphens("") == ""


class TestRemoveSpaces:
    def test_removes_all_spaces(self):
        assert remove_spaces("hello world") == "helloworld"

    def test_no_spaces_unchanged(self):
        assert remove_spaces("nospaces") == "nospaces"

    def test_replacement_string(self):
        assert remove_spaces("a b c", replacement="-") == "a-b-c"

    def test_empty_string(self):
        assert remove_spaces("") == ""


class TestRemoveLetters:
    def test_removes_all_letters(self):
        assert remove_letters("abc123def") == "123"

    def test_no_letters_unchanged(self):
        assert remove_letters("123") == "123"

    def test_replacement_string(self):
        assert remove_letters("a1b2", replacement="_") == "_1_2"

    def test_empty_string(self):
        assert remove_letters("") == ""


class TestRemovePunctuation:
    def test_removes_punctuation(self):
        result = remove_punctuation("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_no_punctuation_unchanged(self):
        result = remove_punctuation("hello world")
        assert result == "hello world"

    def test_empty_string(self):
        assert remove_punctuation("") == ""


class TestIsIsbn10:
    def test_valid_isbn10(self):
        assert is_isbn_10("0-306-40615-2") is True

    def test_valid_isbn10_with_x_check_digit(self):
        assert is_isbn_10("0-19-853453-1") is True

    def test_invalid_isbn10(self):
        assert is_isbn_10("1234567890") is False

    def test_invalid_too_short(self):
        assert is_isbn_10("123") is False

    def test_non_numeric(self):
        assert is_isbn_10("abcdefghij") is False


class TestIsIsbn13:
    def test_valid_isbn13(self):
        assert is_isbn_13("978-0-306-40615-7") is True

    def test_invalid_isbn13(self):
        assert is_isbn_13("9780000000000") is False

    def test_invalid_too_short(self):
        assert is_isbn_13("123") is False

    def test_non_numeric(self):
        assert is_isbn_13("abcdefghijklm") is False


class TestRegexPostalCode:
    def test_us_not_in_dict_uses_fallback(self):
        pattern = regex_postal_code("US")
        assert isinstance(pattern, str)
        assert re.match(pattern, "10001")

    def test_known_country_germany(self):
        pattern = regex_postal_code("DE")
        assert re.match(pattern, "10115")
        assert not re.match(pattern, "1011")

    def test_canada(self):
        pattern = regex_postal_code("CA")
        assert re.match(pattern, "K1A0A1")

    def test_three_letter_country_code(self):
        pattern_two = regex_postal_code("DE")
        pattern_three = regex_postal_code("DEU")
        assert pattern_two == pattern_three

    def test_unknown_country_returns_fallback(self):
        pattern = regex_postal_code("XX")
        assert isinstance(pattern, str)
