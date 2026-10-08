import math
import re

import pytest

from pointblank._spec_utils import (
    check_credit_card,
    check_email,
    check_iban,
    check_ipv4_address,
    check_ipv6_address,
    check_isbn,
    check_mac,
    check_phone,
    check_postal_code,
    check_swift_bic,
    check_url,
    check_vin,
    is_credit_card,
    is_isbn_10,
    is_isbn_13,
    is_vin,
    luhn,
    regex_iban,
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


class TestRegexIban:
    def test_no_country_returns_generic(self):
        pattern = regex_iban()
        assert "A-Z" in pattern

    def test_known_country_gb(self):
        pattern = regex_iban("GB")
        assert "GB" in pattern

    def test_three_letter_country_aut(self):
        pattern = regex_iban("AUT")
        assert "AT" in pattern

    def test_unknown_country_returns_generic(self):
        pattern = regex_iban("ZZ")
        assert isinstance(pattern, str)
        assert "A-Z" in pattern


class TestLuhn:
    def test_valid_luhn(self):
        assert luhn("4532015112830366") is True

    def test_invalid_luhn(self):
        assert luhn("1234567890123456") is False

    def test_luhn_non_numeric_returns_false(self):
        assert luhn("453201511283036x") is False


class TestIsIsbn10Extended:
    def test_isbn10_x_check_digit(self):
        assert is_isbn_10("007462542X") is True

    def test_isbn10_invalid_format_non_digit(self):
        assert is_isbn_10("abcdefghij") is False


class TestIsIsbn13Extended:
    def test_valid_isbn13_no_hyphens(self):
        assert is_isbn_13("9780306406157") is True

    def test_isbn13_non_numeric_raises_false(self):
        assert is_isbn_13("978abc3060615") is False


class TestCheckIsbn:
    def test_valid_isbn10_in_list(self):
        assert check_isbn(["0-306-40615-2"]) == [True]

    def test_valid_isbn13_in_list(self):
        assert check_isbn(["978-0-306-40615-7"]) == [True]

    def test_none_returns_false(self):
        assert check_isbn([None]) == [False]

    def test_nan_returns_false(self):
        assert check_isbn([float("nan")]) == [False]

    def test_wrong_length_returns_false(self):
        assert check_isbn(["12345"]) == [False]

    def test_mixed_list(self):
        results = check_isbn(["0-306-40615-2", None, "12345"])
        assert results == [True, False, False]


class TestIsVin:
    def test_valid_vin(self):
        assert is_vin("1HGBH41JXMN109186") is True

    def test_invalid_vin_wrong_length(self):
        assert is_vin("1HGBH41") is False

    def test_invalid_vin_check_digit(self):
        assert is_vin("1HGBH41JXMN109187") is False


class TestCheckVin:
    def test_valid_vin_in_list(self):
        assert check_vin(["1HGBH41JXMN109186"]) == [True]

    def test_none_returns_false(self):
        assert check_vin([None]) == [False]

    def test_nan_returns_false(self):
        assert check_vin([float("nan")]) == [False]

    def test_invalid_vin_in_list(self):
        assert check_vin(["BADVIN"]) == [False]


class TestIsCreditCard:
    def test_valid_visa(self):
        assert is_credit_card("4532015112830366") is True

    def test_invalid_non_numeric(self):
        assert is_credit_card("ABCDEFGHIJKLMNOP") is False

    def test_invalid_too_short(self):
        assert is_credit_card("123456789012") is False

    def test_invalid_luhn_fails(self):
        assert is_credit_card("4532015112830360") is False


class TestCheckCreditCard:
    def test_valid_card_in_list(self):
        assert check_credit_card(["4532015112830366"]) == [True]

    def test_none_returns_false(self):
        assert check_credit_card([None]) == [False]

    def test_nan_returns_false(self):
        assert check_credit_card([float("nan")]) == [False]

    def test_invalid_in_list(self):
        assert check_credit_card(["1234567890"]) == [False]


class TestCheckIban:
    def test_valid_gb_iban(self):
        assert check_iban(["GB29NWBK60161331926819"]) == [True]

    def test_invalid_iban(self):
        assert check_iban(["NOTANIBAN"]) == [False]

    def test_none_returns_false(self):
        assert check_iban([None]) == [False]

    def test_nan_returns_false(self):
        assert check_iban([float("nan")]) == [False]

    def test_with_country_code(self):
        result = check_iban(["GB29NWBK60161331926819"], country="GB")
        assert result == [True]


class TestCheckPostalCode:
    def test_valid_german_postal(self):
        assert check_postal_code(["10115"], country="DE") == [True]

    def test_invalid_german_postal(self):
        assert check_postal_code(["1011"], country="DE") == [False]

    def test_none_returns_false(self):
        assert check_postal_code([None], country="DE") == [False]

    def test_nan_returns_false(self):
        assert check_postal_code([float("nan")], country="DE") == [False]


class TestCheckUrl:
    def test_valid_http_url(self):
        assert check_url(["https://www.example.com"]) == [True]

    def test_invalid_url(self):
        assert check_url(["not a url"]) == [False]

    def test_none_returns_false(self):
        assert check_url([None]) == [False]

    def test_nan_returns_false(self):
        assert check_url([float("nan")]) == [False]


class TestCheckIpv4:
    def test_valid_ipv4(self):
        assert check_ipv4_address(["192.168.1.1"]) == [True]

    def test_invalid_ipv4(self):
        assert check_ipv4_address(["999.999.999.999"]) == [False]

    def test_none_returns_false(self):
        assert check_ipv4_address([None]) == [False]

    def test_nan_returns_false(self):
        assert check_ipv4_address([float("nan")]) == [False]


class TestCheckIpv6:
    def test_valid_ipv6(self):
        assert check_ipv6_address(["2001:0db8:85a3:0000:0000:8a2e:0370:7334"]) == [True]

    def test_invalid_ipv6(self):
        assert check_ipv6_address(["not::an::ipv6"]) == [False]

    def test_none_returns_false(self):
        assert check_ipv6_address([None]) == [False]

    def test_nan_returns_false(self):
        assert check_ipv6_address([float("nan")]) == [False]


class TestCheckEmail:
    def test_valid_email(self):
        assert check_email(["user@example.com"]) == [True]

    def test_invalid_email(self):
        assert check_email(["notanemail"]) == [False]

    def test_none_returns_false(self):
        assert check_email([None]) == [False]

    def test_nan_returns_false(self):
        assert check_email([float("nan")]) == [False]


class TestCheckPhone:
    def test_valid_phone(self):
        assert check_phone(["+1-555-123-4567"]) == [True]

    def test_invalid_phone(self):
        assert check_phone(["abc"]) == [False]

    def test_none_returns_false(self):
        assert check_phone([None]) == [False]

    def test_nan_returns_false(self):
        assert check_phone([float("nan")]) == [False]


class TestCheckMac:
    def test_valid_mac(self):
        assert check_mac(["AA:BB:CC:DD:EE:FF"]) == [True]

    def test_invalid_mac(self):
        assert check_mac(["not-a-mac"]) == [False]

    def test_none_returns_false(self):
        assert check_mac([None]) == [False]

    def test_nan_returns_false(self):
        assert check_mac([float("nan")]) == [False]


class TestCheckSwiftBic:
    def test_valid_swift(self):
        assert check_swift_bic(["DEUTDEDB"]) == [True]

    def test_invalid_swift(self):
        assert check_swift_bic(["123"]) == [False]

    def test_none_returns_false(self):
        assert check_swift_bic([None]) == [False]

    def test_nan_returns_false(self):
        assert check_swift_bic([float("nan")]) == [False]
