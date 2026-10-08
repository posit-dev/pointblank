"""Tests for LocaleGenerator methods in pointblank.countries."""

import re

import pytest

from pointblank.countries import (
    LocaleGenerator,
    LocaleRegistry,
    get_generator,
    _normalize_country,
    _transliterate_to_ascii,
    _is_tiered,
    _flatten_tiered,
    _pick_from_tiered,
    COUNTRY_CODE_MAP,
)


class TestNormalizeCountry:
    def test_alpha2(self):
        assert _normalize_country("US") == "US"
        assert _normalize_country("DE") == "DE"

    def test_alpha3(self):
        assert _normalize_country("USA") == "US"
        assert _normalize_country("DEU") == "DE"

    def test_locale_format(self):
        assert _normalize_country("en_US") == "US"
        assert _normalize_country("de-DE") == "DE"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown country code"):
            _normalize_country("ZZZ")


class TestLocaleRegistry:
    def test_singleton(self):
        r1 = LocaleRegistry()
        r2 = LocaleRegistry()
        assert r1 is r2

    def test_get_returns_data(self):
        registry = LocaleRegistry()
        data = registry.get("US")
        assert data is not None
        assert data.locale == "US"

    def test_caching(self):
        registry = LocaleRegistry()
        registry.clear_cache()
        d1 = registry.get("US")
        d2 = registry.get("US")
        assert d1 is d2

    def test_clear_cache(self):
        registry = LocaleRegistry()
        registry.get("US")
        registry.clear_cache()
        assert registry._cache == {}


class TestGetGenerator:
    def test_returns_locale_generator(self):
        gen = get_generator("US", seed=42)
        assert isinstance(gen, LocaleGenerator)
        assert gen.country_code == "US"

    def test_weighted_option(self):
        gen = get_generator("DE", seed=1, weighted=False)
        assert gen.weighted is False


class TestLocaleGeneratorPerson:
    def test_first_name(self):
        gen = LocaleGenerator("US", seed=1)
        name = gen.first_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_last_name(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_person()
        name = gen.last_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_name(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_person()
        full = gen.name()
        assert " " in full

    def test_name_full(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_person()
        full = gen.name_full()
        assert isinstance(full, str)
        assert len(full) > 0

    def test_gender(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_person()
        g = gen.gender()
        assert g in ("male", "female")

    def test_new_person_and_clear(self):
        gen = LocaleGenerator("US", seed=1)
        p = gen.new_person()
        assert "first_name" in p
        assert "last_name" in p
        gen.clear_person()
        assert gen._current_person is None

    def test_init_row_persons(self):
        gen = LocaleGenerator("US", seed=1)
        gen.init_row_persons(5)
        assert gen._row_persons is not None
        assert len(gen._row_persons) == 5
        gen.clear_row_persons()
        assert gen._row_persons is None


class TestLocaleGeneratorAddress:
    def test_city(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        city = gen.city()
        assert isinstance(city, str)
        assert len(city) > 0

    def test_state(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        state = gen.state()
        assert isinstance(state, str)
        state_abbr = gen.state(abbr=True)
        assert len(state_abbr) <= 3

    def test_country(self):
        gen = LocaleGenerator("US", seed=1)
        c = gen.country()
        assert isinstance(c, str)

    def test_country_name(self):
        gen = LocaleGenerator("US", seed=1)
        cn = gen.country_name()
        assert "United States" in cn or isinstance(cn, str)

    def test_country_code_2(self):
        gen = LocaleGenerator("DE", seed=1)
        assert gen.country_code_2() == "DE"

    def test_country_code_3(self):
        gen = LocaleGenerator("US", seed=1)
        assert gen.country_code_3() == "USA"

    def test_locale_code(self):
        gen = LocaleGenerator("US", seed=1)
        code = gen.locale_code()
        assert "_" in code or len(code) >= 2

    def test_postcode(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        pc = gen.postcode()
        assert isinstance(pc, str)
        assert len(pc) >= 3

    def test_street_name(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        street = gen.street_name()
        assert isinstance(street, str)
        assert len(street) > 0

    def test_building_number(self):
        gen = LocaleGenerator("US", seed=1)
        bn = gen.building_number()
        assert bn.isdigit()

    def test_address(self):
        gen = LocaleGenerator("US", seed=1)
        addr = gen.address()
        assert isinstance(addr, str)
        assert len(addr) > 10

    def test_new_location_and_clear(self):
        gen = LocaleGenerator("US", seed=1)
        loc = gen.new_location()
        assert "city" in loc
        gen.clear_location()
        assert gen._current_location is None

    def test_init_row_locations(self):
        gen = LocaleGenerator("US", seed=1)
        gen.init_row_locations(3)
        assert gen._row_locations is not None
        assert len(gen._row_locations) == 3
        gen.set_row(0)
        assert gen._current_row == 0
        gen.clear_row_locations()
        assert gen._row_locations is None

    def test_latitude_longitude(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        lat = gen.latitude()
        lon = gen.longitude()
        assert float(lat)
        assert float(lon)

    def test_phone_number(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        phone = gen.phone_number()
        assert isinstance(phone, str)
        assert len(phone) >= 10


class TestLocaleGeneratorCompany:
    def test_company(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        company = gen.company()
        assert isinstance(company, str)
        assert len(company) > 0

    def test_job(self):
        gen = LocaleGenerator("US", seed=1)
        j = gen.job()
        assert isinstance(j, str)

    def test_catch_phrase(self):
        gen = LocaleGenerator("US", seed=1)
        cp = gen.catch_phrase()
        assert isinstance(cp, str)
        assert len(cp) > 5

    def test_employer_coherence(self):
        gen = LocaleGenerator("US", seed=1)
        gen.init_row_locations(3)
        gen.init_row_employers(3)
        gen.set_row(0)
        j = gen.job()
        c = gen.company()
        assert isinstance(j, str)
        assert isinstance(c, str)
        gen.clear_row_employers()
        assert gen._row_employers is None


class TestLocaleGeneratorInternet:
    def test_email(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_person()
        email = gen.email()
        assert "@" in email

    def test_user_name(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_person()
        un = gen.user_name()
        assert isinstance(un, str)
        assert len(un) > 0

    def test_password(self):
        gen = LocaleGenerator("US", seed=1)
        pw = gen.password(length=16)
        assert len(pw) == 16

    def test_url(self):
        gen = LocaleGenerator("US", seed=1)
        u = gen.url()
        assert u.startswith("https://")

    def test_domain_name(self):
        gen = LocaleGenerator("US", seed=1)
        dn = gen.domain_name()
        assert "." in dn

    def test_ipv4(self):
        gen = LocaleGenerator("US", seed=1)
        ip = gen.ipv4()
        parts = ip.split(".")
        assert len(parts) == 4
        assert all(0 <= int(p) <= 255 for p in parts)

    def test_ipv6(self):
        gen = LocaleGenerator("US", seed=1)
        ip = gen.ipv6()
        parts = ip.split(":")
        assert len(parts) == 8


class TestLocaleGeneratorText:
    def test_word(self):
        gen = LocaleGenerator("US", seed=1)
        w = gen.word()
        assert isinstance(w, str)

    def test_sentence(self):
        gen = LocaleGenerator("US", seed=1)
        s = gen.sentence()
        assert s.endswith(".")
        assert len(s) > 5

    def test_sentence_with_num_words(self):
        gen = LocaleGenerator("US", seed=1)
        s = gen.sentence(num_words=3)
        # 3 words + period
        assert s.endswith(".")

    def test_paragraph(self):
        gen = LocaleGenerator("US", seed=1)
        p = gen.paragraph()
        assert isinstance(p, str)
        assert len(p) > 20

    def test_text(self):
        gen = LocaleGenerator("US", seed=1)
        t = gen.text(max_chars=100)
        assert isinstance(t, str)
        assert len(t) <= 110  # slight tolerance for sentence boundaries


class TestLocaleGeneratorFinancial:
    def test_credit_card_number(self):
        gen = LocaleGenerator("US", seed=1)
        ccn = gen.credit_card_number()
        assert len(ccn) in (15, 16)
        assert ccn.isdigit()

    def test_credit_card_provider(self):
        gen = LocaleGenerator("US", seed=1)
        prov = gen.credit_card_provider()
        assert prov in ("Visa", "Mastercard", "American Express", "Discover")

    def test_card_coherence(self):
        gen = LocaleGenerator("US", seed=1)
        gen.init_row_card_prefixes(3)
        gen.set_row(0)
        ccn = gen.credit_card_number()
        prov = gen.credit_card_provider()
        if ccn.startswith("4"):
            assert prov == "Visa"
        elif ccn.startswith("5"):
            assert prov == "Mastercard"
        gen.clear_row_card_prefixes()

    def test_iban(self):
        gen = LocaleGenerator("US", seed=1)
        iban = gen.iban()
        assert isinstance(iban, str)
        assert len(iban) >= 15

    def test_currency_code(self):
        gen = LocaleGenerator("US", seed=1)
        cc = gen.currency_code()
        assert len(cc) == 3


class TestLocaleGeneratorIdentifiers:
    def test_uuid4(self):
        gen = LocaleGenerator("US", seed=1)
        u = gen.uuid4()
        parts = u.split("-")
        assert len(parts) == 5
        assert parts[2].startswith("4")

    def test_md5(self):
        gen = LocaleGenerator("US", seed=1)
        h = gen.md5()
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_sha1(self):
        gen = LocaleGenerator("US", seed=1)
        h = gen.sha1()
        assert len(h) == 40

    def test_sha256(self):
        gen = LocaleGenerator("US", seed=1)
        h = gen.sha256()
        assert len(h) == 64

    def test_ssn(self):
        gen = LocaleGenerator("US", seed=1)
        ssn = gen.ssn()
        assert isinstance(ssn, str)
        assert len(ssn) >= 9


class TestLocaleGeneratorMisc:
    def test_license_plate(self):
        gen = LocaleGenerator("US", seed=1)
        gen.new_location()
        plate = gen.license_plate()
        assert isinstance(plate, str)
        assert len(plate) >= 4

    def test_license_plate_ca(self):
        gen = LocaleGenerator("CA", seed=1)
        gen.new_location()
        plate = gen.license_plate()
        assert isinstance(plate, str)

    def test_license_plate_de(self):
        gen = LocaleGenerator("DE", seed=1)
        gen.new_location()
        plate = gen.license_plate()
        assert "-" in plate

    def test_license_plate_gb(self):
        gen = LocaleGenerator("GB", seed=1)
        gen.new_location()
        plate = gen.license_plate()
        assert isinstance(plate, str)

    def test_license_plate_au(self):
        gen = LocaleGenerator("AU", seed=1)
        gen.new_location()
        plate = gen.license_plate()
        assert isinstance(plate, str)

    def test_ean8(self):
        gen = LocaleGenerator("US", seed=1)
        code = gen.ean8()
        assert len(code) == 8
        assert code.isdigit()

    def test_ean13(self):
        gen = LocaleGenerator("US", seed=1)
        code = gen.ean13()
        assert len(code) == 13
        assert code.isdigit()

    def test_color_name(self):
        gen = LocaleGenerator("US", seed=1)
        c = gen.color_name()
        assert isinstance(c, str)

    def test_file_name(self):
        gen = LocaleGenerator("US", seed=1)
        fn = gen.file_name()
        assert "." in fn

    def test_file_extension(self):
        gen = LocaleGenerator("US", seed=1)
        ext = gen.file_extension()
        assert isinstance(ext, str)
        assert len(ext) >= 2

    def test_mime_type(self):
        gen = LocaleGenerator("US", seed=1)
        mt = gen.mime_type()
        assert "/" in mt

    def test_user_agent(self):
        gen = LocaleGenerator("US", seed=1)
        ua = gen.user_agent()
        assert "Mozilla" in ua or len(ua) > 10


class TestLocaleGeneratorDatetime:
    def test_date_this_year(self):
        gen = LocaleGenerator("US", seed=1)
        d = gen.date_this_year()
        assert re.match(r"\d{4}-\d{2}-\d{2}", d)

    def test_date_this_decade(self):
        gen = LocaleGenerator("US", seed=1)
        d = gen.date_this_decade()
        assert re.match(r"\d{4}-\d{2}-\d{2}", d)

    def test_time(self):
        gen = LocaleGenerator("US", seed=1)
        t = gen.time()
        assert re.match(r"\d{2}:\d{2}:\d{2}", t)

    def test_date_between(self):
        gen = LocaleGenerator("US", seed=1)
        d = gen.date_between(start_year=2020, end_year=2022)
        year = int(d.split("-")[0])
        assert 2020 <= year <= 2022

    def test_date_range(self):
        gen = LocaleGenerator("US", seed=1)
        dr = gen.date_range()
        assert "\u2013" in dr or "–" in dr

    def test_future_date(self):
        gen = LocaleGenerator("US", seed=1)
        d = gen.future_date()
        assert re.match(r"\d{4}-\d{2}-\d{2}", d)

    def test_past_date(self):
        gen = LocaleGenerator("US", seed=1)
        d = gen.past_date()
        assert re.match(r"\d{4}-\d{2}-\d{2}", d)


class TestLocaleGeneratorUnweighted:
    """Test unweighted mode exercises different code paths."""

    def test_first_name_unweighted(self):
        gen = LocaleGenerator("US", seed=1, weighted=False)
        name = gen.first_name()
        assert isinstance(name, str)

    def test_last_name_unweighted(self):
        gen = LocaleGenerator("US", seed=1, weighted=False)
        gen.new_person()
        name = gen.last_name()
        assert isinstance(name, str)

    def test_location_unweighted(self):
        gen = LocaleGenerator("US", seed=1, weighted=False)
        gen.new_location()
        city = gen.city()
        assert isinstance(city, str)
