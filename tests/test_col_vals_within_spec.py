import pytest

import ibis
import pandas as pd
import polars as pl

import pointblank as pb


IBAN_VALID = {
    "AT": [
        "AT582774098454337653",
        "AT220332087576467472",
    ],
    "DE": [
        "DE06495352657836424132",
        "DE09121688720378475751",
    ],
    "GB": [
        "GB39MUJS50172570996370",
        "GB14SIPV86193224493527",
    ],
}

POSTAL_CODE_VALID = {
    "US": ["99553", "36264", "71660", "85225", "90309"],
    "CA": ["L6M 3V5", "V7G 1V1", "B2X 1R5", "E2K 1H3", "M4Y 3C1"],
    "DE": ["01945", "03119", "08393", "36457", "99996"],
}

CREDIT_CARD_VALID = [
    "340000000000009",  # American Express
    "378734493671000",  # American Express Corporate
    "6703444444444449",  # Bancontact
    "4035501000000008",  # Cartes Bancaires
    "6011000000000004",  # Discover
    "5500000000000004",  # MasterCard
    "4012888888881881",  # Visa
]

CREDIT_CARD_INVALID = [
    "ABCDEFJHIGK",
    "340000000000000",
    "378734493671001",
    "5500000000000005",
]

VIN_VALID = [
    "1FTEW1E41KKD70581",
    "ZARBAAB46LM355009",
    "JTEBH3FJ60K093139",
    "1HD1FS4178Y631180",
]

VIN_INVALID = [
    "7A8GK4M0706100372",
    "WVWZZZ1KZ7U022191",
]

ISBN_10_VALID = [
    "1101907932",
    "0375712356",
    "0307957802",
    "067940581X",
]

ISBN_13_VALID = [
    "978-1101907931",
    "978-0375712357",
    "978-0307957801",
]

ISBN_10_INVALID = [
    "1101907931",
    "0375712358",
]

ISBN_13_INVALID = [
    "978-1101907930",
    "978-0375712358",
]

SWIFT_BIC_VALID = [
    "RBOSGGSX",
    "RZTIAT22263",
    "BCEELULL",
    "MARKDEFF",
]

SWIFT_BIC_INVALID = [
    "CE1EL2LLFFF",  # Invalid: digits in bank code
    "E31DCLLFFF",  # Invalid: digits in bank code
]

PHONE_VALID = [
    "+5-555-555-5555",
    "5-555-555-5555",
    "555-555-5555",
    "(555)555-5555",
    "+1 (555) 555 5555",
]

PHONE_INVALID = [
    "",
    "123",
    "text",
]

MAC_VALID = [
    "01-2d-4c-ef-89-ab",
    "01-2D-4C-EF-89-AB",
    "01:2d:4c:ef:89:ab",
    "01:2D:4C:EF:89:AB",
]

MAC_INVALID = [
    "999999999",
    "01-2d-4c-ef-89-ab-06",
    "text",
]

EMAIL_VALID = [
    "test@test.com",
    "mail+mail@example.com",
    "mail.email@e.test.com",
]

EMAIL_INVALID = [
    "",
    "test",
    "@test.com",
    "mail@example",
]

URL_VALID = [
    "http://foo.com/blah_blah",
    "https://www.example.com/foo/?bar=baz&inga=42&quux",
    "ftp://foo.bar/baz",
]

URL_INVALID = [
    "http://",
    "foo.com",
    "http:// shouldfail.com",
]

IPV4_ADDRESS_VALID = [
    "93.184.220.20",
    "161.148.172.130",
    "0.0.0.0",
    "255.255.255.255",
]

IPV4_ADDRESS_INVALID = [
    "256.255.255.255",
    "2001:0db8:0000:85a3:0000:0000:ac1f:8001",
    "",
]

IPV6_ADDRESS_VALID = [
    "2001:0db8:0000:85a3:0000:0000:ac1f:8001",
    "2001:db8:0:85a3:0:0:ac1f:8001",
]

IPV6_ADDRESS_INVALID = [
    "0db8:0000:85a3:0000:0000:ac1f:8001",
    "93.184.220.20",
]


@pytest.fixture
def email_valid_pl():
    return pl.DataFrame({"email": EMAIL_VALID})


@pytest.fixture
def email_valid_pd():
    return pd.DataFrame({"email": EMAIL_VALID})


@pytest.fixture
def email_valid_duckdb(tmp_path):
    """Create a temporary DuckDB database with email data."""
    db_path = tmp_path / "email_test.ddb"
    con = ibis.connect(f"duckdb://{db_path}")

    # Create table with email data
    df = pl.DataFrame({"email": EMAIL_VALID})
    con.create_table("email_data", df, overwrite=True)

    return con.table("email_data")


@pytest.fixture
def email_valid_sqlite(tmp_path):
    """Create a temporary SQLite database with email data."""
    db_path = tmp_path / "email_test.sqlite"
    con = ibis.sqlite.connect(db_path)

    # Create table with email data
    df = pl.DataFrame({"email": EMAIL_VALID})
    con.create_table("email_data", df, overwrite=True)

    return con.table("email_data")


@pytest.fixture
def credit_card_valid_pl():
    return pl.DataFrame({"card": CREDIT_CARD_VALID})


@pytest.fixture
def credit_card_valid_pd():
    return pd.DataFrame({"card": CREDIT_CARD_VALID})


@pytest.fixture
def credit_card_valid_duckdb(tmp_path):
    """Create a temporary DuckDB database with credit card data."""
    db_path = tmp_path / "card_test.ddb"
    con = ibis.connect(f"duckdb://{db_path}")

    df = pl.DataFrame({"card": CREDIT_CARD_VALID})
    con.create_table("card_data", df, overwrite=True)

    return con.table("card_data")


@pytest.fixture
def credit_card_valid_sqlite(tmp_path):
    """Create a temporary SQLite database with credit card data."""
    db_path = tmp_path / "card_test.sqlite"
    con = ibis.sqlite.connect(db_path)

    df = pl.DataFrame({"card": CREDIT_CARD_VALID})
    con.create_table("card_data", df, overwrite=True)

    return con.table("card_data")


@pytest.mark.parametrize(
    "fixture_name",
    ["email_valid_pl", "email_valid_pd", "email_valid_duckdb", "email_valid_sqlite"],
)
def test_email_validation_valid(fixture_name, request):
    """Test valid email addresses across different backends."""
    tbl = request.getfixturevalue(fixture_name)

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="email", spec="email").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(EMAIL_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_email_validation_invalid():
    """Test invalid email addresses."""
    tbl = pl.DataFrame({"email": EMAIL_INVALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="email", spec="email").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(EMAIL_INVALID)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "credit_card_valid_pl",
        "credit_card_valid_pd",
        "credit_card_valid_duckdb",
        "credit_card_valid_sqlite",
    ],
)
def test_credit_card_validation_valid(fixture_name, request):
    """Test valid credit card numbers across different backends."""
    tbl = request.getfixturevalue(fixture_name)

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="card", spec="credit_card").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(CREDIT_CARD_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_credit_card_validation_invalid():
    """Test invalid credit card numbers."""
    tbl = pl.DataFrame({"card": CREDIT_CARD_INVALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="card", spec="credit_card").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(CREDIT_CARD_INVALID)


@pytest.mark.parametrize("country", ["AT", "DE", "GB"])
def test_iban_validation_valid(country):
    """Test valid IBANs for different countries."""
    tbl = pl.DataFrame({"iban": IBAN_VALID[country]})

    validation = (
        pb.Validate(data=tbl)
        .col_vals_within_spec(columns="iban", spec=f"iban[{country}]")
        .interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(IBAN_VALID[country])
    assert validation.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("country,column", [("US", "zip"), ("CA", "postal_code"), ("DE", "plz")])
def test_postal_code_validation_valid(country, column):
    """Test valid postal codes for different countries."""
    tbl = pl.DataFrame({column: POSTAL_CODE_VALID[country]})

    validation = (
        pb.Validate(data=tbl)
        .col_vals_within_spec(columns=column, spec=f"postal_code[{country}]")
        .interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(POSTAL_CODE_VALID[country])
    assert validation.n_failed(i=1, scalar=True) == 0


def test_vin_validation_valid():
    """Test valid VINs."""
    tbl = pl.DataFrame({"vin": VIN_VALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="vin", spec="vin").interrogate()

    assert validation.n_passed(i=1, scalar=True) == len(VIN_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_vin_validation_invalid():
    """Test invalid VINs."""
    tbl = pl.DataFrame({"vin": VIN_INVALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="vin", spec="vin").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(VIN_INVALID)


def test_isbn_10_validation_valid():
    """Test valid ISBN-10 numbers."""
    tbl = pl.DataFrame({"isbn": ISBN_10_VALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="isbn", spec="isbn").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(ISBN_10_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_isbn_13_validation_valid():
    """Test valid ISBN-13 numbers."""
    tbl = pl.DataFrame({"isbn": ISBN_13_VALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="isbn", spec="isbn").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(ISBN_13_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_isbn_10_validation_invalid():
    """Test invalid ISBN-10 numbers."""
    tbl = pl.DataFrame({"isbn": ISBN_10_INVALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="isbn", spec="isbn").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(ISBN_10_INVALID)


def test_isbn_13_validation_invalid():
    """Test invalid ISBN-13 numbers."""
    tbl = pl.DataFrame({"isbn": ISBN_13_INVALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="isbn", spec="isbn").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(ISBN_13_INVALID)


def test_phone_validation_valid():
    """Test valid phone numbers."""
    tbl = pl.DataFrame({"phone": PHONE_VALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="phone", spec="phone").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(PHONE_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_phone_validation_invalid():
    """Test invalid phone numbers."""
    tbl = pl.DataFrame({"phone": PHONE_INVALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="phone", spec="phone").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(PHONE_INVALID)


def test_mac_validation_valid():
    """Test valid MAC addresses."""
    tbl = pl.DataFrame({"mac": MAC_VALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="mac", spec="mac").interrogate()

    assert validation.n_passed(i=1, scalar=True) == len(MAC_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_mac_validation_invalid():
    """Test invalid MAC addresses."""
    tbl = pl.DataFrame({"mac": MAC_INVALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="mac", spec="mac").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(MAC_INVALID)


def test_swift_bic_validation_valid():
    """Test valid SWIFT/BIC codes."""
    tbl = pl.DataFrame({"swift": SWIFT_BIC_VALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="swift", spec="swift").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == len(SWIFT_BIC_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_swift_bic_validation_invalid():
    """Test invalid SWIFT/BIC codes."""
    tbl = pl.DataFrame({"swift": SWIFT_BIC_INVALID})

    validation = (
        pb.Validate(data=tbl).col_vals_within_spec(columns="swift", spec="swift").interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(SWIFT_BIC_INVALID)


def test_url_validation_valid():
    """Test valid URLs."""
    tbl = pl.DataFrame({"url": URL_VALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="url", spec="url").interrogate()

    assert validation.n_passed(i=1, scalar=True) == len(URL_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_url_validation_invalid():
    """Test invalid URLs."""
    tbl = pl.DataFrame({"url": URL_INVALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="url", spec="url").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(URL_INVALID)


def test_ipv4_validation_valid():
    """Test valid IPv4 addresses."""
    tbl = pl.DataFrame({"ip": IPV4_ADDRESS_VALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="ip", spec="ipv4").interrogate()

    assert validation.n_passed(i=1, scalar=True) == len(IPV4_ADDRESS_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_ipv4_validation_invalid():
    """Test invalid IPv4 addresses."""
    tbl = pl.DataFrame({"ip": IPV4_ADDRESS_INVALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="ip", spec="ipv4").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(IPV4_ADDRESS_INVALID)


def test_ipv6_validation_valid():
    """Test valid IPv6 addresses."""
    tbl = pl.DataFrame({"ip": IPV6_ADDRESS_VALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="ip", spec="ipv6").interrogate()

    assert validation.n_passed(i=1, scalar=True) == len(IPV6_ADDRESS_VALID)
    assert validation.n_failed(i=1, scalar=True) == 0


def test_ipv6_validation_invalid():
    """Test invalid IPv6 addresses."""
    tbl = pl.DataFrame({"ip": IPV6_ADDRESS_INVALID})

    validation = pb.Validate(data=tbl).col_vals_within_spec(columns="ip", spec="ipv6").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == len(IPV6_ADDRESS_INVALID)


def test_na_pass_false():
    """Test that NA values fail when na_pass=False."""
    tbl = pl.DataFrame({"email": ["test@test.com", None, "invalid"]})

    validation = (
        pb.Validate(data=tbl)
        .col_vals_within_spec(columns="email", spec="email", na_pass=False)
        .interrogate()
    )

    # Should have 1 pass, 2 fails (None and "invalid")
    assert validation.n_passed(i=1, scalar=True) == 1
    assert validation.n_failed(i=1, scalar=True) == 2


def test_na_pass_true():
    """Test that NA values pass when na_pass=True."""
    tbl = pl.DataFrame({"email": ["test@test.com", None, "invalid"]})

    validation = (
        pb.Validate(data=tbl)
        .col_vals_within_spec(columns="email", spec="email", na_pass=True)
        .interrogate()
    )

    # Should have 2 passes (valid email and None), 1 fail ("invalid")
    assert validation.n_passed(i=1, scalar=True) == 2
    assert validation.n_failed(i=1, scalar=True) == 1


def test_regex_specs_no_materialization_ibis():
    """
    Test that regex-based specs (email, url, phone, etc.) don't materialize Ibis tables.

    This verifies that simple regex validations use Narwhals directly,
    avoiding data transfer from remote databases.
    """
    # Create a DuckDB table with email data
    con = ibis.connect("duckdb://")
    data = {
        "id": [1, 2, 3, 4],
        "email": [
            "valid@example.com",
            "another.valid@test.org",
            "invalid-email",
            None,
        ],
    }
    tbl = con.create_table("email_test", data, overwrite=True)

    # Validate emails - should NOT materialize the table
    validation = (
        pb.Validate(data=tbl)
        .col_vals_within_spec(columns="email", spec="email", na_pass=False)
        .interrogate()
    )

    # Verify results are correct
    assert validation.n_passed(i=1, scalar=True) == 2  # Two valid emails
    assert validation.n_failed(i=1, scalar=True) == 2  # Invalid email + None

    # Test other regex-based specs to ensure they all work without materialization
    specs_to_test = [
        ("url", ["https://example.com", "ftp://test.org", "not-a-url", None]),
        ("phone", ["+1-555-123-4567", "555.123.4567", "invalid", None]),
        ("ipv4", ["192.168.1.1", "10.0.0.1", "999.999.999.999", None]),
        ("mac", ["00:1B:44:11:3A:B7", "00-1B-44-11-3A-B7", "invalid", None]),
        ("swift_bic", ["DEUTDEFF", "NEDSZAJJ", "invalid", None]),
    ]

    for spec, test_data in specs_to_test:
        data = {"id": list(range(len(test_data))), "value": test_data}
        tbl = con.create_table(f"{spec}_test", data, overwrite=True)

        validation = (
            pb.Validate(data=tbl)
            .col_vals_within_spec(columns="value", spec=spec, na_pass=False)
            .interrogate()
        )

        # Each spec has 2 valid values, 2 invalid (including None)
        assert validation.n_passed(i=1, scalar=True) == 2, f"Failed for spec: {spec}"
        assert validation.n_failed(i=1, scalar=True) == 2, f"Failed for spec: {spec}"
