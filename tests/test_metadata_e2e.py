from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import pointblank as pb

# Path to the fixtures directory
FIXTURES = Path(__file__).parent / "metadata_fixtures"

pyreadstat = pytest.importorskip("pyreadstat")
lxml = pytest.importorskip("lxml")


# ===========================================================================
# SPSS .sav
# ===========================================================================


class TestSpssEndToEnd:
    """End-to-end tests for SPSS .sav file import."""

    @pytest.fixture()
    def meta(self):
        return pb.import_metadata(str(FIXTURES / "survey_data.sav"))

    def test_auto_detect_format(self, meta):
        assert meta.source_format == "spss"

    def test_dataset_name(self, meta):
        assert meta.dataset_name == "survey_data"

    def test_variable_count(self, meta):
        assert len(meta.variables) == 7

    def test_variable_names(self, meta):
        names = [v.name for v in meta.variables]
        assert names == [
            "respondent_id",
            "age",
            "gender",
            "education",
            "income",
            "satisfaction",
            "region",
        ]

    def test_variable_labels(self, meta):
        labels = {v.name: v.label for v in meta.variables}
        assert labels["age"] == "Age in Years"
        assert labels["gender"] == "Gender Identity"
        assert labels["income"] == "Annual Household Income (USD)"

    def test_dtypes(self, meta):
        dtypes = {v.name: v.dtype for v in meta.variables}
        assert dtypes["region"] == "String"
        # Numeric variables come in as Float64 from SPSS
        assert dtypes["age"] == "Float64"
        assert dtypes["income"] == "Float64"

    def test_codelists_extracted(self, meta):
        assert len(meta.codelists) == 3
        assert "gender_values" in meta.codelists
        assert "education_values" in meta.codelists
        assert "satisfaction_values" in meta.codelists

    def test_gender_codelist_values(self, meta):
        cl = meta.codelists["gender_values"]
        assert set(cl.to_set()) == {1.0, 2.0, 3.0}
        labels = cl.to_dict()
        assert labels[1.0] == "Male"
        assert labels[2.0] == "Female"
        assert labels[3.0] == "Non-binary"

    def test_education_codelist_values(self, meta):
        cl = meta.codelists["education_values"]
        assert len(cl.to_set()) == 5

    def test_to_schema(self, meta):
        schema = meta.to_schema()
        assert len(schema.columns) == 7
        col_dict = dict(schema.columns)
        assert "respondent_id" in col_dict
        assert "region" in col_dict

    def test_to_validate_valid_data(self, meta):
        """Validate data that conforms to the SPSS metadata."""
        df = pl.DataFrame(
            {
                "respondent_id": [1001.0, 1002.0, 1003.0],
                "age": [28.0, 45.0, 62.0],
                "gender": [1.0, 2.0, 3.0],
                "education": [3.0, 4.0, 5.0],
                "income": [45000.0, 72000.0, 95000.0],
                "satisfaction": [4.0, 5.0, 3.0],
                "region": ["NE", "SE", "MW"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        # Should have schema + codelist checks
        assert len(validation.validation_info) >= 4
        # All value label checks should pass since data matches
        for v in validation.validation_info:
            if v.assertion_type == "col_vals_in_set":
                assert v.n_failed == 0, f"Step {v.i} failed unexpectedly"

    def test_to_validate_bad_data(self, meta):
        """Detect invalid codelist values."""
        df = pl.DataFrame(
            {
                "respondent_id": [1001.0, 1002.0],
                "age": [28.0, 45.0],
                "gender": [1.0, 99.0],  # 99 is not in codelist
                "education": [3.0, 4.0],
                "income": [45000.0, 72000.0],
                "satisfaction": [4.0, 5.0],
                "region": ["NE", "SE"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        # Gender codelist check should fail
        gender_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "gender"
        ]
        assert len(gender_steps) == 1
        assert gender_steps[0].n_failed == 1


# ===========================================================================
# SAS Transport .xpt
# ===========================================================================


class TestXptEndToEnd:
    """End-to-end tests for SAS Transport .xpt file import."""

    @pytest.fixture()
    def meta(self):
        return pb.import_metadata(str(FIXTURES / "dm.xpt"))

    def test_auto_detect_format(self, meta):
        assert meta.source_format == "xpt"

    def test_dataset_name(self, meta):
        assert meta.dataset_name == "DM"

    def test_variable_count(self, meta):
        assert len(meta.variables) == 12

    def test_variable_labels(self, meta):
        labels = {v.name: v.label for v in meta.variables}
        assert labels["STUDYID"] == "Study Identifier"
        assert labels["USUBJID"] == "Unique Subject Identifier"
        assert labels["AGE"] == "Age"

    def test_max_lengths_extracted(self, meta):
        """SAS Transport variables have defined max lengths."""
        lengths = {v.name: v.max_length for v in meta.variables if v.max_length}
        assert "STUDYID" in lengths
        assert "USUBJID" in lengths
        assert lengths["USUBJID"] == 10  # matches our fixture data width

    def test_to_schema(self, meta):
        schema = meta.to_schema()
        col_dict = dict(schema.columns)
        assert "STUDYID" in col_dict
        assert "AGE" in col_dict
        assert col_dict["AGE"] == "Float64"

    def test_to_validate_valid_data(self, meta):
        """Full validation of conforming DM data."""
        df = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["XYZ789-101", "XYZ789-102", "XYZ789-103"],
                "SUBJID": ["101", "102", "103"],
                "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "RFENDTC": ["2024-07-15", "2024-07-20", "2024-08-01"],
                "SITEID": ["S01", "S01", "S02"],
                "AGE": [45.0, 62.0, 38.0],
                "SEX": ["M", "F", "M"],
                "RACE": ["WHITE", "BLACK", "ASIAN"],
                "ARMCD": ["TRT", "PBO", "TRT"],
                "ARM": ["Active 10mg", "Placebo", "Active 10mg"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        # Schema match should pass
        schema_steps = [
            v for v in validation.validation_info if v.assertion_type == "col_schema_match"
        ]
        assert len(schema_steps) == 1
        assert schema_steps[0].n_failed == 0


# ===========================================================================
# Stata .dta
# ===========================================================================


class TestStataEndToEnd:
    """End-to-end tests for Stata .dta file import."""

    @pytest.fixture()
    def meta(self):
        return pb.import_metadata(str(FIXTURES / "economics_panel.dta"))

    def test_auto_detect_format(self, meta):
        assert meta.source_format == "stata"

    def test_dataset_name(self, meta):
        assert meta.dataset_name == "economics_panel"

    def test_variable_count(self, meta):
        assert len(meta.variables) == 6

    def test_variable_labels(self, meta):
        labels = {v.name: v.label for v in meta.variables}
        assert labels["country_id"] == "Country Identifier"
        assert labels["gdp_growth"] == "GDP Growth Rate (%)"

    def test_codelists(self, meta):
        assert len(meta.codelists) == 1
        assert "region_values" in meta.codelists
        cl = meta.codelists["region_values"]
        labels = cl.to_dict()
        assert labels[1] == "North America"
        assert labels[2] == "Europe"
        assert labels[3] == "Asia-Pacific"

    def test_to_validate_valid_data(self, meta):
        """Data with valid region codes passes validation."""
        df = pl.DataFrame(
            {
                "country_id": [1.0, 2.0, 3.0],
                "year": [2022.0, 2022.0, 2022.0],
                "gdp_growth": [5.7, 4.2, 6.0],
                "unemployment": [6.3, 5.5, 5.8],
                "inflation": [3.5, 4.2, 2.9],
                "region": [1.0, 2.0, 3.0],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        region_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "region"
        ]
        assert len(region_steps) == 1
        assert region_steps[0].n_failed == 0

    def test_to_validate_invalid_region(self, meta):
        """Invalid region code is detected."""
        df = pl.DataFrame(
            {
                "country_id": [1.0, 2.0],
                "year": [2022.0, 2022.0],
                "gdp_growth": [5.7, 4.2],
                "unemployment": [6.3, 5.5],
                "inflation": [3.5, 4.2],
                "region": [1.0, 99.0],  # 99 is invalid
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        region_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "region"
        ]
        assert region_steps[0].n_failed == 1


# ===========================================================================
# Frictionless Data Package
# ===========================================================================


class TestFrictionlessEndToEnd:
    """End-to-end tests for Frictionless Data Package import."""

    @pytest.fixture()
    def meta(self):
        return pb.import_metadata(str(FIXTURES / "datapackage.json"), format="frictionless")

    def test_format_detected(self, meta):
        assert meta.source_format == "frictionless"

    def test_dataset_name(self, meta):
        assert meta.dataset_name == "transactions"

    def test_variable_count(self, meta):
        assert len(meta.variables) == 8

    def test_constraints_parsed(self, meta):
        vars_dict = {v.name: v for v in meta.variables}
        # transaction_id: required, unique
        assert vars_dict["transaction_id"].required is True
        assert vars_dict["transaction_id"].unique is True
        # amount: required, min 0.01, max 99999.99
        assert vars_dict["amount"].required is True
        assert vars_dict["amount"].min_val == 0.01
        assert vars_dict["amount"].max_val == 99999.99
        # quantity: required, min 1, max 1000
        assert vars_dict["quantity"].required is True
        assert vars_dict["quantity"].min_val == 1.0
        assert vars_dict["quantity"].max_val == 1000.0
        # category: enum
        assert vars_dict["category"].allowed_values == [
            "electronics",
            "clothing",
            "food",
            "home",
            "sports",
        ]
        # email: pattern
        assert vars_dict["email"].pattern is not None

    def test_to_validate_valid_data(self, meta):
        """Conforming sales data passes all checks."""
        df = pl.DataFrame(
            {
                "transaction_id": ["TXN-001", "TXN-002", "TXN-003"],
                "customer_id": ["CUST-12345", "CUST-67890", "CUST-11111"],
                "amount": [29.99, 149.50, 9.99],
                "quantity": [1, 3, 1],
                "category": ["electronics", "clothing", "food"],
                "sale_date": ["2024-01-15", "2024-02-20", "2024-03-10"],
                "discount_pct": [0.0, 10.0, 5.0],
                "email": ["alice@example.com", "bob@corp.io", "charlie@mail.org"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        # Not-null, in-set, between, regex checks should all pass
        for v in validation.validation_info:
            if v.assertion_type in (
                "col_vals_not_null",
                "col_vals_in_set",
                "col_vals_between",
                "col_vals_regex",
            ):
                assert v.n_failed == 0, (
                    f"Step {v.i} ({v.assertion_type}, col={v.column}) failed with {v.n_failed}"
                )

    def test_to_validate_bad_category(self, meta):
        """Invalid category value is caught."""
        df = pl.DataFrame(
            {
                "transaction_id": ["TXN-001", "TXN-002"],
                "customer_id": ["CUST-12345", "CUST-67890"],
                "amount": [29.99, 149.50],
                "quantity": [1, 3],
                "category": ["electronics", "INVALID"],
                "sale_date": ["2024-01-15", "2024-02-20"],
                "discount_pct": [0.0, 10.0],
                "email": ["alice@example.com", "bob@corp.io"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        cat_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "category"
        ]
        assert len(cat_steps) == 1
        assert cat_steps[0].n_failed == 1

    def test_to_validate_out_of_range(self, meta):
        """Amount outside valid range is detected."""
        df = pl.DataFrame(
            {
                "transaction_id": ["TXN-001", "TXN-002"],
                "customer_id": ["CUST-12345", "CUST-67890"],
                "amount": [29.99, 100000.0],  # exceeds max of 99999.99
                "quantity": [1, 3],
                "category": ["electronics", "clothing"],
                "sale_date": ["2024-01-15", "2024-02-20"],
                "discount_pct": [0.0, 10.0],
                "email": ["alice@example.com", "bob@corp.io"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        amount_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_between" and v.column == "amount"
        ]
        assert len(amount_steps) == 1
        assert amount_steps[0].n_failed == 1


# ===========================================================================
# Frictionless Table Schema (standalone)
# ===========================================================================


class TestTableSchemaEndToEnd:
    """End-to-end tests for standalone Frictionless Table Schema."""

    @pytest.fixture()
    def meta(self):
        return pb.import_metadata(str(FIXTURES / "table_schema.json"), format="table_schema")

    def test_format(self, meta):
        assert meta.source_format == "frictionless"

    def test_variable_count(self, meta):
        assert len(meta.variables) == 6

    def test_constraints(self, meta):
        vars_dict = {v.name: v for v in meta.variables}
        # sensor_id: required, pattern
        assert vars_dict["sensor_id"].required is True
        assert vars_dict["sensor_id"].pattern == r"^SNS-[0-9]{4}$"
        # battery_pct: required, 0-100
        assert vars_dict["battery_pct"].required is True
        assert vars_dict["battery_pct"].min_val == 0.0
        assert vars_dict["battery_pct"].max_val == 100.0
        # status: enum
        assert vars_dict["status"].allowed_values == ["active", "maintenance", "offline", "error"]

    def test_to_validate_valid_data(self, meta):
        df = pl.DataFrame(
            {
                "sensor_id": ["SNS-0001", "SNS-0002", "SNS-0003"],
                "reading_time": [
                    "2024-06-01T08:00:00",
                    "2024-06-01T09:00:00",
                    "2024-06-01T10:00:00",
                ],
                "temperature": [22.5, 23.1, 18.7],
                "pressure_hpa": [1013.25, 1012.8, 1014.0],
                "battery_pct": [95, 88, 72],
                "status": ["active", "active", "maintenance"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        for v in validation.validation_info:
            if v.assertion_type in (
                "col_vals_not_null",
                "col_vals_in_set",
                "col_vals_between",
                "col_vals_regex",
            ):
                assert v.n_failed == 0, f"Step {v.i} ({v.assertion_type}) failed"

    def test_to_validate_bad_sensor_pattern(self, meta):
        """Sensor IDs not matching pattern are caught."""
        df = pl.DataFrame(
            {
                "sensor_id": ["SNS-0001", "BAD-FORMAT", "SNS-0003"],
                "reading_time": [
                    "2024-06-01T08:00:00",
                    "2024-06-01T09:00:00",
                    "2024-06-01T10:00:00",
                ],
                "temperature": [22.5, 23.1, 18.7],
                "pressure_hpa": [1013.25, 1012.8, 1014.0],
                "battery_pct": [95, 88, 72],
                "status": ["active", "active", "maintenance"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        regex_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_regex" and v.column == "sensor_id"
        ]
        assert len(regex_steps) == 1
        assert regex_steps[0].n_failed == 1


# ===========================================================================
# CSVW (CSV on the Web)
# ===========================================================================


class TestCsvwEndToEnd:
    """End-to-end tests for CSVW metadata import."""

    @pytest.fixture()
    def meta(self):
        return pb.import_metadata(str(FIXTURES / "weather_csvw.json"), format="csvw")

    def test_format(self, meta):
        assert meta.source_format == "csvw"

    def test_dataset_name(self, meta):
        assert meta.dataset_name == "weather_observations"

    def test_variable_count(self, meta):
        assert len(meta.variables) == 7

    def test_constraints(self, meta):
        vars_dict = {v.name: v for v in meta.variables}
        assert vars_dict["station_id"].required is True
        assert vars_dict["temperature_c"].required is True
        assert vars_dict["temperature_c"].min_val == -50.0
        assert vars_dict["temperature_c"].max_val == 60.0
        assert vars_dict["humidity_pct"].max_val == 100.0

    def test_to_validate_valid_data(self, meta):
        df = pl.DataFrame(
            {
                "station_id": ["WS-001", "WS-002", "WS-003"],
                "timestamp": ["2024-06-01T08:00", "2024-06-01T09:00", "2024-06-01T10:00"],
                "temperature_c": [22.5, 23.1, 18.7],
                "humidity_pct": [65.0, 62.0, 78.0],
                "wind_speed_kmh": [12.5, 15.0, 8.0],
                "precipitation_mm": [0.0, 0.0, 0.2],
                "condition": ["clear", "clear", "cloudy"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        # Range checks should pass for valid weather data
        between_steps = [
            v for v in validation.validation_info if v.assertion_type == "col_vals_between"
        ]
        for step in between_steps:
            assert step.n_failed == 0, f"Range check on {step.column} failed"

    def test_to_validate_temperature_out_of_range(self, meta):
        """Temperature above 60C is caught."""
        df = pl.DataFrame(
            {
                "station_id": ["WS-001", "WS-002"],
                "timestamp": ["2024-06-01T08:00", "2024-06-01T09:00"],
                "temperature_c": [22.5, 65.0],  # 65 exceeds max of 60
                "humidity_pct": [65.0, 62.0],
                "wind_speed_kmh": [12.5, 15.0],
                "precipitation_mm": [0.0, 0.0],
                "condition": ["clear", "clear"],
            }
        )
        validation = meta.to_validate(data=df).interrogate()
        temp_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_between" and v.column == "temperature_c"
        ]
        assert len(temp_steps) == 1
        assert temp_steps[0].n_failed == 1


# ===========================================================================
# CDISC Define-XML
# ===========================================================================


class TestDefineXmlEndToEnd:
    """End-to-end tests for CDISC Define-XML import."""

    @pytest.fixture()
    def package(self):
        return pb.import_metadata(str(FIXTURES / "define.xml"), format="cdisc_define")

    def test_returns_package(self, package):
        from pointblank.metadata import MetadataPackage

        assert isinstance(package, MetadataPackage)

    def test_domains_found(self, package):
        assert "DM" in package
        assert "AE" in package

    def test_dm_variables(self, package):
        dm = package["DM"]
        assert dm.dataset_label == "Demographics"
        names = [v.name for v in dm.variables]
        assert "STUDYID" in names
        assert "USUBJID" in names
        assert "SEX" in names
        assert "AGE" in names

    def test_dm_required_variables(self, package):
        dm = package["DM"]
        required = [v.name for v in dm.variables if v.required]
        assert "STUDYID" in required
        assert "DOMAIN" in required
        assert "USUBJID" in required
        assert "SUBJID" in required

    def test_dm_codelists(self, package):
        dm = package["DM"]
        # SEX and RACE should have codelist references
        sex_var = next(v for v in dm.variables if v.name == "SEX")
        assert sex_var.codelist_ref is not None
        # The codelist should be in the metadata
        assert len(dm.codelists) >= 2  # SEX and RACE at minimum

    def test_codelist_values(self, package):
        dm = package["DM"]
        # Find the SEX codelist
        sex_cl = None
        for cl in dm.codelists.values():
            if "sex" in cl.name.lower() or "Sex" in (cl.label or ""):
                sex_cl = cl
                break
        assert sex_cl is not None
        assert set(sex_cl.to_set()) == {"M", "F", "U"}

    def test_ae_variables(self, package):
        ae = package["AE"]
        assert ae.dataset_label == "Adverse Events"
        names = [v.name for v in ae.variables]
        assert "AETERM" in names
        assert "AEDECOD" in names
        assert "AESEV" in names

    def test_dm_to_validate(self, package):
        """Full validation of DM data using Define-XML metadata."""
        dm = package["DM"]
        df = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["XYZ789-101", "XYZ789-102", "XYZ789-103"],
                "SUBJID": ["101", "102", "103"],
                "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "RFENDTC": ["2024-07-15", "2024-07-20", "2024-08-01"],
                "SITEID": ["S01", "S01", "S02"],
                "AGE": [45, 62, 38],
                "AGEU": ["YEARS"] * 3,
                "SEX": ["M", "F", "M"],
                "RACE": ["WHITE", "ASIAN", "BLACK OR AFRICAN AMERICAN"],
                "ARMCD": ["TRT", "PBO", "TRT"],
                "ARM": ["Active 10mg", "Placebo", "Active 10mg"],
            }
        )
        validation = dm.to_validate(data=df).interrogate()
        # Required vars should be non-null, codelists should pass
        for v in validation.validation_info:
            if v.assertion_type in ("col_vals_not_null", "col_vals_in_set"):
                assert v.n_failed == 0, (
                    f"Step {v.i} ({v.assertion_type}, col={v.column}) failed with {v.n_failed}"
                )

    def test_dm_to_validate_bad_sex(self, package):
        """Invalid SEX value caught via Define-XML codelist."""
        dm = package["DM"]
        df = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 2,
                "DOMAIN": ["DM"] * 2,
                "USUBJID": ["XYZ789-101", "XYZ789-102"],
                "SUBJID": ["101", "102"],
                "RFSTDTC": ["2024-01-15", "2024-01-20"],
                "RFENDTC": ["2024-07-15", "2024-07-20"],
                "SITEID": ["S01", "S01"],
                "AGE": [45, 62],
                "AGEU": ["YEARS"] * 2,
                "SEX": ["M", "X"],  # X is not in codelist
                "RACE": ["WHITE", "ASIAN"],
                "ARMCD": ["TRT", "PBO"],
                "ARM": ["Active 10mg", "Placebo"],
            }
        )
        validation = dm.to_validate(data=df).interrogate()
        sex_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "SEX"
        ]
        assert len(sex_steps) == 1
        assert sex_steps[0].n_failed == 1


# ===========================================================================
# CDISC Controlled Terminology
# ===========================================================================


class TestCdiscCtEndToEnd:
    """End-to-end tests for CDISC Controlled Terminology import."""

    @pytest.fixture()
    def package(self):
        return pb.import_metadata(str(FIXTURES / "sdtm_ct.xml"), format="cdisc_ct")

    def test_returns_package(self, package):
        from pointblank.metadata import MetadataPackage

        assert isinstance(package, MetadataPackage)

    def test_codelists_found(self, package):
        # Should find SEX, SEVERITY, NY, RACE, ROUTE
        assert len(package) == 5

    def test_sex_codelist(self, package):
        """SEX codelist has correct values and is non-extensible."""
        sex_item = package["Sex"]
        sex_cl = list(sex_item.codelists.values())[0]
        assert set(sex_cl.to_set()) == {"F", "M", "U", "UNDIFFERENTIATED"}
        assert sex_cl.extensible is False

    def test_race_codelist_extensible(self, package):
        """RACE codelist is extensible."""
        race_item = package["Race"]
        race_cl = list(race_item.codelists.values())[0]
        assert race_cl.extensible is True
        assert "WHITE" in race_cl.to_set()
        assert "ASIAN" in race_cl.to_set()

    def test_severity_codelist(self, package):
        """SEVERITY codelist has MILD/MODERATE/SEVERE."""
        sev_item = package["Severity/Intensity Scale for Adverse Events"]
        sev_cl = list(sev_item.codelists.values())[0]
        assert set(sev_cl.to_set()) == {"MILD", "MODERATE", "SEVERE"}

    def test_use_codelist_for_validation(self, package):
        """Use extracted codelist in a validation workflow."""
        sex_item = package["Sex"]
        sex_cl = list(sex_item.codelists.values())[0]

        # Valid data
        good_df = pl.DataFrame({"SEX": ["M", "F", "U", "M", "F"]})
        validation = (
            pb.Validate(data=good_df)
            .col_vals_in_set(columns="SEX", set=sex_cl.to_set())
            .interrogate()
        )
        assert validation.all_passed()

        # Invalid data
        bad_df = pl.DataFrame({"SEX": ["M", "F", "X", "UNKNOWN"]})
        validation = (
            pb.Validate(data=bad_df)
            .col_vals_in_set(columns="SEX", set=sex_cl.to_set())
            .interrogate()
        )
        assert validation.validation_info[0].n_failed == 2


# ===========================================================================
# SDTM Domain Templates (end-to-end with real-ish data)
# ===========================================================================


class TestSdtmEndToEnd:
    """End-to-end tests for SDTM domain validation with realistic data."""

    def test_dm_valid_data(self):
        """Complete DM dataset passes SDTM validation."""
        from pointblank.metadata import validate_sdtm

        dm_data = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 4,
                "DOMAIN": ["DM"] * 4,
                "USUBJID": ["XYZ789-101", "XYZ789-102", "XYZ789-103", "XYZ789-104"],
                "SUBJID": ["101", "102", "103", "104"],
                "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-10"],
                "RFENDTC": ["2024-07-15", "2024-07-20", "2024-08-01", "2024-08-10"],
                "SITEID": ["S01", "S01", "S02", "S02"],
                "AGE": [45, 62, 38, 55],
                "AGEU": ["YEARS"] * 4,
                "SEX": ["M", "F", "M", "F"],
                "RACE": ["WHITE", "BLACK", "ASIAN", "WHITE"],
                "ARMCD": ["TRT", "PBO", "TRT", "PBO"],
                "ARM": ["Active 10mg", "Placebo", "Active 10mg", "Placebo"],
                "COUNTRY": ["USA", "USA", "GBR", "GBR"],
            }
        )
        validation = validate_sdtm(data=dm_data, domain="DM").interrogate()
        # All required-not-null checks should pass
        null_steps = [
            v for v in validation.validation_info if v.assertion_type == "col_vals_not_null"
        ]
        for step in null_steps:
            assert step.n_failed == 0, f"Not-null check on step {step.i} failed"

    def test_dm_detects_null_required(self):
        """Null value in required field is caught."""
        from pointblank.metadata import validate_sdtm

        dm_data = pl.DataFrame(
            {
                "STUDYID": ["XYZ789", None, "XYZ789"],
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["XYZ789-101", "XYZ789-102", "XYZ789-103"],
                "SUBJID": ["101", "102", "103"],
                "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "RFENDTC": ["2024-07-15", "2024-07-20", "2024-08-01"],
                "SITEID": ["S01", "S01", "S02"],
            }
        )
        validation = validate_sdtm(data=dm_data, domain="DM").interrogate()
        # STUDYID not-null should fail
        studyid_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_not_null" and v.column == "STUDYID"
        ]
        assert len(studyid_steps) == 1
        assert studyid_steps[0].n_failed == 1

    def test_dm_detects_bad_date_format(self):
        """Non-ISO 8601 date in --DTC variable is caught."""
        from pointblank.metadata import validate_sdtm

        dm_data = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["XYZ789-101", "XYZ789-102", "XYZ789-103"],
                "SUBJID": ["101", "102", "103"],
                "RFSTDTC": ["2024-01-15", "01/20/2024", "2024-02-01"],  # bad format
                "RFENDTC": ["2024-07-15", "2024-07-20", "Aug 1, 2024"],  # bad format
                "SITEID": ["S01", "S01", "S02"],
            }
        )
        validation = validate_sdtm(data=dm_data, domain="DM").interrogate()
        regex_steps = [
            v for v in validation.validation_info if v.assertion_type == "col_vals_regex"
        ]
        # At least one date regex check should fail
        total_failures = sum(v.n_failed for v in regex_steps)
        assert total_failures >= 2  # at least 2 bad dates

    def test_ae_valid_data(self):
        """AE domain with valid data passes key checks."""
        from pointblank.metadata import validate_sdtm

        ae_data = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 4,
                "DOMAIN": ["AE"] * 4,
                "USUBJID": ["XYZ789-101", "XYZ789-101", "XYZ789-102", "XYZ789-102"],
                "AESEQ": [1, 2, 1, 2],
                "AETERM": ["HEADACHE", "NAUSEA", "FATIGUE", "DIZZINESS"],
                "AEDECOD": ["Headache", "Nausea", "Fatigue", "Dizziness"],
                "AESTDTC": ["2024-02-01", "2024-02-15", "2024-02-05", "2024-03-01"],
                "AEENDTC": ["2024-02-03", "2024-02-17", "2024-02-10", "2024-03-05"],
                "AESEV": ["MILD", "MODERATE", "MILD", "MILD"],
                "AESER": ["N", "N", "N", "N"],
            }
        )
        validation = validate_sdtm(data=ae_data, domain="AE").interrogate()
        # AESEQ should be positive (passes)
        seq_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_gt" and v.column == "AESEQ"
        ]
        assert len(seq_steps) == 1
        assert seq_steps[0].n_failed == 0

    def test_ae_wrong_domain_value(self):
        """Wrong DOMAIN value is caught."""
        from pointblank.metadata import validate_sdtm

        ae_data = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 2,
                "DOMAIN": ["AE", "XX"],  # XX is wrong
                "USUBJID": ["XYZ789-101", "XYZ789-102"],
                "AESEQ": [1, 1],
                "AETERM": ["HEADACHE", "NAUSEA"],
                "AEDECOD": ["Headache", "Nausea"],
                "AESTDTC": ["2024-02-01", "2024-02-15"],
                "AEENDTC": ["2024-02-03", "2024-02-17"],
            }
        )
        validation = validate_sdtm(data=ae_data, domain="AE").interrogate()
        domain_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "DOMAIN"
        ]
        assert len(domain_steps) == 1
        assert domain_steps[0].n_failed == 1


# ===========================================================================
# ADaM Templates (end-to-end with real-ish data)
# ===========================================================================


class TestAdamEndToEnd:
    """End-to-end tests for ADaM validation with realistic data."""

    def test_adsl_valid_data(self):
        """Complete ADSL passes ADaM validation."""
        from pointblank.metadata import validate_adam

        adsl = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 4,
                "USUBJID": [f"XYZ789-{i:03d}" for i in range(1, 5)],
                "SUBJID": [f"{i:03d}" for i in range(1, 5)],
                "SITEID": ["S01", "S01", "S02", "S02"],
                "TRT01P": ["Drug A", "Placebo", "Drug A", "Placebo"],
                "TRT01A": ["Drug A", "Placebo", "Drug A", "Placebo"],
                "AGE": [45, 62, 38, 55],
                "AGEU": ["YEARS"] * 4,
                "SEX": ["M", "F", "M", "F"],
                "RACE": ["WHITE", "BLACK", "ASIAN", "WHITE"],
                "SAFFL": ["Y", "Y", "Y", "Y"],
                "ITTFL": ["Y", "Y", "Y", "Y"],
                "EFFFL": ["Y", "Y", "N", "Y"],
                "TRTSDT": ["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-10"],
                "TRTEDT": ["2024-06-15", "2024-06-20", "2024-07-01", "2024-07-10"],
            }
        )
        validation = validate_adam(data=adsl, dataset="ADSL").interrogate()
        # Population flags should pass (all Y/N)
        flag_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column in ("SAFFL", "ITTFL", "EFFFL")
        ]
        for step in flag_steps:
            assert step.n_failed == 0, f"Flag {step.column} check failed"

    def test_adsl_bad_population_flag(self):
        """Invalid population flag value is caught."""
        from pointblank.metadata import validate_adam

        adsl = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "USUBJID": ["XYZ789-001", "XYZ789-002", "XYZ789-003"],
                "SUBJID": ["001", "002", "003"],
                "SITEID": ["S01", "S01", "S02"],
                "TRT01P": ["Drug A", "Placebo", "Drug A"],
                "TRT01A": ["Drug A", "Placebo", "Drug A"],
                "SAFFL": ["Y", "MAYBE", "N"],  # "MAYBE" is invalid
                "ITTFL": ["Y", "Y", "Y"],
                "TRTSDT": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "TRTEDT": ["2024-06-15", "2024-06-20", "2024-07-01"],
            }
        )
        validation = validate_adam(data=adsl, dataset="ADSL").interrogate()
        saffl_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "SAFFL"
        ]
        assert len(saffl_steps) == 1
        assert saffl_steps[0].n_failed == 1

    def test_adtte_valid_data(self):
        """ADTTE with valid censoring and time values passes."""
        from pointblank.metadata import validate_adam

        adtte = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 4,
                "USUBJID": [f"XYZ789-{i:03d}" for i in range(1, 5)],
                "PARAMCD": ["OS"] * 4,
                "PARAM": ["Overall Survival"] * 4,
                "AVAL": [365.0, 180.0, 540.0, 270.0],
                "CNSR": [0, 1, 0, 1],
                "STARTDT": ["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-10"],
                "ADT": ["2025-01-15", "2024-07-20", "2025-07-01", "2024-11-10"],
                "TRTA": ["Drug A", "Placebo", "Drug A", "Placebo"],
            }
        )
        validation = validate_adam(data=adtte, dataset="ADTTE").interrogate()
        # CNSR in {0, 1} should pass
        cnsr_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "CNSR"
        ]
        assert len(cnsr_steps) == 1
        assert cnsr_steps[0].n_failed == 0
        # AVAL >= 0 should pass
        aval_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_ge" and v.column == "AVAL"
        ]
        assert len(aval_steps) == 1
        assert aval_steps[0].n_failed == 0

    def test_adtte_bad_cnsr(self):
        """Invalid CNSR value (must be 0 or 1) is caught."""
        from pointblank.metadata import validate_adam

        adtte = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "USUBJID": ["XYZ789-001", "XYZ789-002", "XYZ789-003"],
                "PARAMCD": ["OS"] * 3,
                "PARAM": ["Overall Survival"] * 3,
                "AVAL": [365.0, 180.0, 540.0],
                "CNSR": [0, 1, 2],  # 2 is invalid
                "STARTDT": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "ADT": ["2025-01-15", "2024-07-20", "2025-07-01"],
                "TRTA": ["Drug A", "Placebo", "Drug A"],
            }
        )
        validation = validate_adam(data=adtte, dataset="ADTTE").interrogate()
        cnsr_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and v.column == "CNSR"
        ]
        assert len(cnsr_steps) == 1
        assert cnsr_steps[0].n_failed == 1

    def test_adtte_negative_time(self):
        """Negative AVAL (time-to-event) is caught."""
        from pointblank.metadata import validate_adam

        adtte = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "USUBJID": ["XYZ789-001", "XYZ789-002", "XYZ789-003"],
                "PARAMCD": ["OS"] * 3,
                "PARAM": ["Overall Survival"] * 3,
                "AVAL": [365.0, -10.0, 540.0],  # -10 is invalid
                "CNSR": [0, 1, 0],
                "STARTDT": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "ADT": ["2025-01-15", "2024-07-20", "2025-07-01"],
                "TRTA": ["Drug A", "Placebo", "Drug A"],
            }
        )
        validation = validate_adam(data=adtte, dataset="ADTTE").interrogate()
        aval_steps = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_ge" and v.column == "AVAL"
        ]
        assert len(aval_steps) == 1
        assert aval_steps[0].n_failed == 1

    def test_bds_paramcd_length(self):
        """PARAMCD exceeding 8 characters is caught."""
        from pointblank.metadata import validate_adam

        bds = pl.DataFrame(
            {
                "STUDYID": ["XYZ789"] * 3,
                "USUBJID": ["XYZ789-001"] * 3,
                "PARAMCD": ["ALT", "AST", "TOOLONGCD"],  # > 8 chars
                "PARAM": ["Alanine Aminotransferase", "Aspartate Aminotransferase", "Bad Param"],
                "AVAL": [25.0, 30.0, 12.0],
                "TRTA": ["Drug A"] * 3,
            }
        )
        validation = validate_adam(data=bds, dataset="BDS").interrogate()
        # Should have a col_vals_expr step that checks length
        expr_steps = [v for v in validation.validation_info if v.assertion_type == "col_vals_expr"]
        assert len(expr_steps) >= 1
        # At least one row should fail (TOOLONGCD is 9 chars)
        total_failures = sum(v.n_failed for v in expr_steps)
        assert total_failures >= 1


# ===========================================================================
# Export and round-trip
# ===========================================================================


class TestExportRoundTrip:
    """Test exporting metadata and re-importing for round-trip fidelity."""

    def test_sdtm_to_frictionless_roundtrip(self, tmp_path):
        """Export SDTM metadata as Frictionless, re-import, verify."""
        from pointblank.metadata import sdtm_to_metadata

        # Convert DM template to MetadataImport
        dm_meta = sdtm_to_metadata(domain="DM", study_id="XYZ789")

        # Export to Frictionless
        output_path = tmp_path / "dm_schema.json"
        pb.export_metadata(dm_meta, str(output_path), format="frictionless")

        # File exists and is valid JSON
        assert output_path.exists()
        import json

        with open(output_path) as f:
            exported = json.load(f)
        assert "fields" in exported
        assert len(exported["fields"]) == len(dm_meta.variables)

        # Re-import
        reimported = pb.import_metadata(str(output_path), format="table_schema")
        assert len(reimported.variables) == len(dm_meta.variables)

        # Variable names should be preserved
        orig_names = {v.name for v in dm_meta.variables}
        reimp_names = {v.name for v in reimported.variables}
        assert orig_names == reimp_names

    def test_xpt_metadata_to_validation_roundtrip(self):
        """Import .xpt metadata, validate actual .xpt data, all passes."""
        import pyreadstat
        import pandas as pd

        # Import metadata from the fixture
        meta = pb.import_metadata(str(FIXTURES / "dm.xpt"))

        # Read the actual data from the same .xpt file
        df_pandas, _ = pyreadstat.read_xport(str(FIXTURES / "dm.xpt"))
        df = pl.from_pandas(df_pandas)

        # Validate the data against its own metadata - should pass
        validation = meta.to_validate(data=df).interrogate()
        # Schema match should pass (same file!)
        schema_steps = [
            v for v in validation.validation_info if v.assertion_type == "col_schema_match"
        ]
        assert len(schema_steps) == 1
        assert schema_steps[0].n_failed == 0
