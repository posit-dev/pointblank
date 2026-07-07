import pytest
from pathlib import Path

from pointblank.metadata._types import (
    Codelist,
    CodelistEntry,
    MetadataImport,
    MetadataPackage,
    MissingValueCode,
    VariableMetadata,
)
from pointblank.metadata._import import import_metadata, _detect_format


@pytest.fixture
def sample_variable():
    """A sample `VariableMetadata` for testing."""
    return VariableMetadata(
        name="age",
        label="Respondent Age",
        dtype="Int64",
        required=True,
        min_val=0,
        max_val=120,
    )


@pytest.fixture
def sample_codelist():
    """A sample `Codelist` for testing."""
    return Codelist(
        name="sex_codes",
        label="Sex",
        source="Test",
        codes=[
            CodelistEntry(value=1, label="Male"),
            CodelistEntry(value=2, label="Female"),
            CodelistEntry(value=9, label="Unknown", is_deprecated=True),
        ],
    )


@pytest.fixture
def sample_metadata(sample_variable, sample_codelist):
    """A sample `MetadataImport` for testing."""
    return MetadataImport(
        source_format="test",
        source_path="/tmp/test.sav",
        dataset_name="test_data",
        dataset_label="Test Dataset",
        variables=[
            sample_variable,
            VariableMetadata(
                name="sex",
                label="Sex",
                dtype="Int64",
                allowed_values=[1, 2],
                codelist_ref="sex_codes",
                value_labels={1: "Male", 2: "Female"},
            ),
            VariableMetadata(
                name="name",
                label="Respondent Name",
                dtype="String",
                max_length=50,
            ),
        ],
        codelists={"sex_codes": sample_codelist},
    )


@pytest.fixture
def spss_file(tmp_path):
    """Create a small SPSS `.sav` file for testing."""
    pyreadstat = pytest.importorskip("pyreadstat")
    import pandas as pd

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 45, 60, 22],
            "gender": [1, 2, 1, 2, 1],
            "city": ["NYC", "LA", "CHI", "NYC", "LA"],
        }
    )

    filepath = tmp_path / "test_survey.sav"

    # Write with metadata
    pyreadstat.write_sav(
        df,
        str(filepath),
        column_labels=["Subject ID", "Age in years", "Gender", "City of residence"],
        variable_value_labels={"gender": {1: "Male", 2: "Female"}},
    )

    return filepath


@pytest.fixture
def xpt_file(tmp_path):
    """Create a small SAS Transport `.xpt` file for testing."""
    pyreadstat = pytest.importorskip("pyreadstat")
    import pandas as pd

    df = pd.DataFrame(
        {
            "USUBJID": ["STUDY-001", "STUDY-002", "STUDY-003"],
            "AGE": [55, 42, 67],
            "SEX": ["M", "F", "M"],
            "RACE": ["WHITE", "BLACK", "ASIAN"],
        }
    )

    filepath = tmp_path / "dm.xpt"

    pyreadstat.write_xport(
        df,
        str(filepath),
        column_labels=[
            "Unique Subject Identifier",
            "Age",
            "Sex",
            "Race",
        ],
        table_name="DM",
        file_label="Demographics",
    )

    return filepath


@pytest.fixture
def stata_file(tmp_path):
    """Create a small Stata `.dta` file for testing."""
    pyreadstat = pytest.importorskip("pyreadstat")
    import pandas as pd

    df = pd.DataFrame(
        {
            "income": [50000.0, 75000.0, 100000.0, 45000.0],
            "education": [1, 2, 3, 2],
            "region": ["NE", "SW", "NE", "MW"],
        }
    )

    filepath = tmp_path / "economic_data.dta"

    pyreadstat.write_dta(
        df,
        str(filepath),
        column_labels=["Annual Income", "Education Level", "Region"],
        variable_value_labels={"education": {1: "High School", 2: "Bachelor", 3: "Graduate"}},
    )

    return filepath


# =============================================================================
# Tests: Core types
# =============================================================================


class TestCodelistEntry:
    """Tests for `CodelistEntry` dataclass."""

    def test_basic_entry(self):
        entry = CodelistEntry(value=1, label="Male")
        assert entry.value == 1
        assert entry.label == "Male"
        assert entry.is_deprecated is False

    def test_deprecated_entry(self):
        entry = CodelistEntry(value=99, label="Unknown", is_deprecated=True)
        assert entry.is_deprecated is True


class TestCodelist:
    """Tests for Codelist dataclass."""

    def test_to_set(self, sample_codelist):
        # Should exclude deprecated entries
        result = sample_codelist.to_set()
        assert 1 in result
        assert 2 in result
        assert 9 not in result  # deprecated

    def test_to_dict(self, sample_codelist):
        result = sample_codelist.to_dict()
        assert result[1] == "Male"
        assert result[2] == "Female"
        assert result[9] == "Unknown"  # to_dict includes all entries

    def test_len(self, sample_codelist):
        assert len(sample_codelist) == 3

    def test_empty_codelist(self):
        cl = Codelist(name="empty")
        assert len(cl) == 0
        assert cl.to_set() == []
        assert cl.to_dict() == {}


class TestMissingValueCode:
    """Tests for `MissingValueCode` dataclass."""

    def test_basic_missing_code(self):
        mvc = MissingValueCode(value=-99, label="Not asked", category="user_missing")
        assert mvc.value == -99
        assert mvc.label == "Not asked"
        assert mvc.category == "user_missing"


class TestVariableMetadata:
    """Tests for VariableMetadata dataclass."""

    def test_basic_variable(self, sample_variable):
        assert sample_variable.name == "age"
        assert sample_variable.label == "Respondent Age"
        assert sample_variable.dtype == "Int64"
        assert sample_variable.required is True
        assert sample_variable.min_val == 0
        assert sample_variable.max_val == 120

    def test_defaults(self):
        var = VariableMetadata(name="x")
        assert var.label is None
        assert var.dtype is None
        assert var.required is False
        assert var.unique is False
        assert var.min_val is None
        assert var.max_val is None
        assert var.allowed_values is None
        assert var.missing_values is None


class TestMetadataImport:
    """Tests for `MetadataImport` dataclass."""

    def test_basic_properties(self, sample_metadata):
        assert sample_metadata.source_format == "test"
        assert sample_metadata.dataset_name == "test_data"
        assert len(sample_metadata) == 3
        assert len(sample_metadata.codelists) == 1

    def test_variable_names(self, sample_metadata):
        assert sample_metadata.variable_names == ["age", "sex", "name"]

    def test_get_variable(self, sample_metadata):
        var = sample_metadata.get_variable("age")
        assert var.name == "age"
        assert var.label == "Respondent Age"

    def test_get_variable_not_found(self, sample_metadata):
        with pytest.raises(KeyError, match="No variable named 'missing'"):
            sample_metadata.get_variable("missing")

    def test_get_codelist(self, sample_metadata):
        cl = sample_metadata.get_codelist("sex_codes")
        assert cl.name == "sex_codes"
        assert len(cl) == 3

    def test_get_codelist_not_found(self, sample_metadata):
        with pytest.raises(KeyError, match="No codelist named 'nonexistent'"):
            sample_metadata.get_codelist("nonexistent")

    def test_summary(self, sample_metadata):
        s = sample_metadata.summary()
        assert "Metadata Import (test)" in s
        assert "test_data" in s
        assert "Variables: 3" in s
        assert "age" in s
        assert "sex" in s

    def test_str(self, sample_metadata):
        assert "Metadata Import" in str(sample_metadata)

    def test_repr(self, sample_metadata):
        r = repr(sample_metadata)
        assert "MetadataImport" in r
        assert "source_format='test'" in r
        assert "variables=3" in r


class TestMetadataPackage:
    """Tests for MetadataPackage dataclass."""

    def test_basic_package(self, sample_metadata):
        pkg = MetadataPackage(
            name="Test Study",
            items={"DM": sample_metadata},
        )
        assert len(pkg) == 1
        assert "DM" in pkg
        assert pkg["DM"] is sample_metadata

    def test_get_domain(self, sample_metadata):
        pkg = MetadataPackage(items={"DM": sample_metadata, "AE": sample_metadata})
        dm = pkg.get_domain("DM")
        assert dm is sample_metadata

    def test_get_domain_not_found(self, sample_metadata):
        pkg = MetadataPackage(items={"DM": sample_metadata})
        with pytest.raises(KeyError, match="No domain/dataset named 'AE'"):
            pkg.get_domain("AE")

    def test_keys(self, sample_metadata):
        pkg = MetadataPackage(items={"DM": sample_metadata, "AE": sample_metadata})
        assert set(pkg.keys()) == {"DM", "AE"}

    def test_iter(self, sample_metadata):
        pkg = MetadataPackage(items={"DM": sample_metadata, "AE": sample_metadata})
        assert list(pkg) == ["DM", "AE"]

    def test_summary(self, sample_metadata):
        pkg = MetadataPackage(
            name="Test Study",
            items={"DM": sample_metadata},
        )
        s = pkg.summary()
        assert "Metadata Package" in s
        assert "Test Study" in s
        assert "[DM]" in s


# =============================================================================
# Tests: Format detection
# =============================================================================


class TestFormatDetection:
    """Tests for format auto-detection."""

    def test_detect_spss(self):
        assert _detect_format("data.sav") == "spss"
        assert _detect_format("/path/to/survey.sav") == "spss"
        assert _detect_format("file.zsav") == "spss"

    def test_detect_xpt(self):
        assert _detect_format("dm.xpt") == "xpt"
        assert _detect_format("/study/data/ae.xpt") == "xpt"

    def test_detect_stata(self):
        assert _detect_format("economic.dta") == "stata"

    def test_detect_unknown(self):
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_format("data.csv")

    def test_detect_from_path_object(self):
        assert _detect_format(Path("survey.sav")) == "spss"


# =============================================================================
# Tests: Import dispatcher
# =============================================================================


class TestImportMetadata:
    """Tests for the import_metadata dispatcher."""

    def test_unsupported_format(self, tmp_path):
        fake = tmp_path / "test.xyz"
        fake.write_text("dummy")
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            import_metadata(fake)

    def test_explicit_unsupported_format(self, tmp_path):
        fake = tmp_path / "test.dat"
        fake.write_text("data")
        with pytest.raises(ValueError, match="Unsupported metadata format"):
            import_metadata(fake, format="unknown_format")

    def test_non_path_input(self):
        with pytest.raises(TypeError, match="Expected a file path"):
            import_metadata({"key": "value"})

    def test_missing_pyreadstat(self, tmp_path, monkeypatch):
        """Test that a helpful error is raised when pyreadstat is missing."""
        import importlib

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        fake_sav = tmp_path / "test.sav"
        fake_sav.write_bytes(b"\x00" * 100)

        def mock_import(name, *args, **kwargs):
            if name == "pyreadstat":
                raise ImportError("No module named 'pyreadstat'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        # Clear the module from cache if loaded
        import sys

        if "pyreadstat" in sys.modules:
            monkeypatch.delitem(sys.modules, "pyreadstat")

        with pytest.raises(ImportError, match="pyreadstat"):
            import_metadata(fake_sav)


# =============================================================================
# Tests: SPSS reader
# =============================================================================


class TestSPSSReader:
    """Tests for SPSS .sav metadata reading."""

    def test_read_spss_basic(self, spss_file):
        meta = import_metadata(spss_file)

        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "spss"
        assert meta.source_path == str(spss_file)
        assert meta.dataset_name == "test_survey"

    def test_read_spss_variables(self, spss_file):
        meta = import_metadata(spss_file)

        assert len(meta.variables) == 4
        names = meta.variable_names
        assert "id" in names
        assert "age" in names
        assert "gender" in names
        assert "city" in names

    def test_read_spss_labels(self, spss_file):
        meta = import_metadata(spss_file)

        age_var = meta.get_variable("age")
        assert age_var.label == "Age in years"

        gender_var = meta.get_variable("gender")
        assert gender_var.label == "Gender"

    def test_read_spss_value_labels(self, spss_file):
        meta = import_metadata(spss_file)

        gender_var = meta.get_variable("gender")
        assert gender_var.value_labels is not None
        assert gender_var.value_labels[1] == "Male"
        assert gender_var.value_labels[2] == "Female"

    def test_read_spss_allowed_values(self, spss_file):
        meta = import_metadata(spss_file)

        gender_var = meta.get_variable("gender")
        assert gender_var.allowed_values is not None
        assert set(gender_var.allowed_values) == {1, 2}

    def test_read_spss_codelists(self, spss_file):
        meta = import_metadata(spss_file)

        # Should have a codelist for gender
        assert len(meta.codelists) >= 1
        gender_cl = meta.get_codelist("gender_values")
        assert gender_cl.to_set() == [1, 2]

    def test_read_spss_dtypes(self, spss_file):
        meta = import_metadata(spss_file)

        id_var = meta.get_variable("id")
        assert id_var.dtype in ("Float64", "Int64")  # SPSS stores as float by default

        city_var = meta.get_variable("city")
        assert city_var.dtype == "String"

    def test_read_spss_string_max_length(self, spss_file):
        meta = import_metadata(spss_file)

        city_var = meta.get_variable("city")
        # String variables should have max_length from SPSS width
        assert city_var.max_length is not None
        assert city_var.max_length > 0

    def test_read_spss_explicit_format(self, spss_file):
        """Test that explicit format='spss' works."""
        meta = import_metadata(spss_file, format="spss")
        assert meta.source_format == "spss"

    def test_read_spss_sav_alias(self, spss_file):
        """Test that format='sav' works as alias."""
        meta = import_metadata(spss_file, format="sav")
        assert meta.source_format == "spss"


# =============================================================================
# Tests: SAS Transport reader
# =============================================================================


class TestXPTReader:
    """Tests for SAS Transport .xpt metadata reading."""

    def test_read_xpt_basic(self, xpt_file):
        meta = import_metadata(xpt_file)

        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "xpt"
        assert meta.source_path == str(xpt_file)

    def test_read_xpt_variables(self, xpt_file):
        meta = import_metadata(xpt_file)

        assert len(meta.variables) == 4
        names = meta.variable_names
        assert "USUBJID" in names
        assert "AGE" in names
        assert "SEX" in names
        assert "RACE" in names

    def test_read_xpt_labels(self, xpt_file):
        meta = import_metadata(xpt_file)

        usubjid_var = meta.get_variable("USUBJID")
        assert usubjid_var.label == "Unique Subject Identifier"

    def test_read_xpt_dtypes(self, xpt_file):
        meta = import_metadata(xpt_file)

        age_var = meta.get_variable("AGE")
        assert age_var.dtype in ("Float64", "Int64")

        sex_var = meta.get_variable("SEX")
        assert sex_var.dtype == "String"

    def test_read_xpt_dataset_info(self, xpt_file):
        meta = import_metadata(xpt_file)

        # dataset_name from table_name or file stem
        assert meta.dataset_name is not None


# =============================================================================
# Tests: Stata reader
# =============================================================================


class TestStataReader:
    """Tests for Stata .dta metadata reading."""

    def test_read_stata_basic(self, stata_file):
        meta = import_metadata(stata_file)

        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "stata"
        assert meta.source_path == str(stata_file)
        assert meta.dataset_name == "economic_data"

    def test_read_stata_variables(self, stata_file):
        meta = import_metadata(stata_file)

        assert len(meta.variables) == 3
        names = meta.variable_names
        assert "income" in names
        assert "education" in names
        assert "region" in names

    def test_read_stata_labels(self, stata_file):
        meta = import_metadata(stata_file)

        income_var = meta.get_variable("income")
        assert income_var.label == "Annual Income"

    def test_read_stata_value_labels(self, stata_file):
        meta = import_metadata(stata_file)

        edu_var = meta.get_variable("education")
        assert edu_var.value_labels is not None
        assert edu_var.value_labels[1] == "High School"
        assert edu_var.value_labels[2] == "Bachelor"
        assert edu_var.value_labels[3] == "Graduate"

    def test_read_stata_codelists(self, stata_file):
        meta = import_metadata(stata_file)

        assert "education_values" in meta.codelists
        cl = meta.get_codelist("education_values")
        assert set(cl.to_set()) == {1, 2, 3}


# =============================================================================
# Tests: to_schema() conversion
# =============================================================================


class TestToSchema:
    """Tests for MetadataImport.to_schema() conversion."""

    def test_basic_schema_conversion(self, sample_metadata):
        schema = sample_metadata.to_schema()
        from pointblank.schema import Schema

        assert isinstance(schema, Schema)

    def test_schema_has_columns(self, sample_metadata):
        schema = sample_metadata.to_schema()

        # Schema should have entries for all variables
        assert schema.columns is not None
        col_names = [col[0] for col in schema.columns]
        assert "age" in col_names
        assert "sex" in col_names
        assert "name" in col_names

    def test_schema_from_spss(self, spss_file):
        meta = import_metadata(spss_file)
        schema = meta.to_schema()

        assert schema.columns is not None
        col_names = [col[0] for col in schema.columns]
        assert "id" in col_names
        assert "age" in col_names
        assert "gender" in col_names
        assert "city" in col_names

    def test_schema_from_xpt(self, xpt_file):
        meta = import_metadata(xpt_file)
        schema = meta.to_schema()

        assert schema.columns is not None
        col_names = [col[0] for col in schema.columns]
        assert "USUBJID" in col_names
        assert "AGE" in col_names


# =============================================================================
# Tests: to_validate() conversion
# =============================================================================


class TestToValidate:
    """Tests for MetadataImport.to_validate() conversion."""

    def test_basic_validate_conversion(self, spss_file):
        pyreadstat = pytest.importorskip("pyreadstat")
        import pandas as pd

        meta = import_metadata(spss_file)
        df, _ = pyreadstat.read_sav(str(spss_file))

        validation = meta.to_validate(data=df)

        from pointblank.validate import Validate

        assert isinstance(validation, Validate)

    def test_validate_has_label(self, spss_file):
        pyreadstat = pytest.importorskip("pyreadstat")

        meta = import_metadata(spss_file)
        df, _ = pyreadstat.read_sav(str(spss_file))

        validation = meta.to_validate(data=df)
        assert "spss" in validation.label.lower() or "test_survey" in validation.label.lower()

    def test_validate_custom_label(self, spss_file):
        pyreadstat = pytest.importorskip("pyreadstat")

        meta = import_metadata(spss_file)
        df, _ = pyreadstat.read_sav(str(spss_file))

        validation = meta.to_validate(data=df, label="Custom Label")
        assert validation.label == "Custom Label"

    def test_validate_generates_steps(self, spss_file):
        pyreadstat = pytest.importorskip("pyreadstat")

        meta = import_metadata(spss_file)
        df, _ = pyreadstat.read_sav(str(spss_file))

        validation = meta.to_validate(data=df)

        # Should have at least the schema check + value label checks
        assert len(validation.validation_info) > 0

    def test_validate_with_value_labels_generates_in_set(self, spss_file):
        """Columns with value labels should produce col_vals_in_set steps."""
        pyreadstat = pytest.importorskip("pyreadstat")

        meta = import_metadata(spss_file)
        df, _ = pyreadstat.read_sav(str(spss_file))

        validation = meta.to_validate(data=df)

        # Check that there are col_vals_in_set steps
        step_types = [step.assertion_type for step in validation.validation_info]
        assert "col_vals_in_set" in step_types

    def test_validate_interrogate(self, spss_file):
        """Full round-trip: import metadata → generate validation → interrogate."""
        pyreadstat = pytest.importorskip("pyreadstat")

        meta = import_metadata(spss_file)
        df, _ = pyreadstat.read_sav(str(spss_file))

        validation = meta.to_validate(data=df).interrogate()

        # Validation should complete without errors
        assert validation is not None
        assert len(validation.validation_info) > 0


# =============================================================================
# Frictionless + CSVW Tests
# =============================================================================


@pytest.fixture
def frictionless_table_schema(tmp_path):
    """Create a Frictionless Table Schema JSON file."""
    import json

    schema = {
        "fields": [
            {
                "name": "id",
                "type": "integer",
                "title": "Record ID",
                "constraints": {"required": True, "unique": True},
            },
            {
                "name": "name",
                "type": "string",
                "title": "Full Name",
                "description": "The person's full name",
                "constraints": {"required": True, "minLength": 1, "maxLength": 100},
            },
            {
                "name": "age",
                "type": "integer",
                "title": "Age",
                "constraints": {"minimum": 0, "maximum": 150},
            },
            {
                "name": "score",
                "type": "number",
                "title": "Test Score",
                "constraints": {"minimum": 0.0, "maximum": 100.0},
            },
            {
                "name": "status",
                "type": "string",
                "title": "Status",
                "constraints": {"enum": ["active", "inactive", "pending"]},
            },
            {
                "name": "email",
                "type": "string",
                "constraints": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
            },
            {
                "name": "joined",
                "type": "date",
                "title": "Join Date",
            },
        ],
        "primaryKey": "id",
        "missingValues": ["", "NA", "N/A"],
    }

    filepath = tmp_path / "schema.json"
    with open(filepath, "w") as f:
        json.dump(schema, f)

    return filepath


@pytest.fixture
def frictionless_datapackage(tmp_path):
    """Create a Frictionless Data Package JSON file with multiple resources."""
    import json

    package = {
        "name": "test-package",
        "description": "A test data package",
        "version": "1.0.0",
        "resources": [
            {
                "name": "users",
                "description": "User records",
                "schema": {
                    "fields": [
                        {
                            "name": "user_id",
                            "type": "integer",
                            "constraints": {"required": True, "unique": True},
                        },
                        {"name": "username", "type": "string", "title": "Username"},
                        {"name": "active", "type": "boolean"},
                    ],
                    "primaryKey": "user_id",
                },
            },
            {
                "name": "orders",
                "description": "Order records",
                "schema": {
                    "fields": [
                        {
                            "name": "order_id",
                            "type": "integer",
                            "constraints": {"required": True},
                        },
                        {"name": "user_id", "type": "integer"},
                        {"name": "amount", "type": "number"},
                        {"name": "order_date", "type": "date"},
                    ],
                },
            },
        ],
    }

    filepath = tmp_path / "datapackage.json"
    with open(filepath, "w") as f:
        json.dump(package, f)

    return filepath


@pytest.fixture
def csvw_metadata(tmp_path):
    """Create a CSVW metadata JSON-LD file."""
    import json

    metadata = {
        "url": "observations.csv",
        "dc:title": "Weather Observations",
        "dc:description": "Daily weather observations",
        "tableSchema": {
            "columns": [
                {
                    "name": "date",
                    "titles": "Observation Date",
                    "datatype": "date",
                    "required": True,
                },
                {
                    "name": "temperature",
                    "titles": "Temperature (C)",
                    "dc:description": "Air temperature in Celsius",
                    "datatype": {
                        "base": "decimal",
                        "minimum": -50.0,
                        "maximum": 60.0,
                    },
                },
                {
                    "name": "humidity",
                    "titles": "Relative Humidity (%)",
                    "datatype": {
                        "base": "integer",
                        "minInclusive": 0,
                        "maxInclusive": 100,
                    },
                },
                {
                    "name": "station_id",
                    "titles": "Station ID",
                    "datatype": {
                        "base": "string",
                        "maxLength": 10,
                    },
                },
                {
                    "name": "notes",
                    "titles": "Notes",
                    "datatype": "string",
                    "null": ["", "NA", "missing"],
                },
            ],
            "primaryKey": "date",
        },
    }

    filepath = tmp_path / "observations.csv-metadata.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f)

    return filepath


@pytest.fixture
def csvw_tablegroup(tmp_path):
    """Create a CSVW TableGroup metadata file."""
    import json

    metadata = {
        "tables": [
            {
                "url": "countries.csv",
                "dc:title": "Countries",
                "tableSchema": {
                    "columns": [
                        {"name": "code", "datatype": "string", "required": True},
                        {"name": "name", "datatype": "string"},
                        {"name": "population", "datatype": "integer"},
                    ],
                    "primaryKey": "code",
                },
            },
            {
                "url": "cities.csv",
                "dc:title": "Cities",
                "tableSchema": {
                    "columns": [
                        {"name": "city_name", "datatype": "string"},
                        {"name": "country_code", "datatype": "string"},
                        {
                            "name": "latitude",
                            "datatype": {"base": "decimal", "minimum": -90, "maximum": 90},
                        },
                        {
                            "name": "longitude",
                            "datatype": {"base": "decimal", "minimum": -180, "maximum": 180},
                        },
                    ],
                },
            },
        ],
    }

    filepath = tmp_path / "geo-metadata.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f)

    return filepath


# =============================================================================
# Tests: Frictionless Table Schema
# =============================================================================


class TestFrictionlessTableSchema:
    """Tests for Frictionless Table Schema import."""

    def test_import_basic(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema, format="frictionless")

        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "frictionless"
        assert meta.source_path == str(frictionless_table_schema)

    def test_auto_detect_json(self, frictionless_table_schema):
        """JSON files with 'fields' should auto-detect as frictionless."""
        meta = import_metadata(frictionless_table_schema)
        assert meta.source_format == "frictionless"

    def test_variables_count(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)
        assert len(meta.variables) == 7

    def test_variable_types(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        id_var = meta.get_variable("id")
        assert id_var.dtype == "Int64"

        name_var = meta.get_variable("name")
        assert name_var.dtype == "String"

        score_var = meta.get_variable("score")
        assert score_var.dtype == "Float64"

        joined_var = meta.get_variable("joined")
        assert joined_var.dtype == "Date"

    def test_constraints_required(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        id_var = meta.get_variable("id")
        assert id_var.required is True

        name_var = meta.get_variable("name")
        assert name_var.required is True

        age_var = meta.get_variable("age")
        assert age_var.required is False

    def test_constraints_unique(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        id_var = meta.get_variable("id")
        assert id_var.unique is True

    def test_constraints_min_max(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        age_var = meta.get_variable("age")
        assert age_var.min_val == 0
        assert age_var.max_val == 150

        score_var = meta.get_variable("score")
        assert score_var.min_val == 0.0
        assert score_var.max_val == 100.0

    def test_constraints_length(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        name_var = meta.get_variable("name")
        assert name_var.min_length == 1
        assert name_var.max_length == 100

    def test_constraints_enum(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        status_var = meta.get_variable("status")
        assert status_var.allowed_values == ["active", "inactive", "pending"]

    def test_constraints_pattern(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        email_var = meta.get_variable("email")
        assert email_var.pattern == r"^[^@]+@[^@]+\.[^@]+$"

    def test_primary_key_implies_required_unique(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        id_var = meta.get_variable("id")
        assert id_var.required is True
        assert id_var.unique is True

    def test_labels_and_descriptions(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        name_var = meta.get_variable("name")
        assert name_var.label == "Full Name"
        assert name_var.description == "The person's full name"

    def test_missing_values(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        # Package-level missing values should propagate
        age_var = meta.get_variable("age")
        assert age_var.missing_values is not None
        assert "NA" in age_var.missing_values
        assert "N/A" in age_var.missing_values

    def test_codelists_from_enum(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)

        assert "status_enum" in meta.codelists
        cl = meta.get_codelist("status_enum")
        assert set(cl.to_set()) == {"active", "inactive", "pending"}

    def test_to_schema(self, frictionless_table_schema):
        meta = import_metadata(frictionless_table_schema)
        schema = meta.to_schema()

        from pointblank.schema import Schema

        assert isinstance(schema, Schema)
        col_names = [col[0] for col in schema.columns]
        assert "id" in col_names
        assert "name" in col_names

    def test_to_validate(self, frictionless_table_schema):
        import polars as pl

        meta = import_metadata(frictionless_table_schema)
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [85.5, 92.0, 78.3],
                "status": ["active", "inactive", "active"],
                "email": ["a@b.com", "c@d.org", "e@f.net"],
                "joined": ["2020-01-01", "2020-06-15", "2021-03-20"],
            }
        )

        validation = meta.to_validate(data=df)
        assert len(validation.validation_info) > 0

        # Should have col_vals_in_set for status enum
        step_types = [s.assertion_type for s in validation.validation_info]
        assert "col_vals_in_set" in step_types
        assert "col_vals_not_null" in step_types


# =============================================================================
# Tests: Frictionless Data Package
# =============================================================================


class TestFrictionlessDataPackage:
    """Tests for Frictionless Data Package import."""

    def test_multi_resource_returns_package(self, frictionless_datapackage):
        result = import_metadata(frictionless_datapackage)

        assert isinstance(result, MetadataPackage)
        assert len(result) == 2
        assert "users" in result
        assert "orders" in result

    def test_package_metadata(self, frictionless_datapackage):
        result = import_metadata(frictionless_datapackage)

        assert result.name == "test-package"
        assert result.description == "A test data package"
        assert result.version == "1.0.0"

    def test_select_resource_by_name(self, frictionless_datapackage):
        result = import_metadata(frictionless_datapackage, resource="users")

        assert isinstance(result, MetadataImport)
        assert result.dataset_name == "users"
        assert len(result.variables) == 3

    def test_select_resource_by_index(self, frictionless_datapackage):
        result = import_metadata(frictionless_datapackage, resource=0)

        assert isinstance(result, MetadataImport)
        assert len(result.variables) == 3

    def test_resource_variables(self, frictionless_datapackage):
        result = import_metadata(frictionless_datapackage)

        users_meta = result["users"]
        assert users_meta.variable_names == ["user_id", "username", "active"]

        orders_meta = result["orders"]
        assert "order_id" in orders_meta.variable_names
        assert "amount" in orders_meta.variable_names

    def test_resource_constraints(self, frictionless_datapackage):
        result = import_metadata(frictionless_datapackage)

        users_meta = result["users"]
        user_id_var = users_meta.get_variable("user_id")
        assert user_id_var.required is True
        assert user_id_var.unique is True

    def test_invalid_resource_name(self, frictionless_datapackage):
        with pytest.raises(ValueError, match="not found"):
            import_metadata(frictionless_datapackage, resource="nonexistent")

    def test_invalid_resource_index(self, frictionless_datapackage):
        with pytest.raises(IndexError, match="out of range"):
            import_metadata(frictionless_datapackage, resource=99)


# =============================================================================
# Tests: CSVW
# =============================================================================


class TestCSVWReader:
    """Tests for CSVW metadata reading."""

    def test_import_basic(self, csvw_metadata):
        meta = import_metadata(csvw_metadata, format="csvw")

        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "csvw"

    def test_auto_detect_csvw(self, csvw_metadata):
        """CSVW files with 'tableSchema' should auto-detect."""
        meta = import_metadata(csvw_metadata)
        assert meta.source_format == "csvw"

    def test_variables_count(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)
        assert len(meta.variables) == 5

    def test_variable_types(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        date_var = meta.get_variable("date")
        assert date_var.dtype == "Date"

        temp_var = meta.get_variable("temperature")
        assert temp_var.dtype == "Float64"

        humidity_var = meta.get_variable("humidity")
        assert humidity_var.dtype == "Int64"

        station_var = meta.get_variable("station_id")
        assert station_var.dtype == "String"

    def test_constraints_from_datatype(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        temp_var = meta.get_variable("temperature")
        assert temp_var.min_val == -50.0
        assert temp_var.max_val == 60.0

        humidity_var = meta.get_variable("humidity")
        assert humidity_var.min_val == 0
        assert humidity_var.max_val == 100

    def test_max_length(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        station_var = meta.get_variable("station_id")
        assert station_var.max_length == 10

    def test_primary_key(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        date_var = meta.get_variable("date")
        assert date_var.required is True
        assert date_var.unique is True

    def test_null_markers(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        notes_var = meta.get_variable("notes")
        assert notes_var.missing_values is not None
        assert "NA" in notes_var.missing_values
        assert "missing" in notes_var.missing_values

    def test_dataset_info(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        assert meta.dataset_name == "observations"
        assert meta.dataset_label == "Weather Observations"

    def test_description(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)

        temp_var = meta.get_variable("temperature")
        assert temp_var.description == "Air temperature in Celsius"

    def test_tablegroup_returns_package(self, csvw_tablegroup):
        result = import_metadata(csvw_tablegroup, format="csvw")

        assert isinstance(result, MetadataPackage)
        assert len(result) == 2
        assert "countries" in result
        assert "cities" in result

    def test_tablegroup_variables(self, csvw_tablegroup):
        result = import_metadata(csvw_tablegroup, format="csvw")

        countries = result["countries"]
        assert "code" in countries.variable_names
        assert "population" in countries.variable_names

        cities = result["cities"]
        assert "latitude" in cities.variable_names
        lat_var = cities.get_variable("latitude")
        assert lat_var.min_val == -90
        assert lat_var.max_val == 90

    def test_to_schema(self, csvw_metadata):
        meta = import_metadata(csvw_metadata)
        schema = meta.to_schema()

        col_names = [col[0] for col in schema.columns]
        assert "date" in col_names
        assert "temperature" in col_names


# =============================================================================
# Tests: Frictionless Export
# =============================================================================


class TestFrictionlessExport:
    """Tests for exporting metadata to Frictionless Table Schema."""

    def test_basic_export(self):
        from pointblank.metadata._export import export_metadata

        meta = MetadataImport(
            source_format="test",
            variables=[
                VariableMetadata(name="id", dtype="Int64", required=True, unique=True),
                VariableMetadata(name="name", dtype="String", max_length=100),
                VariableMetadata(name="score", dtype="Float64", min_val=0, max_val=100),
            ],
        )

        result = export_metadata(meta, format="frictionless")

        assert isinstance(result, dict)
        assert "fields" in result
        assert len(result["fields"]) == 3

    def test_export_field_types(self):
        from pointblank.metadata._export import export_metadata

        meta = MetadataImport(
            source_format="test",
            variables=[
                VariableMetadata(name="x", dtype="Int64"),
                VariableMetadata(name="y", dtype="Float64"),
                VariableMetadata(name="z", dtype="String"),
                VariableMetadata(name="d", dtype="Date"),
                VariableMetadata(name="b", dtype="Boolean"),
            ],
        )

        result = export_metadata(meta, format="frictionless")
        fields = {f["name"]: f for f in result["fields"]}

        assert fields["x"]["type"] == "integer"
        assert fields["y"]["type"] == "number"
        assert fields["z"]["type"] == "string"
        assert fields["d"]["type"] == "date"
        assert fields["b"]["type"] == "boolean"

    def test_export_constraints(self):
        from pointblank.metadata._export import export_metadata

        meta = MetadataImport(
            source_format="test",
            variables=[
                VariableMetadata(
                    name="age",
                    dtype="Int64",
                    required=True,
                    min_val=0,
                    max_val=150,
                ),
                VariableMetadata(
                    name="status",
                    dtype="String",
                    allowed_values=["a", "b", "c"],
                ),
                VariableMetadata(
                    name="email",
                    dtype="String",
                    pattern=r"^.+@.+$",
                ),
            ],
        )

        result = export_metadata(meta, format="frictionless")
        fields = {f["name"]: f for f in result["fields"]}

        assert fields["age"]["constraints"]["required"] is True
        assert fields["age"]["constraints"]["minimum"] == 0
        assert fields["age"]["constraints"]["maximum"] == 150
        assert fields["status"]["constraints"]["enum"] == ["a", "b", "c"]
        assert fields["email"]["constraints"]["pattern"] == r"^.+@.+$"

    def test_export_primary_key(self):
        from pointblank.metadata._export import export_metadata

        meta = MetadataImport(
            source_format="test",
            variables=[
                VariableMetadata(name="id", dtype="Int64", required=True, unique=True),
                VariableMetadata(name="value", dtype="Float64"),
            ],
        )

        result = export_metadata(meta, format="frictionless")
        assert result["primaryKey"] == "id"

    def test_export_to_file(self, tmp_path):
        import json
        from pointblank.metadata._export import export_metadata

        meta = MetadataImport(
            source_format="test",
            dataset_label="Test Dataset",
            dataset_description="A test",
            variables=[
                VariableMetadata(name="x", dtype="Int64"),
            ],
        )

        filepath = tmp_path / "output.json"
        result = export_metadata(meta, destination=str(filepath), format="frictionless")

        assert filepath.exists()
        with open(filepath) as f:
            written = json.load(f)
        assert written == result
        assert written["title"] == "Test Dataset"
        assert written["description"] == "A test"

    def test_round_trip(self, frictionless_table_schema):
        """Import then export should preserve structure."""
        from pointblank.metadata._export import export_metadata

        meta = import_metadata(frictionless_table_schema)
        result = export_metadata(meta, format="frictionless")

        # Re-import the exported schema
        import json

        roundtrip_path = frictionless_table_schema.parent / "roundtrip.json"
        with open(roundtrip_path, "w") as f:
            json.dump(result, f)

        meta2 = import_metadata(roundtrip_path, format="frictionless")

        # Should have the same variables
        assert meta2.variable_names == meta.variable_names

        # Constraints should be preserved
        id_var = meta2.get_variable("id")
        assert id_var.required is True
        assert id_var.unique is True

        age_var = meta2.get_variable("age")
        assert age_var.min_val == 0
        assert age_var.max_val == 150

    def test_export_via_public_api(self):
        """Test that export_metadata is accessible from pb namespace."""
        import pointblank as pb

        meta = pb.MetadataImport(
            source_format="test",
            variables=[pb.VariableMetadata(name="x", dtype="Int64")],
        )
        result = pb.export_metadata(meta, format="frictionless")
        assert "fields" in result


# =============================================================================
# Tests: Format detection for JSON files
# =============================================================================


class TestJSONFormatDetection:
    """Tests for JSON format auto-detection."""

    def test_detect_frictionless_fields(self, frictionless_table_schema):
        from pointblank.metadata._import import _detect_format

        assert _detect_format(frictionless_table_schema) == "frictionless"

    def test_detect_frictionless_resources(self, frictionless_datapackage):
        from pointblank.metadata._import import _detect_format

        assert _detect_format(frictionless_datapackage) == "frictionless"

    def test_detect_csvw_tableschema(self, csvw_metadata):
        from pointblank.metadata._import import _detect_format

        assert _detect_format(csvw_metadata) == "csvw"

    def test_detect_csvw_tablegroup(self, csvw_tablegroup):
        from pointblank.metadata._import import _detect_format

        assert _detect_format(csvw_tablegroup) == "csvw"

    def test_ambiguous_json_raises(self, tmp_path):
        """A JSON file with no recognizable structure should raise."""
        import json
        from pointblank.metadata._import import _detect_format

        filepath = tmp_path / "unknown.json"
        with open(filepath, "w") as f:
            json.dump({"key": "value"}, f)

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_format(filepath)

    def test_invalid_json_raises(self, tmp_path):
        from pointblank.metadata._import import _detect_format

        filepath = tmp_path / "bad.json"
        with open(filepath, "w") as f:
            f.write("not json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            _detect_format(filepath)


# =============================================================================
# CDISC Define-XML Tests
# =============================================================================


class TestDefineXMLReader:
    """Tests for CDISC Define-XML 2.0/2.1 metadata reader."""

    @pytest.fixture
    def define_xml_single_domain(self, tmp_path):
        """A minimal Define-XML with a single DM (Demographics) domain."""
        xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     FileType="Snapshot"
     FileOID="DEFINE-FILE-001"
     CreationDateTime="2024-06-15T10:00:00">
  <Study OID="STUDY-001">
    <MetaDataVersion OID="MDV-001" Name="Study Metadata v1">
      <CodeList OID="CL.SEX" Name="SEX" DataType="text">
        <CodeListItem CodedValue="M">
          <Decode><TranslatedText>Male</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="F">
          <Decode><TranslatedText>Female</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="U">
          <Decode><TranslatedText>Unknown</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>
      <CodeList OID="CL.NY" Name="NY" DataType="text">
        <EnumeratedItem CodedValue="N"/>
        <EnumeratedItem CodedValue="Y"/>
      </CodeList>
      <ItemDef OID="IT.DM.STUDYID" Name="STUDYID" DataType="text" Length="20">
        <Description><TranslatedText>Study Identifier</TranslatedText></Description>
      </ItemDef>
      <ItemDef OID="IT.DM.USUBJID" Name="USUBJID" DataType="text" Length="40">
        <Description><TranslatedText>Unique Subject Identifier</TranslatedText></Description>
      </ItemDef>
      <ItemDef OID="IT.DM.SEX" Name="SEX" DataType="text" Length="2">
        <Description><TranslatedText>Sex</TranslatedText></Description>
        <CodeListRef CodeListOID="CL.SEX"/>
      </ItemDef>
      <ItemDef OID="IT.DM.AGE" Name="AGE" DataType="integer" Length="3" SignificantDigits="0">
        <Description><TranslatedText>Age</TranslatedText></Description>
      </ItemDef>
      <ItemDef OID="IT.DM.BRTHDTC" Name="BRTHDTC" DataType="partialDate" Length="10">
        <Description><TranslatedText>Date/Time of Birth</TranslatedText></Description>
      </ItemDef>
      <MethodDef OID="MT.AGE" Name="Algorithm for AGE">
        <Description><TranslatedText>AGE = floor((RFSTDTC - BRTHDTC) / 365.25)</TranslatedText></Description>
      </MethodDef>
      <ItemGroupDef OID="IG.DM" Name="DM" Domain="DM" Repeating="No" Purpose="Tabulation" SASDatasetName="DM">
        <Description><TranslatedText>Demographics</TranslatedText></Description>
        <ItemRef ItemOID="IT.DM.STUDYID" Mandatory="Yes" Role="Identifier" OrderNumber="1"/>
        <ItemRef ItemOID="IT.DM.USUBJID" Mandatory="Yes" Role="Identifier" OrderNumber="2"/>
        <ItemRef ItemOID="IT.DM.SEX" Mandatory="Yes" Role="Topic" OrderNumber="3"/>
        <ItemRef ItemOID="IT.DM.AGE" Mandatory="No" Role="Qualifier" OrderNumber="4"/>
        <ItemRef ItemOID="IT.DM.BRTHDTC" Mandatory="No" Role="Timing" OrderNumber="5"/>
      </ItemGroupDef>
    </MetaDataVersion>
  </Study>
</ODM>
"""
        filepath = tmp_path / "define.xml"
        filepath.write_text(xml_content)
        return filepath

    @pytest.fixture
    def define_xml_multi_domain(self, tmp_path):
        """A Define-XML with multiple domains (DM and AE)."""
        xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.1"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     FileType="Snapshot"
     FileOID="DEFINE-FILE-002"
     CreationDateTime="2024-07-01T12:00:00">
  <Study OID="STUDY-002">
    <MetaDataVersion OID="MDV-002" Name="Multi-Domain Study v2.1">
      <CodeList OID="CL.AESER" Name="AESER" DataType="text">
        <EnumeratedItem CodedValue="Y"/>
        <EnumeratedItem CodedValue="N"/>
      </CodeList>
      <ItemDef OID="IT.DM.STUDYID" Name="STUDYID" DataType="text" Length="20">
        <Description><TranslatedText>Study Identifier</TranslatedText></Description>
      </ItemDef>
      <ItemDef OID="IT.DM.USUBJID" Name="USUBJID" DataType="text" Length="40">
        <Description><TranslatedText>Unique Subject Identifier</TranslatedText></Description>
      </ItemDef>
      <ItemDef OID="IT.AE.AETERM" Name="AETERM" DataType="text" Length="200">
        <Description><TranslatedText>Reported Term for the Adverse Event</TranslatedText></Description>
      </ItemDef>
      <ItemDef OID="IT.AE.AESER" Name="AESER" DataType="text" Length="2">
        <Description><TranslatedText>Serious Event</TranslatedText></Description>
        <CodeListRef CodeListOID="CL.AESER"/>
      </ItemDef>
      <ItemDef OID="IT.AE.AESTDTC" Name="AESTDTC" DataType="datetime" Length="20">
        <Description><TranslatedText>Start Date/Time of Adverse Event</TranslatedText></Description>
      </ItemDef>
      <ItemGroupDef OID="IG.DM" Name="DM" Domain="DM" Repeating="No" Purpose="Tabulation">
        <Description><TranslatedText>Demographics</TranslatedText></Description>
        <ItemRef ItemOID="IT.DM.STUDYID" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.DM.USUBJID" Mandatory="Yes" Role="Identifier"/>
      </ItemGroupDef>
      <ItemGroupDef OID="IG.AE" Name="AE" Domain="AE" Repeating="Yes" Purpose="Tabulation">
        <Description><TranslatedText>Adverse Events</TranslatedText></Description>
        <ItemRef ItemOID="IT.DM.STUDYID" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.DM.USUBJID" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.AE.AETERM" Mandatory="Yes" Role="Topic"/>
        <ItemRef ItemOID="IT.AE.AESER" Mandatory="No" Role="Qualifier"/>
        <ItemRef ItemOID="IT.AE.AESTDTC" Mandatory="No" Role="Timing"/>
      </ItemGroupDef>
    </MetaDataVersion>
  </Study>
</ODM>
"""
        filepath = tmp_path / "define_multi.xml"
        filepath.write_text(xml_content)
        return filepath

    def test_read_single_domain(self, define_xml_single_domain):
        """Reading a single-domain Define-XML returns a MetadataImport."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_single_domain)
        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "cdisc_define"
        assert meta.domain == "DM"
        assert meta.dataset_name == "DM"
        assert meta.dataset_label == "Demographics"
        assert meta.study_id == "STUDY-001"
        assert "Define-XML 2.0" in meta.source_version

    def test_single_domain_variables(self, define_xml_single_domain):
        """Variables are correctly extracted with roles, types, and constraints."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_single_domain)
        assert len(meta.variables) == 5

        # Check STUDYID
        studyid = meta.get_variable("STUDYID")
        assert studyid.dtype == "String"
        assert studyid.required is True
        assert studyid.max_length == 20
        assert studyid.label == "Study Identifier"
        assert studyid.cdisc_role == "Identifier"
        assert studyid.cdisc_domain == "DM"

        # Check AGE
        age = meta.get_variable("AGE")
        assert age.dtype == "Int64"
        assert age.required is False
        assert age.cdisc_role == "Qualifier"
        assert age.significant_digits == 0

        # Check BRTHDTC (partial date → String)
        brthdtc = meta.get_variable("BRTHDTC")
        assert brthdtc.dtype == "String"
        assert brthdtc.cdisc_role == "Timing"

    def test_single_domain_codelists(self, define_xml_single_domain):
        """Codelists are extracted and linked to variables."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_single_domain)

        # SEX should have a codelist reference
        sex = meta.get_variable("SEX")
        assert sex.codelist_ref == "SEX"
        assert sex.allowed_values == ["M", "F", "U"]

        # Check the codelist object
        assert "SEX" in meta.codelists
        cl = meta.codelists["SEX"]
        assert len(cl) == 3
        assert cl.to_dict() == {"M": "Male", "F": "Female", "U": "Unknown"}

    def test_multi_domain_returns_package(self, define_xml_multi_domain):
        """Multiple domains in one file returns a MetadataPackage."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        result = _read_define_xml_metadata(define_xml_multi_domain)
        assert isinstance(result, MetadataPackage)
        assert len(result) == 2
        assert "DM" in result
        assert "AE" in result

    def test_multi_domain_select_one(self, define_xml_multi_domain):
        """Selecting a specific dataset returns a MetadataImport."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_multi_domain, dataset="AE")
        assert isinstance(meta, MetadataImport)
        assert meta.domain == "AE"
        assert meta.dataset_name == "AE"
        assert len(meta.variables) == 5

        # Check AE-specific variable
        aeterm = meta.get_variable("AETERM")
        assert aeterm.required is True
        assert aeterm.cdisc_role == "Topic"
        assert aeterm.max_length == 200

    def test_multi_domain_case_insensitive(self, define_xml_multi_domain):
        """Dataset selection is case-insensitive."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_multi_domain, dataset="dm")
        assert isinstance(meta, MetadataImport)
        assert meta.domain == "DM"

    def test_multi_domain_invalid_dataset(self, define_xml_multi_domain):
        """Requesting a non-existent dataset raises KeyError."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        with pytest.raises(KeyError, match="Dataset 'XY' not found"):
            _read_define_xml_metadata(define_xml_multi_domain, dataset="XY")

    def test_define_version_21_detected(self, define_xml_multi_domain):
        """Define-XML 2.1 is detected from namespace."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        result = _read_define_xml_metadata(define_xml_multi_domain)
        dm = result["DM"]
        assert "2.1" in dm.source_version

    def test_define_xml_to_schema(self, define_xml_single_domain):
        """to_schema() generates a valid Pointblank Schema."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_single_domain)
        schema = meta.to_schema()
        col_dict = dict(schema.columns)
        assert "STUDYID" in col_dict
        assert "AGE" in col_dict
        assert col_dict["AGE"] == "Int64"
        assert col_dict["SEX"] == "String"

    def test_define_xml_to_validate(self, define_xml_single_domain):
        """to_validate() generates validation steps from Define-XML constraints."""
        import pandas as pd
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        meta = _read_define_xml_metadata(define_xml_single_domain)
        df = pd.DataFrame(
            {
                "STUDYID": ["STUDY-001"],
                "USUBJID": ["SUBJ-001"],
                "SEX": ["M"],
                "AGE": [45],
                "BRTHDTC": ["1979-03-15"],
            }
        )
        validation = meta.to_validate(data=df)
        # Should have schema match + constraint steps
        assert len(validation.validation_info) > 0

    def test_define_xml_not_found(self, tmp_path):
        """FileNotFoundError for missing file."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

        with pytest.raises(FileNotFoundError):
            _read_define_xml_metadata(tmp_path / "nonexistent.xml")

    def test_define_xml_enumerated_items(self, define_xml_single_domain):
        """EnumeratedItems (value = label) are parsed correctly via _parse_codelists."""
        from pointblank.metadata._readers_cdisc import _read_define_xml_metadata, _ensure_lxml
        from lxml import etree

        # Parse the XML directly to access all codelists (not just domain-referenced ones)
        tree = etree.parse(str(define_xml_single_domain))
        root = tree.getroot()
        from pointblank.metadata._readers_cdisc import _detect_define_version, _parse_codelists

        ns, _ = _detect_define_version(root)
        mdv = root.find(".//odm:Study/odm:MetaDataVersion", ns)
        all_codelists = _parse_codelists(mdv, ns)

        # NY codelist uses EnumeratedItem (value = label)
        assert "CL.NY" in all_codelists
        ny_cl = all_codelists["CL.NY"]
        assert len(ny_cl) == 2
        assert ny_cl.to_set() == ["N", "Y"]
        # For EnumeratedItem, value and label should be the same
        assert ny_cl.to_dict() == {"N": "N", "Y": "Y"}


class TestCDISCCTReader:
    """Tests for CDISC Controlled Terminology reader."""

    @pytest.fixture
    def ct_file(self, tmp_path):
        """A minimal CDISC Controlled Terminology ODM-XML file."""
        xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:nciodm="http://ncicb.nci.nih.gov/xml/odm/EVS/CDISC"
     FileType="Snapshot"
     FileOID="CT-2024-09-27"
     CreationDateTime="2024-09-27T00:00:00">
  <Study OID="CDISC-CT">
    <MetaDataVersion OID="CDISC_CT_2024-09-27" Name="CDISC SDTM CT 2024-09-27"
                     Description="CDISC Controlled Terminology for SDTM">
      <CodeList OID="CL.C66731.SEX" Name="SEX" DataType="text"
                nciodm:CodeListExtensible="No">
        <Description><TranslatedText>Sex or gender</TranslatedText></Description>
        <EnumeratedItem CodedValue="F" nciodm:PreferredTerm="Female"
                        nciodm:CDISCSynonym="Female;FEMALE"/>
        <EnumeratedItem CodedValue="M" nciodm:PreferredTerm="Male"
                        nciodm:CDISCSynonym="Male;MALE"/>
        <EnumeratedItem CodedValue="U" nciodm:PreferredTerm="Unknown"
                        nciodm:CDISCSynonym="Unknown;UNKNOWN"/>
        <EnumeratedItem CodedValue="UNDIFFERENTIATED" nciodm:PreferredTerm="Undifferentiated"/>
      </CodeList>
      <CodeList OID="CL.C66742.NY" Name="NY" DataType="text"
                nciodm:CodeListExtensible="No">
        <Description><TranslatedText>Yes/No Response</TranslatedText></Description>
        <EnumeratedItem CodedValue="N" nciodm:PreferredTerm="No"/>
        <EnumeratedItem CodedValue="Y" nciodm:PreferredTerm="Yes"/>
      </CodeList>
      <CodeList OID="CL.C66769.RACE" Name="RACE" DataType="text"
                nciodm:CodeListExtensible="Yes">
        <Description><TranslatedText>Race</TranslatedText></Description>
        <EnumeratedItem CodedValue="AMERICAN INDIAN OR ALASKA NATIVE"
                        nciodm:PreferredTerm="American Indian or Alaska Native"/>
        <EnumeratedItem CodedValue="ASIAN" nciodm:PreferredTerm="Asian"/>
        <EnumeratedItem CodedValue="BLACK OR AFRICAN AMERICAN"
                        nciodm:PreferredTerm="Black or African American"/>
        <EnumeratedItem CodedValue="WHITE" nciodm:PreferredTerm="White"/>
        <EnumeratedItem CodedValue="MULTIPLE" nciodm:PreferredTerm="Multiple"/>
      </CodeList>
    </MetaDataVersion>
  </Study>
</ODM>
"""
        filepath = tmp_path / "sdtm_ct_2024-09-27.xml"
        filepath.write_text(xml_content)
        return filepath

    def test_read_all_codelists(self, ct_file):
        """Reading without filter returns all codelists as MetadataPackage."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        result = _read_cdisc_ct_metadata(ct_file)
        assert isinstance(result, MetadataPackage)
        assert len(result) == 3
        assert "SEX" in result
        assert "NY" in result
        assert "RACE" in result

    def test_read_single_codelist(self, ct_file):
        """Reading with codelist= filter returns a single MetadataImport."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        meta = _read_cdisc_ct_metadata(ct_file, codelist="SEX")
        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "cdisc_ct"
        assert "SEX" in meta.codelists

        cl = meta.codelists["SEX"]
        assert len(cl) == 4
        assert cl.to_set() == ["F", "M", "U", "UNDIFFERENTIATED"]

    def test_codelist_preferred_terms(self, ct_file):
        """NCI PreferredTerm is used as the label for entries."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        meta = _read_cdisc_ct_metadata(ct_file, codelist="SEX")
        cl = meta.codelists["SEX"]
        labels = cl.to_dict()
        assert labels["F"] == "Female"
        assert labels["M"] == "Male"

    def test_codelist_synonyms(self, ct_file):
        """CDISCSynonym is parsed into synonyms list."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        meta = _read_cdisc_ct_metadata(ct_file, codelist="SEX")
        cl = meta.codelists["SEX"]
        # Find the Female entry
        female_entry = next(e for e in cl.codes if e.value == "F")
        assert female_entry.synonyms == ["Female", "FEMALE"]

    def test_codelist_extensible(self, ct_file):
        """Non-extensible and extensible codelists are distinguished."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        result = _read_cdisc_ct_metadata(ct_file)

        sex_meta = result["SEX"]
        assert sex_meta.codelists["SEX"].extensible is False

        race_meta = result["RACE"]
        assert race_meta.codelists["RACE"].extensible is True

    def test_codelist_not_found(self, ct_file):
        """Requesting a non-existent codelist raises KeyError."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        with pytest.raises(KeyError, match="Codelist 'MISSING' not found"):
            _read_cdisc_ct_metadata(ct_file, codelist="MISSING")

    def test_ct_package_metadata(self, ct_file):
        """Package-level metadata is populated."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        result = _read_cdisc_ct_metadata(ct_file)
        assert result.name == "CDISC SDTM CT 2024-09-27"
        assert result.version == "2024-09-27"

    def test_ct_file_not_found(self, tmp_path):
        """FileNotFoundError for missing file."""
        from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

        with pytest.raises(FileNotFoundError):
            _read_cdisc_ct_metadata(tmp_path / "nonexistent.xml")


class TestXMLFormatDetection:
    """Tests for XML auto-detection (Define-XML vs CT)."""

    def test_detect_define_xml(self, tmp_path):
        """Detect Define-XML from def namespace."""
        xml = """\
<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0">
  <Study OID="S1"><MetaDataVersion OID="M1" Name="Test"/></Study>
</ODM>
"""
        filepath = tmp_path / "test.xml"
        filepath.write_text(xml)
        assert _detect_format(filepath) == "cdisc_define"

    def test_detect_ct_from_nci_ns(self, tmp_path):
        """Detect CDISC CT from NCI namespace."""
        xml = """\
<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:nciodm="http://ncicb.nci.nih.gov/xml/odm/EVS/CDISC">
  <Study OID="S1"><MetaDataVersion OID="M1" Name="Test"/></Study>
</ODM>
"""
        filepath = tmp_path / "ct.xml"
        filepath.write_text(xml)
        assert _detect_format(filepath) == "cdisc_ct"

    def test_detect_define_from_filename(self, tmp_path):
        """Filename heuristic for define.xml."""
        xml = """\
<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3">
  <Study OID="S1"><MetaDataVersion OID="M1" Name="Test"/></Study>
</ODM>
"""
        filepath = tmp_path / "define.xml"
        filepath.write_text(xml)
        assert _detect_format(filepath) == "cdisc_define"

    def test_detect_ct_from_filename(self, tmp_path):
        """Filename heuristic for terminology files."""
        xml = """\
<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3">
  <Study OID="S1"><MetaDataVersion OID="M1" Name="Test"/></Study>
</ODM>
"""
        filepath = tmp_path / "sdtm_terminology_2024.xml"
        filepath.write_text(xml)
        assert _detect_format(filepath) == "cdisc_ct"

    def test_detect_generic_odm_as_ct(self, tmp_path):
        """A generic ODM file without specific hints is detected as CT."""
        xml = """\
<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3">
  <Study OID="S1"><MetaDataVersion OID="M1" Name="Test"/></Study>
</ODM>
"""
        filepath = tmp_path / "study_data.xml"
        filepath.write_text(xml)
        assert _detect_format(filepath) == "cdisc_ct"


class TestCDISCImportMetadataIntegration:
    """Test import_metadata() with CDISC format routing."""

    @pytest.fixture
    def define_file(self, tmp_path):
        xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     FileType="Snapshot" FileOID="F1" CreationDateTime="2024-01-01">
  <Study OID="S1">
    <MetaDataVersion OID="M1" Name="Test">
      <ItemDef OID="IT.SUBJ" Name="SUBJID" DataType="text" Length="10">
        <Description><TranslatedText>Subject ID</TranslatedText></Description>
      </ItemDef>
      <ItemGroupDef OID="IG.DM" Name="DM" Domain="DM" Repeating="No">
        <ItemRef ItemOID="IT.SUBJ" Mandatory="Yes" Role="Identifier"/>
      </ItemGroupDef>
    </MetaDataVersion>
  </Study>
</ODM>
"""
        filepath = tmp_path / "define.xml"
        filepath.write_text(xml_content)
        return filepath

    def test_import_with_explicit_format(self, define_file):
        """import_metadata() with format='cdisc_define' routes correctly."""
        meta = import_metadata(define_file, format="cdisc_define")
        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "cdisc_define"

    def test_import_with_auto_detect(self, define_file):
        """import_metadata() auto-detects Define-XML from content."""
        meta = import_metadata(define_file)
        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "cdisc_define"

    def test_import_ct_with_format(self, tmp_path):
        """import_metadata() with format='cdisc_ct' routes correctly."""
        xml = """\
<?xml version="1.0"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:nciodm="http://ncicb.nci.nih.gov/xml/odm/EVS/CDISC">
  <Study OID="CT">
    <MetaDataVersion OID="M1" Name="Test CT">
      <CodeList OID="CL.YN" Name="YN" DataType="text">
        <EnumeratedItem CodedValue="Y"/>
        <EnumeratedItem CodedValue="N"/>
      </CodeList>
    </MetaDataVersion>
  </Study>
</ODM>
"""
        filepath = tmp_path / "ct.xml"
        filepath.write_text(xml)
        result = import_metadata(filepath, format="cdisc_ct")
        assert isinstance(result, MetadataPackage)
        assert "YN" in result


# =============================================================================
# CDISC SDTM Domain Templates & Validation
# =============================================================================


class TestSDTMDomainTemplates:
    """Tests for SDTM domain template definitions."""

    def test_list_sdtm_domains(self):
        """list_sdtm_domains returns all supported domains."""
        from pointblank.metadata._sdtm_templates import list_sdtm_domains

        domains = list_sdtm_domains()
        assert "DM" in domains
        assert "AE" in domains
        assert "LB" in domains
        assert "VS" in domains
        assert "EX" in domains
        assert "DS" in domains
        assert "MH" in domains
        assert "CM" in domains
        assert len(domains) == 8

    def test_get_dm_template(self):
        """DM domain template has correct structure."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        dm = get_sdtm_domain("DM")
        assert dm.domain == "DM"
        assert dm.label == "Demographics"
        assert dm.domain_class == "Special Purpose"
        assert dm.repeating is False
        assert "STUDYID" in dm.natural_keys
        assert "USUBJID" in dm.natural_keys

    def test_dm_required_variables(self):
        """DM has the required variables from IG 3.4."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        dm = get_sdtm_domain("DM")
        req_vars = dm.required_variables
        assert "STUDYID" in req_vars
        assert "DOMAIN" in req_vars
        assert "USUBJID" in req_vars
        assert "SUBJID" in req_vars
        assert "ARMCD" in req_vars
        assert "ARM" in req_vars
        assert "COUNTRY" in req_vars
        assert "SEX" in req_vars  # SEX is Req in DM

    def test_dm_identifier_variables(self):
        """DM identifiers are correctly classified."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        dm = get_sdtm_domain("DM")
        id_vars = dm.identifier_variables
        assert "STUDYID" in id_vars
        assert "DOMAIN" in id_vars
        assert "USUBJID" in id_vars

    def test_ae_template(self):
        """AE domain template has correct structure."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        ae = get_sdtm_domain("AE")
        assert ae.domain == "AE"
        assert ae.domain_class == "Events"
        assert ae.repeating is True
        assert "AETERM" in ae.required_variables
        assert "AESEQ" in ae.required_variables

    def test_lb_template(self):
        """LB domain template has correct structure."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        lb = get_sdtm_domain("LB")
        assert lb.domain == "LB"
        assert lb.domain_class == "Findings"
        assert lb.repeating is True
        assert "LBTESTCD" in lb.required_variables
        assert "LBTEST" in lb.required_variables

    def test_get_variable(self):
        """get_variable returns spec by name."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        dm = get_sdtm_domain("DM")
        sex_spec = dm.get_variable("SEX")
        assert sex_spec is not None
        assert sex_spec.label == "Sex"
        assert sex_spec.dtype == "Char"
        assert sex_spec.controlled_term == "SEX"
        assert sex_spec.max_length == 2

    def test_get_variable_not_found(self):
        """get_variable returns None for unknown variable."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        dm = get_sdtm_domain("DM")
        assert dm.get_variable("NONEXIST") is None

    def test_case_insensitive_lookup(self):
        """Domain lookup is case-insensitive."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        dm1 = get_sdtm_domain("dm")
        dm2 = get_sdtm_domain("DM")
        assert dm1.domain == dm2.domain

    def test_invalid_domain_raises(self):
        """Unknown domain raises KeyError."""
        from pointblank.metadata._sdtm_templates import get_sdtm_domain

        with pytest.raises(KeyError, match="not supported"):
            get_sdtm_domain("ZZ")


class TestValidateSDTMStructure:
    """Tests for structural validation against SDTM templates."""

    def test_valid_dm_structure(self):
        """A valid DM dataset passes structural validation."""
        import pandas as pd
        from pointblank.metadata._sdtm_templates import validate_sdtm_structure

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "DOMAIN": ["DM", "DM"],
                "USUBJID": ["S1-001", "S1-002"],
                "SUBJID": ["001", "002"],
                "SEX": ["M", "F"],
                "ARMCD": ["TRT", "PBO"],
                "ARM": ["Treatment", "Placebo"],
                "SITEID": ["SITE1", "SITE1"],
                "COUNTRY": ["USA", "USA"],
            }
        )
        result = validate_sdtm_structure(dm, domain="DM")
        assert result["valid"] is True
        assert result["missing_required"] == []
        assert result["domain_mismatch"] is False

    def test_missing_required_variable(self):
        """Missing required variable is detected."""
        import pandas as pd
        from pointblank.metadata._sdtm_templates import validate_sdtm_structure

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["DM"],
                "USUBJID": ["S1-001"],
                # SUBJID is missing (required)
                "SEX": ["M"],
                "ARMCD": ["TRT"],
                "ARM": ["Treatment"],
                "COUNTRY": ["USA"],
            }
        )
        result = validate_sdtm_structure(dm, domain="DM")
        assert result["valid"] is False
        assert "SUBJID" in result["missing_required"]

    def test_domain_value_mismatch(self):
        """Incorrect DOMAIN column value is detected."""
        import pandas as pd
        from pointblank.metadata._sdtm_templates import validate_sdtm_structure

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["AE"],  # Wrong!
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SEX": ["M"],
                "ARMCD": ["TRT"],
                "ARM": ["Treatment"],
                "SITEID": ["SITE1"],
                "COUNTRY": ["USA"],
            }
        )
        result = validate_sdtm_structure(dm, domain="DM")
        assert result["valid"] is False
        assert result["domain_mismatch"] is True

    def test_strict_mode_reports_expected(self):
        """Strict mode reports missing Expected variables."""
        import pandas as pd
        from pointblank.metadata._sdtm_templates import validate_sdtm_structure

        # Minimal DM with only required vars
        dm = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["DM"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SEX": ["M"],
                "ARMCD": ["TRT"],
                "ARM": ["Treatment"],
                "SITEID": ["SITE1"],
                "COUNTRY": ["USA"],
            }
        )
        result = validate_sdtm_structure(dm, domain="DM", strict=True)
        # AGE is Expected in DM
        assert "AGE" in result["missing_expected"]

    def test_strict_mode_reports_unknown(self):
        """Strict mode reports unknown (non-template) variables."""
        import pandas as pd
        from pointblank.metadata._sdtm_templates import validate_sdtm_structure

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["DM"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SEX": ["M"],
                "ARMCD": ["TRT"],
                "ARM": ["Treatment"],
                "SITEID": ["SITE1"],
                "COUNTRY": ["USA"],
                "CUSTOM_VAR": ["X"],  # Not in template
            }
        )
        result = validate_sdtm_structure(dm, domain="DM", strict=True)
        assert "CUSTOM_VAR" in result["unknown_variables"]


class TestSDTMToMetadata:
    """Tests for converting SDTM templates to MetadataImport."""

    def test_basic_conversion(self):
        """sdtm_to_metadata returns a valid MetadataImport."""
        from pointblank.metadata._sdtm_validate import sdtm_to_metadata

        meta = sdtm_to_metadata("DM")
        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "cdisc_sdtm"
        assert meta.domain == "DM"
        assert meta.dataset_name == "DM"
        assert meta.dataset_label == "Demographics"
        assert len(meta.variables) > 0

    def test_variable_types_mapped(self):
        """SDTM Char/Num types are mapped to String/Float64."""
        from pointblank.metadata._sdtm_validate import sdtm_to_metadata

        meta = sdtm_to_metadata("DM")
        studyid = meta.get_variable("STUDYID")
        assert studyid.dtype == "String"
        age = meta.get_variable("AGE")
        assert age.dtype == "Float64"

    def test_required_flag_preserved(self):
        """Required variables have required=True."""
        from pointblank.metadata._sdtm_validate import sdtm_to_metadata

        meta = sdtm_to_metadata("AE")
        aeterm = meta.get_variable("AETERM")
        assert aeterm.required is True
        aesev = meta.get_variable("AESEV")
        assert aesev.required is False

    def test_to_schema(self):
        """to_schema() works on SDTM-generated metadata."""
        from pointblank.metadata._sdtm_validate import sdtm_to_metadata

        meta = sdtm_to_metadata("AE")
        schema = meta.to_schema()
        col_dict = dict(schema.columns)
        assert "AETERM" in col_dict
        assert col_dict["AESEQ"] == "Float64"  # Num → Float64

    def test_study_id_passed_through(self):
        """study_id parameter is preserved."""
        from pointblank.metadata._sdtm_validate import sdtm_to_metadata

        meta = sdtm_to_metadata("DM", study_id="ABC-123")
        assert meta.study_id == "ABC-123"

    def test_import_metadata_sdtm_format(self, tmp_path):
        """import_metadata with format='cdisc_sdtm' uses template."""
        # Need a dummy file for the path
        dummy = tmp_path / "dm.xpt"
        dummy.write_bytes(b"")
        meta = import_metadata(dummy, format="cdisc_sdtm", domain="DM")
        assert isinstance(meta, MetadataImport)
        assert meta.domain == "DM"
        assert meta.source_format == "cdisc_sdtm"


class TestValidateSDTM:
    """Tests for the validate_sdtm() validation generator."""

    def test_basic_validation(self):
        """validate_sdtm generates a Validate object."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "DOMAIN": ["DM", "DM"],
                "USUBJID": ["S1-001", "S1-002"],
                "SUBJID": ["001", "002"],
                "SEX": ["M", "F"],
                "ARMCD": ["TRT", "PBO"],
                "ARM": ["Treatment", "Placebo"],
                "SITEID": ["SITE1", "SITE1"],
                "COUNTRY": ["USA", "USA"],
            }
        )
        from pointblank.validate import Validate

        validation = validate_sdtm(dm, domain="DM")
        assert isinstance(validation, Validate)
        # Should have validation steps
        assert len(validation.validation_info) > 0

    def test_required_vars_checked(self):
        """Required variables get col_vals_not_null checks."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["DM"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SEX": ["M"],
                "ARMCD": ["TRT"],
                "ARM": ["Treatment"],
                "SITEID": ["SITE1"],
                "COUNTRY": ["USA"],
            }
        )
        validation = validate_sdtm(dm, domain="DM")
        # Check that not-null assertions are generated for required vars
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_not_null" in assertion_types

    def test_domain_value_checked(self):
        """DOMAIN column is checked against expected value."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        ae = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["AE"],
                "USUBJID": ["S1-001"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AEDECOD": ["HEADACHE"],
            }
        )
        validation = validate_sdtm(ae, domain="AE")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_in_set" in assertion_types

    def test_seq_positivity_checked(self):
        """Sequence number (--SEQ) is checked for positivity."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        ae = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["AE"],
                "USUBJID": ["S1-001"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AEDECOD": ["HEADACHE"],
            }
        )
        validation = validate_sdtm(ae, domain="AE")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_gt" in assertion_types

    def test_iso8601_date_checked(self):
        """--DTC variables get ISO 8601 regex checks."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        ae = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["AE"],
                "USUBJID": ["S1-001"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AEDECOD": ["HEADACHE"],
                "AESTDTC": ["2024-06-15"],
                "AEENDTC": ["2024-06-20"],
            }
        )
        validation = validate_sdtm(ae, domain="AE")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_regex" in assertion_types

    def test_iso8601_partial_dates_pass(self):
        """Partial ISO 8601 dates should pass the regex check."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        ae = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1", "S1", "S1"],
                "DOMAIN": ["AE", "AE", "AE", "AE"],
                "USUBJID": ["S1-001", "S1-001", "S1-001", "S1-001"],
                "AESEQ": [1, 2, 3, 4],
                "AETERM": ["Headache", "Nausea", "Rash", "Fatigue"],
                "AEDECOD": ["HEADACHE", "NAUSEA", "RASH", "FATIGUE"],
                "AESTDTC": ["2024", "2024-06", "2024-06-15", "2024-06-15T10:30:00"],
            }
        )
        validation = validate_sdtm(ae, domain="AE").interrogate()
        # All validation results should pass (no failing rows)
        # The partial dates are valid ISO 8601 per CDISC
        for info in validation.validation_info:
            if info.assertion_type == "col_vals_regex":
                assert info.n_failed == 0

    def test_no_dates_check_disabled(self):
        """check_dates=False skips ISO 8601 validation."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        ae = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["AE"],
                "USUBJID": ["S1-001"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AEDECOD": ["HEADACHE"],
                "AESTDTC": ["NOT-A-DATE"],
            }
        )
        validation = validate_sdtm(ae, domain="AE", check_dates=False)
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_regex" not in assertion_types

    def test_custom_label(self):
        """Custom label is applied to the Validate object."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        dm = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "DOMAIN": ["DM"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SEX": ["M"],
                "ARMCD": ["TRT"],
                "ARM": ["Treatment"],
                "SITEID": ["SITE1"],
                "COUNTRY": ["USA"],
            }
        )
        validation = validate_sdtm(dm, domain="DM", label="My Custom Label")
        assert validation.label == "My Custom Label"

    def test_interrogate_passes_valid_data(self):
        """Full interrogation passes with valid SDTM data."""
        import pandas as pd
        from pointblank.metadata._sdtm_validate import validate_sdtm

        dm = pd.DataFrame(
            {
                "STUDYID": ["STUDY1", "STUDY1"],
                "DOMAIN": ["DM", "DM"],
                "USUBJID": ["STUDY1-001", "STUDY1-002"],
                "SUBJID": ["001", "002"],
                "RFSTDTC": ["2024-01-15", "2024-01-20"],
                "SEX": ["M", "F"],
                "AGE": [45.0, 38.0],
                "ARMCD": ["TRT", "PBO"],
                "ARM": ["Treatment", "Placebo"],
                "SITEID": ["SITE01", "SITE01"],
                "COUNTRY": ["USA", "USA"],
            }
        )
        validation = validate_sdtm(dm, domain="DM").interrogate()
        # All checks should pass
        for info in validation.validation_info:
            if info.assertion_type in ("col_vals_not_null", "col_vals_in_set"):
                assert info.n_failed == 0


# =============================================================================
# CDISC ADaM Templates & Validation
# =============================================================================


class TestADaMDatasetTemplates:
    """Tests for ADaM dataset template definitions."""

    def test_list_adam_datasets(self):
        """list_adam_datasets returns all supported datasets."""
        from pointblank.metadata._adam_templates import list_adam_datasets

        datasets = list_adam_datasets()
        assert "ADSL" in datasets
        assert "BDS" in datasets
        assert "ADAE" in datasets
        assert "ADTTE" in datasets
        assert len(datasets) == 4

    def test_get_adsl_template(self):
        """ADSL template has correct structure."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adsl = get_adam_dataset("ADSL")
        assert adsl.name == "ADSL"
        assert adsl.dataset_class == "ADSL"
        assert "STUDYID" in adsl.natural_keys
        assert "USUBJID" in adsl.natural_keys

    def test_adsl_required_variables(self):
        """ADSL has the correct required variables."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adsl = get_adam_dataset("ADSL")
        req = adsl.required_variables
        assert "STUDYID" in req
        assert "USUBJID" in req
        assert "SUBJID" in req
        assert "SITEID" in req
        assert "TRT01P" in req

    def test_adsl_population_flags(self):
        """ADSL template has population flag variables."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adsl = get_adam_dataset("ADSL")
        flags = adsl.population_flags
        assert "SAFFL" in flags
        assert "ITTFL" in flags
        assert "EFFFL" in flags
        assert "RANDFL" in flags

    def test_bds_template(self):
        """BDS template has correct structure."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        bds = get_adam_dataset("BDS")
        assert bds.dataset_class == "BDS"
        assert "PARAMCD" in bds.required_variables
        assert "PARAM" in bds.required_variables
        assert "AVAL" in bds.required_variables

    def test_adae_template(self):
        """ADAE template has correct structure."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adae = get_adam_dataset("ADAE")
        assert adae.dataset_class == "ADAE"
        assert "AETERM" in adae.required_variables
        assert "AEDECOD" in adae.required_variables
        assert "AESEQ" in adae.required_variables

    def test_adtte_template(self):
        """ADTTE template has correct structure."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adtte = get_adam_dataset("ADTTE")
        assert adtte.dataset_class == "ADTTE"
        assert "CNSR" in adtte.required_variables
        assert "AVAL" in adtte.required_variables
        assert "STARTDT" in adtte.required_variables
        assert "PARAMCD" in adtte.required_variables

    def test_case_insensitive_lookup(self):
        """Dataset lookup is case-insensitive."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adsl1 = get_adam_dataset("adsl")
        adsl2 = get_adam_dataset("ADSL")
        assert adsl1.name == adsl2.name

    def test_invalid_dataset_raises(self):
        """Unknown dataset raises KeyError."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        with pytest.raises(KeyError, match="not supported"):
            get_adam_dataset("INVALID")

    def test_get_variable(self):
        """get_variable returns spec by name."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adsl = get_adam_dataset("ADSL")
        saffl = adsl.get_variable("SAFFL")
        assert saffl is not None
        assert saffl.is_population_flag is True
        assert saffl.controlled_term == "NY"
        assert saffl.max_length == 1

    def test_conditional_variables(self):
        """conditional_variables returns Cond-core vars."""
        from pointblank.metadata._adam_templates import get_adam_dataset

        adsl = get_adam_dataset("ADSL")
        cond = adsl.conditional_variables
        assert "SAFFL" in cond  # Population flags are conditional
        assert "AGE" in cond


class TestValidateADaMStructure:
    """Tests for structural validation against ADaM templates."""

    def test_valid_adsl_structure(self):
        """A valid ADSL dataset passes structural validation."""
        import pandas as pd
        from pointblank.metadata._adam_templates import validate_adam_structure

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-002"],
                "SUBJID": ["001", "002"],
                "SITEID": ["SITE1", "SITE1"],
                "TRT01P": ["Drug A", "Placebo"],
                "SAFFL": ["Y", "Y"],
                "ITTFL": ["Y", "Y"],
                "AGE": [45, 38],
                "SEX": ["M", "F"],
            }
        )
        result = validate_adam_structure(adsl, dataset="ADSL")
        assert result["valid"] is True
        assert result["missing_required"] == []
        assert "SAFFL" in result["population_flags_found"]
        assert "ITTFL" in result["population_flags_found"]

    def test_missing_required_variable(self):
        """Missing required variable is detected."""
        import pandas as pd
        from pointblank.metadata._adam_templates import validate_adam_structure

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SITEID": ["SITE1"],
                # TRT01P is missing (required)
                "SAFFL": ["Y"],
            }
        )
        result = validate_adam_structure(adsl, dataset="ADSL")
        assert result["valid"] is False
        assert "TRT01P" in result["missing_required"]

    def test_missing_population_flag_warning(self):
        """ADSL without any population flag generates an issue."""
        import pandas as pd
        from pointblank.metadata._adam_templates import validate_adam_structure

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SITEID": ["SITE1"],
                "TRT01P": ["Drug A"],
            }
        )
        result = validate_adam_structure(adsl, dataset="ADSL")
        assert any("population flag" in issue for issue in result["issues"])

    def test_strict_mode_reports_conditional(self):
        """Strict mode reports missing conditional variables."""
        import pandas as pd
        from pointblank.metadata._adam_templates import validate_adam_structure

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SITEID": ["SITE1"],
                "TRT01P": ["Drug A"],
                "SAFFL": ["Y"],
            }
        )
        result = validate_adam_structure(adsl, dataset="ADSL", strict=True)
        # AGE is conditionally required
        assert "AGE" in result["missing_conditional"]

    def test_bds_structure_valid(self):
        """Valid BDS dataset passes."""
        import pandas as pd
        from pointblank.metadata._adam_templates import validate_adam_structure

        advs = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-001"],
                "PARAMCD": ["SYSBP", "DIABP"],
                "PARAM": ["Systolic Blood Pressure", "Diastolic Blood Pressure"],
                "AVAL": [120.0, 80.0],
            }
        )
        result = validate_adam_structure(advs, dataset="BDS")
        assert result["valid"] is True


class TestADaMToMetadata:
    """Tests for converting ADaM templates to MetadataImport."""

    def test_basic_conversion(self):
        """adam_to_metadata returns a valid MetadataImport."""
        from pointblank.metadata._adam_validate import adam_to_metadata

        meta = adam_to_metadata("ADSL")
        assert isinstance(meta, MetadataImport)
        assert meta.source_format == "cdisc_adam"
        assert meta.domain == "ADSL"
        assert meta.dataset_name == "ADSL"
        assert len(meta.variables) > 0

    def test_variable_types_mapped(self):
        """ADaM Char/Num types are mapped to String/Float64."""
        from pointblank.metadata._adam_validate import adam_to_metadata

        meta = adam_to_metadata("ADSL")
        studyid = meta.get_variable("STUDYID")
        assert studyid.dtype == "String"
        age = meta.get_variable("AGE")
        assert age.dtype == "Float64"

    def test_to_schema(self):
        """to_schema() works on ADaM-generated metadata."""
        from pointblank.metadata._adam_validate import adam_to_metadata

        meta = adam_to_metadata("BDS")
        schema = meta.to_schema()
        col_dict = dict(schema.columns)
        assert "PARAMCD" in col_dict
        assert col_dict["AVAL"] == "Float64"

    def test_import_metadata_adam_format(self, tmp_path):
        """import_metadata with format='cdisc_adam' uses template."""
        dummy = tmp_path / "adsl.xpt"
        dummy.write_bytes(b"")
        meta = import_metadata(dummy, format="cdisc_adam", dataset="ADSL")
        assert isinstance(meta, MetadataImport)
        assert meta.domain == "ADSL"
        assert meta.source_format == "cdisc_adam"


class TestValidateADaM:
    """Tests for the validate_adam() validation generator."""

    def test_basic_adsl_validation(self):
        """validate_adam generates a Validate object for ADSL."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam
        from pointblank.validate import Validate

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-002"],
                "SUBJID": ["001", "002"],
                "SITEID": ["SITE1", "SITE1"],
                "TRT01P": ["Drug A", "Placebo"],
                "SAFFL": ["Y", "Y"],
            }
        )
        validation = validate_adam(adsl, dataset="ADSL")
        assert isinstance(validation, Validate)
        assert len(validation.validation_info) > 0

    def test_population_flags_checked(self):
        """Population flag columns get Y/N value checks."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-002"],
                "SUBJID": ["001", "002"],
                "SITEID": ["SITE1", "SITE1"],
                "TRT01P": ["Drug A", "Placebo"],
                "SAFFL": ["Y", "Y"],
                "ITTFL": ["Y", "N"],
            }
        )
        validation = validate_adam(adsl, dataset="ADSL")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_in_set" in assertion_types

    def test_adsl_trt01p_not_null(self):
        """ADSL validates TRT01P is non-null."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SITEID": ["SITE1"],
                "TRT01P": ["Drug A"],
                "SAFFL": ["Y"],
            }
        )
        validation = validate_adam(adsl, dataset="ADSL")
        # TRT01P not_null should be there (both as required and as ADSL-specific)
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_not_null" in assertion_types

    def test_adtte_cnsr_values(self):
        """ADTTE validates CNSR is 0 or 1."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adtte = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-002"],
                "PARAMCD": ["OS", "OS"],
                "PARAM": ["Overall Survival", "Overall Survival"],
                "AVAL": [120.0, 85.0],
                "STARTDT": [19724.0, 19724.0],
                "ADT": [19844.0, 19809.0],
                "CNSR": [0, 1],
            }
        )
        validation = validate_adam(adtte, dataset="ADTTE")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        # CNSR should be checked with in_set
        assert "col_vals_in_set" in assertion_types
        # AVAL should be >= 0
        assert "col_vals_ge" in assertion_types

    def test_adtte_interrogate_valid(self):
        """ADTTE valid data passes interrogation."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adtte = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-002"],
                "PARAMCD": ["OS", "OS"],
                "PARAM": ["Overall Survival", "Overall Survival"],
                "AVAL": [120.0, 85.0],
                "STARTDT": [19724.0, 19724.0],
                "ADT": [19844.0, 19809.0],
                "CNSR": [0, 1],
            }
        )
        validation = validate_adam(adtte, dataset="ADTTE").interrogate()
        for info in validation.validation_info:
            assert info.n_failed == 0

    def test_adae_trtemfl_checked(self):
        """ADAE validates TRTEMFL is Y or N."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adae = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AEDECOD": ["HEADACHE"],
                "TRTEMFL": ["Y"],
            }
        )
        validation = validate_adam(adae, dataset="ADAE")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_in_set" in assertion_types

    def test_adae_aeseq_positive(self):
        """ADAE validates AESEQ > 0."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adae = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AEDECOD": ["HEADACHE"],
            }
        )
        validation = validate_adam(adae, dataset="ADAE")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_gt" in assertion_types

    def test_bds_paramcd_length_checked(self):
        """BDS validates PARAMCD length <= 8."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        advs = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "PARAMCD": ["SYSBP"],
                "PARAM": ["Systolic Blood Pressure"],
                "AVAL": [120.0],
            }
        )
        validation = validate_adam(advs, dataset="BDS")
        assertion_types = [v.assertion_type for v in validation.validation_info]
        assert "col_vals_expr" in assertion_types

    def test_custom_label(self):
        """Custom label is applied."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1"],
                "USUBJID": ["S1-001"],
                "SUBJID": ["001"],
                "SITEID": ["SITE1"],
                "TRT01P": ["Drug A"],
                "SAFFL": ["Y"],
            }
        )
        validation = validate_adam(adsl, dataset="ADSL", label="My Label")
        assert validation.label == "My Label"

    def test_population_flags_invalid_values_fail(self):
        """Population flags with invalid values fail validation."""
        import pandas as pd
        from pointblank.metadata._adam_validate import validate_adam

        adsl = pd.DataFrame(
            {
                "STUDYID": ["S1", "S1"],
                "USUBJID": ["S1-001", "S1-002"],
                "SUBJID": ["001", "002"],
                "SITEID": ["SITE1", "SITE1"],
                "TRT01P": ["Drug A", "Placebo"],
                "SAFFL": ["Y", "INVALID"],  # Invalid value
            }
        )
        validation = validate_adam(adsl, dataset="ADSL").interrogate()
        # Find the SAFFL in_set check
        saffl_checks = [
            v
            for v in validation.validation_info
            if v.assertion_type == "col_vals_in_set" and "SAFFL" in str(v.column)
        ]
        assert len(saffl_checks) > 0
        assert saffl_checks[0].n_failed > 0


class TestLoadMetadataExample:
    """Tests for the `load_metadata_example()` bundled-file accessor."""

    def test_returns_existing_path_for_each_example(self):
        """Every advertised example resolves to a file that exists on disk."""
        from pointblank.metadata._import import _METADATA_EXAMPLES, load_metadata_example

        for name in _METADATA_EXAMPLES:
            path = load_metadata_example(name)
            assert isinstance(path, Path)
            assert path.exists()
            assert path.name == name

    def test_invalid_name_raises_with_available_options(self):
        """An unknown example name raises ValueError listing valid options."""
        from pointblank.metadata._import import load_metadata_example

        with pytest.raises(ValueError, match="is not valid"):
            load_metadata_example("does_not_exist.xml")

    def test_define_example_imports(self):
        """The bundled Define-XML example imports into a usable MetadataPackage."""
        from pointblank.metadata._import import import_metadata, load_metadata_example

        package = import_metadata(load_metadata_example("define.xml"), format="cdisc_define")
        assert isinstance(package, MetadataPackage)
        assert "DM" in package.keys()

    def test_exposed_at_top_level(self):
        """`load_metadata_example` is exported from the top-level package."""
        import pointblank as pb

        assert hasattr(pb, "load_metadata_example")
