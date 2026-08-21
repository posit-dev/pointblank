"""Tests for MCP tools that lack dedicated coverage."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from pointblank.mcp.server import mcp


# ── MCP Prompt function tests ────────────────────────────────────────────────


def test_prompt_load_dataframe_returns_tuple():
    from pointblank.mcp._prompts import prompt_load_dataframe

    result = prompt_load_dataframe()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_prompt_load_dataframe_custom_args():
    from pointblank.mcp._prompts import prompt_load_dataframe

    result = prompt_load_dataframe(input_path="data.csv", df_id="my_df")
    assert isinstance(result, tuple)


def test_prompt_create_validator_returns_tuple():
    from pointblank.mcp._prompts import prompt_create_validator

    result = prompt_create_validator()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_prompt_create_validator_with_thresholds():
    from pointblank.mcp._prompts import prompt_create_validator

    result = prompt_create_validator(thresholds_dict_example={"warning": 0.05})
    assert isinstance(result, tuple)


def test_prompt_add_validation_step_example_returns_tuple():
    from pointblank.mcp._prompts import prompt_add_validation_step_example

    result = prompt_add_validation_step_example()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_prompt_get_validation_step_output_returns_tuple():
    from pointblank.mcp._prompts import prompt_get_validation_step_output

    result = prompt_get_validation_step_output()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_prompt_interrogate_validator_returns_tuple():
    from pointblank.mcp._prompts import prompt_interrogate_validator

    result = prompt_interrogate_validator()
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.fixture(scope="module")
def mcp_server():
    """Provides the FastMCP server instance."""
    return mcp


@pytest.fixture
def sample_data():
    """DataFrame with variety of types and some nulls."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", None, "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [85.5, None, 78.5, 88.0, 95.0],
            "status": ["active", "active", "inactive", "active", "inactive"],
        }
    )


@pytest.fixture
def temp_csv_file(sample_data):
    """Creates a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# Server Health & Info Tools
# =============================================================================


@pytest.mark.asyncio
async def test_server_health_check(mcp_server):
    """Test that server_health_check returns well-formed health info."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("server_health_check")

        assert not result.is_error
        data = result.data
        assert data["server_status"] == "healthy"
        assert "system_info" in data
        assert "backend_status" in data
        assert data["backend_status"]["pandas_available"] is True
        assert "resource_usage" in data
        assert "capabilities" in data
        assert "validation_types_count" in data["capabilities"]
        assert "templates_available" in data["capabilities"]


@pytest.mark.asyncio
async def test_get_pointblank_api_reference_validation_methods(mcp_server):
    """Test API reference for validation methods."""
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_pointblank_api_reference", {"category": "validation_methods"}
        )
        assert not result.is_error
        text = result.data
        assert "col_vals_gt" in text
        assert "col_vals_between" in text
        assert "rows_distinct" in text


@pytest.mark.asyncio
async def test_get_pointblank_api_reference_thresholds(mcp_server):
    """Test API reference for thresholds."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("get_pointblank_api_reference", {"category": "thresholds"})
        assert not result.is_error
        text = result.data
        assert "warning" in text
        assert "error" in text
        assert "critical" in text


@pytest.mark.asyncio
async def test_get_pointblank_api_reference_all(mcp_server):
    """Test API reference for all categories."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("get_pointblank_api_reference", {"category": "all"})
        assert not result.is_error
        text = result.data
        assert "col_vals_gt" in text
        assert "warning" in text
        assert "Common Patterns" in text


@pytest.mark.asyncio
async def test_get_pointblank_api_reference_unknown_category(mcp_server):
    """Test API reference with unknown category."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("get_pointblank_api_reference", {"category": "nonexistent"})
        assert not result.is_error
        text = result.data
        assert "Unknown category" in text


@pytest.mark.asyncio
async def test_list_available_backends(mcp_server):
    """Test list_available_backends tool."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("list_available_backends")
        assert not result.is_error
        data = result.data
        assert "available_backends" in data
        assert "pandas" in data["available_backends"]
        assert "backend_details" in data
        assert data["backend_details"]["pandas"]["available"] is True


# =============================================================================
# Table Visualization Tools
# =============================================================================


@pytest.mark.asyncio
async def test_preview_table(mcp_server, temp_csv_file):
    """Test preview_table generates HTML output."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool(
            "preview_table", {"dataframe_id": df_id, "n_head": 3, "n_tail": 2}
        )
        assert not result.is_error
        text = result.data
        assert "preview generated successfully" in text.lower()
        assert "5" in text  # total rows


@pytest.mark.asyncio
async def test_preview_table_not_found(mcp_server):
    """Test preview_table raises error for missing DataFrame."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("preview_table", {"dataframe_id": "nonexistent_df"})


@pytest.mark.asyncio
async def test_missing_values_table(mcp_server, temp_csv_file):
    """Test missing_values_table generates analysis."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("missing_values_table", {"dataframe_id": df_id})
        assert not result.is_error
        text = result.data
        assert "missing values analysis generated" in text.lower()


@pytest.mark.asyncio
async def test_missing_values_table_not_found(mcp_server):
    """Test missing_values_table raises error for missing DataFrame."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("missing_values_table", {"dataframe_id": "nonexistent_df"})


@pytest.mark.asyncio
async def test_column_summary_table(mcp_server, temp_csv_file):
    """Test column_summary_table generates summary."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("column_summary_table", {"dataframe_id": df_id})
        assert not result.is_error
        text = result.data
        assert "column summary table generated" in text.lower()


@pytest.mark.asyncio
async def test_column_summary_table_not_found(mcp_server):
    """Test column_summary_table raises error for missing DataFrame."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("column_summary_table", {"dataframe_id": "nonexistent_df"})


# =============================================================================
# Validation Assistant Tool
# =============================================================================


@pytest.mark.asyncio
async def test_validation_assistant(mcp_server, temp_csv_file):
    """Test validation_assistant generates data-aware suggestions."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool(
            "validation_assistant", {"dataframe_id": df_id, "validation_goal": "general"}
        )
        assert not result.is_error
        text = result.data

        # Should contain column analysis
        assert "Column Analysis" in text
        assert "id" in text
        assert "name" in text
        assert "age" in text

        # Should contain suggested validation code
        assert "import pointblank as pb" in text
        assert "pb.Validate(data)" in text

        # Should detect nulls in name/score columns and suggest not_null for non-null columns
        assert "col_vals_not_null" in text

        # Should detect numeric range for age
        assert "age" in text


@pytest.mark.asyncio
async def test_validation_assistant_not_found(mcp_server):
    """Test validation_assistant raises error for missing DataFrame."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("validation_assistant", {"dataframe_id": "nonexistent_df"})


# =============================================================================
# Data Analysis Tools
# =============================================================================


@pytest.mark.asyncio
async def test_profile_dataframe(mcp_server, temp_csv_file):
    """Test profile_dataframe returns profiling results."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("profile_dataframe", {"df_id": df_id, "sample_size": 0})
        assert not result.is_error
        data = result.data
        assert data["status"] == "success"
        assert data["df_id"] == df_id
        assert "profile" in data


@pytest.mark.asyncio
async def test_profile_dataframe_not_found(mcp_server):
    """Test profile_dataframe raises for missing DataFrame."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("profile_dataframe", {"df_id": "nonexistent_df"})


@pytest.mark.asyncio
async def test_profile_dataframe_with_sampling(mcp_server, temp_csv_file):
    """Test profile_dataframe with sampling enabled."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("profile_dataframe", {"df_id": df_id, "sample_size": 3})
        assert not result.is_error
        data = result.data
        assert data["status"] == "success"


# =============================================================================
# Validation Template Tool
# =============================================================================


@pytest.mark.asyncio
async def test_apply_validation_template(mcp_server, temp_csv_file):
    """Test applying a basic_quality template."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        validator_result = await client.call_tool("create_validator", {"df_id": df_id})
        validator_id = validator_result.data.validator_id

        result = await client.call_tool(
            "apply_validation_template",
            {
                "validator_id": validator_id,
                "template_name": "basic_quality",
                "column_mapping": {
                    "id_column": "id",
                    "required_column": "name",
                },
            },
        )
        assert not result.is_error
        data = result.data
        assert data["template_name"] == "basic_quality"
        assert data["total_validations"] > 0


@pytest.mark.asyncio
async def test_apply_validation_template_unknown(mcp_server, temp_csv_file):
    """Test applying an unknown template raises error."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        validator_result = await client.call_tool("create_validator", {"df_id": df_id})
        validator_id = validator_result.data.validator_id

        with pytest.raises(ToolError):
            await client.call_tool(
                "apply_validation_template",
                {
                    "validator_id": validator_id,
                    "template_name": "nonexistent_template",
                    "column_mapping": {},
                },
            )


# =============================================================================
# Delete Tools
# =============================================================================


@pytest.mark.asyncio
async def test_delete_dataframe(mcp_server, temp_csv_file):
    """Test deleting a loaded DataFrame."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("delete_dataframe", {"df_id": df_id})
        assert not result.is_error
        assert result.data["status"] == "success"

        # Verify it's gone
        list_result = await client.call_tool("list_loaded_dataframes")
        assert df_id not in list_result.data["loaded_dataframes"]


@pytest.mark.asyncio
async def test_delete_dataframe_not_found(mcp_server):
    """Test deleting a nonexistent DataFrame raises error."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("delete_dataframe", {"df_id": "nonexistent_df"})


@pytest.mark.asyncio
async def test_delete_validator(mcp_server, temp_csv_file):
    """Test deleting a validator."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        validator_result = await client.call_tool("create_validator", {"df_id": df_id})
        validator_id = validator_result.data.validator_id

        result = await client.call_tool("delete_validator", {"validator_id": validator_id})
        assert not result.is_error
        assert result.data["status"] == "success"

        # Verify it's gone
        list_result = await client.call_tool("list_active_validators")
        assert validator_id not in list_result.data["active_validators"]


@pytest.mark.asyncio
async def test_delete_validator_not_found(mcp_server):
    """Test deleting a nonexistent validator raises error."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("delete_validator", {"validator_id": "nonexistent_val"})


# =============================================================================
# ID Validation
# =============================================================================


@pytest.mark.asyncio
async def test_invalid_resource_id_rejected(mcp_server, temp_csv_file):
    """Test that resource IDs with special characters are rejected."""
    async with Client(mcp_server) as client:
        # Load with a valid ID first
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        assert not load_result.is_error

        # Try to load with an invalid ID containing special chars
        with pytest.raises(ToolError):
            await client.call_tool(
                "load_dataframe", {"input_path": temp_csv_file, "df_id": "../../etc/passwd"}
            )

        # Try with spaces
        with pytest.raises(ToolError):
            await client.call_tool(
                "load_dataframe", {"input_path": temp_csv_file, "df_id": "my data frame"}
            )


@pytest.mark.asyncio
async def test_valid_resource_ids_accepted(mcp_server, temp_csv_file):
    """Test that valid resource IDs with allowed characters work."""
    async with Client(mcp_server) as client:
        # Underscore
        result = await client.call_tool(
            "load_dataframe", {"input_path": temp_csv_file, "df_id": "my_data_frame"}
        )
        assert not result.is_error

        # Hyphen
        result = await client.call_tool(
            "load_dataframe", {"input_path": temp_csv_file, "df_id": "my-data-frame"}
        )
        assert not result.is_error

        # Alphanumeric
        result = await client.call_tool(
            "load_dataframe", {"input_path": temp_csv_file, "df_id": "df123ABC"}
        )
        assert not result.is_error


# ── Direct unit tests for _utils.py ────────────────────────────────────────────


class TestValidateResourceId:
    def test_valid_id_returned_stripped(self):
        from pointblank.mcp._utils import validate_resource_id

        assert validate_resource_id("  my_id  ") == "my_id"

    def test_empty_id_raises(self):
        from pointblank.mcp._utils import validate_resource_id

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_resource_id("")

    def test_whitespace_only_raises(self):
        from pointblank.mcp._utils import validate_resource_id

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_resource_id("   ")

    def test_invalid_chars_raises(self):
        from pointblank.mcp._utils import validate_resource_id

        with pytest.raises(ValueError, match="Invalid"):
            validate_resource_id("bad id!")

    def test_custom_resource_type_in_error(self):
        from pointblank.mcp._utils import validate_resource_id

        with pytest.raises(ValueError, match="validator"):
            validate_resource_id("", resource_type="validator")


class TestValidateInputPath:
    def test_traversal_raises(self):
        from pointblank.mcp._utils import validate_input_path

        with pytest.raises(ValueError, match="traversal"):
            validate_input_path("../some/file.csv")

    def test_invalid_extension_raises(self, tmp_path):
        from pointblank.mcp._utils import validate_input_path

        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not allowed"):
            validate_input_path(str(f))

    def test_file_not_found_raises(self, tmp_path):
        from pointblank.mcp._utils import validate_input_path

        with pytest.raises(FileNotFoundError):
            validate_input_path(str(tmp_path / "nonexistent.csv"))

    def test_valid_csv_returns_path(self, tmp_path):
        from pointblank.mcp._utils import validate_input_path

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        result = validate_input_path(str(f))
        assert result == f.resolve()


class TestValidateOutputPath:
    def test_traversal_raises(self):
        from pointblank.mcp._utils import validate_output_path

        with pytest.raises(ValueError, match="traversal"):
            validate_output_path("../out/file.html", {".html"})

    def test_invalid_extension_raises(self, tmp_path):
        from pointblank.mcp._utils import validate_output_path

        with pytest.raises(ValueError, match="not allowed"):
            validate_output_path(str(tmp_path / "out.txt"), {".html"})

    def test_system_dir_raises(self):
        from pointblank.mcp._utils import validate_output_path

        with pytest.raises(ValueError, match="system directory"):
            validate_output_path("/usr/something.html", {".html"})

    def test_valid_output_path_creates_parent(self, tmp_path):
        from pointblank.mcp._utils import validate_output_path

        out = tmp_path / "subdir" / "out.html"
        result = validate_output_path(str(out), {".html"})
        assert result.parent.exists()


class TestSaveDataframeToCsv:
    def test_save_pandas_df(self, tmp_path):
        from pointblank.mcp._utils import save_dataframe_to_csv
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = tmp_path / "out.csv"
        save_dataframe_to_csv(df, out)
        assert out.exists()
        content = out.read_text()
        assert "a" in content

    def test_save_polars_df(self, tmp_path):
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not available")
        from pointblank.mcp._utils import save_dataframe_to_csv

        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = tmp_path / "out.csv"
        save_dataframe_to_csv(df, out)
        assert out.exists()


class TestOpenBrowserConditionally:
    def test_testing_mode_suppresses_browser(self):
        from pointblank.mcp._utils import open_browser_conditionally
        from pointblank.mcp._config import TESTING_MODE

        assert TESTING_MODE is True
        open_browser_conditionally("https://example.com")


class TestSaveHtmlAndOpen:
    def test_testing_mode_returns_message_without_file(self):
        from pointblank.mcp._utils import save_html_and_open

        result = save_html_and_open("<p>hello</p>", "Test Title", "test_prefix")
        assert "HTML generated" in result
        assert "testing" in result.lower()


class TestGenerateValidationReportHtml:
    def test_testing_mode_returns_path_string(self):
        import pandas as pd
        import pointblank as pb
        from pointblank.mcp._utils import generate_validation_report_html

        df = pd.DataFrame({"a": [1, 2, 3]})
        v = pb.Validate(df).col_vals_not_null(columns="a").interrogate()
        result = generate_validation_report_html(v, "test_validator")
        assert isinstance(result, str)
        assert "pointblank_validation_report" in result


class TestCleanForJsonSerialization:
    def test_nan_becomes_none(self):
        import math
        from pointblank.mcp._utils import clean_for_json_serialization

        result = clean_for_json_serialization(float("nan"))
        assert result is None

    def test_inf_becomes_string(self):
        from pointblank.mcp._utils import clean_for_json_serialization

        result = clean_for_json_serialization(float("inf"))
        assert result == "inf"

    def test_normal_float_unchanged(self):
        from pointblank.mcp._utils import clean_for_json_serialization

        assert clean_for_json_serialization(3.14) == 3.14

    def test_dict_cleaned_recursively(self):
        import math
        from pointblank.mcp._utils import clean_for_json_serialization

        result = clean_for_json_serialization({"a": float("nan"), "b": 1})
        assert result == {"a": None, "b": 1}

    def test_list_cleaned_recursively(self):
        from pointblank.mcp._utils import clean_for_json_serialization

        result = clean_for_json_serialization([float("nan"), float("inf"), 1])
        assert result == [None, "inf", 1]

    def test_non_float_passthrough(self):
        from pointblank.mcp._utils import clean_for_json_serialization

        assert clean_for_json_serialization("hello") == "hello"
        assert clean_for_json_serialization(42) == 42


class TestGeneratePythonCodeForValidator:
    def test_with_df_path(self):
        import pandas as pd
        import pointblank as pb
        from pointblank.mcp._utils import generate_python_code_for_validator

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        v = (
            pb.Validate(df)
            .col_vals_not_null(columns="a")
            .col_vals_between(columns="b", left=1, right=10)
            .rows_distinct()
            .col_exists(columns="a")
            .interrogate()
        )
        code = generate_python_code_for_validator(v, "v1", df_path="data.csv")
        assert "import pointblank as pb" in code
        assert "data.csv" in code
        assert "col_vals_not_null" in code

    def test_without_df_path(self):
        import pandas as pd
        import pointblank as pb
        from pointblank.mcp._utils import generate_python_code_for_validator

        df = pd.DataFrame({"a": [1, 2]})
        v = pb.Validate(df).col_vals_ge(columns="a", value=0).interrogate()
        code = generate_python_code_for_validator(v, "v1")
        assert "your_data.csv" in code
        assert "col_vals_ge" in code

    def test_various_assertion_types(self):
        import pandas as pd
        import pointblank as pb
        from pointblank.mcp._utils import generate_python_code_for_validator

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1, 2]})
        v = (
            pb.Validate(df)
            .col_vals_gt(columns="a", value=0)
            .col_vals_le(columns="a", value=10)
            .col_vals_lt(columns="a", value=100)
            .col_vals_in_set(columns="b", set=["x", "y"])
            .col_vals_regex(columns="b", pattern=r"[a-z]+")
            .interrogate()
        )
        code = generate_python_code_for_validator(v, "v2")
        assert "col_vals_gt" in code
        assert "col_vals_le" in code
        assert "col_vals_lt" in code
        assert "col_vals_in_set" in code
        assert "col_vals_regex" in code


class TestGetAvailableBackends:
    def test_returns_list_with_pandas(self):
        from pointblank.mcp._utils import get_available_backends

        backends = get_available_backends()
        assert isinstance(backends, list)
        assert "pandas" in backends
