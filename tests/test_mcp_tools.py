"""Tests for MCP tools that lack dedicated coverage."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from pointblank.mcp.server import mcp


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
async def test_analyze_data_quality(mcp_server, temp_csv_file):
    """Test analyze_data_quality returns scan results."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("analyze_data_quality", {"df_id": df_id})
        assert not result.is_error
        data = result.data
        assert data["status"] == "success"
        assert data["df_id"] == df_id
        assert "analysis" in data


@pytest.mark.asyncio
async def test_analyze_data_quality_not_found(mcp_server):
    """Test analyze_data_quality raises for missing DataFrame."""
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("analyze_data_quality", {"df_id": "nonexistent_df"})


@pytest.mark.asyncio
async def test_profile_dataframe(mcp_server, temp_csv_file):
    """Test profile_dataframe returns profiling results."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("profile_dataframe", {"df_id": df_id, "sample_size": 0})
        assert not result.is_error
        data = result.data
        # DataScan JSON output should be a dict or list
        assert data is not None


@pytest.mark.asyncio
async def test_profile_dataframe_with_sampling(mcp_server, temp_csv_file):
    """Test profile_dataframe with sampling enabled."""
    async with Client(mcp_server) as client:
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        result = await client.call_tool("profile_dataframe", {"df_id": df_id, "sample_size": 3})
        assert not result.is_error


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
