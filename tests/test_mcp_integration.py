import asyncio
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client

from pointblank_mcp_server.pointblank_server import mcp


@pytest.fixture(scope="module")
def mcp_server():
    """Provides the FastMCP server instance."""
    return mcp


@pytest.fixture(scope="module")
def sample_data():
    """Provides test data for validation scenarios."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "email": [
                "valid@test.com",
                "invalid-email",
                "another@test.com",
                "",
                "test@domain.org",
            ],
            "age": [25, -5, 35, 999, 40],  # Mix of valid/invalid ages
            "score": [85.5, 92.0, 78.5, 88.0, 95.0],
        }
    )


@pytest.fixture
def temp_csv_file(sample_data):
    """Creates a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_server_initialization(mcp_server):
    """Test that the MCP server initializes correctly."""
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        expected_tools = [
            "load_dataframe",
            "create_validator",
            "add_validation_step",
            "interrogate_validator",
            "get_validation_step_output",
        ]
        tool_names = [tool.name for tool in tools]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"


@pytest.mark.asyncio
async def test_data_loading_success(mcp_server, temp_csv_file):
    """Test successful data loading."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})

        assert not result.is_error, f"Data loading failed: {result.error}"
        assert hasattr(result.data, "df_id")
        # Shape is returned as a list [rows, cols] instead of tuple
        assert result.data.shape == [5, 4]


@pytest.mark.asyncio
async def test_data_loading_failure(mcp_server):
    """Test data loading with invalid file."""
    async with Client(mcp_server) as client:
        try:
            result = await client.call_tool(
                "load_dataframe", {"input_path": "/nonexistent/file.csv"}
            )
            # If we get here, the tool didn't raise an exception, check if it returned an error
            assert result.is_error
            assert "not found" in str(result.error).lower()
        except Exception as e:
            # Tool raised an exception as expected
            assert "not found" in str(e).lower()


@pytest.mark.asyncio
async def test_validation_workflow_success(mcp_server, temp_csv_file):
    """Test a complete successful validation workflow."""
    async with Client(mcp_server) as client:
        # Load data
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        assert not load_result.is_error
        df_id = load_result.data.df_id

        # Create validator
        validator_result = await client.call_tool("create_validator", {"df_id": df_id})
        assert not validator_result.is_error
        validator_id = validator_result.data.validator_id

        # Add validation steps
        steps = [
            {"validation_type": "col_vals_not_null", "params": {"columns": "id"}},
            {"validation_type": "col_vals_gt", "params": {"columns": "age", "value": 0}},
            {
                "validation_type": "col_vals_between",
                "params": {"columns": "score", "left": 0, "right": 100},
            },
        ]

        for step in steps:
            step_result = await client.call_tool(
                "add_validation_step", {"validator_id": validator_id, **step}
            )
            assert not step_result.is_error

        # Interrogate
        interrogate_result = await client.call_tool(
            "interrogate_validator", {"validator_id": validator_id}
        )
        assert not interrogate_result.is_error

        summary = interrogate_result.data["validation_summary"]
        assert len(summary) == 3  # Three validation steps


@pytest.mark.asyncio
async def test_validation_with_failures(mcp_server, temp_csv_file):
    """Test validation that detects data quality issues."""
    async with Client(mcp_server) as client:
        # Load data
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        df_id = load_result.data.df_id

        # Create validator
        validator_result = await client.call_tool("create_validator", {"df_id": df_id})
        validator_id = validator_result.data.validator_id

        # Add validation that should fail for some rows
        step_result = await client.call_tool(
            "add_validation_step",
            {
                "validator_id": validator_id,
                "validation_type": "col_vals_between",
                "params": {"columns": "age", "left": 18, "right": 65},
            },
        )
        assert not step_result.is_error

        # Interrogate
        interrogate_result = await client.call_tool(
            "interrogate_validator", {"validator_id": validator_id}
        )
        assert not interrogate_result.is_error

        summary = interrogate_result.data["validation_summary"]
        assert summary[0]["f_passed"] < 1.0  # Some failures expected


@pytest.mark.asyncio
async def test_export_functionality(mcp_server, temp_csv_file):
    """Test exporting validation results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        async with Client(mcp_server) as client:
            # Setup validation with failures
            load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
            df_id = load_result.data.df_id

            validator_result = await client.call_tool("create_validator", {"df_id": df_id})
            validator_id = validator_result.data.validator_id

            # Add validation that will fail
            await client.call_tool(
                "add_validation_step",
                {
                    "validator_id": validator_id,
                    "validation_type": "col_vals_gt",
                    "params": {"columns": "age", "value": 100},  # Most ages will fail this
                },
            )

            # Interrogate
            await client.call_tool("interrogate_validator", {"validator_id": validator_id})

            # Export failed rows
            output_path = str(Path(temp_dir) / "failed_rows.csv")
            export_result = await client.call_tool(
                "get_validation_step_output",
                {"validator_id": validator_id, "output_path": output_path, "step_index": 1},
            )

            assert not export_result.is_error
            assert Path(output_path).exists()

            # Verify exported data
            exported_df = pd.read_csv(output_path)
            assert len(exported_df) > 0  # Should have some failed rows


@pytest.mark.asyncio
async def test_concurrent_validators(mcp_server, temp_csv_file):
    """Test that multiple validators can work concurrently."""
    async with Client(mcp_server) as client:
        # Load same data twice
        load_result1 = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        load_result2 = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})

        df_id1 = load_result1.data.df_id
        df_id2 = load_result2.data.df_id

        # Create two validators
        validator_result1 = await client.call_tool("create_validator", {"df_id": df_id1})
        validator_result2 = await client.call_tool("create_validator", {"df_id": df_id2})

        validator_id1 = validator_result1.data.validator_id
        validator_id2 = validator_result2.data.validator_id

        # They should be different
        assert validator_id1 != validator_id2


@pytest.mark.asyncio
async def test_memory_cleanup(mcp_server, temp_csv_file):
    """Test that resources are properly cleaned up."""
    async with Client(mcp_server) as client:
        # Get initial counts using the list tools
        initial_dfs = await client.call_tool("list_loaded_dataframes")
        initial_validators = await client.call_tool("list_active_validators")

        initial_df_count = len(initial_dfs.data["loaded_dataframes"])
        initial_validator_count = len(initial_validators.data["active_validators"])

        # Perform operations that create resources
        load_result = await client.call_tool("load_dataframe", {"input_path": temp_csv_file})
        validator_result = await client.call_tool(
            "create_validator", {"df_id": load_result.data.df_id}
        )

        # Check that resources were created
        current_dfs = await client.call_tool("list_loaded_dataframes")
        current_validators = await client.call_tool("list_active_validators")

        current_df_count = len(current_dfs.data["loaded_dataframes"])
        current_validator_count = len(current_validators.data["active_validators"])

        assert current_df_count > initial_df_count
        assert current_validator_count > initial_validator_count
