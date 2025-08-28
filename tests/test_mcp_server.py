from pathlib import Path

import pandas as pd
import pytest

# Import the fastmcp Client and your mcp application instance
from fastmcp import Client
from pointblank_mcp_server.pointblank_server import mcp

# --- Fixtures ---


@pytest.fixture(scope="module")
def mcp_server():
    """Provides the FastMCP server instance for in-memory testing."""
    return mcp


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 9, 35, 99, 40],  # Contains values that will pass and fail validation
            "score": [85, 92, 78, 88, 95],
        }
    )


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory, sample_df) -> str:
    """Creates a temporary CSV file with sample data for the tests."""
    file_path = tmp_path_factory.mktemp("data") / "test_data.csv"
    sample_df.to_csv(file_path, index=False)
    return str(file_path)


# --- Test Functions ---


@pytest.mark.asyncio
async def test_list_available_tools(mcp_server):
    """Tests if the MCP is up and correctly lists its registered tools."""
    print("\n=== Testing Tool Listing ===")
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        print(f"✅ Found {len(tool_names)} tools.")
        assert "load_dataframe" in tool_names
        assert "create_validator" in tool_names
        assert "add_validation_step" in tool_names
        assert "interrogate_validator" in tool_names
        assert "get_validation_step_output" in tool_names
        print("✅ All expected tools are registered.")


@pytest.mark.asyncio
async def test_full_validation_workflow(mcp_server, csv_file: str, tmp_path: Path):
    """
    Tests a complete user workflow using in-memory calls to the MCP server.
    """
    print("\n=== Testing Full Validation Workflow ===")

    # The client connects directly to the mcp object in memory
    async with Client(mcp_server) as client:
        # 1. Load a DataFrame from the CSV file
        print("🔧 Step 1: Calling 'load_dataframe'...")
        result = await client.call_tool("load_dataframe", {"input_path": csv_file})
        assert not result.is_error
        df_info = result.data
        assert tuple(df_info.shape) == (5, 4)
        df_id = df_info.df_id
        print(f"✅ DataFrame loaded with ID: {df_id}")

        # 2. Create a validator for the loaded DataFrame
        print("\n🔧 Step 2: Calling 'create_validator'...")
        result = await client.call_tool("create_validator", {"df_id": df_id})
        assert not result.is_error
        validator_info = result.data
        validator_id = validator_info.validator_id
        print(f"✅ Validator created with ID: {validator_id}")

        # 3. Add validation steps
        print("\n🔧 Step 3: Calling 'add_validation_step'...")
        # This is STEP 1
        step1_params = {"columns": "id"}
        result = await client.call_tool(
            "add_validation_step",
            {
                "validator_id": validator_id,
                "validation_type": "col_vals_not_null",
                "params": step1_params,
            },
        )
        assert not result.is_error
        print("✅ Added passing step: 'id' column is not null")

        # This is STEP 2
        step2_params = {"columns": "age", "value": 10}
        result = await client.call_tool(
            "add_validation_step",
            {
                "validator_id": validator_id,
                "validation_type": "col_vals_lt",
                "params": step2_params,
            },
        )
        assert not result.is_error
        print("✅ Added failing step: 'age' column values < 10")

        # 4. Interrogate the validator
        print("\n🔧 Step 4: Calling 'interrogate_validator'...")
        result = await client.call_tool("interrogate_validator", {"validator_id": validator_id})
        assert not result.is_error
        interrogate_result = result.data
        summary = interrogate_result["validation_summary"]

        # summary is 0-indexed, so summary[1] is the second step
        assert summary[1]["f_passed"] < 1.0
        print("✅ Interrogation complete. Found expected failures.")

        # 5. Get the output for the specific failing step (step_index=2)
        print("\n🔧 Step 5: Calling 'get_validation_step_output' for a specific failing step...")
        failed_step_path = str(tmp_path / "failed_step_output.csv")
        result = await client.call_tool(
            "get_validation_step_output",
            {
                "validator_id": validator_id,
                "output_path": failed_step_path,
                # FINAL FIX: pointblank.get_data_extracts() is likely 1-indexed. Use 2 for the second step.
                "step_index": 2,
            },
        )
        assert not result.is_error
        output_info = result.data
        assert output_info.output_file is not None, f"No CSV was written: {output_info.message}"
        assert Path(output_info.output_file).exists()
        failed_df = pd.read_csv(output_info.output_file)
        assert len(failed_df) == 4  # Should contain the 4 rows that failed the age check
        print(f"✅ Failing rows for step 2 correctly saved to '{output_info.output_file}'")

        # 6. Get the output for all passing data across the entire run
        print("\n🔧 Step 6: Calling 'get_validation_step_output' for all passing data...")
        passed_run_path = str(tmp_path / "passed_run_output.csv")
        result = await client.call_tool(
            "get_validation_step_output",
            {
                "validator_id": validator_id,
                "output_path": passed_run_path,
                "sundered_type": "pass",  # No step_index, so this is used
            },
        )
        assert not result.is_error
        pass_info = result.data
        assert pass_info.output_file is not None, f"No CSV was written: {pass_info.message}"
        assert Path(pass_info.output_file).exists()
        pass_df = pd.read_csv(pass_info.output_file)
        assert len(pass_df) == 1  # Only the row with age=9 passed all validations
        assert pass_df.iloc[0]["age"] == 9
        print(f"✅ All passing rows correctly saved to '{pass_info.output_file}'")
        print("\n🎉 Full workflow test passed!")
