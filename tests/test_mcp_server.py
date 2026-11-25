from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mcp", reason="mcp package not installed", exc_type=ImportError)

from fastmcp import Client
from fastmcp.exceptions import ToolError
from pointblank.mcp.server import mcp


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


@pytest.mark.asyncio
async def test_list_available_tools(mcp_server):
    """Tests if the MCP is up and correctly lists its registered tools."""

    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        assert "load_dataframe" in tool_names
        assert "create_validator" in tool_names
        assert "add_validation_step" in tool_names
        assert "interrogate_validator" in tool_names
        assert "get_validation_step_output" in tool_names


@pytest.mark.asyncio
async def test_full_validation_workflow(mcp_server, csv_file: str, tmp_path: Path):
    """Tests a complete user workflow using in-memory calls to the MCP server."""

    # The client connects directly to the mcp object in memory
    async with Client(mcp_server) as client:
        # 1. Load a DataFrame from the CSV file

        result = await client.call_tool("load_dataframe", {"input_path": csv_file})

        assert not result.is_error

        df_info = result.data

        assert tuple(df_info.shape) == (5, 4)

        df_id = df_info.df_id

        # 2. Create a validator for the loaded DataFrame

        result = await client.call_tool("create_validator", {"df_id": df_id})

        assert not result.is_error

        validator_info = result.data
        validator_id = validator_info.validator_id

        # 3. Add validation steps

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

        # 4. Interrogate the validator

        result = await client.call_tool("interrogate_validator", {"validator_id": validator_id})

        assert not result.is_error

        interrogate_result = result.data
        summary = interrogate_result["validation_summary"]

        # Verify our new functionality - Python code generation
        assert "python_code" in interrogate_result

        python_code = interrogate_result["python_code"]

        assert isinstance(python_code, str)
        assert "import pointblank as pb" in python_code
        assert "pb.Validate(df)" in python_code
        assert ".col_vals_not_null(columns='id')" in python_code
        assert ".col_vals_lt(columns='age', value=10)" in python_code
        assert ".interrogate()" in python_code

        # Verify instructions are provided
        assert "instructions" in interrogate_result

        instructions = interrogate_result["instructions"]

        assert "html_report" in instructions
        assert "python_code" in instructions

        # summary is 0-indexed, so summary[1] is the second step
        assert summary[1]["f_passed"] < 1.0

        # 5. Get the output for the specific failing step (step_index=2)

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

        # 6. Get the output for all passing data across the entire run

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


@pytest.mark.asyncio
async def test_get_validation_step_output_polars_backend(mcp_server, csv_file: str, tmp_path: Path):
    """Test validation output export for a Polars-backed validator."""
    pytest.importorskip("polars")

    async with Client(mcp_server) as client:
        load_result = await client.call_tool(
            "load_dataframe", {"input_path": csv_file, "backend": "polars"}
        )
        assert not load_result.is_error
        df_id = load_result.data.df_id

        validator_result = await client.call_tool("create_validator", {"df_id": df_id})
        assert not validator_result.is_error
        validator_id = validator_result.data.validator_id

        step_result = await client.call_tool(
            "add_validation_step",
            {
                "validator_id": validator_id,
                "validation_type": "col_vals_gt",
                "params": {"columns": "age", "value": 30},
            },
        )
        assert not step_result.is_error

        interrogate_result = await client.call_tool(
            "interrogate_validator", {"validator_id": validator_id}
        )
        assert not interrogate_result.is_error

        step_output_path = str(tmp_path / "polars_step_output.csv")
        step_output_result = await client.call_tool(
            "get_validation_step_output",
            {
                "validator_id": validator_id,
                "output_path": step_output_path,
                "step_index": 1,
            },
        )
        assert not step_output_result.is_error
        assert step_output_result.data.output_file is not None
        assert Path(step_output_result.data.output_file).exists()
        assert len(pd.read_csv(step_output_result.data.output_file)) == 2

        sundered_output_path = str(tmp_path / "polars_sundered_output.csv")
        sundered_output_result = await client.call_tool(
            "get_validation_step_output",
            {
                "validator_id": validator_id,
                "output_path": sundered_output_path,
                "sundered_type": "fail",
            },
        )
        assert not sundered_output_result.is_error
        assert sundered_output_result.data.output_file is not None
        assert Path(sundered_output_result.data.output_file).exists()
        assert len(pd.read_csv(sundered_output_result.data.output_file)) == 2


async def _polars_validator(client, csv_file: str, value: int) -> str:
    """Load the CSV with Polars and add one `col_vals_gt()` step on `age` (not interrogated)."""
    df_id = (
        await client.call_tool("load_dataframe", {"input_path": csv_file, "backend": "polars"})
    ).data.df_id
    validator_id = (await client.call_tool("create_validator", {"df_id": df_id})).data.validator_id
    await client.call_tool(
        "add_validation_step",
        {
            "validator_id": validator_id,
            "validation_type": "col_vals_gt",
            "params": {"columns": "age", "value": value},
        },
    )
    return validator_id


@pytest.mark.asyncio
async def test_get_validation_step_output_polars_no_failures(
    mcp_server, csv_file: str, tmp_path: Path
):
    """Test that a Polars-backed validator with no failing rows doesn't write a CSV."""
    pytest.importorskip("polars")

    async with Client(mcp_server) as client:
        validator_id = await _polars_validator(client, csv_file, value=0)
        await client.call_tool("interrogate_validator", {"validator_id": validator_id})

        for args, message in [
            ({"step_index": 1}, "No data extract available for step 1."),
            ({"sundered_type": "fail"}, "No sundered data available for type 'fail'."),
        ]:
            result = await client.call_tool(
                "get_validation_step_output",
                {"validator_id": validator_id, "output_path": str(tmp_path / "out.csv"), **args},
            )
            assert result.data.output_file is None
            assert result.data.message == message

        assert not (tmp_path / "out.csv").exists()


@pytest.mark.asyncio
async def test_get_validation_step_output_interrogates_first(
    mcp_server, csv_file: str, tmp_path: Path
):
    """Test that the output of a validator that wasn't interrogated is still exported."""
    pytest.importorskip("polars")

    async with Client(mcp_server) as client:
        validator_id = await _polars_validator(client, csv_file, value=30)

        result = await client.call_tool(
            "get_validation_step_output",
            {
                "validator_id": validator_id,
                "output_path": str(tmp_path / "out.csv"),
                "step_index": 1,
            },
        )
        assert len(pd.read_csv(result.data.output_file)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("step_index", [0, 2, -1])
async def test_get_validation_step_output_step_index_out_of_range(
    mcp_server, csv_file: str, tmp_path: Path, step_index: int
):
    """Test that a step index outside of the validation's steps gives a clear error."""
    pytest.importorskip("polars")

    async with Client(mcp_server) as client:
        validator_id = await _polars_validator(client, csv_file, value=30)

        with pytest.raises(ToolError, match="must be a step number from 1 to 1"):
            await client.call_tool(
                "get_validation_step_output",
                {
                    "validator_id": validator_id,
                    "output_path": str(tmp_path / "out.csv"),
                    "step_index": step_index,
                },
            )
