import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pointblank as pb
from pathlib import Path
import uuid
import json

from pointblank_server import (
    AppContext,
    app_lifespan,
    FastMCP,
    load_dataframe,
    create_validator,
    add_validation_step,
    get_validation_step_output,
    interrogate_validator
)

# Fixtures
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', None],
        'age': [25, 30, 35, 40],
        'score': [85, 92, 78, 88]
    })

@pytest.fixture
def csv_file(tmp_path, sample_df):
    file_path = tmp_path / "test.csv"
    sample_df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def app_context(sample_df):
    ctx = AppContext()
    df_id = "test_df"
    ctx.loaded_dataframes[df_id] = sample_df
    return ctx

@pytest.fixture
def validator_context(app_context):
    validator_id = "test_validator"
    validator = pb.Validate(
        data=app_context.loaded_dataframes["test_df"],
        tbl_name="test_table"
    )
    # Add a validation step to ensure interrogate works
    validator.col_exists("age")
    app_context.active_validators[validator_id] = validator
    return app_context, validator_id

# Mock FastMCP context
@pytest.fixture
def mock_context(app_context):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app_context
    ctx.report_progress = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx

# Tests
def test_load_dataframe_success(csv_file, mock_context):
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        result = load_dataframe(csv_file)

        assert "df_id" in result
        assert result["status"] == "DataFrame loaded successfully."
        assert result["shape"] == (4, 4)
        assert set(result["columns"]) == {"id", "name", "age", "score"}

        df_id = result["df_id"]
        assert df_id in mock_context.request_context.lifespan_context.loaded_dataframes

def test_load_dataframe_invalid_file(mock_context):
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        with pytest.raises(FileNotFoundError):
            load_dataframe("invalid_path.csv")

def test_create_validator_success(app_context, mock_context):
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        result = create_validator("test_df")

        assert "validator_id" in result
        assert result["status"] == "Validator created successfully."

        validator_id = result["validator_id"]
        assert validator_id in mock_context.request_context.lifespan_context.active_validators

def test_create_validator_invalid_df_id(mock_context):
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        with pytest.raises(ValueError) as excinfo:
            create_validator("invalid_df_id")
        assert "not found" in str(excinfo.value)

def test_add_validation_step_success(validator_context, mock_context):
    app_context, validator_id = validator_context
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        params = {"columns": "age", "value": 50}
        result = add_validation_step(validator_id, "col_vals_lt", params)

        assert result["validator_id"] == validator_id
        assert "success" in result["status"]

        validator = app_context.active_validators[validator_id]
        assert len(validator.n()) > 0

def test_add_validation_step_invalid_type(validator_context, mock_context):
    _, validator_id = validator_context
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        with pytest.raises(ValueError) as excinfo:
            add_validation_step(validator_id, "invalid_type", {})
        assert "Unsupported validation_type" in str(excinfo.value)

@pytest.mark.asyncio
async def test_interrogate_validator_success(validator_context, mock_context, tmp_path):
    app_context, validator_id = validator_context
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        result = await interrogate_validator(validator_id)

        assert "validation_summary" in result
        report = json.loads(result["validation_summary"])
        assert isinstance(report, list)
        assert len(report) > 0

@pytest.mark.asyncio
async def test_get_validation_step_output_csv(validator_context, mock_context, tmp_path):
    app_context, validator_id = validator_context
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        # Add a validation step that will fail
        add_validation_step(validator_id, "col_vals_lt", {"columns": "age", "value": 10})
        await interrogate_validator(validator_id)

        output_path = str(tmp_path / "output.csv")
        result = await get_validation_step_output(
            validator_id,
            output_path,
            step_index=2  # Use step 2 for the failing validation
        )

        assert result["status"] == "success"
        if result["output_file"]:  # Handle case where no extract exists
            assert Path(result["output_file"]).exists()
        else:
            assert "No data extract" in result["message"]

@pytest.mark.asyncio
async def test_get_validation_step_output_png(validator_context, mock_context, tmp_path):
    app_context, validator_id = validator_context
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        # Add a validation step that will fail
        add_validation_step(validator_id, "col_vals_lt", {"columns": "age", "value": 10})
        await interrogate_validator(validator_id)

        output_path = str(tmp_path / "output.png")
        result = await get_validation_step_output(
            validator_id,
            output_path,
            step_index=2  # Use step 2 for the failing validation
        )

        assert result["status"] == "success"
        assert Path(result["output_file"]).exists()

@pytest.mark.asyncio
async def test_get_validation_step_output_no_step_index(validator_context, mock_context, tmp_path):
    app_context, validator_id = validator_context
    with patch("pointblank_server.mcp.get_context", return_value=mock_context):
        # Add a validation step that will fail
        add_validation_step(validator_id, "col_vals_lt", {"columns": "age", "value": 10})
        await interrogate_validator(validator_id)

        output_path = str(tmp_path / "output.csv")
        result = await get_validation_step_output(
            validator_id,
            output_path,
            sundered_type="fail"
        )

        assert result["status"] == "success"
        assert Path(result["output_file"]).exists()
        assert "saved" in result["message"]

@pytest.mark.asyncio
async def test_app_context_lifespan():
    mock_server = MagicMock()

    # Test async context manager
    async with app_lifespan(mock_server) as ctx:
        assert isinstance(ctx, AppContext)
        assert ctx.loaded_dataframes == {}
        assert ctx.active_validators == {}

        # Add some data
        ctx.loaded_dataframes["test"] = pd.DataFrame()
        ctx.active_validators["val"] = pb.Validate(data=pd.DataFrame())

    # Verify cleanup
    assert ctx.loaded_dataframes == {}
    assert ctx.active_validators == {}
