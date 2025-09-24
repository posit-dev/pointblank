import pytest
import pandas as pd
import polars as pl
from unittest.mock import Mock, patch, MagicMock

import pointblank as pb
from pointblank.validate import Validate
from pointblank._constants import MODEL_PROVIDERS


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data_pd():
    """Sample pandas DataFrame for testing."""

    return pd.DataFrame(
        {
            "email": ["valid@example.com", "invalid-email", "user@test.org", "bad-format"],
            "name": ["John Doe", "", "Jane Smith", "Bob Johnson"],
            "age": [25, 30, 35, 40],
            "score": [85, 90, 78, 92],
        }
    )


@pytest.fixture
def sample_data_pl():
    """Sample polars DataFrame for testing."""

    return pl.DataFrame(
        {
            "email": ["valid@example.com", "invalid-email", "user@test.org", "bad-format"],
            "name": ["John Doe", "", "Jane Smith", "Bob Johnson"],
            "age": [25, 30, 35, 40],
            "score": [85, 90, 78, 92],
        }
    )


@pytest.fixture
def mock_ai_validation_results():
    """Mock AI validation results for testing."""

    return [True, False, True, False]  # 2 pass, 2 fail


# ============================================================================
# Test Basic Functionality
# ============================================================================


def test_prompt_method_exists():
    """Test that the prompt method exists on Validate class."""

    validate = Validate(data=[])
    assert hasattr(validate, "prompt")
    assert callable(getattr(validate, "prompt"))


def test_prompt_method_signature():
    """Test that prompt method has correct signature and required parameters."""

    validate = Validate(data=[])

    # Should work with required parameters
    result = validate.prompt(prompt="Test prompt", model="anthropic:claude-3-sonnet-20240229")

    assert isinstance(result, Validate)
    assert result is validate  # Should return self for chaining


def test_prompt_parameter_validation():
    """Test prompt parameter validation."""

    validate = Validate(data=[])

    # Empty prompt should raise ValueError
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        validate.prompt(prompt="", model="anthropic:claude-3-sonnet-20240229")

    # Non-string prompt should raise ValueError
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        validate.prompt(prompt=123, model="anthropic:claude-3-sonnet-20240229")

    # Whitespace-only prompt should raise ValueError
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        validate.prompt(prompt="   ", model="anthropic:claude-3-sonnet-20240229")


def test_model_parameter_validation():
    """Test model parameter parsing and validation."""

    validate = Validate(data=[])

    # Invalid model format should raise ValueError
    with pytest.raises(ValueError, match="Model must be in format 'provider:model_name'"):
        validate.prompt(prompt="Test prompt", model="invalid-format")

    # Unsupported provider should raise ValueError
    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        validate.prompt(prompt="Test prompt", model="invalid:model-name")

    # Valid providers should work
    for provider in MODEL_PROVIDERS:
        result = validate.prompt(prompt="Test prompt", model=f"{provider}:some-model")
        assert isinstance(result, Validate)


def test_batch_size_validation():
    """Test batch_size parameter validation."""

    validate = Validate(data=[])

    # Non-integer batch_size should raise ValueError
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate.prompt(
            prompt="Test prompt", model="anthropic:claude-3-sonnet-20240229", batch_size="invalid"
        )

    # Zero or negative batch_size should raise ValueError
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate.prompt(
            prompt="Test prompt", model="anthropic:claude-3-sonnet-20240229", batch_size=0
        )

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate.prompt(
            prompt="Test prompt", model="anthropic:claude-3-sonnet-20240229", batch_size=-5
        )


def test_max_concurrent_validation():
    """Test max_concurrent parameter validation."""

    validate = Validate(data=[])

    # Non-integer max_concurrent should raise ValueError
    with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
        validate.prompt(
            prompt="Test prompt",
            model="anthropic:claude-3-sonnet-20240229",
            max_concurrent="invalid",
        )

    # Zero or negative max_concurrent should raise ValueError
    with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
        validate.prompt(
            prompt="Test prompt", model="anthropic:claude-3-sonnet-20240229", max_concurrent=0
        )


def test_columns_subset_parameter():
    """Test columns_subset parameter handling."""

    validate = Validate(data=[])

    # Single column as string should be converted to list
    result = validate.prompt(
        prompt="Test prompt", model="anthropic:claude-3-sonnet-20240229", columns_subset="email"
    )

    assert isinstance(result, Validate)

    # Check that the validation was added correctly
    assert len(validate.validation_info) == 1
    assert validate.validation_info[0].column == ["email"]

    # List of columns should remain as list
    validate2 = Validate(data=[])
    result2 = validate2.prompt(
        prompt="Test prompt",
        model="anthropic:claude-3-sonnet-20240229",
        columns_subset=["email", "name"],
    )

    assert isinstance(result2, Validate)
    assert validate2.validation_info[0].column == ["email", "name"]


def test_validation_info_creation():
    """Test that ValidationInfo is created correctly."""

    validate = Validate(data=[])

    validate.prompt(
        prompt="Test validation prompt",
        model="openai:gpt-4",
        columns_subset=["email", "name"],
        batch_size=500,
        max_concurrent=2,
    )

    assert len(validate.validation_info) == 1
    val_info = validate.validation_info[0]

    assert val_info.assertion_type == "prompt"
    assert val_info.column == ["email", "name"]
    assert val_info.values["prompt"] == "Test validation prompt"
    assert val_info.values["llm_provider"] == "openai"
    assert val_info.values["llm_model"] == "gpt-4"
    assert val_info.values["batch_size"] == 500
    assert val_info.values["max_concurrent"] == 2


# ============================================================================
# Test Integration with Validation Framework
# ============================================================================


@patch("pointblank._interrogation.interrogate_prompt")
def test_prompt_integration_with_interrogate_pandas(
    mock_interrogate, sample_data_pd, mock_ai_validation_results
):
    """Test prompt method integration with interrogation for pandas data."""
    # Mock the interrogation function to return DataFrame with pb_is_good_ column

    mock_result_df = sample_data_pd.copy()
    mock_result_df["pb_is_good_"] = mock_ai_validation_results
    mock_interrogate.return_value = mock_result_df

    validate = Validate(data=sample_data_pd)
    result = validate.prompt(
        prompt="Each row should have a valid email address and non-empty name",
        model="anthropic:claude-3-sonnet-20240229",
        columns_subset=["email", "name"],
    ).interrogate()

    # Check that interrogation was called
    mock_interrogate.assert_called_once()

    # Verify the validation results exist
    assert hasattr(result, "validation_info")
    assert len(result.validation_info) == 1

    # Check that the validation step was configured correctly
    val_info = result.validation_info[0]
    assert val_info.assertion_type == "prompt"
    assert val_info.column == ["email", "name"]

    # Check that the validation has results (n_passed, n_failed should be set)
    assert val_info.n_passed is not None
    assert val_info.n_failed is not None


@patch("pointblank._interrogation.interrogate_prompt")
def test_prompt_integration_with_interrogate_polars(
    mock_interrogate, sample_data_pl, mock_ai_validation_results
):
    """Test prompt method integration with interrogation for polars data."""
    # Mock the interrogation function to return DataFrame with pb_is_good_ column

    mock_result_df = sample_data_pl.with_columns(pb_is_good_=pl.Series(mock_ai_validation_results))
    mock_interrogate.return_value = mock_result_df

    validate = Validate(data=sample_data_pl)
    result = validate.prompt(
        prompt="Each row should have a valid email address and non-empty name",
        model="openai:gpt-4o-mini",
        columns_subset=["email", "name"],
    ).interrogate()

    # Check that interrogation was called
    mock_interrogate.assert_called_once()

    # Verify the validation results exist
    assert hasattr(result, "validation_info")
    assert len(result.validation_info) == 1

    # Check that the validation has results
    val_info = result.validation_info[0]
    assert val_info.n_passed is not None
    assert val_info.n_failed is not None


def test_prompt_method_chaining():
    """Test that prompt method supports method chaining."""

    validate = Validate(data=[])

    result = validate.prompt(
        prompt="First validation", model="anthropic:claude-3-sonnet-20240229"
    ).prompt(prompt="Second validation", model="openai:gpt-4")

    assert isinstance(result, Validate)
    assert result is validate
    assert len(validate.validation_info) == 2

    # Check both validations were added
    assert validate.validation_info[0].values["prompt"] == "First validation"
    assert validate.validation_info[1].values["prompt"] == "Second validation"


def test_prompt_with_thresholds():
    """Test prompt method with threshold configuration."""

    validate = Validate(data=[])

    thresholds = pb.Thresholds(warning=0.1, error=0.2, critical=0.3)

    result = validate.prompt(
        prompt="Test with thresholds",
        model="anthropic:claude-3-sonnet-20240229",
        thresholds=thresholds,
    )

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]
    assert val_info.thresholds is not None
    assert val_info.thresholds.warning == 0.1
    assert val_info.thresholds.error == 0.2
    assert val_info.thresholds.critical == 0.3


def test_prompt_with_actions():
    """Test prompt method with actions configuration."""

    validate = Validate(data=[])

    actions = pb.Actions(
        warning="Warning: some validation issues", error="Error: significant validation problems"
    )

    result = validate.prompt(
        prompt="Test with actions", model="anthropic:claude-3-sonnet-20240229", actions=actions
    )

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]
    assert val_info.actions is not None

    # Actions might store values as lists
    warning_val = val_info.actions.warning
    error_val = val_info.actions.error

    if isinstance(warning_val, list):
        assert warning_val == ["Warning: some validation issues"]
    else:
        assert warning_val == "Warning: some validation issues"

    if isinstance(error_val, list):
        assert error_val == ["Error: significant validation problems"]
    else:
        assert error_val == "Error: significant validation problems"


def test_prompt_with_preprocessing():
    """Test prompt method with preprocessing function."""

    validate = Validate(data=[])

    def preprocess_data(df):
        # Simple preprocessing function
        return df.head(2) if hasattr(df, "head") else df[:2]

    result = validate.prompt(
        prompt="Test with preprocessing",
        model="anthropic:claude-3-sonnet-20240229",
        pre=preprocess_data,
    )

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]
    assert val_info.pre is preprocess_data


def test_prompt_with_brief():
    """Test prompt method with brief description."""

    validate = Validate(data=[])

    result = validate.prompt(
        prompt="Test with brief",
        model="anthropic:claude-3-sonnet-20240229",
        brief="Custom brief description",
    )

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]
    assert val_info.brief == "Custom brief description"


def test_prompt_inactive():
    """Test prompt method with active=False."""

    validate = Validate(data=[])

    result = validate.prompt(
        prompt="Inactive test", model="anthropic:claude-3-sonnet-20240229", active=False
    )

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]
    assert val_info.active is False


# ============================================================================
# Test Different Model Providers
# ============================================================================


@pytest.mark.parametrize(
    "provider,model",
    [
        ("anthropic", "claude-3-sonnet-20240229"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("ollama", "llama2"),
        ("bedrock", "anthropic.claude-3-sonnet-20240229-v1:0"),
    ],
)
def test_prompt_with_different_providers(provider, model):
    """Test prompt method with different model providers."""

    validate = Validate(data=[])

    result = validate.prompt(prompt="Test with different providers", model=f"{provider}:{model}")

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]
    assert val_info.values["llm_provider"] == provider
    assert val_info.values["llm_model"] == model


# ============================================================================
# Test Error Handling
# ============================================================================


def test_prompt_with_invalid_parameters():
    """Test prompt method with various invalid parameter combinations."""

    validate = Validate(data=[])

    # Test with all invalid parameters at once
    with pytest.raises(ValueError):
        validate.prompt(
            prompt="",  # Empty prompt
            model="invalid-format",  # Invalid model format
            batch_size=-1,  # Invalid batch size
            max_concurrent=0,  # Invalid max_concurrent
        )


def test_prompt_parameter_types():
    """Test prompt method parameter type validation."""

    validate = Validate(data=[])

    # Test non-string model parameter
    with pytest.raises(AttributeError):
        validate.prompt(
            prompt="Test prompt",
            model=123,  # Should be string
        )


# ============================================================================
# Test Complex Scenarios
# ============================================================================


@patch("pointblank._interrogation.interrogate_prompt")
def test_complex_validation_scenario(mock_interrogate, sample_data_pd):
    """Test a complex validation scenario with multiple configurations."""

    # Mock different results for different validation steps
    first_results = [True, False, True, False]
    second_results = [False, True, False, True]

    mock_result_df1 = sample_data_pd.copy()
    mock_result_df1["pb_is_good_"] = first_results

    mock_result_df2 = sample_data_pd.copy()
    mock_result_df2["pb_is_good_"] = second_results

    mock_interrogate.side_effect = [mock_result_df1, mock_result_df2]

    validate = Validate(data=sample_data_pd, thresholds=pb.Thresholds(warning=0.2, error=0.4))

    result = (
        validate.prompt(
            prompt="Check email format and name presence",
            model="openai:gpt-4o-mini",
            columns_subset=["email", "name"],
            batch_size=2,
            max_concurrent=1,
            thresholds=pb.Thresholds(warning=0.1, error=0.3),
            brief="Email and name validation",
        )
        .prompt(
            prompt="Check if age is reasonable for the context",
            model="anthropic:claude-3-sonnet-20240229",
            columns_subset=["age"],
            batch_size=1,
            actions=pb.Actions(error="Age validation failed"),
            brief="Age reasonableness check",
        )
        .interrogate()
    )

    # Verify both validations were called
    assert mock_interrogate.call_count == 2

    # Check validation configuration
    assert len(result.validation_info) == 2

    # First validation
    val1 = result.validation_info[0]
    assert val1.assertion_type == "prompt"
    assert val1.column == ["email", "name"]
    assert val1.values["batch_size"] == 2
    assert val1.values["max_concurrent"] == 1
    assert val1.brief == "Email and name validation"

    # Second validation
    val2 = result.validation_info[1]
    assert val2.assertion_type == "prompt"
    assert val2.column == ["age"]
    assert val2.values["batch_size"] == 1

    # Actions might store values as lists
    error_val = val2.actions.error
    if isinstance(error_val, list):
        assert error_val == ["Age validation failed"]
    else:
        assert error_val == "Age validation failed"
    assert val2.brief == "Age reasonableness check"


def test_prompt_with_all_parameters():
    """Test prompt method with all possible parameters configured."""

    validate = Validate(data=[])

    def preprocess_func(df):
        return df

    thresholds = pb.Thresholds(warning=0.1, error=0.2, critical=0.3)
    actions = pb.Actions(
        warning="Warning triggered", error="Error triggered", critical="Critical triggered"
    )

    result = validate.prompt(
        prompt="Comprehensive validation test",
        model="openai:gpt-4o",
        columns_subset=["col1", "col2", "col3"],
        batch_size=100,
        max_concurrent=5,
        pre=preprocess_func,
        segments=["segment_col"],
        thresholds=thresholds,
        actions=actions,
        brief="Comprehensive test brief",
        active=True,
    )

    assert isinstance(result, Validate)
    val_info = validate.validation_info[0]

    # Verify all parameters were set correctly
    assert val_info.assertion_type == "prompt"
    assert val_info.column == ["col1", "col2", "col3"]
    assert val_info.values["prompt"] == "Comprehensive validation test"
    assert val_info.values["llm_provider"] == "openai"
    assert val_info.values["llm_model"] == "gpt-4o"
    assert val_info.values["batch_size"] == 100
    assert val_info.values["max_concurrent"] == 5
    assert val_info.pre is preprocess_func
    assert val_info.segments == ["segment_col"]
    assert val_info.thresholds is thresholds
    assert val_info.actions is actions
    assert val_info.brief == "Comprehensive test brief"
    assert val_info.active is True
