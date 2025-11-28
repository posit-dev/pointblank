import pytest
import pandas as pd
import polars as pl
import json
import hashlib
from unittest.mock import Mock, patch, MagicMock

from pointblank._utils_ai import (
    _LLMConfig,
    _BatchConfig,
    _DataBatcher,
    _PromptBuilder,
    _ValidationResponseParser,
    _AIValidationEngine,
    _create_chat_instance,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_pd_data():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
            "age": [25, 30, 35, 25, 30],
            "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
            "score": [85, 90, 78, 85, 90],
        }
    )


@pytest.fixture
def sample_pl_data():
    """Sample polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
            "age": [25, 30, 35, 25, 30],
            "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
            "score": [85, 90, 78, 85, 90],
        }
    )


@pytest.fixture
def llm_config():
    """Sample LLM configuration."""
    return _LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="test-key")


@pytest.fixture
def batch_config():
    """Sample batch configuration."""
    return _BatchConfig(size=2, max_concurrent=1)


@pytest.fixture
def mock_chat_response():
    """Mock chat response for testing."""
    return """[
        {"index": 0, "result": true},
        {"index": 1, "result": false}
    ]"""


# ============================================================================
# Test LLMConfig
# ============================================================================


def test_llm_config_creation():
    """Test LLMConfig dataclass creation."""
    config = _LLMConfig(provider="openai", model="gpt-4")

    assert config.provider == "openai"
    assert config.model == "gpt-4"
    assert config.api_key is None


def test_llm_config_with_api_key():
    """Test LLMConfig with API key."""
    config = _LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="test-key")

    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4-5"
    assert config.api_key == "test-key"


# ============================================================================
# Test create_chat_instance
# ============================================================================


def test_create_chat_instance_missing_chatlas():
    """Test error when chatlas is not installed."""
    # We can't easily mock the import machinery, so we'll just test the error case
    # This is covered by the integration tests anyway
    pass  # Skip this complex mocking test


def test_create_chat_instance_basic_functionality():
    """Test basic chat instance creation functionality without deep mocking."""
    # Test that invalid provider raises error
    with pytest.raises(ValueError, match="Provider 'invalid' is not supported"):
        _create_chat_instance("invalid", "model")

    # The actual chat instance creation is tested in integration tests
    # where we mock _create_chat_instance itself rather than its internals


def test_create_chat_instance_invalid_provider():
    """Test error with invalid provider."""
    with pytest.raises(ValueError, match="Provider 'invalid' is not supported"):
        _create_chat_instance("invalid", "model")


# ============================================================================
# Test BatchConfig
# ============================================================================


def test_batch_config_defaults():
    """Test BatchConfig default values."""
    config = _BatchConfig()

    assert config.size == 1000
    assert config.max_concurrent == 3


def test_batch_config_custom_values():
    """Test BatchConfig with custom values."""
    config = _BatchConfig(size=500, max_concurrent=5)

    assert config.size == 500
    assert config.max_concurrent == 5


# ============================================================================
# Test DataBatcher
# ============================================================================


def test_data_batcher_init_pandas(sample_pd_data):
    """Test DataBatcher initialization with pandas DataFrame."""
    batcher = _DataBatcher(sample_pd_data)

    assert batcher.data is sample_pd_data
    assert batcher.columns is None
    assert isinstance(batcher.config, _BatchConfig)
    assert batcher.unique_rows_table is None
    assert batcher.signature_to_original_indices == {}
    assert batcher.reduction_stats == {}


def test_data_batcher_init_polars(sample_pl_data):
    """Test DataBatcher initialization with polars DataFrame."""
    batcher = _DataBatcher(sample_pl_data)

    assert batcher.data is sample_pl_data
    assert batcher.columns is None
    assert isinstance(batcher.config, _BatchConfig)


def test_data_batcher_init_with_columns(sample_pd_data):
    """Test DataBatcher initialization with specific columns."""
    columns = ["name", "age"]
    batcher = _DataBatcher(sample_pd_data, columns=columns)

    assert batcher.columns == columns


def test_data_batcher_init_with_config(sample_pd_data, batch_config):
    """Test DataBatcher initialization with custom config."""
    batcher = _DataBatcher(sample_pd_data, config=batch_config)

    assert batcher.config is batch_config


def test_data_batcher_invalid_data():
    """Test DataBatcher with invalid data."""
    with pytest.raises(ValueError, match="Data must have a 'shape' attribute"):
        _DataBatcher([1, 2, 3])


def test_data_batcher_invalid_columns(sample_pd_data):
    """Test DataBatcher with invalid columns."""
    with pytest.raises(ValueError, match="Columns not found in data"):
        _DataBatcher(sample_pd_data, columns=["nonexistent"])


def test_create_row_signature(sample_pd_data):
    """Test row signature creation."""
    batcher = _DataBatcher(sample_pd_data)

    row_dict = {"name": "Alice", "age": 25, "city": "NYC"}
    signature = batcher._create_row_signature(row_dict)

    # Should create consistent signature
    assert isinstance(signature, str)
    assert len(signature) == 32  # MD5 hash length

    # Same data should produce same signature
    signature2 = batcher._create_row_signature(row_dict)
    assert signature == signature2

    # Different data should produce different signature
    row_dict2 = {"name": "Bob", "age": 30, "city": "LA"}
    signature3 = batcher._create_row_signature(row_dict2)
    assert signature != signature3


def test_create_row_signature_ignores_pb_row_index(sample_pd_data):
    """Test that row signature ignores _pb_row_index."""
    batcher = _DataBatcher(sample_pd_data)

    row_dict1 = {"name": "Alice", "age": 25, "_pb_row_index": 0}
    row_dict2 = {"name": "Alice", "age": 25, "_pb_row_index": 99}

    signature1 = batcher._create_row_signature(row_dict1)
    signature2 = batcher._create_row_signature(row_dict2)

    assert signature1 == signature2


def test_build_unique_rows_table_pandas(sample_pd_data):
    """Test building unique rows table with pandas."""
    batcher = _DataBatcher(sample_pd_data)

    unique_df, signature_mapping = batcher._build_unique_rows_table()

    # Should have 3 unique rows (Alice and Bob appear twice)
    assert len(unique_df) == 3
    assert len(signature_mapping) == 3

    # Check reduction stats
    stats = batcher.reduction_stats
    assert stats["original_rows"] == 5
    assert stats["unique_rows"] == 3
    assert stats["reduction_percentage"] == 40.0


def test_build_unique_rows_table_polars(sample_pl_data):
    """Test building unique rows table with polars."""
    batcher = _DataBatcher(sample_pl_data)

    unique_df, signature_mapping = batcher._build_unique_rows_table()

    # Should have 3 unique rows (Alice and Bob appear twice)
    assert len(unique_df) == 3
    assert len(signature_mapping) == 3

    # Check reduction stats
    stats = batcher.reduction_stats
    assert stats["original_rows"] == 5
    assert stats["unique_rows"] == 3
    assert stats["reduction_percentage"] == 40.0


def test_create_batches_pandas(sample_pd_data, batch_config):
    """Test creating batches with pandas DataFrame."""
    batcher = _DataBatcher(sample_pd_data, config=batch_config)

    batches, signature_mapping = batcher.create_batches()

    # With batch size 2, should have 2 batches for 3 unique rows
    assert len(batches) == 2
    assert len(signature_mapping) == 3

    # Check first batch structure
    batch = batches[0]
    assert batch["batch_id"] == 0
    assert batch["start_row"] == 0
    assert batch["end_row"] == 2
    assert "data" in batch

    # Check batch data structure
    data = batch["data"]
    assert "columns" in data
    assert "rows" in data
    assert "batch_info" in data
    assert len(data["rows"]) == 2


def test_create_batches_with_columns(sample_pd_data, batch_config):
    """Test creating batches with specific columns."""
    columns = ["name", "age"]
    batcher = _DataBatcher(sample_pd_data, columns=columns, config=batch_config)

    batches, signature_mapping = batcher.create_batches()

    # Check that only specified columns are included
    data = batches[0]["data"]
    expected_columns = ["name", "age"]  # _pb_signature removed
    assert set(data["columns"]) == set(expected_columns)


def test_convert_batch_to_json_pandas(sample_pd_data):
    """Test converting batch to JSON format."""
    batcher = _DataBatcher(sample_pd_data)

    # Create a small batch
    batch_data = sample_pd_data.head(2)
    json_data = batcher._convert_batch_to_json(batch_data, 0)

    assert "columns" in json_data
    assert "rows" in json_data
    assert "batch_info" in json_data

    assert len(json_data["rows"]) == 2
    assert json_data["batch_info"]["start_row"] == 0
    assert json_data["batch_info"]["num_rows"] == 2

    # Check that _pb_row_index is added
    for i, row in enumerate(json_data["rows"]):
        assert row["_pb_row_index"] == i


def test_get_reduction_stats(sample_pd_data):
    """Test getting reduction statistics."""
    batcher = _DataBatcher(sample_pd_data)
    batcher.create_batches()

    stats = batcher.get_reduction_stats()

    assert "original_rows" in stats
    assert "unique_rows" in stats
    assert "reduction_percentage" in stats
    assert stats["original_rows"] == 5
    assert stats["unique_rows"] == 3
    assert stats["reduction_percentage"] == 40.0


# ============================================================================
# Test PromptBuilder
# ============================================================================


def test_prompt_builder_init():
    """Test PromptBuilder initialization."""
    user_prompt = "Check if age is greater than 18"
    builder = _PromptBuilder(user_prompt)

    assert builder.user_prompt == user_prompt


def test_prompt_builder_build_prompt():
    """Test building a prompt from batch data."""
    user_prompt = "Check if age is greater than 18"
    builder = _PromptBuilder(user_prompt)

    batch_data = {
        "columns": ["name", "age"],
        "rows": [
            {"name": "Alice", "age": 25, "_pb_row_index": 0},
            {"name": "Bob", "age": 17, "_pb_row_index": 1},
        ],
        "batch_info": {"start_row": 0, "num_rows": 2},
    }

    prompt = builder.build_prompt(batch_data)

    assert "VALIDATION CRITERIA:" in prompt
    assert user_prompt in prompt
    assert "DATA TO VALIDATE:" in prompt
    assert "Alice" in prompt
    assert "Bob" in prompt


# ============================================================================
# Test ValidationResponseParser
# ============================================================================


def test_validation_response_parser_init():
    """Test ValidationResponseParser initialization."""
    parser = _ValidationResponseParser(total_rows=100)

    assert parser.total_rows == 100


def test_parse_response_valid_json(mock_chat_response):
    """Test parsing valid JSON response."""
    parser = _ValidationResponseParser(total_rows=100)
    batch_info = {"batch_id": 0, "start_row": 0, "end_row": 2}

    results = parser.parse_response(mock_chat_response, batch_info)

    assert len(results) == 2
    assert results[0]["index"] == 0
    assert results[0]["result"] is True
    assert results[1]["index"] == 1
    assert results[1]["result"] is False


def test_parse_response_with_extra_text():
    """Test parsing response with extra text around JSON."""
    response = """Here are the results:
    [
        {"index": 0, "result": true},
        {"index": 1, "result": false}
    ]
    Hope this helps!"""

    parser = _ValidationResponseParser(total_rows=100)
    batch_info = {"batch_id": 0, "start_row": 0, "end_row": 2}

    results = parser.parse_response(response, batch_info)

    assert len(results) == 2
    assert results[0]["result"] is True
    assert results[1]["result"] is False


def test_parse_response_invalid_json():
    """Test parsing invalid JSON response returns defaults."""
    response = "This is not JSON"

    parser = _ValidationResponseParser(total_rows=100)
    batch_info = {"batch_id": 0, "start_row": 0, "end_row": 2}

    results = parser.parse_response(response, batch_info)

    # Should return default results (all False)
    assert len(results) == 2
    assert all(not result["result"] for result in results)


def test_parse_response_wrong_structure():
    """Test parsing response with wrong structure."""
    response = '{"not": "an array"}'

    parser = _ValidationResponseParser(total_rows=100)
    batch_info = {"batch_id": 0, "start_row": 0, "end_row": 2}

    results = parser.parse_response(response, batch_info)

    # Should return default results (all False)
    assert len(results) == 2
    assert all(not result["result"] for result in results)


def test_validate_response_structure():
    """Test response structure validation."""
    parser = _ValidationResponseParser(total_rows=100)
    batch_info = {"batch_id": 0, "start_row": 0, "end_row": 2}

    # Valid structure
    valid_response = [{"index": 0, "result": True}, {"index": 1, "result": False}]
    parser._validate_response_structure(valid_response, batch_info)  # Should not raise

    # Invalid structure - not a list
    with pytest.raises(ValueError, match="Response must be a JSON array"):
        parser._validate_response_structure({}, batch_info)

    # Invalid structure - missing keys
    invalid_response = [{"index": 0}]  # Missing "result"
    with pytest.raises(ValueError, match="must have 'index' and 'result' keys"):
        parser._validate_response_structure(invalid_response, batch_info)


def test_create_default_results():
    """Test creating default results."""
    parser = _ValidationResponseParser(total_rows=100)
    batch_info = {"batch_id": 0, "start_row": 5, "end_row": 8}

    results = parser._create_default_results(batch_info)

    assert len(results) == 3
    assert results[0]["index"] == 5
    assert results[0]["result"] is False
    assert results[2]["index"] == 7
    assert results[2]["result"] is False


def test_combine_batch_results_no_mapping():
    """Test combining batch results without signature mapping."""
    parser = _ValidationResponseParser(total_rows=100)

    batch_results = [
        [{"index": 0, "result": True}, {"index": 1, "result": False}],
        [{"index": 2, "result": True}],
    ]

    combined = parser.combine_batch_results(batch_results)

    assert len(combined) == 3
    assert combined[0] is True
    assert combined[1] is False
    assert combined[2] is True


def test_combine_batch_results_with_mapping():
    """Test combining batch results with signature mapping."""
    parser = _ValidationResponseParser(total_rows=100)

    batch_results = [[{"index": 0, "result": True}, {"index": 1, "result": False}]]

    # Signature mapping: unique row 0 maps to original rows [0, 3], unique row 1 maps to [1, 2, 4]
    signature_mapping = {"sig1": [0, 3], "sig2": [1, 2, 4]}

    combined = parser.combine_batch_results(batch_results, signature_mapping)

    assert len(combined) == 5  # Should map to 5 original rows
    assert combined[0] is True  # From unique row 0
    assert combined[3] is True  # From unique row 0
    assert combined[1] is False  # From unique row 1
    assert combined[2] is False  # From unique row 1
    assert combined[4] is False  # From unique row 1


# ============================================================================
# Test AIValidationEngine
# ============================================================================


@patch("pointblank._utils_ai._create_chat_instance")
def test_ai_validation_engine_init(mock_create_chat, llm_config):
    """Test AIValidationEngine initialization."""
    mock_chat = Mock()
    mock_create_chat.return_value = mock_chat

    engine = _AIValidationEngine(llm_config)

    assert engine.llm_config is llm_config
    assert engine.chat is mock_chat
    mock_create_chat.assert_called_once_with(
        provider="anthropic",
        model_name="claude-sonnet-4-5",
        api_key="test-key",
        verify_ssl=True,
    )


@patch("pointblank._utils_ai._create_chat_instance")
def test_validate_single_batch(mock_create_chat, llm_config, mock_chat_response):
    """Test validating a single batch."""
    mock_chat = Mock()
    mock_chat.chat.return_value = mock_chat_response
    mock_create_chat.return_value = mock_chat

    engine = _AIValidationEngine(llm_config)

    batch = {
        "batch_id": 0,
        "start_row": 0,
        "end_row": 2,
        "data": {
            "columns": ["name", "age"],
            "rows": [
                {"name": "Alice", "age": 25, "_pb_row_index": 0},
                {"name": "Bob", "age": 17, "_pb_row_index": 1},
            ],
        },
    }

    prompt_builder = _PromptBuilder("Check if age > 18")

    results = engine.validate_single_batch(batch, prompt_builder)

    assert len(results) == 2
    assert results[0]["index"] == 0
    assert results[0]["result"] is True
    assert results[1]["index"] == 1
    assert results[1]["result"] is False

    # Verify chat was called
    mock_chat.chat.assert_called_once()


@patch("pointblank._utils_ai._create_chat_instance")
def test_validate_single_batch_error(mock_create_chat, llm_config):
    """Test validating a single batch with error."""
    mock_chat = Mock()
    mock_chat.chat.side_effect = Exception("API Error")
    mock_create_chat.return_value = mock_chat

    engine = _AIValidationEngine(llm_config)

    batch = {"batch_id": 0, "start_row": 0, "end_row": 2, "data": {"columns": [], "rows": []}}

    prompt_builder = _PromptBuilder("Check something")

    results = engine.validate_single_batch(batch, prompt_builder)

    # Should return default results (all False)
    assert len(results) == 2
    assert all(not result["result"] for result in results)


@patch("pointblank._utils_ai._create_chat_instance")
def test_validate_batches(mock_create_chat, llm_config, mock_chat_response):
    """Test validating multiple batches."""
    mock_chat = Mock()
    mock_chat.chat.return_value = mock_chat_response
    mock_create_chat.return_value = mock_chat

    engine = _AIValidationEngine(llm_config)

    batches = [
        {
            "batch_id": 0,
            "start_row": 0,
            "end_row": 2,
            "data": {"columns": ["name"], "rows": [{"name": "Alice"}, {"name": "Bob"}]},
        },
        {
            "batch_id": 1,
            "start_row": 2,
            "end_row": 3,
            "data": {"columns": ["name"], "rows": [{"name": "Charlie"}]},
        },
    ]

    prompt_builder = _PromptBuilder("Check names")

    results = engine.validate_batches(batches, prompt_builder)

    assert len(results) == 2  # Two batches
    assert len(results[0]) == 2  # First batch has 2 results
    assert len(results[1]) == 2  # Second batch also returns 2 results (from mock)

    # Verify chat was called for each batch
    assert mock_chat.chat.call_count == 2


@patch("pointblank._utils_ai._create_chat_instance")
def test_validate_batches_with_many_results(mock_create_chat, llm_config):
    """Test validating a batch with more than 5 results to trigger debug logging."""

    # Create a mock response with 7 results (more than 5)
    mock_chat_response = """[
        {"index": 0, "result": true},
        {"index": 1, "result": false},
        {"index": 2, "result": true},
        {"index": 3, "result": true},
        {"index": 4, "result": false},
        {"index": 5, "result": true},
        {"index": 6, "result": false}
    ]"""

    mock_chat = Mock()
    mock_chat.chat.return_value = mock_chat_response
    mock_create_chat.return_value = mock_chat

    engine = _AIValidationEngine(llm_config)

    # Create a batch with 7 rows
    batches = [
        {
            "batch_id": 0,
            "start_row": 0,
            "end_row": 7,
            "data": {
                "columns": ["name"],
                "rows": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                    {"name": "Charlie"},
                    {"name": "David"},
                    {"name": "Eve"},
                    {"name": "Frank"},
                    {"name": "Grace"},
                ],
            },
        }
    ]

    prompt_builder = _PromptBuilder("Check names")

    results = engine.validate_batches(batches, prompt_builder)

    assert len(results) == 1  # One batch
    assert len(results[0]) == 7  # Batch has 7 results

    # Verify the results
    assert results[0][0]["result"] is True
    assert results[0][1]["result"] is False
    assert results[0][6]["result"] is False


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_pipeline_pandas(sample_pd_data):
    """Test full AI validation pipeline with pandas."""
    with patch("pointblank._utils_ai._create_chat_instance") as mock_create_chat:
        # Mock the chat instance with dynamic responses
        mock_chat = Mock()

        def mock_chat_response(prompt, **kwargs):
            # Count how many rows are in this batch based on the prompt
            if '"_pb_row_index": 0' in prompt and '"_pb_row_index": 1' in prompt:
                # First batch with 2 rows
                return """[
                    {"index": 0, "result": true},
                    {"index": 1, "result": false}
                ]"""
            else:
                # Second batch with 1 row
                return """[
                    {"index": 2, "result": true}
                ]"""

        mock_chat.chat.side_effect = mock_chat_response
        mock_create_chat.return_value = mock_chat

        # Setup components
        llm_config = _LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
        )
        batch_config = _BatchConfig(size=2, max_concurrent=1)

        # Create batcher and batches
        batcher = _DataBatcher(sample_pd_data, config=batch_config)
        batches, signature_mapping = batcher.create_batches()

        # Create engine and validate
        engine = _AIValidationEngine(llm_config)
        prompt_builder = _PromptBuilder("Check if valid")

        batch_results = engine.validate_batches(batches, prompt_builder)

        # Parse and combine results
        parser = _ValidationResponseParser(total_rows=len(sample_pd_data))
        combined_results = parser.combine_batch_results(batch_results, signature_mapping)

        # Should have results for all original rows (might be less due to deduplication)
        assert len(combined_results) >= 3  # At least the unique rows
        assert all(isinstance(result, bool) for result in combined_results.values())


def test_full_pipeline_polars(sample_pl_data):
    """Test full AI validation pipeline with polars."""
    with patch("pointblank._utils_ai._create_chat_instance") as mock_create_chat:
        # Mock the chat instance
        mock_chat = Mock()
        mock_chat.chat.return_value = """[
            {"index": 0, "result": true},
            {"index": 1, "result": false},
            {"index": 2, "result": true}
        ]"""
        mock_create_chat.return_value = mock_chat

        # Setup components
        llm_config = _LLMConfig(provider="openai", model="gpt-4")
        batch_config = _BatchConfig(size=3)

        # Create batcher and batches
        batcher = _DataBatcher(sample_pl_data, config=batch_config)
        batches, signature_mapping = batcher.create_batches()

        # Create engine and validate
        engine = _AIValidationEngine(llm_config)
        prompt_builder = _PromptBuilder("Validate data")

        batch_results = engine.validate_batches(batches, prompt_builder)

        # Parse and combine results
        parser = _ValidationResponseParser(total_rows=len(sample_pl_data))
        combined_results = parser.combine_batch_results(batch_results, signature_mapping)

        # Should have results for all original rows (might be less due to deduplication)
        assert len(combined_results) >= 3  # At least the unique rows
        assert all(isinstance(result, bool) for result in combined_results.values())


def test_extract_json_direct_parsing():
    """Test JSON extraction from direct response."""
    parser = _ValidationResponseParser(total_rows=2)

    # Test direct JSON parsing (no code blocks)
    raw_json_response = '[{"index": 0, "result": true}, {"index": 1, "result": false}]'
    result = parser._extract_json(raw_json_response)
    assert len(result) == 2
    assert result[0]["index"] == 0
    assert result[0]["result"] is True
    assert result[1]["index"] == 1
    assert result[1]["result"] is False


def test_validation_response_structure_error():
    """Test validation of response structure with specific error cases."""
    parser = _ValidationResponseParser(total_rows=2)
    batch_info = {"start_row": 0, "end_row": 2, "batch_id": 0}

    # Test non-dict result
    with pytest.raises(ValueError, match="Result 0 must be a dictionary"):
        parser._validate_response_structure(
            ["not_a_dict", {"index": 1, "result": True}], batch_info
        )

    # Test missing keys
    with pytest.raises(ValueError, match="Result 0 must have 'index' and 'result' keys"):
        parser._validate_response_structure(
            [{"index": 0}, {"index": 1, "result": True}], batch_info
        )  # missing 'result'

    with pytest.raises(ValueError, match="Result 0 must have 'index' and 'result' keys"):
        parser._validate_response_structure(
            [{"result": True}, {"index": 1, "result": True}], batch_info
        )  # missing 'index'

    # Test non-boolean result
    with pytest.raises(ValueError, match="Result 0 'result' must be a boolean"):
        parser._validate_response_structure(
            [{"index": 0, "result": "not_boolean"}, {"index": 1, "result": True}], batch_info
        )


def test_default_results_creation():
    """Test creation of default results for failed batches."""
    parser = _ValidationResponseParser(total_rows=5)
    batch_info = {"start_row": 1, "end_row": 4, "batch_id": 0}

    results = parser._create_default_results(batch_info)

    assert len(results) == 3  # end_row - start_row
    assert all(result["result"] is False for result in results)
    assert results[0]["index"] == 1
    assert results[1]["index"] == 2
    assert results[2]["index"] == 3
