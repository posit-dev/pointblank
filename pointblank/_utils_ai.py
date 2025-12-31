from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import narwhals as nw

from pointblank._constants import MODEL_PROVIDERS

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Configuration and Chat Interface
# ============================================================================


@dataclass
class _LLMConfig:
    """Configuration for LLM provider.

    Parameters
    ----------
    provider
        LLM provider name (e.g., 'anthropic', 'openai', 'ollama', 'bedrock').
    model
        Model name (e.g., 'claude-sonnet-4-5', 'gpt-4').
    api_key
        API key for the provider. If None, will be read from environment.
    verify_ssl
        Whether to verify SSL certificates when making requests. Defaults to True.
    """

    provider: str
    model: str
    api_key: Optional[str] = None
    verify_ssl: bool = True


def _create_chat_instance(
    provider: str, model_name: str, api_key: Optional[str] = None, verify_ssl: bool = True
):
    """
    Create a chatlas chat instance for the specified provider.

    Parameters
    ----------
    provider
        The provider name (e.g., 'anthropic', 'openai', 'ollama', 'bedrock').
    model_name
        The model name for the provider.
    api_key
        Optional API key. If None, will be read from environment.
    verify_ssl
        Whether to verify SSL certificates when making requests. Defaults to True.

    Returns
    -------
    Chat instance from chatlas.
    """
    # Check if chatlas is installed
    try:
        import chatlas  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError(
            "The `chatlas` package is required for AI validation. "
            "Please install it using `pip install chatlas`."
        )

    # Validate provider
    if provider not in MODEL_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' is not supported. "
            f"Supported providers: {', '.join(MODEL_PROVIDERS)}"
        )

    # System prompt with role definition and instructions
    system_prompt = """You are a data validation assistant. Your task is to analyze rows of data and determine if they meet the specified validation criteria.

INSTRUCTIONS:
- Analyze each row in the provided data
- For each row, determine if it meets the validation criteria (True) or not (False)
- Return ONLY a JSON array with validation results
- Each result should have: {"index": <row_index>, "result": <true_or_false>}
- Do not include any explanatory text, only the JSON array
- The row_index should match the "_pb_row_index" field from the input data

EXAMPLE OUTPUT FORMAT:
[
  {"index": 0, "result": true},
  {"index": 1, "result": false},
  {"index": 2, "result": true}
]"""

    # Create httpx client with SSL verification settings
    try:
        import httpx  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "The `httpx` package is required for SSL configuration. "
            "Please install it using `pip install httpx`."
        )

    http_client = httpx.AsyncClient(verify=verify_ssl)

    # Create provider-specific chat instance
    if provider == "anthropic":  # pragma: no cover
        # Check that the anthropic package is installed
        try:
            import anthropic  # noqa  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "The `anthropic` package is required to use AI validation with "
                "`anthropic`. Please install it using `pip install anthropic`."
            )

        from chatlas import ChatAnthropic

        chat = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
            kwargs={"http_client": http_client},
        )

    elif provider == "openai":  # pragma: no cover
        # Check that the openai package is installed
        try:
            import openai  # noqa
        except ImportError:
            raise ImportError(
                "The `openai` package is required to use AI validation with "
                "`openai`. Please install it using `pip install openai`."
            )

        from chatlas import ChatOpenAI

        chat = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
            kwargs={"http_client": http_client},
        )

    elif provider == "ollama":  # pragma: no cover
        # Check that the openai package is installed (required for Ollama)
        try:
            import openai  # noqa
        except ImportError:
            raise ImportError(
                "The `openai` package is required to use AI validation with "
                "`ollama`. Please install it using `pip install openai`."
            )

        from chatlas import ChatOllama

        chat = ChatOllama(
            model=model_name,
            system_prompt=system_prompt,
            kwargs={"http_client": http_client},
        )

    elif provider == "bedrock":  # pragma: no cover
        from chatlas import ChatBedrockAnthropic

        chat = ChatBedrockAnthropic(
            model=model_name,
            system_prompt=system_prompt,
            kwargs={"http_client": http_client},
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return chat


# ============================================================================
# Data Batching and Optimization
# ============================================================================


@dataclass
class _BatchConfig:
    """Configuration for AI validation batching.

    Parameters
    ----------
    size
        Batch size for processing rows.
    max_concurrent
        Maximum number of concurrent LLM requests.
    """

    size: int = 1000
    max_concurrent: int = 3


class _DataBatcher:
    """Optimized batching of data for AI validation with row signature memoization."""

    def __init__(
        self,
        data: Any,
        columns: Optional[List[str]] = None,
        config: Optional[_BatchConfig] = None,
    ):
        """
        Initialize the optimized data batcher.

        Parameters
        ----------
        data
            The data frame to batch.
        columns
            Optional list of columns to include in batches. If None, all columns are included.
        config
            Optional batch configuration. If None, default configuration is used.
        """
        self.data = data
        self.columns = columns
        self.config = config or _BatchConfig()
        self._validate_data()

        # Memoization structures
        self.unique_rows_table = None
        self.signature_to_original_indices = {}
        self.reduction_stats = {}

    def _validate_data(self) -> None:
        """Validate that the data is supported."""
        if not hasattr(self.data, "shape"):
            raise ValueError("Data must have a 'shape' attribute")

        # Get data with narwhals for compatibility
        self._nw_data = nw.from_native(self.data)

        if self.columns:
            # Validate that specified columns exist
            available_columns = self._nw_data.columns
            missing_columns = set(self.columns) - set(available_columns)
            if missing_columns:
                raise ValueError(f"Columns not found in data: {missing_columns}")

    def _create_row_signature(self, row_dict: Dict[str, Any]) -> str:
        """
        Create a unique signature for a row based on selected columns.

        Parameters
        ----------
        row_dict
            Dictionary representing a row.

        Returns
        -------
        str
            Unique signature for the row.
        """
        # Create deterministic signature from sorted column values
        signature_data = {k: v for k, v in row_dict.items() if k != "_pb_row_index"}
        signature_str = json.dumps(signature_data, sort_keys=True, default=str)
        return hashlib.md5(signature_str.encode()).hexdigest()

    def _build_unique_rows_table(self) -> Tuple[Any, Dict[str, List[int]]]:
        """
        Build unique rows table and mapping back to original indices.

        Returns
        -------
        Tuple[Any, Dict[str, List[int]]]
            Unique rows table and signature-to-indices mapping.
        """
        nw_data = self._nw_data

        # Select columns if specified
        if self.columns:
            nw_data = nw_data.select(self.columns)

        # Convert to native for easier manipulation
        native_data = nw.to_native(nw_data)

        # Get all rows as dictionaries
        if hasattr(native_data, "to_dicts"):
            # Polars DataFrame
            all_rows = native_data.to_dicts()
        elif hasattr(native_data, "to_dict"):
            # Pandas DataFrame
            all_rows = native_data.to_dict("records")
        else:  # pragma: no cover
            # Fallback: manual conversion
            all_rows = []
            columns = nw_data.columns
            for i in range(len(native_data)):
                row_dict = {}
                for col in columns:
                    row_dict[col] = (
                        native_data[col].iloc[i]
                        if hasattr(native_data[col], "iloc")
                        else native_data[col][i]
                    )
                all_rows.append(row_dict)

        # Build signature mapping
        signature_to_indices = {}
        unique_rows = []
        unique_signatures = set()

        for original_idx, row_dict in enumerate(all_rows):
            signature = self._create_row_signature(row_dict)

            if signature not in signature_to_indices:
                signature_to_indices[signature] = []

            signature_to_indices[signature].append(original_idx)

            # Add to unique rows if not seen before
            if signature not in unique_signatures:
                unique_signatures.add(signature)
                # Add signature tracking for later joining
                row_dict["_pb_signature"] = signature
                unique_rows.append(row_dict)

        # Convert unique rows back to dataframe
        if unique_rows:
            if hasattr(native_data, "with_columns"):  # Polars
                import polars as pl

                unique_df = pl.DataFrame(unique_rows)
            elif hasattr(native_data, "assign"):  # Pandas
                import pandas as pd

                unique_df = pd.DataFrame(unique_rows)
            else:  # pragma: no cover
                # This is tricky for generic case, but let's try
                unique_df = unique_rows  # Fallback to list of dicts
        else:  # pragma: no cover
            unique_df = native_data.head(0)  # Empty dataframe with same structure

        # Store reduction stats
        original_count = len(all_rows)
        unique_count = len(unique_rows)
        reduction_pct = (1 - unique_count / original_count) * 100 if original_count > 0 else 0

        self.reduction_stats = {
            "original_rows": original_count,
            "unique_rows": unique_count,
            "reduction_percentage": reduction_pct,
        }

        logger.info(
            f"Row signature optimization: {original_count} → {unique_count} rows ({reduction_pct:.1f}% reduction)"
        )

        return unique_df, signature_to_indices

    def create_batches(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        """
        Create optimized batches using unique row signatures.

        Returns
        -------
        Tuple[List[Dict[str, Any]], Dict[str, List[int]]]
            Batches for unique rows and signature-to-indices mapping.
        """
        # Build unique rows table and signature mapping
        unique_rows_table, signature_to_indices = self._build_unique_rows_table()
        self.unique_rows_table = unique_rows_table
        self.signature_to_original_indices = signature_to_indices

        # Create batches from unique rows table
        if hasattr(unique_rows_table, "shape"):
            total_rows = unique_rows_table.shape[0]
        else:  # pragma: no cover
            total_rows = len(unique_rows_table)

        batches = []
        batch_id = 0

        # Convert to narwhals if needed
        if not hasattr(unique_rows_table, "columns"):  # pragma: no cover
            nw_unique = nw.from_native(unique_rows_table)
        else:
            nw_unique = unique_rows_table

        for start_row in range(0, total_rows, self.config.size):
            end_row = min(start_row + self.config.size, total_rows)

            # Get the batch data
            if hasattr(nw_unique, "__getitem__"):
                batch_data = nw_unique[start_row:end_row]
            else:  # pragma: no cover
                # Fallback for list of dicts
                batch_data = unique_rows_table[start_row:end_row]

            # Convert to JSON-serializable format
            batch_json = self._convert_batch_to_json(batch_data, start_row)

            batches.append(
                {
                    "batch_id": batch_id,
                    "start_row": start_row,
                    "end_row": end_row,
                    "data": batch_json,
                }
            )

            batch_id += 1

        logger.info(f"Created {len(batches)} batches for {total_rows} unique rows")
        return batches, signature_to_indices

    def _convert_batch_to_json(self, batch_data, start_row: int) -> Dict[str, Any]:
        """
        Convert a batch of unique data to JSON format for LLM consumption.
        """
        # Handle different input types
        if isinstance(batch_data, list):
            # List of dictionaries
            rows = []
            columns = list(batch_data[0].keys()) if batch_data else []

            for i, row_dict in enumerate(batch_data):
                # Remove signature column from LLM input but keep for joining
                clean_row = {k: v for k, v in row_dict.items() if k != "_pb_signature"}
                clean_row["_pb_row_index"] = start_row + i
                rows.append(clean_row)

            # Remove signature from columns list
            columns = [col for col in columns if col != "_pb_signature"]

        else:
            # DataFrame-like object
            columns = [col for col in batch_data.columns if col != "_pb_signature"]
            rows = []

            # batch_data is already native format from slicing
            native_batch = batch_data

            # Handle different data frame types
            if hasattr(native_batch, "to_dicts"):
                # Polars DataFrame
                batch_dicts = native_batch.to_dicts()
            elif hasattr(native_batch, "to_dict"):
                # Pandas DataFrame
                batch_dicts = native_batch.to_dict("records")
            else:  # pragma: no cover
                # Fallback: manual conversion
                batch_dicts = []
                for i in range(len(native_batch)):
                    row_dict = {}
                    for col in columns:
                        row_dict[col] = (
                            native_batch[col].iloc[i]
                            if hasattr(native_batch[col], "iloc")
                            else native_batch[col][i]
                        )
                    batch_dicts.append(row_dict)

            # Clean up rows and add indices
            for i, row_dict in enumerate(batch_dicts):
                clean_row = {k: v for k, v in row_dict.items() if k != "_pb_signature"}
                clean_row["_pb_row_index"] = start_row + i
                rows.append(clean_row)

        return {
            "columns": columns,
            "rows": rows,
            "batch_info": {
                "start_row": start_row,
                "num_rows": len(rows),
                "columns_count": len(columns),
            },
        }

    def get_reduction_stats(self) -> Dict[str, Any]:
        """Get statistics about the row reduction optimization."""
        return self.reduction_stats.copy()


# ============================================================================
# Prompt Building
# ============================================================================


class _PromptBuilder:
    """
    Builds user messages for AI validation.

    Works in conjunction with the system prompt set on the chat instance.
    The system prompt contains role definition and general instructions,
    while this class builds user messages with specific validation criteria and data.
    """

    USER_MESSAGE_TEMPLATE = """VALIDATION CRITERIA:
{user_prompt}

DATA TO VALIDATE:
{data_json}"""

    def __init__(self, user_prompt: str):
        """
        Initialize the prompt builder.

        Parameters
        ----------
        user_prompt
            The user's validation prompt describing what to check.
        """
        self.user_prompt = user_prompt

    def build_prompt(self, batch_data: Dict[str, Any]) -> str:
        """
        Build a user message for a data batch.

        Parameters
        ----------
        batch_data
            The batch data dictionary from DataBatcher.

        Returns
        -------
        str
            The user message for the LLM.
        """
        data_json = json.dumps(batch_data, indent=2, default=str)

        return self.USER_MESSAGE_TEMPLATE.format(user_prompt=self.user_prompt, data_json=data_json)


# ============================================================================
# Response Parsing
# ============================================================================


class _ValidationResponseParser:
    """Parses AI validation responses."""

    def __init__(self, total_rows: int):
        """
        Initialize the response parser.

        Parameters
        ----------
        total_rows
            Total number of rows being validated.
        """
        self.total_rows = total_rows

    def parse_response(self, response: str, batch_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse an LLM response for a batch.

        Parameters
        ----------
        response
            The raw response from the LLM.
        batch_info
            Information about the batch being processed.

        Returns
        -------
        List[Dict[str, Any]]
            List of parsed results with 'index' and 'result' keys.
        """
        try:
            # Try to extract JSON from the response
            json_response = self._extract_json(response)

            # Validate the structure
            self._validate_response_structure(json_response, batch_info)

            return json_response

        except Exception as e:
            logger.error(
                f"Failed to parse response for batch {batch_info.get('batch_id', 'unknown')}: {e}"
            )
            logger.error(f"Raw response: {response}")

            # Return default results (all False) for this batch
            return self._create_default_results(batch_info)

    def _extract_json(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Clean up the response
        response = response.strip()

        # Look for JSON array patterns
        import re

        json_pattern = r"\[.*?\]"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            # Try to parse the first match
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:  # pragma: no cover
                # If that fails, try the raw response
                return json.loads(response)
        else:
            # Try to parse the raw response
            return json.loads(response)

    def _validate_response_structure(
        self, json_response: List[Dict[str, Any]], batch_info: Dict[str, Any]
    ) -> None:
        """Validate that the response has the correct structure."""
        if not isinstance(json_response, list):
            raise ValueError("Response must be a JSON array")

        expected_rows = batch_info["end_row"] - batch_info["start_row"]
        if len(json_response) != expected_rows:
            logger.warning(
                f"Expected {expected_rows} results, got {len(json_response)} for batch {batch_info.get('batch_id')}"
            )

        for i, result in enumerate(json_response):
            if not isinstance(result, dict):
                raise ValueError(f"Result {i} must be a dictionary")

            if "index" not in result or "result" not in result:
                raise ValueError(f"Result {i} must have 'index' and 'result' keys")

            if not isinstance(result["result"], bool):
                raise ValueError(f"Result {i} 'result' must be a boolean")

    def _create_default_results(self, batch_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create default results (all False) for a batch."""
        results = []
        for i in range(batch_info["start_row"], batch_info["end_row"]):
            results.append({"index": i, "result": False})
        return results

    def combine_batch_results(
        self,
        batch_results: List[List[Dict[str, Any]]],
        signature_mapping: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[int, bool]:
        """
        Combine results from multiple batches and project to original rows using signature mapping.

        Parameters
        ----------
        batch_results
            List of batch results from parse_response.
        signature_mapping
            Optional mapping from row signatures to original row indices for memoization.

        Returns
        -------
        Dict[int, bool]
            Dictionary mapping original row index to validation result.
        """
        logger.debug(f"🔀 Combining results from {len(batch_results)} batches")

        if signature_mapping:
            logger.debug(
                f"🎯 Using signature mapping optimization for {len(signature_mapping)} unique signatures"
            )

        # First, collect results from unique rows
        unique_results = {}
        total_processed = 0

        for batch_idx, batch_result in enumerate(batch_results):
            logger.debug(f"   Batch {batch_idx}: {len(batch_result)} results")

            for result in batch_result:
                index = result["index"]
                validation_result = result["result"]
                unique_results[index] = validation_result
                total_processed += 1

                # Log first few results
                if len(unique_results) <= 3 or total_processed <= 3:
                    logger.debug(f"     Unique row {index}: {validation_result}")

        # If no signature mapping, return unique results as-is (fallback to original behavior)
        if not signature_mapping:
            logger.debug("📊 No signature mapping - returning unique results")
            passed_count = sum(1 for v in unique_results.values() if v)
            failed_count = len(unique_results) - passed_count
            logger.debug(f"   - Final count: {passed_count} passed, {failed_count} failed")
            return unique_results

        # Project unique results back to all original rows using signature mapping
        combined_results = {}

        # We need to map from unique row indices back to signatures, then to original indices
        # This requires rebuilding the signatures from the unique rows
        # For now, let's assume the unique_results indices correspond to signature order
        signature_list = list(signature_mapping.keys())

        for unique_idx, validation_result in unique_results.items():
            if unique_idx < len(signature_list):
                signature = signature_list[unique_idx]
                original_indices = signature_mapping[signature]

                logger.debug(
                    f"   Projecting result {validation_result} from unique row {unique_idx} to {len(original_indices)} original rows"
                )

                # Project this result to all original rows with this signature
                for original_idx in original_indices:
                    combined_results[original_idx] = validation_result
            else:  # pragma: no cover
                logger.warning(f"Unique index {unique_idx} out of range for signature mapping")

        logger.debug("📊 Projected results summary:")
        logger.debug(f"   - Unique rows processed: {len(unique_results)}")
        logger.debug(f"   - Original rows mapped: {len(combined_results)}")
        logger.debug(
            f"   - Index range: {min(combined_results.keys()) if combined_results else 'N/A'} to {max(combined_results.keys()) if combined_results else 'N/A'}"
        )

        passed_count = sum(1 for v in combined_results.values() if v)
        failed_count = len(combined_results) - passed_count
        logger.debug(f"   - Final count: {passed_count} passed, {failed_count} failed")

        return combined_results


# ============================================================================
# AI Validation Engine
# ============================================================================


class _AIValidationEngine:
    """Main engine for AI-powered validation using chatlas."""

    def __init__(self, llm_config: _LLMConfig):
        """
        Initialize the AI validation engine.

        Parameters
        ----------
        llm_config
            Configuration for the LLM provider.
        """
        self.llm_config = llm_config
        self.chat = _create_chat_instance(
            provider=llm_config.provider,
            model_name=llm_config.model,
            api_key=llm_config.api_key,
            verify_ssl=llm_config.verify_ssl,
        )

    def validate_batches(
        self, batches: List[Dict[str, Any]], prompt_builder: Any, max_concurrent: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Validate multiple batches.

        Parameters
        ----------
        batches
            List of batch dictionaries from DataBatcher.
        prompt_builder
            PromptBuilder instance for generating prompts.
        max_concurrent
            Maximum number of concurrent requests (ignored for now with chatlas).

        Returns
        -------
        List[List[Dict[str, Any]]]
            List of batch results, each containing validation results.
        """

        def validate_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                # Debug: Log batch information
                logger.debug(f"🔍 Processing batch {batch['batch_id']}")
                logger.debug(f"   - Rows: {batch['start_row']} to {batch['end_row'] - 1}")
                logger.debug(
                    f"   - Data shape: {batch['data'].shape if hasattr(batch['data'], 'shape') else 'N/A'}"
                )

                # Build the prompt for this batch
                prompt = prompt_builder.build_prompt(batch["data"])

                # Debug: Log the prompt being sent to LLM
                logger.debug(f"📤 LLM Prompt for batch {batch['batch_id']}:")
                logger.debug("--- PROMPT START ---")
                logger.debug(prompt)
                logger.debug("--- PROMPT END ---")

                # Get response from LLM using chatlas (synchronous)
                response = str(self.chat.chat(prompt, stream=False, echo="none"))

                # Debug: Log the raw LLM response
                logger.debug(f"📥 LLM Response for batch {batch['batch_id']}:")
                logger.debug("--- RESPONSE START ---")
                logger.debug(response)
                logger.debug("--- RESPONSE END ---")

                # Parse the response
                parser = _ValidationResponseParser(total_rows=1000)  # This will be set properly
                results = parser.parse_response(response, batch)

                # Debug: Log parsed results  # pragma: no cover
                logger.debug(f"📊 Parsed results for batch {batch['batch_id']}:")
                for i, result in enumerate(results[:5]):  # Show first 5 results
                    logger.debug(f"   Row {result['index']}: {result['result']}")
                if len(results) > 5:
                    logger.debug(f"   ... and {len(results) - 5} more results")

                passed_count = sum(1 for r in results if r["result"])
                failed_count = len(results) - passed_count
                logger.debug(f"   Summary: {passed_count} passed, {failed_count} failed")

                logger.info(f"Successfully validated batch {batch['batch_id']}")
                return results

            except Exception as e:  # pragma: no cover
                logger.error(
                    f"Failed to validate batch {batch['batch_id']}: {e}"
                )  # pragma: no cover
                # Return default results (all False) for failed batches
                default_results = []  # pragma: no cover
                for i in range(batch["start_row"], batch["end_row"]):  # pragma: no cover
                    default_results.append({"index": i, "result": False})  # pragma: no cover
                return default_results  # pragma: no cover

        # Execute all batch validations sequentially (chatlas is synchronous)
        final_results = []
        for batch in batches:
            result = validate_batch(batch)
            final_results.append(result)

        return final_results

    def validate_single_batch(
        self, batch: Dict[str, Any], prompt_builder: Any
    ) -> List[Dict[str, Any]]:
        """
        Validate a single batch.

        Parameters
        ----------
        batch
            Batch dictionary from DataBatcher.
        prompt_builder
            PromptBuilder instance for generating prompts.

        Returns
        -------
        List[Dict[str, Any]]
            Validation results for the batch.
        """
        try:
            # Build the prompt for this batch
            prompt = prompt_builder.build_prompt(batch["data"])

            # Get response from LLM using chatlas (synchronous)
            response = str(self.chat.chat(prompt, stream=False, echo="none"))

            # Parse the response
            parser = _ValidationResponseParser(total_rows=1000)  # This will be set properly
            results = parser.parse_response(response, batch)

            logger.info(f"Successfully validated batch {batch['batch_id']}")
            return results

        except Exception as e:
            logger.error(f"Failed to validate batch {batch['batch_id']}: {e}")
            # Return default results (all False) for failed batch
            default_results = []
            for i in range(batch["start_row"], batch["end_row"]):
                default_results.append({"index": i, "result": False})
            return default_results
