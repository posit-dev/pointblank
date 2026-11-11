from __future__ import annotations

import ast
import datetime
import decimal
import json
import math
import operator
import re
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Optional, Union
from zoneinfo import ZoneInfo

import yaml

from pointblank._agg import is_valid_agg
from pointblank._utils import _is_lib_present
from pointblank.missing import MissingSpec, _missing_spec_to_code
from pointblank.thresholds import Actions
from pointblank.validate import Validate, load_dataset

if TYPE_CHECKING:
    from pointblank.ablation import Ablation


def _missing_spec_from_dict(spec_def: dict) -> MissingSpec:
    """Build a `MissingSpec` from a YAML mapping."""
    if not isinstance(spec_def, dict):
        raise YAMLValidationError(
            f"A missing spec must be a mapping, got {type(spec_def).__name__}."
        )
    return MissingSpec(
        reasons=spec_def.get("reasons", {}),
        categories=spec_def.get("categories"),
        null_is_missing=spec_def.get("null_is_missing", True),
        null_reason=spec_def.get("null_reason", "unknown"),
        description=spec_def.get("description"),
    )


# A column reference written directly as a step value (e.g., `value: pb.col("b")`)
_COLUMN_REF_PATTERN = re.compile(r"\s*pb\.(col|ref)\(.*\)\s*", re.DOTALL)


class YAMLValidationError(Exception):
    """Exception raised for YAML validation errors."""

    pass


# How Python code in a YAML config is handled: "deny" raises an error (the default, since the
# code can do anything the Python session can), "allow" runs it (`allow_python=True`), and
# "source" keeps it as source text without running it (used by `yaml_to_python()`)
_PythonMode = Literal["deny", "allow", "source"]

# The `pb` functions that YAML can use without `allow_python=True`: they only build column
# references, column selectors, segment groups, and `active=` checks from literal arguments, or
# load one of the bundled datasets
_DECLARATIVE_PB_FUNCTIONS = frozenset(
    {
        "load_dataset",
        "col",
        "ref",
        "seg_group",
        "has_columns",
        "has_rows",
        "starts_with",
        "ends_with",
        "contains",
        "matches",
        "everything",
        "first_n",
        "last_n",
    }
)

# Other calls that YAML can use without `allow_python=True`: they build values that YAML has no
# literal for, such as `Decimal("2.2")`, `datetime.time(9, 30)`, or a datetime with
# `tzinfo=ZoneInfo("America/New_York")` (as written by `to_yaml()`)
_DECLARATIVE_VALUE_CALLS = {
    "Decimal": decimal.Decimal,
    "ZoneInfo": ZoneInfo,
    "datetime.date": datetime.date,
    "datetime.time": datetime.time,
    "datetime.datetime": datetime.datetime,
    "datetime.timedelta": datetime.timedelta,
}

# Operators that combine column selectors (e.g., `pb.starts_with("a") & pb.ends_with("z")`) or
# negate a number
_DECLARATIVE_BINARY_OPS = {
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.Sub: operator.sub,
}
_DECLARATIVE_UNARY_OPS = {
    ast.Invert: operator.invert,
    ast.USub: operator.neg,
}


class _NotDeclarative(Exception):
    """Raised when YAML code is outside the subset that runs without `allow_python=True`."""


class _PythonSource:
    """Python code from a YAML config, kept as source text instead of being run."""

    def __init__(self, code: str) -> None:
        self.code = code.strip()

    def __repr__(self) -> str:
        return self.code

    __str__ = __repr__


def _eval_declarative_node(node: ast.AST) -> Any:
    """Evaluate an expression node from the declarative subset, or raise `_NotDeclarative`."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_eval_declarative_node(item) for item in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {
            _eval_declarative_node(k): _eval_declarative_node(v)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp) and type(node.op) in _DECLARATIVE_BINARY_OPS:
        return _DECLARATIVE_BINARY_OPS[type(node.op)](
            _eval_declarative_node(node.left), _eval_declarative_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _DECLARATIVE_UNARY_OPS:
        return _DECLARATIVE_UNARY_OPS[type(node.op)](_eval_declarative_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and not any(isinstance(arg, ast.Starred) for arg in node.args)
        and all(kw.arg is not None for kw in node.keywords)
    ):
        func = _declarative_function(node.func)
        args = [_eval_declarative_node(arg) for arg in node.args]
        kwargs = {kw.arg: _eval_declarative_node(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)
    raise _NotDeclarative


def _declarative_function(node: ast.AST) -> Any:
    """Get the function a declarative call refers to, or raise `_NotDeclarative`."""
    import pointblank as pb

    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        name = f"{node.value.id}.{node.attr}"
    else:
        raise _NotDeclarative

    prefix, _, attr = name.rpartition(".")
    if prefix == "pb" and attr in _DECLARATIVE_PB_FUNCTIONS:
        return getattr(pb, attr)
    if name in _DECLARATIVE_VALUE_CALLS:
        return _DECLARATIVE_VALUE_CALLS[name]
    raise _NotDeclarative


def _eval_declarative_code(code: str) -> Any:
    """Evaluate code that is in the declarative subset, or raise `_NotDeclarative`.

    The subset is a single expression made of literals, calls to the `pb` functions in
    `_DECLARATIVE_PB_FUNCTIONS` (e.g., `pb.col("b")` or `pb.has_columns("a", "b")`), and calls
    to the constructors in `_DECLARATIVE_VALUE_CALLS` (e.g., `Decimal("2.2")`). It's evaluated by
    walking the syntax tree, so nothing else can be reached.
    """
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError:
        raise _NotDeclarative
    return _eval_declarative_node(tree.body)


def _python_not_allowed_error(code: str) -> YAMLValidationError:
    snippet = " ".join(code.split())
    if len(snippet) > 80:
        snippet = snippet[:77] + "..."
    return YAMLValidationError(
        f"This YAML config contains Python code, which isn't run unless you allow it: "
        f"`{snippet}`. Python code in a YAML config can do anything your Python session can "
        "(read and write files, run commands), so only allow it for configs you trust: use "
        "`yaml_interrogate(..., allow_python=True)` or `pb run --allow-python`."
    )


def _eval_python_code(
    code: str,
    namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    python_mode: _PythonMode = "deny",
) -> Any:
    """Evaluate Python code embedded in a YAML configuration.

    Code in the declarative subset (see `_eval_declarative_code()`) is always evaluated. Other
    code raises an error with `python_mode="deny"`, is kept as `_PythonSource` text with
    `python_mode="source"`, and is run as ordinary Python with `python_mode="allow"`. Running it
    is not sandboxed: the code has the full access of the Python session, with `pb`, `pl`, `pd`,
    `Path`, `datetime`, `Decimal`, `ZoneInfo`, and any `namespaces=` modules already imported.

    Parameters
    ----------
    code
        The Python code to evaluate.
    namespaces
        Modules to make available to the code, as a mapping of aliases to module names or a list
        of module names.
    python_mode
        How to handle code outside the declarative subset: `"deny"`, `"allow"`, or `"source"`.

    Returns
    -------
    Any
        The result of evaluating the Python code (its last expression, for multi-line code).

    Raises
    ------
    YAMLValidationError
        If the code isn't allowed to run, or running it fails.
    """
    if not isinstance(code, str):
        raise YAMLValidationError(f"Python code in YAML must be a string, got: {code!r}")

    if python_mode == "source":
        return _PythonSource(code)

    try:
        return _eval_declarative_code(code)
    except _NotDeclarative:
        pass
    except Exception as e:
        raise YAMLValidationError(f"Error evaluating '{code.strip()}': {e}")

    if python_mode != "allow":
        raise _python_not_allowed_error(code)

    import pointblank as pb

    namespace: dict[str, Any] = {
        "Path": Path,
        "pb": pb,
        "datetime": datetime,
        "Decimal": decimal.Decimal,
        "ZoneInfo": ZoneInfo,
    }

    if _is_lib_present("polars"):
        import polars as pl

        namespace["pl"] = pl

    if _is_lib_present("pandas"):
        import pandas as pd

        namespace["pd"] = pd

    if namespaces:
        for alias, module_name in (
            namespaces.items() if isinstance(namespaces, dict) else ((m, m) for m in namespaces)
        ):
            try:
                namespace[alias] = import_module(str(module_name))
            except ImportError as e:  # TODO: This is basically redundant, remove?
                raise ImportError(
                    f"Could not import requested namespace '{module_name}': {e}"
                ) from e

    try:
        # First try to parse as expression for simple cases
        try:
            parsed = ast.parse(code, mode="eval")
            return eval(compile(parsed, "<string>", "eval"), namespace)
        except SyntaxError:
            # If that fails, try as a statement (for more complex code)
            # For multi-statement code, we need to capture the result of the last expression
            parsed = ast.parse(code, mode="exec")

            # Check if the last node is an expression
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                # Split the last expression from the statements
                statements = parsed.body[:-1]
                last_expr = parsed.body[-1].value

                # Execute the statements first
                if statements:
                    statements_module = ast.Module(body=statements, type_ignores=[])
                    exec(compile(statements_module, "<string>", "exec"), namespace)

                # Then evaluate the last expression and return its value
                expr_module = ast.Expression(body=last_expr)
                return eval(compile(expr_module, "<string>", "eval"), namespace)
            else:
                # No expression at the end, just execute statements
                exec(compile(parsed, "<string>", "exec"), namespace)
                return None

    except Exception as e:
        raise YAMLValidationError(f"Error executing Python code '{code}': {e}")


def _process_python_expressions(
    value: Any,
    namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    python_mode: _PythonMode = "deny",
) -> Any:
    """Process Python code snippets embedded in YAML values.

    Code is evaluated with `_eval_python_code()`, so `python_mode=` decides whether code outside
    the declarative subset raises an error, runs, or is kept as source text.

    This function supports the python: block syntax for embedding Python code:

    python: |
      import polars as pl
      pl.scan_csv("data.csv").head(10)

    Note: col_vals_expr() also supports a shortcut syntax where the expr parameter
    can be written directly without the python: wrapper: +

    col_vals_expr:
      expr: |
        pl.col("column") > 0

    Parameters
    ----------
    value
        The value to process, can be any YAML type.

    Returns
    -------
    Any
        The processed value with Python expressions evaluated.

    Examples
    --------
    # Returns the result of the Python expression
    >>> import polars as pl
    >>> expr = _process_python_expressions(
    ...     {"python": "pl.scan_csv('data.csv').head(10)"}, python_mode="allow"
    ... )
    >>> assert isinstance(expr, pl.LazyFrame)

    # Returns the result of multiline Python code
    >>> expr = _process_python_expressions(
    ...     {"python": "import polars as pl\\npl.scan_csv('data.csv')"}, python_mode="allow"
    ... )
    >>> assert isinstance(expr, pl.LazyFrame)
    """
    if isinstance(value, dict):
        # Handle python: block syntax
        if "python" in value and len(value) == 1:
            code = value["python"]
            return _eval_python_code(code, namespaces=namespaces, python_mode=python_mode)

        # Recursively process dictionary values
        return {
            k: _process_python_expressions(v, namespaces=namespaces, python_mode=python_mode)
            for k, v in value.items()
        }

    elif isinstance(value, list):
        # Recursively process list items
        return [
            _process_python_expressions(item, namespaces=namespaces, python_mode=python_mode)
            for item in value
        ]

    else:
        # Return primitive types unchanged
        return value


class YAMLValidator:
    """Validates YAML configuration and converts to Validate objects."""

    # Map YAML method names to Python method names
    validation_method_map = {
        "col_exists": "col_exists",
        "col_vals_gt": "col_vals_gt",
        "col_vals_ge": "col_vals_ge",
        "col_vals_lt": "col_vals_lt",
        "col_vals_le": "col_vals_le",
        "col_vals_eq": "col_vals_eq",
        "col_vals_ne": "col_vals_ne",
        "col_vals_between": "col_vals_between",
        "col_vals_outside": "col_vals_outside",
        "col_vals_regex": "col_vals_regex",
        "col_vals_in_set": "col_vals_in_set",
        "col_vals_in_table": "col_vals_in_table",
        "col_vals_not_in_set": "col_vals_not_in_set",
        "col_vals_not_null": "col_vals_not_null",
        "col_vals_null": "col_vals_null",
        "col_vals_expr": "col_vals_expr",
        "col_vals_increasing": "col_vals_increasing",
        "col_vals_decreasing": "col_vals_decreasing",
        "col_vals_within_spec": "col_vals_within_spec",
        "col_vals_str_len": "col_vals_str_len",
        "col_vals_scored": "col_vals_scored",
        "col_vals_chat_format": "col_vals_chat_format",
        "col_vals_text_quality": "col_vals_text_quality",
        "col_vals_in_language": "col_vals_in_language",
        "col_vals_token_len": "col_vals_token_len",
        "col_vals_no_pii": "col_vals_no_pii",
        "col_pct_null": "col_pct_null",
        "col_pct_missing": "col_pct_missing",
        "col_missing_coded": "col_missing_coded",
        "col_missing_only_coded": "col_missing_only_coded",
        "col_missing_consistent": "col_missing_consistent",
        "rows_distinct": "rows_distinct",
        "rows_near_distinct": "rows_near_distinct",
        "rows_not_contaminated": "rows_not_contaminated",
        "rows_complete": "rows_complete",
        "col_count_match": "col_count_match",
        "row_count_match": "row_count_match",
        "col_schema_match": "col_schema_match",
        "data_freshness": "data_freshness",
        "tbl_match": "tbl_match",
        "tbl_reconcile": "tbl_reconcile",
        "prompt": "prompt",
        "conjointly": "conjointly",
        "specially": "specially",
    }

    def __init__(self) -> None:
        """Initialize the YAML validator."""
        pass

    def load_config(self, source: Union[str, Path]) -> dict:
        """Load and validate YAML configuration.

        Parameters
        ----------
        source
            YAML string or Path to YAML file.

        Returns
        -------
        dict
            Parsed and validated configuration dictionary.

        Raises
        ------
        YAMLValidationError
            If the YAML is invalid or malformed.
        """
        try:
            if isinstance(source, (str, Path)):
                if isinstance(source, Path):
                    # It's definitely a file path
                    with open(source, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                elif isinstance(source, str):
                    # Check if it looks like YAML content
                    stripped = source.strip()
                    if (
                        stripped.startswith(("tbl:", "steps:"))
                        or "\n" in stripped
                        or ":" in stripped
                    ):
                        # Looks like YAML content
                        config = yaml.safe_load(source)
                    else:
                        # Assume it's a file path
                        with open(source, "r", encoding="utf-8") as f:
                            config = yaml.safe_load(f)
            else:
                raise YAMLValidationError(
                    f"Invalid source type: {type(source)}. Only YAML strings and file paths supported."
                )

            if not isinstance(config, dict):
                raise YAMLValidationError("YAML must contain a dictionary at the root level")

            self._validate_schema(config)
            return config

        except yaml.YAMLError as e:
            raise YAMLValidationError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise YAMLValidationError(f"Error loading YAML configuration: {e}")

    def _validate_schema(self, config: dict) -> None:
        """Validate the YAML configuration schema.

        Parameters
        ----------
        config
            Configuration dictionary to validate.

        Raises
        ------
        YAMLValidationError
            If the schema is invalid.
        """
        # Define known top-level keys
        known_keys = {
            "tbl",
            "steps",
            "tbl_name",
            "label",
            "missing_specs",
            "thresholds",
            "actions",
            "final_actions",
            "brief",
            "lang",
            "locale",
            "df_library",
            "owner",
            "consumers",
            "version",
            "reference",
            "ablation",
        }

        # Warn about unknown top-level keys (likely typos)
        unknown_keys = set(config.keys()) - known_keys
        if unknown_keys:
            raise YAMLValidationError(
                f"Unknown top-level key(s): {sorted(unknown_keys)}. "
                f"Valid keys are: {sorted(known_keys)}"
            )

        if "ablation" in config:
            _check_ablation_block(config["ablation"])

        # Check required fields
        if "tbl" not in config:
            raise YAMLValidationError("YAML must contain 'tbl' field")

        if "steps" not in config:
            raise YAMLValidationError("YAML must contain 'steps' field")

        if not isinstance(config["steps"], list):
            raise YAMLValidationError("'steps' must be a list")  # pragma: no cover

        if len(config["steps"]) == 0:
            raise YAMLValidationError("'steps' cannot be empty")

        # Validate thresholds if present
        if "thresholds" in config:
            thresholds = config["thresholds"]
            if not isinstance(thresholds, dict):
                raise YAMLValidationError("'thresholds' must be a dictionary")

            for key, value in thresholds.items():
                if key not in ["warning", "error", "critical"]:
                    raise YAMLValidationError(
                        f"Invalid threshold key: {key}. Must be 'warning', 'error', or 'critical'"
                    )

                if not isinstance(value, (int, float)):
                    raise YAMLValidationError(f"Threshold '{key}' must be a number")

                if value < 0:
                    raise YAMLValidationError(f"Threshold '{key}' must be non-negative")

        # Validate actions if present
        if "actions" in config:
            actions = config["actions"]
            if not isinstance(actions, dict):
                raise YAMLValidationError("'actions' must be a dictionary")

            for key, value in actions.items():
                if key not in ["warning", "error", "critical", "default", "highest_only"]:
                    raise YAMLValidationError(
                        f"Invalid action key: {key}. Must be 'warning', 'error', 'critical', "
                        f"'default', or 'highest_only'"
                    )

                if key == "highest_only":
                    if not isinstance(value, bool):
                        raise YAMLValidationError(f"Action '{key}' must be a boolean")
                else:
                    # Action values can be strings or have python: block syntax for callables
                    if not isinstance(value, (str, dict, list)):
                        raise YAMLValidationError(
                            f"Action '{key}' must be a string, dictionary (for python: block), "
                            f"or list of strings/dictionaries"
                        )

    def _load_data_source(
        self,
        tbl_spec: str,
        df_library: Literal["polars", "pandas", "duckdb"],
        python_mode: _PythonMode = "deny",
    ) -> Any:
        """Load data source based on table specification.

        Parameters
        ----------
        tbl_spec
            Data source specification. Can be (1) a dataset name for `load_dataset()`, (2) a CSV file
            path (relative or absolute), (3) a Parquet file path (relative or absolute), or (4) a
            Python code snippet to be executed for dynamic data loading.
        df_library
            DataFrame library to use for loading datasets and CSV files. Options: "polars", "pandas", "duckdb".
        python_mode
            How to handle Python code in the specification (see `_eval_python_code()`).

        Returns
        -------
            Loaded data object.

        Raises
        ------
        YAMLValidationError
            If data source cannot be loaded.
        """
        from pointblank.validate import _process_data

        try:
            # First, try to process as Python expression
            processed_tbl_spec = _process_python_expressions(tbl_spec, python_mode=python_mode)

            # If processing returned a different object (not a string), use it directly
            if processed_tbl_spec is not tbl_spec or not isinstance(processed_tbl_spec, str):
                return processed_tbl_spec

            # Check if it's a CSV file and handle with specified library
            if isinstance(processed_tbl_spec, str) and processed_tbl_spec.endswith(".csv"):
                return self._load_csv_file(processed_tbl_spec, df_library)

            # Use the centralized data processing pipeline from validate.py
            # This handles Parquet files and other data sources
            processed_data = _process_data(processed_tbl_spec)

            # If _process_data returns the original string unchanged,
            # then it's not a file path, so try load_dataset with specified library
            if processed_data is processed_tbl_spec and isinstance(processed_tbl_spec, str):
                return load_dataset(processed_tbl_spec, tbl_type=df_library)
            else:
                return processed_data  # pragma: no cover

        except YAMLValidationError:
            raise
        except Exception as e:  # pragma: no cover
            raise YAMLValidationError(f"Failed to load data source '{tbl_spec}': {e}")

    def _load_csv_file(self, file_path: str, df_library: str) -> Any:
        """Load CSV file using the specified DataFrame library.

        Parameters
        ----------
        file_path
            Path to the CSV file.
        df_library
            DataFrame library to use: "polars", "pandas", or "duckdb".

        Returns
        -------
            Loaded DataFrame object.

        Raises
        ------
        YAMLValidationError
            If CSV file cannot be loaded or library is not available.
        """
        import os

        if not os.path.exists(file_path):
            raise YAMLValidationError(f"CSV file not found: {file_path}")

        try:
            if df_library == "polars":
                if not _is_lib_present("polars"):
                    raise YAMLValidationError("Polars library is not available")
                import polars as pl

                return pl.read_csv(file_path)

            elif df_library == "pandas":
                if not _is_lib_present("pandas"):
                    raise YAMLValidationError("Pandas library is not available")
                import pandas as pd

                return pd.read_csv(file_path)

            elif df_library == "duckdb":
                # For DuckDB, we'll use the existing _process_data since it handles DuckDB
                from pointblank.validate import _process_data  # pragma: no cover

                return _process_data(file_path)  # pragma: no cover

            else:
                raise YAMLValidationError(
                    f"Unsupported df_library: {df_library}. Use 'polars', 'pandas', or 'duckdb'"
                )

        except Exception as e:  # pragma: no cover
            raise YAMLValidationError(
                f"Failed to load CSV file '{file_path}' with {df_library}: {e}"
            )

    def _parse_column_spec(self, columns_expr: Any) -> list[str]:
        """Parse column specification from YAML.

        Handles standard YAML syntax for columns.

        Parameters
        ----------
        columns_expr
            Column specification (list, or string).

        Returns
        -------
        list[str]
            List of column names.
        """
        if isinstance(columns_expr, list):
            return [str(col) for col in columns_expr]

        if isinstance(columns_expr, str):
            # Single column name
            return [columns_expr]

        # Fallback: convert to string
        return [str(columns_expr)]

    def _parse_schema_spec(self, schema_spec: Any) -> Any:
        """Parse schema specification from YAML.

        Converts dictionary-based schema definitions into Schema objects.

        Column specifications support multiple formats:
        - Scalar strings: "column_name" (name only, no type checking)
        - Lists with name and type: ["column_name", "data_type"]
        - Lists with name only: ["column_name"] (equivalent to scalar)

        Parameters
        ----------
        schema_spec
            Schema specification as a dictionary with 'columns' field.

        Returns
        -------
        Schema
            A Schema object created from the specification.

        Raises
        ------
        YAMLValidationError
            If schema specification is invalid.
        """
        from pointblank.schema import Schema

        # Handle dictionary specification only
        if isinstance(schema_spec, dict):
            if "columns" in schema_spec:
                # Convert columns list to a `Schema` object
                columns_spec = schema_spec["columns"]

                if not isinstance(columns_spec, list):
                    raise YAMLValidationError(
                        "Schema 'columns' must be a list of column specifications"
                    )

                # Convert YAML column specs to `Schema` format
                schema_columns = []
                for col_spec in columns_spec:
                    if isinstance(col_spec, list):
                        if len(col_spec) == 1:
                            # Column name only: ["column_name"]
                            schema_columns.append((col_spec[0],))
                        elif len(col_spec) == 2:
                            # Column name and type: ["column_name", "type"]
                            schema_columns.append((col_spec[0], col_spec[1]))
                        else:
                            raise YAMLValidationError(
                                f"Column specification must have 1-2 elements, got: {col_spec}"
                            )
                    elif isinstance(col_spec, str):
                        # Just column name as string
                        schema_columns.append((col_spec,))
                    else:
                        raise YAMLValidationError(
                            f"Invalid column specification type: {type(col_spec)}"
                        )

                # Create Schema object
                return Schema(columns=schema_columns)
            else:
                raise YAMLValidationError("Schema specification must contain 'columns' field")
        else:
            raise YAMLValidationError(
                f"Schema specification must be a dictionary, got: {type(schema_spec)}"
            )

    def _parse_segments_spec(self, segments: Any) -> Any:
        """Parse a `segments=` specification from YAML.

        A string is a column to segment by. In a list, each string is a column and each two-item
        `[column, value]` list is a `(column, value)` segment (the value may itself be a list of
        values). This mirrors how `to_yaml()` writes segments, since YAML has no tuples.
        """
        if not isinstance(segments, list):
            return segments

        parsed: list[Any] = []
        for item in segments:
            if isinstance(item, (str, tuple)):
                parsed.append(item)
            elif isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
                parsed.append(tuple(item))
            else:
                raise YAMLValidationError(
                    f"Invalid 'segments' item: {item!r}. Use a column name, or a two-item "
                    "[column, value] list for a specific segment."
                )
        return parsed

    def _parse_missing_specs(self, config: dict) -> dict[str, MissingSpec]:
        """Parse the top-level `missing_specs` block into named `MissingSpec` objects."""
        raw = config.get("missing_specs")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise YAMLValidationError("'missing_specs' must be a dictionary of named specs")
        return {name: _missing_spec_from_dict(spec_def) for name, spec_def in raw.items()}

    def _resolve_missing(
        self, value: Any, missing_specs: Optional[dict[str, MissingSpec]]
    ) -> MissingSpec:
        """Resolve a step's `missing=` value to a `MissingSpec`.

        The value can be a named reference into the top-level `missing_specs` block, an inline
        mapping defining a spec, or an already-constructed `MissingSpec`.
        """
        if isinstance(value, MissingSpec):
            return value
        if isinstance(value, str):
            if not missing_specs or value not in missing_specs:
                available = sorted(missing_specs.keys()) if missing_specs else []
                raise YAMLValidationError(
                    f"Unknown missing spec '{value}'. Define it under the top-level "
                    f"'missing_specs' block. Available: {available}"
                )
            return missing_specs[value]
        if isinstance(value, dict):
            return _missing_spec_from_dict(value)
        raise YAMLValidationError(
            f"Invalid 'missing' value: {value!r}. Use a named reference, an inline mapping, "
            "or a MissingSpec."
        )

    @staticmethod
    def _parse_scorer_spec(spec: dict) -> Any:
        """Make a `pb.scorers` adapter from its YAML form, `{<adapter>: {<settings>}}`."""
        from pointblank import scorers

        if len(spec) != 1 or next(iter(spec)) not in scorers.ADAPTERS:
            raise YAMLValidationError(
                "A `col_vals_scored` scorer must be written as `{<adapter>: {...}}`, where the "
                f"adapter is one of {', '.join(scorers.ADAPTERS)} (or as a `python:` block); got "
                f"{spec!r}."
            )
        kind, settings = next(iter(spec.items()))
        settings = settings or {}
        if not isinstance(settings, dict):
            raise YAMLValidationError(f"The settings of the `{kind}` scorer must be a dictionary.")
        try:
            return getattr(scorers, kind)(**settings)
        except (TypeError, ValueError) as e:
            raise YAMLValidationError(f"The `{kind}` scorer couldn't be made: {e}") from e

    def _parse_validation_step(
        self,
        step_config: Union[str, dict],
        namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
        missing_specs: Optional[dict[str, MissingSpec]] = None,
        python_mode: _PythonMode = "deny",
    ) -> tuple[str, dict]:
        """Parse a single validation step from YAML configuration.

        Parameters
        ----------
        step_config
            Step configuration (string for parameterless steps, dict for others).
        python_mode
            How to handle Python code in the step's parameters (see `_eval_python_code()`).

        Returns
        -------
        tuple[str, dict]
            Tuple of (method_name, parameters).

        Raises
        ------
        YAMLValidationError
            If step configuration is invalid.
        """
        if isinstance(step_config, str):
            # Simple step with no parameters (e.g., "rows_distinct")
            method_name = step_config
            parameters = {}
        elif isinstance(step_config, dict):
            # Step with parameters
            if len(step_config) != 1:
                raise YAMLValidationError(
                    "Step configuration must contain exactly one validation method, "
                    f"got: {list(step_config.keys())}"
                )

            method_name = list(step_config.keys())[0]
            parameters = step_config[method_name] or {}

            if not isinstance(parameters, dict):
                raise YAMLValidationError(f"Parameters for '{method_name}' must be a dictionary")
        else:
            raise YAMLValidationError(f"Invalid step configuration type: {type(step_config)}")

        # Validate that we know this method (static map or dynamic aggregate method)
        if method_name not in self.validation_method_map and not is_valid_agg(method_name):
            available_methods = list(self.validation_method_map.keys()) + [
                "col_sum_*",
                "col_avg_*",
                "col_sd_*",
            ]
            raise YAMLValidationError(
                f"Unknown validation method '{method_name}'. Available methods: {available_methods}"
            )

        # Process Python expressions in all parameters
        processed_parameters = {}
        for key, value in parameters.items():
            # Special case: `col_vals_expr()`'s `expr=` parameter can use shortcut syntax
            if method_name == "col_vals_expr" and key == "expr" and isinstance(value, str):
                # Treat string directly as Python code (shortcut syntax)
                processed_parameters[key] = _eval_python_code(
                    value, namespaces=namespaces, python_mode=python_mode
                )
            # Special case: `pre=` parameter can use shortcut syntax (like `expr=`)
            elif key == "pre" and isinstance(value, str):
                # Treat string directly as Python code (shortcut syntax)
                processed_parameters[key] = _eval_python_code(
                    value, namespaces=namespaces, python_mode=python_mode
                )
            # Special case: `active=` parameter can use shortcut syntax for callables
            # (e.g., `active: pb.has_columns("col_a")` or `active: false`)
            elif key == "active" and isinstance(value, str):
                processed_parameters[key] = _eval_python_code(
                    value, namespaces=namespaces, python_mode=python_mode
                )
            # Special case: a column reference given as a comparison value can be written directly
            # (e.g., `value: pb.col("b")`) as well as in a `python:` block
            elif (
                key in ("value", "left", "right")
                and isinstance(value, str)
                and _COLUMN_REF_PATTERN.fullmatch(value)
            ):
                processed_parameters[key] = _eval_python_code(
                    value, namespaces=namespaces, python_mode=python_mode
                )
            elif key == "missing":
                # Pass the raw value through (a spec name, inline mapping, or MissingSpec); it is
                # resolved to a MissingSpec below, after the loop
                processed_parameters[key] = value
            else:
                # Normal processing (requires python: block syntax)
                processed_parameters[key] = _process_python_expressions(
                    value, namespaces=namespaces, python_mode=python_mode
                )
        parameters = processed_parameters

        # Resolve a `missing=` parameter (used by col_pct_missing, col_missing_coded) into a
        # MissingSpec, looking up named references in the top-level `missing_specs` block
        if "missing" in parameters:
            parameters["missing"] = self._resolve_missing(parameters["missing"], missing_specs)

        # Convert `columns=` specification
        if "columns" in parameters:
            parameters["columns"] = self._parse_column_spec(parameters["columns"])

        #
        # Convert special parameter formats
        #

        # Convert `columns_subset=` if present (from `rows_[distinct|complete]()`)
        if "columns_subset" in parameters:
            parameters["columns_subset"] = self._parse_column_spec(parameters["columns_subset"])

        # Convert `schema=` if present (for `col_schema_match()`)
        if "schema" in parameters and method_name == "col_schema_match":
            parameters["schema"] = self._parse_schema_spec(parameters["schema"])

        # Handle `conjointly()` expressions: convert list to separate positional arguments
        if method_name == "conjointly" and "expressions" in parameters:
            expressions = parameters.pop("expressions")  # Remove from parameters
            if isinstance(expressions, list):
                # Convert string expressions to lambda functions
                lambda_expressions = []
                for expr in expressions:
                    if isinstance(expr, str):
                        lambda_expressions.append(_eval_python_code(expr, python_mode=python_mode))
                    else:
                        lambda_expressions.append(expr)  # pragma: no cover
                # Pass expressions as positional arguments (stored as special key)
                parameters["_conjointly_expressions"] = lambda_expressions
            else:
                raise YAMLValidationError("conjointly 'expressions' must be a list")

        # Handle `specially()` expr parameter: support shortcut syntax
        if method_name == "specially" and "expr" in parameters:
            expr_value = parameters["expr"]
            if isinstance(expr_value, str):
                # Treat string directly as Python code (shortcut syntax)
                parameters["expr"] = _eval_python_code(expr_value, python_mode=python_mode)

        # A `col_vals_scored()` scorer can be written as `{<adapter>: {...}}`, which calls the
        # `pb.scorers` function of that name with the settings given (e.g.,
        # `scorer: {classifier: {hf_model: ..., label: ...}}`)
        if method_name == "col_vals_scored" and isinstance(parameters.get("scorer"), dict):
            parameters["scorer"] = self._parse_scorer_spec(parameters["scorer"])

        # Convert `actions=` if present (ensure it's an Actions object)
        if "actions" in parameters:
            if isinstance(parameters["actions"], dict):
                parameters["actions"] = Actions(**parameters["actions"])

        # Handle `inclusive=` parameter for `col_vals_[inside|outside]()` (convert list to tuple)
        if "inclusive" in parameters and isinstance(parameters["inclusive"], list):
            parameters["inclusive"] = tuple(parameters["inclusive"])

        # A `tol=` given as a two-item list is a (lower, upper) tolerance tuple
        if "tol" in parameters and isinstance(parameters["tol"], list):
            parameters["tol"] = tuple(parameters["tol"])

        # Convert `segments=`, where each `[column, value]` pair in a list is a segment tuple
        if "segments" in parameters:
            parameters["segments"] = self._parse_segments_spec(parameters["segments"])

        # Resolve the method name: static map takes priority, then dynamic aggregate methods
        if method_name in self.validation_method_map:
            resolved_name = self.validation_method_map[method_name]
        else:
            # Dynamic aggregate method (col_sum_gt, col_avg_eq, etc.)
            resolved_name = method_name

        return resolved_name, parameters

    def build_validation(
        self,
        config: dict,
        namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
        tbl: Any = None,
        allow_python: bool = False,
    ) -> Validate:
        """Convert YAML config to Validate object.

        Parameters
        ----------
        config
            Validated configuration dictionary.
        tbl
            A table to validate in place of the config's `tbl` (which is then not loaded).
        allow_python
            Whether to run Python code in the config (see `yaml_interrogate()`). Without it, a
            config with Python code raises a `YAMLValidationError`.

        Returns
        -------
        Validate
            Validate object with configured validation steps.
        """
        python_mode: _PythonMode = "allow" if allow_python else "deny"

        # Load data source with specified library
        df_library = config.get("df_library", "polars")
        data = (
            tbl
            if tbl is not None
            else self._load_data_source(config["tbl"], df_library, python_mode=python_mode)
        )

        # Create Validate object
        validate_kwargs = {}

        # Set table name if provided
        if "tbl_name" in config:
            validate_kwargs["tbl_name"] = config["tbl_name"]

        # Set label if provided
        if "label" in config:
            validate_kwargs["label"] = config["label"]

        # Set thresholds if provided
        if "thresholds" in config:
            validate_kwargs["thresholds"] = config["thresholds"]

        # Set actions if provided
        if "actions" in config:
            # Process actions: handle `python:` block syntax for callables
            processed_actions = _process_python_expressions(
                config["actions"], namespaces=namespaces, python_mode=python_mode
            )
            # Convert to Actions object
            validate_kwargs["actions"] = Actions(**processed_actions)

        # Set language if provided
        if "lang" in config:
            validate_kwargs["lang"] = config["lang"]

        # Set locale if provided
        if "locale" in config:
            validate_kwargs["locale"] = config["locale"]

        # Set global brief if provided
        if "brief" in config:
            validate_kwargs["brief"] = config["brief"]

        # Set owner if provided (governance)
        if "owner" in config:
            validate_kwargs["owner"] = config["owner"]

        # Set consumers if provided (governance)
        if "consumers" in config:
            validate_kwargs["consumers"] = config["consumers"]

        # Set version if provided (governance)
        if "version" in config:
            validate_kwargs["version"] = config["version"]

        # Set final_actions if provided
        if "final_actions" in config:
            from pointblank.thresholds import FinalActions

            final_actions_spec = config["final_actions"]
            # Process Python expressions in final_actions
            processed_final_actions = _process_python_expressions(
                final_actions_spec, namespaces=namespaces, python_mode=python_mode
            )
            if isinstance(processed_final_actions, list):
                validate_kwargs["final_actions"] = FinalActions(*processed_final_actions)
            elif callable(processed_final_actions):
                validate_kwargs["final_actions"] = FinalActions(processed_final_actions)
            else:
                validate_kwargs["final_actions"] = processed_final_actions

        # Set reference data if provided
        if "reference" in config:
            ref_spec = config["reference"]
            # Process as a data source (could be a dataset name, file path, or python expression)
            ref_data = self._load_data_source(ref_spec, df_library, python_mode=python_mode)
            validate_kwargs["reference"] = ref_data

        validation = Validate(data, **validate_kwargs)

        # Parse any named missing specs declared at the top level
        missing_specs = self._parse_missing_specs(config)

        # Add validation steps
        for step_config in config["steps"]:
            method_name, parameters = self._parse_validation_step(
                step_config,
                namespaces=namespaces,
                missing_specs=missing_specs,
                python_mode=python_mode,
            )

            # Get the method from the validation object
            method = getattr(validation, method_name)

            # Special handling for conjointly: pass expressions as positional arguments
            if method_name == "conjointly" and "_conjointly_expressions" in parameters:
                expressions = parameters.pop("_conjointly_expressions")
                validation = method(*expressions, **parameters)
            else:
                # Call the method with parameters
                validation = method(**parameters)

        return validation

    def execute_workflow(
        self,
        config: dict,
        namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
        allow_python: bool = False,
    ) -> Validate:
        """Execute a complete YAML validation workflow.

        Parameters
        ----------
        config
            Validated configuration dictionary.
        allow_python
            Whether to run Python code in the config (see `yaml_interrogate()`).

        Returns
        -------
        Validate
            Interrogated Validate object with results.
        """
        # Build the validation plan
        validation = self.build_validation(config, namespaces=namespaces, allow_python=allow_python)

        # Execute interrogation to get results
        validation = validation.interrogate()

        return validation


# The keys of an `ablation:` block, and the strategies it can use with their parameters (those
# in the second set are required)
_ABLATION_KEYS = (
    "evaluate",
    "evaluate_version",
    "metric",
    "higher_is_better",
    "repeats",
    "seed",
    "control",
    "duplicates",
    "duplicates_order_by",
    "max_removed",
    "conf_level",
    "adjust",
    "label",
    "cache",
    "holdout",
    "ci",
    "pass_as",
    "strategies",
)
_ABLATION_STRATEGIES = {
    "drop_each": ({"steps"}, set()),
    "drop_all": ({"steps"}, set()),
    "drop_steps": ({"steps", "name"}, {"steps"}),
    "drop_all_but_each": ({"steps"}, set()),
    "drop_groups": ({"column", "values"}, {"column"}),
    "sweep_score": ({"score", "quantiles", "values", "remove"}, {"score"}),
    "sweep_step": ({"step", "param", "values"}, {"step", "param", "values"}),
    "inject": ({"corruption", "column", "step", "rates", "name"}, {"corruption"}),
    "shapley": ({"steps", "permutations", "seed"}, set()),
}

# An `evaluate:` function given by its import path (e.g., `my_models.training:evaluate`)
_IMPORT_PATH_PATTERN = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_][\w.]*")


def _ablation_strategies(block: dict) -> list[tuple[str, dict]]:
    """The strategies of an `ablation:` block, as `(method, parameters)` pairs."""
    strategies = block.get("strategies") or []
    if not isinstance(strategies, list):
        raise YAMLValidationError("'ablation.strategies' must be a list of strategies.")
    parsed = []
    for strategy in strategies:
        if isinstance(strategy, str):
            method, params = strategy, {}
        elif isinstance(strategy, dict) and len(strategy) == 1:
            method, params = next(iter(strategy.items()))
            params = {} if params is None else params
        else:
            raise YAMLValidationError(
                "Each strategy in 'ablation.strategies' must be a strategy name (e.g., "
                f"`drop_each`) or a mapping of one name to its parameters, got {strategy!r}."
            )
        if method not in _ABLATION_STRATEGIES:
            raise YAMLValidationError(
                f"Unknown ablation strategy {method!r}. The strategies are: "
                f"{', '.join(_ABLATION_STRATEGIES)}."
            )
        if not isinstance(params, dict):
            raise YAMLValidationError(
                f"The parameters of the `{method}` strategy must be a mapping, got {params!r}."
            )
        allowed, required = _ABLATION_STRATEGIES[method]
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise YAMLValidationError(
                f"Unknown parameter(s) {unknown} for the `{method}` strategy. Valid parameters "
                f"are: {sorted(allowed)}."
            )
        missing = sorted(required - set(params))
        if missing:
            raise YAMLValidationError(f"The `{method}` strategy needs the parameter(s) {missing}.")
        parsed.append((method, dict(params)))
    return parsed


def _check_ablation_block(block: Any) -> None:
    """Check the structure of an `ablation:` block."""
    if not isinstance(block, dict):
        raise YAMLValidationError("'ablation' must be a mapping of ablation settings.")
    unknown = sorted(set(block) - set(_ABLATION_KEYS))
    if unknown:
        raise YAMLValidationError(
            f"Unknown key(s) in 'ablation': {unknown}. Valid keys are: {sorted(_ABLATION_KEYS)}"
        )
    evaluate = block.get("evaluate")
    if evaluate is not None and not (
        (isinstance(evaluate, str) and _IMPORT_PATH_PATTERN.fullmatch(evaluate.strip()))
        or (isinstance(evaluate, dict) and set(evaluate) == {"python"})
    ):
        raise YAMLValidationError(
            "'ablation.evaluate' must be the import path of a function (e.g., "
            "`my_models.training:evaluate`) or a `python:` block, got "
            f"{evaluate!r}."
        )
    holdout = block.get("holdout")
    if holdout is not None and not (
        isinstance(holdout, str) or (isinstance(holdout, dict) and set(holdout) == {"python"})
    ):
        raise YAMLValidationError(
            "'ablation.holdout' must be a data source like `tbl:` (a dataset name or a file path) "
            f"or a `python:` block, got {holdout!r}."
        )
    control = block.get("control", "size_matched")
    if not (
        control is None
        or control == "size_matched"
        or (isinstance(control, dict) and set(control) == {"stratified"})
    ):
        raise YAMLValidationError(
            "'ablation.control' must be `size_matched`, `null`, or a mapping like "
            f"`{{stratified: column}}`, got {control!r}."
        )
    _ablation_strategies(block)


def _import_evaluate(path: str) -> Any:
    """Import a function from its import path (e.g., `my_models.training:evaluate`)."""
    module_name, _, attrs = path.strip().partition(":")
    try:
        obj: Any = import_module(module_name)
    except ImportError as e:
        raise YAMLValidationError(
            f"The module {module_name!r} of 'ablation.evaluate' can't be imported: {e}"
        ) from e
    for attr in attrs.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            raise YAMLValidationError(
                f"The module {module_name!r} has no {attrs!r} (in 'ablation.evaluate')."
            ) from None
    return obj


def _ablation_kwargs(
    block: dict,
    namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    python_mode: _PythonMode = "deny",
    resolve_evaluate: bool = True,
) -> dict[str, Any]:
    """The arguments of `Ablation()` from an `ablation:` block."""
    from pointblank.ablation import Stratified

    # (the `holdout:` is a data source, which is loaded with the validation's `tbl:`)
    kwargs = {
        k: v for k, v in block.items() if k not in ("evaluate", "strategies", "control", "holdout")
    }
    if "control" in block:
        control = block["control"]
        if isinstance(control, dict):
            by = control["stratified"]
            kwargs["control"] = (
                _PythonSource(f"pb.ablation.Stratified(by={_literal_code(by)})")
                if python_mode == "source"
                else Stratified(by=by)
            )
        else:
            kwargs["control"] = control

    evaluate = block.get("evaluate")
    if evaluate is not None and resolve_evaluate:
        if python_mode == "source":
            kwargs["evaluate"] = (
                _PythonSource(evaluate["python"])
                if isinstance(evaluate, dict)
                else _PythonSource(evaluate.strip().partition(":")[2])
            )
        elif python_mode != "allow":
            shown = evaluate.strip() if isinstance(evaluate, str) else "a `python:` block"
            raise YAMLValidationError(
                f"The `ablation:` block's `evaluate:` ({shown}) is Python code, which isn't run "
                "unless you allow it. Use `yaml_ablation(..., allow_python=True)` for a config "
                "you trust, or pass the function with `yaml_ablation(..., evaluate=...)`."
            )
        elif isinstance(evaluate, dict):
            kwargs["evaluate"] = _eval_python_code(
                evaluate["python"], namespaces=namespaces, python_mode=python_mode
            )
        else:
            kwargs["evaluate"] = _import_evaluate(evaluate)
        if python_mode == "allow" and not callable(kwargs["evaluate"]):
            raise YAMLValidationError(
                f"'ablation.evaluate' must be a function, got {type(kwargs['evaluate']).__name__}."
            )
    return kwargs


def yaml_ablation(
    yaml: Union[str, Path],
    evaluate: Any = None,
    set_tbl: Any = None,
    holdout: Any = None,
    namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    allow_python: bool = False,
) -> "Ablation":
    """Set up an ablation from a YAML configuration.

    A YAML configuration for [`yaml_interrogate()`](`pointblank.yaml_interrogate`) can have an
    `ablation:` block, which describes an [`Ablation`](`pointblank.Ablation`) of its validation:
    the `evaluate=` function, the settings, and the strategies that add variants. This function
    interrogates the validation and sets up the ablation, ready to
    [`run()`](`pointblank.Ablation.run`) or to [`export()`](`pointblank.Ablation.export`). The
    validation and its ablation can then be kept together in one file, alongside a model's
    training code.

    Parameters
    ----------
    yaml
        YAML configuration as a string or a file path, with an `ablation:` block.
    evaluate
        The `evaluate=` function of the ablation. If given, it's used in place of the block's
        `evaluate:` (which is then not imported), so that the configuration can be used without
        `allow_python=True`.
    set_tbl
        A table to use in place of the configuration's `tbl` (as in `yaml_interrogate()`).
    holdout
        A table to use in place of the block's `holdout:` (the holdout set that the validation's
        steps are also applied to, as in the `holdout=` of `Ablation`).
    namespaces
        Modules to make available to Python code in the configuration (as in
        `yaml_interrogate()`). This only has an effect with `allow_python=True`.
    allow_python
        Whether to run Python code in the configuration. The `evaluate:` of the `ablation:` block
        is Python code (it imports a function and runs it on the training data), so it's only
        used with `allow_python=True`. Without it, either pass `evaluate=` or leave `evaluate:`
        out (e.g., to `~~pointblank.Ablation.export()` the versions of the table for evaluation
        elsewhere).

    Returns
    -------
    Ablation
        The ablation, with its variants added but not yet run.

    Raises
    ------
    YAMLValidationError
        If the configuration is invalid, has no `ablation:` block, or has Python code without
        `allow_python=True`.

    The `ablation:` Block
    ---------------------
    The block has the parameters of `Ablation` (other than `validation=`), along with a list of
    `strategies:`:

    ```yaml
    ablation:
      evaluate: my_models.training:evaluate
      evaluate_version: "2024-06"
      metric: mae
      higher_is_better: false
      repeats: 5
      control:
        stratified: sensor
      cache: .ablation_cache
      holdout: data/holdout.parquet
      strategies:
        - drop_each
        - drop_all_but_each
        - drop_steps:
            steps: [1, 2]
            name: Without step 3
        - drop_groups:
            column: sensor
    ```

    The `evaluate:` is the import path of a function (the module, a colon, and the name of the
    function), or a `python:` block whose result is the function. The `control:` is
    `size_matched` (the default), `null` (for no control), or `stratified:` with one or more
    columns (for a [`Stratified`](`pointblank.ablation.Stratified`) control). The `holdout:` is a
    data source like the `tbl:` of the configuration (a dataset name, a file path, or a `python:`
    block), loaded with its `df_library:`. Each strategy is
    the name of an `Ablation` method that adds variants (including the sweeps `sweep_score` and
    `sweep_step`, the injected defects of `inject`, and the Shapley values of `shapley`), with its
    parameters as a mapping. The strategies are added in order. In YAML, the `corruption:` of
    `inject` is the name of a defect (`null`, `shuffle`, `truncate`, or `duplicate`), not a
    function.

    The block is ignored by `yaml_interrogate()`, so the same configuration can be used both to
    validate the data and to measure what its steps do for a model.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False)
    ```

    Here's a table of temperature readings with some out-of-range values, and a YAML
    configuration with a validation step and an `ablation:` block. The `evaluate()` function is
    passed from Python, so the configuration doesn't need `allow_python=True`.

    ```{python}
    import pointblank as pb
    import polars as pl

    temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
    for k in range(0, 300, 25):
        temps[k] *= 10
    readings = pl.DataFrame({"temp": temps})

    def evaluate(train, *, seed):
        sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
        return abs(sample["temp"].mean() - 20)

    yaml_config = '''
    tbl: readings.csv
    tbl_name: readings
    steps:
    - col_vals_between:
        columns: temp
        left: -10
        right: 45
    ablation:
      metric: error
      higher_is_better: false
      label: Temperature model
      strategies:
        - drop_each
    '''

    ablation = pb.yaml_ablation(yaml_config, evaluate=evaluate, set_tbl=readings)

    ablation.run()
    ```

    The `tbl:` of the configuration is replaced with the table in `set_tbl=`, and the ablation
    was set up from the block, ready to run.

    See Also
    --------
    Ablation : Measure how the rows that fail validation steps affect a model trained on the data.
    yaml_interrogate : Execute a YAML-based validation workflow.
    """
    from pointblank.ablation import Ablation

    if evaluate is not None and not callable(evaluate):
        raise TypeError(f"`evaluate=` must be a function, got {type(evaluate).__name__}.")

    validator = YAMLValidator()
    config = validator.load_config(yaml)
    if "ablation" not in config:
        raise YAMLValidationError("The YAML configuration has no `ablation:` block.")
    block = config["ablation"]
    python_mode: _PythonMode = "allow" if allow_python else "deny"

    # Check the block (and any Python code in it) before the data is loaded and interrogated
    kwargs = _ablation_kwargs(
        block, namespaces=namespaces, python_mode=python_mode, resolve_evaluate=evaluate is None
    )
    if evaluate is not None:
        kwargs["evaluate"] = evaluate

    validation = validator.build_validation(
        config, namespaces=namespaces, tbl=set_tbl, allow_python=allow_python
    ).interrogate()
    if holdout is not None:
        kwargs["holdout"] = holdout
    elif block.get("holdout") is not None:
        kwargs["holdout"] = validator._load_data_source(
            block["holdout"], config.get("df_library", "polars"), python_mode=python_mode
        )

    ablation = Ablation(validation, **kwargs)
    for method, params in _ablation_strategies(block):
        getattr(ablation, method)(**params)
    return ablation


def _ablation_code(block: dict, df_library: str) -> tuple[list[str], list[str], str | None]:
    """
    The imports and the code of the `pb.Ablation()` from an `ablation:` block, and the import alias
    that the code of its `holdout:` needs (`"pl"`, `"pd"` or `None`).
    """
    imports = []
    evaluate = block.get("evaluate")
    if isinstance(evaluate, str):
        module_name, _, attrs = evaluate.strip().partition(":")
        imports.append(f"from {module_name} import {attrs.split('.')[0]}")

    kwargs = _ablation_kwargs(block, python_mode="source")
    args = ["validation"]
    if "evaluate" in kwargs:
        args.append(f"evaluate={kwargs.pop('evaluate')}")
    args += [f"{key}={_literal_code(value)}" for key, value in kwargs.items()]
    holdout = block.get("holdout")
    alias = None
    if isinstance(holdout, dict):
        args.append(f"holdout={holdout['python'].strip()}")
    elif holdout is not None:
        holdout_code, alias = _data_source_code(holdout, df_library)
        args.append(f"holdout={holdout_code}")

    lines = ["ablation = (", "    pb.Ablation("]
    lines += [f"        {arg}," for arg in args]
    lines.append("    )")
    for method, params in _ablation_strategies(block):
        params_str = ", ".join(f"{k}={_literal_code(v)}" for k, v in params.items())
        lines.append(f"    .{method}({params_str})")
    lines.append(")")
    return imports, lines, alias


def _literal_code(value: Any) -> str:
    """Python code for a value from a YAML config, with strings in double quotes."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_literal_code(item) for item in value) + "]"
    if isinstance(value, dict):
        items = (f"{_literal_code(k)}: {_literal_code(v)}" for k, v in value.items())
        return "{" + ", ".join(items) + "}"
    return _value_code(value)


def _value_code(value: Any) -> str:
    """Python code for a value, written as `float("inf")` (etc.) for a non-finite float."""
    if isinstance(value, float) and not math.isfinite(value):
        # `repr()` gives `inf`, `-inf`, or `nan`, which aren't names that Python knows
        return f'float("{value!r}")'
    return repr(value)


def yaml_interrogate(
    yaml: Union[str, Path],
    set_tbl: Any = None,
    namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
    allow_python: bool = False,
) -> Validate:
    """Execute a YAML-based validation workflow.

    This is the main entry point for YAML-based validation workflows. It takes YAML configuration
    (as a string or file path) and returns a validated `Validate` object with interrogation results.

    The YAML configuration defines the data source, validation steps, and optional settings like
    thresholds and labels. This function automatically loads the data, builds the validation plan,
    executes all validation steps, and returns the interrogated results. An `ablation:` block in
    the configuration is ignored here (it's used by [`yaml_ablation()`](`pointblank.yaml_ablation`)),
    so its `evaluate:` function is neither imported nor run.

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.
    set_tbl
        An optional table to override the table specified in the YAML configuration. This allows you
        to apply a YAML-defined validation workflow to a different table than what's specified in
        the configuration. If provided, this table is used in place of the table defined in the
        YAML's `tbl` field (which is then not loaded, so it can be a placeholder or a source
        that isn't available). This can be any supported table type
        including DataFrame objects, Ibis table objects, CSV file paths, Parquet file paths, GitHub
        URLs, or database connection strings.
    namespaces
        Optional module namespaces to make available for Python code execution in YAML
        configurations. Can be a dictionary mapping aliases to module names or a list of module
        names. This only has an effect with `allow_python=True`. See the "Using Namespaces"
        section below for detailed examples.
    allow_python
        Whether to run Python code in the YAML configuration (`python:` blocks, and the `expr:`,
        `pre:`, and `active:` shortcuts). This is `False` by default, which makes a configuration
        with Python code raise a `YAMLValidationError`. Python code in YAML runs with the full
        access of your Python session (it can read and write files or run commands), so only set
        this to `True` for configurations you trust. A few `pb` functions called with literal
        arguments (e.g., `pb.col("b")` or `pb.has_columns("a")`) don't need it. See the "Python
        Code in YAML" section below.

    Returns
    -------
    Validate
        An instance of the `Validate` class that has been configured based on the YAML input. This
        object contains the results of the validation steps defined in the YAML configuration. It
        includes metadata like table name, label, language, and thresholds if specified.

    Raises
    ------
    YAMLValidationError
        If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing
        required fields, unknown validation methods, data loading failures, or Python code in a
        configuration used without `allow_python=True`.

    Python Code in YAML
    -------------------
    A YAML configuration can include Python code: a `python:` block for the data source, a
    parameter value, or an action, and the `expr:`, `pre:`, and `active:` shortcuts for code
    written as a plain string. This code isn't run unless you set `allow_python=True`.

    Python code in YAML is not sandboxed. It runs with the same access as the rest of your Python
    session, so a configuration with Python code should be treated like a Python script: only allow
    it for configurations you wrote or otherwise trust, never for ones supplied by other users or
    systems.

    A small set of `pb` functions can always be used, when called with literal arguments (strings,
    numbers, lists, and so on): [`load_dataset()`](`pointblank.load_dataset`),
    [`col()`](`pointblank.col`), [`ref()`](`pointblank.ref`),
    [`seg_group()`](`pointblank.seg_group`), [`has_columns()`](`pointblank.has_columns`),
    [`has_rows()`](`pointblank.has_rows`), and the column selectors (`starts_with()`, `ends_with()`,
    `contains()`, `~~pointblank.matches()`, `everything()`, `first_n()`, and `last_n()`). So can the
    constructors for values that YAML has no literal for: `Decimal()`, `ZoneInfo()`,
    `datetime.date()`, `datetime.time()`, `datetime.datetime()`, and `datetime.timedelta()`. These
    only load a bundled dataset or build an object, so `value: pb.col("b")`,
    `value: {python: 'Decimal("2.2")'}`, or `active: pb.has_columns("a")` work without
    `allow_python=True`. In code that is allowed to run, `datetime`, `Decimal`, and `ZoneInfo` are
    available along with `pb`, `pl`, `pd`, and `Path`.

    Using Namespaces
    ----------------
    The `namespaces=` parameter enables custom Python modules and functions in YAML configurations.
    This is particularly useful for custom action functions and advanced Python expressions. Since
    these are Python code, both options below need `allow_python=True`.

    **Namespace formats:**

    - Dictionary format: `{"alias": "module.name"}` maps aliases to module names
    - List format: `["module.name", "another.module"]` imports modules directly

    **Option 1: Inline expressions (no namespaces needed)**

    ```{python}
    import pointblank as pb

    # Simple inline custom action
    yaml_config = '''
    tbl: small_table
    thresholds:
      warning: 0.01
    actions:
      warning:
        python: "lambda: print('Custom warning triggered')"
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
    '''

    result = pb.yaml_interrogate(yaml_config, allow_python=True)
    result
    ```

    **Option 2: External functions with namespaces**

    ```{python}
    # Define a custom action function
    def my_custom_action():
        print("Data validation failed: please check your data.")

    # Add to current module for demo
    import sys
    sys.modules[__name__].my_custom_action = my_custom_action

    # YAML that references the external function
    yaml_config = '''
    tbl: small_table
    thresholds:
      warning: 0.01
    actions:
      warning:
        python: actions.my_custom_action
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000  # This will fail
    '''

    # Use namespaces to make the function available
    result = pb.yaml_interrogate(
        yaml_config, namespaces={'actions': '__main__'}, allow_python=True
    )
    result
    ```

    This approach enables modular, reusable validation workflows with custom business logic.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    For the examples here, we'll use YAML configurations to define validation workflows. Let's start
    with a basic YAML workflow that validates the built-in `small_table` dataset.

    ```{python}
    import pointblank as pb

    # Define a basic YAML validation workflow
    yaml_config = '''
    tbl: small_table
    steps:
    - rows_distinct
    - col_exists:
        columns: [date, a, b]
    '''

    # Execute the validation workflow
    result = pb.yaml_interrogate(yaml_config)
    result
    ```

    The validation table shows the results of our YAML-defined workflow. We can see that the
    `~~pointblank.Validate.rows_distinct()` validation failed (because there are duplicate rows in
    the table), while the column existence checks passed.

    Now let's create a more comprehensive validation workflow with thresholds and metadata:

    ```{python}
    # Advanced YAML configuration with thresholds and metadata
    yaml_config = '''
    tbl: small_table
    tbl_name: small_table_demo
    label: Comprehensive data validation
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - col_vals_gt:
        columns: [d]
        value: 100
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    - col_vals_not_null:
        columns: [date, a]
    '''

    # Execute the validation workflow
    result = pb.yaml_interrogate(yaml_config)
    print(f"Table name: {result.tbl_name}")
    print(f"Label: {result.label}")
    print(f"Total validation steps: {len(result.validation_info)}")
    ```

    The validation results now include our custom table name and label. The thresholds we defined
    will determine when validation steps are marked as warnings, errors, or critical failures.

    You can also load YAML configurations from files. Here's how you would work with a YAML file:

    ```{python}
    from pathlib import Path
    import tempfile

    # Create a temporary YAML file for demonstration
    yaml_content = '''
    tbl: small_table
    tbl_name: File-based Validation
    steps:
    - col_vals_between:
        columns: [c]
        left: 1
        right: 10
    - col_vals_in_set:
        columns: [f]
        set: [low, mid, high]
    '''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_file_path = Path(f.name)

    # Load and execute validation from file
    result = pb.yaml_interrogate(yaml_file_path)
    result
    ```

    This approach is particularly useful for storing validation configurations as part of your data
    pipeline or version control system, allowing you to maintain validation rules alongside your
    code.

    ### Governance Metadata

    YAML workflows support governance metadata via `owner`, `consumers`, and `version` top-level
    keys. These are forwarded to the `Validate` constructor and embedded in the validation report:

    ```{python}
    yaml_config = '''
    tbl: small_table
    tbl_name: sales_pipeline
    owner: Data Engineering
    consumers: [Analytics, Finance, Compliance]
    version: "2.1.0"
    steps:
    - col_vals_not_null:
        columns: [a, b]
    '''

    result = pb.yaml_interrogate(yaml_config)
    print(f"Owner: {result.owner}")
    print(f"Consumers: {result.consumers}")
    print(f"Version: {result.version}")
    ```

    ### Aggregate Validations

    YAML supports aggregate validation methods for checking column-level statistics. These methods
    validate that a column's sum, average, or standard deviation meets a threshold:

    ```{python}
    yaml_config = '''
    tbl: small_table
    steps:
    - col_sum_gt:
        columns: [d]
        value: 0
    - col_avg_le:
        columns: [a]
        value: 10
    '''

    result = pb.yaml_interrogate(yaml_config)
    result
    ```

    The 15 available aggregate methods follow the pattern `col_{stat}_{comparator}` where
    `{stat}` is `sum`, `avg`, or `sd` and `{comparator}` is `gt`, `lt`, `ge`,
    `le`, or `eq`.

    ### Data Freshness

    Check that a date/datetime column has recent data using `~~pointblank.Validate.data_freshness`:

    ```yaml
    tbl: events.csv
    steps:
    - data_freshness:
        column: event_date
        max_age: "24h"
    ```

    ### Active Parameter Shortcut

    The `active=` parameter controls whether a validation step runs. It supports boolean values
    and Python expression shortcuts:

    ```yaml
    steps:
    - col_vals_gt:
        columns: [d]
        value: 100
        active: false            # Skip this step

    - col_vals_not_null:
        columns: [a]
        active: true             # Always run (default)

    - col_vals_not_null:
        columns: [e]
        active: pb.has_columns("e")  # Run only if column `e` exists
    ```

    ### Null Percentage Check

    Use `~~pointblank.Validate.col_pct_null` to validate that the percentage of null values in a
    column is within bounds:

    ```yaml
    steps:
    - col_pct_null:
        columns: [a, b]
        value: 0.05
    ```

    ### Using `set_tbl=` to Override the Table

    The `set_tbl=` parameter allows you to override the table specified in the YAML configuration.
    This is useful when you have a template validation workflow but want to apply it to different
    tables:

    ```{python}
    import polars as pl

    # Create a test table with similar structure to small_table
    test_table = pl.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "a": [1, 2, 3],
        "b": ["1-abc-123", "2-def-456", "3-ghi-789"],
        "d": [150, 200, 250]
    })

    # Use the same YAML config but apply it to our test table
    yaml_config = '''
    tbl: small_table  # This will be overridden
    tbl_name: Test Table  # This name will be used
    steps:
    - col_exists:
        columns: [date, a, b, d]
    - col_vals_gt:
        columns: [d]
        value: 100
    '''

    # Execute with table override
    result = pb.yaml_interrogate(yaml_config, set_tbl=test_table)
    print(f"Validation applied to: {result.tbl_name}")
    result
    ```

    This feature makes YAML configurations more reusable and flexible, allowing you to define
    validation logic once and apply it to multiple similar tables.
    """
    validator = YAMLValidator()
    config = validator.load_config(yaml)

    # If `set_tbl=` is provided, build the validation on that table (without loading the config's
    # `tbl`, which may be a placeholder or unavailable) and then interrogate
    if set_tbl is not None:
        validation = validator.build_validation(
            config, namespaces=namespaces, tbl=set_tbl, allow_python=allow_python
        )
        return validation.interrogate()
    else:
        # Standard execution without table override (includes interrogation)
        return validator.execute_workflow(config, namespaces=namespaces, allow_python=allow_python)


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Load YAML configuration from file or string.

    Parameters
    ----------
    file_path
        Path to YAML file or YAML content string

    Returns
    -------
    dict
        Parsed configuration dictionary

    Raises
    ------
    YAMLValidationError
        If the file cannot be loaded or is invalid
    """
    validator = YAMLValidator()
    return validator.load_config(file_path)


def validate_yaml(yaml: Union[str, Path]) -> None:
    """Validate YAML configuration against the expected structure.

    This function validates that a YAML configuration conforms to the expected structure for
    validation workflows. It checks for required fields, proper data types, and valid
    validation method names. This is useful for validating configurations before execution or
    for building configuration editors and validators.

    The function performs comprehensive validation including:

    - required fields ('tbl' and 'steps')
    - proper data types for all fields
    - valid threshold configurations
    - known validation method names
    - proper step configuration structure

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.

    Raises
    ------
    YAMLValidationError
        If the YAML is invalid, malformed, or execution fails. This includes syntax errors,
        missing required fields, unknown validation methods, or data loading failures.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    For the examples here, we'll demonstrate how to validate YAML configurations before using them
    with validation workflows. This is particularly useful for building robust data validation
    systems where you want to catch configuration errors early.

    Let's start with validating a basic configuration:

    ```{python}
    import pointblank as pb

    # Define a basic YAML validation configuration
    yaml_config = '''
    tbl: small_table
    steps:
    - rows_distinct
    - col_exists:
        columns: [a, b]
    '''

    # Validate the configuration: no exception means it's valid
    pb.validate_yaml(yaml_config)
    print("Basic YAML configuration is valid")
    ```

    The function completed without raising an exception, which means our configuration is valid and
    follows the expected structure.

    Now let's validate a more complex configuration with thresholds and metadata:

    ```{python}
    # Complex YAML configuration with all optional fields
    yaml_config = '''
    tbl: small_table
    tbl_name: My Dataset
    label: Quality check
    lang: en
    locale: en
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - rows_distinct
    - col_vals_gt:
        columns: [d]
        value: 100
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    '''

    # Validate the configuration
    pb.validate_yaml(yaml_config)
    print("Complex YAML configuration is valid")

    # Count the validation steps
    import pointblank.yaml as pby
    config = pby.load_yaml_config(yaml_config)
    print(f"Configuration has {len(config['steps'])} validation steps")
    ```

    This configuration includes all the optional metadata fields and complex validation steps,
    demonstrating that the validation handles the full range of supported options.

    Let's see what happens when we try to validate an invalid configuration:

    ```{python}
    # Invalid YAML configuration: missing required 'tbl' field
    invalid_yaml = '''
    steps:
    - rows_distinct
    '''

    try:
        pb.validate_yaml(invalid_yaml)
    except pb.yaml.YAMLValidationError as e:
        print(f"Validation failed: {e}")
    ```

    The validation correctly identifies that our configuration is missing the required `'tbl'`
    field.

    Here's a practical example of using validation in a workflow builder:

    ```{python}
    def safe_yaml_interrogate(yaml_config):
        \"\"\"Safely execute a YAML configuration after validation.\"\"\"
        try:
            # Validate the YAML configuration first
            pb.validate_yaml(yaml_config)
            print("✓ YAML configuration is valid")

            # Then execute the workflow
            result = pb.yaml_interrogate(yaml_config)
            print(f"Validation completed with {len(result.validation_info)} steps")
            return result

        except pb.yaml.YAMLValidationError as e:
            print(f"Configuration error: {e}")
            return None

    # Test with a valid YAML configuration
    test_yaml = '''
    tbl: small_table
    steps:
    - col_vals_between:
        columns: [c]
        left: 1
        right: 10
    '''

    result = safe_yaml_interrogate(test_yaml)
    ```

    This pattern of validating before executing helps build more reliable data validation pipelines
    by catching configuration errors early in the process.

    Note that this function only validates the structure and does not check if the specified data
    source ('tbl') exists or is accessible. Data source validation occurs during execution with
    `yaml_interrogate()`.

    Supported Top-level Keys
    ------------------------
    The following top-level keys are recognized in the YAML configuration:

    - `tbl`: data source specification (required)
    - `steps`: list of validation steps (required)
    - `tbl_name`: human-readable table name
    - `label`: validation description
    - `df_library`: DataFrame library (`"polars"`, `"pandas"`, `"duckdb"`)
    - `lang`: language code
    - `locale`: locale setting
    - `brief`: global brief template
    - `thresholds`: global failure thresholds
    - `actions`: global failure actions
    - `final_actions`: actions triggered after all steps complete
    - `owner`: data owner (governance metadata)
    - `consumers`: data consumers (governance metadata)
    - `version`: validation version string (governance metadata)
    - `reference`: reference table for comparison-based validations
    - `ablation`: an ablation of the validation (used by
      [`yaml_ablation()`](`pointblank.yaml_ablation`), and ignored otherwise)

    Unknown top-level keys are rejected, which catches typos like `tbl_nmae` or `step`.

    Supported Validation Methods
    ----------------------------
    In addition to all standard validation methods (e.g., `~~pointblank.Validate.col_vals_gt`,
    `~~pointblank.Validate.rows_distinct`, `~~pointblank.Validate.col_schema_match`), the following
    methods are also supported:

    - `~~pointblank.Validate.col_pct_null`: check the percentage of null values in a column
    - `~~pointblank.Validate.data_freshness`: check that data is recent
    - aggregate methods: `col_sum_gt`, `col_sum_lt`, `col_sum_ge`, `col_sum_le`,
      `col_sum_eq`, `col_avg_gt`, `col_avg_lt`, `col_avg_ge`, `col_avg_le`,
      `col_avg_eq`, `col_sd_gt`, `col_sd_lt`, `col_sd_ge`, `col_sd_le`, `col_sd_eq`

    See Also
    --------
    yaml_interrogate : execute YAML-based validation workflows
    """
    validator = YAMLValidator()
    config = validator.load_config(yaml)
    # Only validate, don't execute the workflow
    return None


def _data_source_code(spec: str, df_library: str) -> tuple[str, str | None]:
    """
    Python code that loads a `tbl:` or `reference:` data source as `yaml_interrogate()` does.

    Follows `YAMLValidator._load_data_source()` without reading anything: a CSV file is read with
    `df_library`, other paths, globs, URLs and connection strings are passed to `Validate` as
    strings (which it loads in the same way), and a bare name is a `load_dataset()` dataset.
    Returns the code and the import alias it needs (`"pl"`, `"pd"` or `None`).
    """
    is_dataset_name = re.fullmatch(r"[A-Za-z_]\w*", spec) is not None and not Path(spec).exists()
    if is_dataset_name:
        return f'pb.load_dataset("{spec}", tbl_type="{df_library}")', None
    quoted = json.dumps(spec)
    if spec.endswith(".csv") and "://" not in spec:
        if df_library == "polars":
            return f"pl.read_csv({quoted})", "pl"
        if df_library == "pandas":
            return f"pd.read_csv({quoted})", "pd"
    return quoted, None


def yaml_to_python(yaml: Union[str, Path]) -> str:
    """Convert YAML validation configuration to equivalent Python code.

    This function takes a YAML validation configuration and generates the equivalent Python code
    that would produce the same validation workflow. This is useful for documentation, code
    generation, or learning how to translate YAML workflows into programmatic workflows.

    The generated Python code includes all necessary imports, data loading, validation steps,
    and interrogation execution, formatted as executable Python code.

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.

    Returns
    -------
    str
        A formatted Python code string enclosed in markdown code blocks that replicates the YAML
        workflow. The code includes import statements, data loading, validation method calls, and
        interrogation execution.

    Raises
    ------
    YAMLValidationError
        If the YAML is invalid, malformed, or contains unknown validation methods.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Convert a basic YAML configuration to Python code:

    ```{python}
    import pointblank as pb

    # Define a YAML validation workflow
    yaml_config = '''
    tbl: small_table
    tbl_name: Data Quality Check
    steps:
    - col_vals_not_null:
        columns: [a, b]
    - col_vals_gt:
        columns: [c]
        value: 0
    '''

    # Generate equivalent Python code
    python_code = pb.yaml_to_python(yaml_config)
    print(python_code)
    ```

    The generated Python code shows exactly how to replicate the YAML workflow programmatically.
    This is particularly useful when transitioning from YAML-based workflows to code-based
    workflows, or when generating documentation that shows both YAML and Python approaches.

    For more complex workflows with thresholds and metadata:

    ```{python}
    # Advanced YAML configuration
    yaml_config = '''
    tbl: small_table
    tbl_name: Advanced Validation
    label: Production data check
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - col_vals_between:
        columns: [c]
        left: 1
        right: 10
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    '''

    # Generate the equivalent Python code
    python_code = pb.yaml_to_python(yaml_config)
    print(python_code)
    ```

    The generated code includes all configuration parameters, thresholds, and maintains the exact
    same validation logic as the original YAML workflow.

    The `tbl` and `reference` data sources are loaded as `yaml_interrogate()` loads them: a
    dataset name becomes a `pb.load_dataset()` call, a CSV file is read with the `df_library`
    library (e.g., `pl.read_csv()`), and other file paths and connection strings are passed to
    `Validate` as strings.

    Python code in the configuration (`python:` blocks, and the `expr:`, `pre:`, and `active:`
    shortcuts) is copied into the generated code as written, without being run. So, unlike
    [`yaml_interrogate()`](`pointblank.yaml_interrogate`), this function doesn't need
    `allow_python=True`, though you should still read the generated code before running it.

    Governance metadata (`owner`, `consumers`, `version`) and `reference` are also rendered
    in the generated Python code:

    ```{python}
    yaml_config = '''
    tbl: small_table
    tbl_name: Sales Pipeline
    owner: Data Engineering
    consumers: [Analytics, Finance]
    version: "2.1.0"
    steps:
    - col_vals_not_null:
        columns: [a]
    - col_sum_gt:
        columns: [d]
        value: 0
    '''

    python_code = pb.yaml_to_python(yaml_config)
    print(python_code)
    ```

    This function is also useful for educational purposes, helping users understand how YAML
    configurations map to the underlying Python API calls.
    """
    # First, parse the raw YAML to detect Polars/Pandas expressions in the source code
    if isinstance(yaml, Path):
        yaml_content = yaml.read_text()  # pragma: no cover
    elif isinstance(yaml, str):
        # Check if it's a file path (single line, reasonable length, no newlines)
        if len(yaml) < 260 and "\n" not in yaml and Path(yaml).exists():
            yaml_content = Path(yaml).read_text()
        else:
            yaml_content = yaml
    else:
        yaml_content = str(yaml)  # pragma: no cover

    # Track whether we need to import Polars and Pandas by analyzing the raw YAML content
    needs_polars_import = False
    needs_pandas_import = False

    # Check for polars/pandas patterns in the raw YAML content
    if "pd." in yaml_content or "pandas" in yaml_content:
        needs_pandas_import = True
    if "pl." in yaml_content or "polars" in yaml_content:
        needs_polars_import = True

    # Parse the raw YAML to extract original Python expressions before they get processed
    import yaml as yaml_module

    raw_config = yaml_module.safe_load(yaml_content)

    # Extract the original tbl python expression if it exists
    original_tbl_expression = None
    if isinstance(raw_config.get("tbl"), dict) and "python" in raw_config["tbl"]:
        original_tbl_expression = raw_config["tbl"]["python"].strip()

    # Extract the original reference python expression if it exists
    original_reference_expression = None
    if isinstance(raw_config.get("reference"), dict) and "python" in raw_config["reference"]:
        original_reference_expression = raw_config["reference"]["python"].strip()

    # Extract original Actions expressions if they exist
    original_actions_expressions = {}
    if "actions" in raw_config:
        for key, value in raw_config["actions"].items():
            if isinstance(value, dict) and "python" in value:
                original_actions_expressions[key] = value["python"].strip()

    # Define function for recursively extract original Python expressions from step parameters
    def extract_python_expressions(obj, path=""):
        expressions = {}
        if isinstance(obj, dict):
            if "python" in obj and len(obj) == 1:
                expressions[path] = obj["python"].strip()
            else:
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    # Special handling for `expr=`, `pre=`, and `active=` parameters
                    # that can use shortcut syntax
                    if key in ["expr", "pre", "active"] and isinstance(value, str):
                        expressions[new_path] = value.strip()
                    # Special handling for actions that might contain python: expressions
                    elif key == "actions" and isinstance(value, dict):
                        for action_key, action_value in value.items():
                            if isinstance(action_value, dict) and "python" in action_value:
                                expressions[f"{new_path}.{action_key}"] = action_value[
                                    "python"
                                ].strip()
                    else:
                        expressions.update(extract_python_expressions(value, new_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                expressions.update(extract_python_expressions(item, new_path))
        return expressions

    step_expressions = {}
    if "steps" in raw_config:
        for i, step in enumerate(raw_config["steps"]):
            if isinstance(step, dict):
                step_expressions.update(extract_python_expressions(step, f"steps[{i}]"))

    # Load and validate the YAML configuration
    validator = YAMLValidator()
    config = validator.load_config(yaml)

    # Parse any named missing specs so steps referencing them can be rendered
    missing_specs = validator._parse_missing_specs(config)

    # Start building the Python code
    code_lines = []

    # Add imports (we'll determine Polars/Pandas import need during processing)
    imports = ["import pointblank as pb"]

    # Build the chained validation call
    code_lines.append("(")

    # Build validation initialization arguments
    validate_args = []

    # Add data loading as first argument
    tbl_spec = config["tbl"]
    df_library = config.get("df_library", "polars")

    # Use the original Python expression if we extracted it (df_library is ignored in this case)
    if original_tbl_expression:
        validate_args.append(f"data={original_tbl_expression}")
    elif isinstance(tbl_spec, str):
        tbl_code, tbl_import = _data_source_code(tbl_spec, df_library)
        validate_args.append(f"data={tbl_code}")
        needs_polars_import = needs_polars_import or tbl_import == "pl"
        needs_pandas_import = needs_pandas_import or tbl_import == "pd"
    else:
        # Fallback to placeholder if we couldn't extract the original expression
        validate_args.append("data=<python_expression_result>")  # pragma: no cover

    # Add table name if present
    if "tbl_name" in config:
        validate_args.append(f'tbl_name="{config["tbl_name"]}"')

    # Add `label=` if present
    if "label" in config:
        validate_args.append(f'label="{config["label"]}"')

    # Add `thresholds=` if present: format as `pb.Thresholds()` for an idiomatic style
    if "thresholds" in config:
        thresholds_dict = config["thresholds"]
        threshold_params = []
        for key, value in thresholds_dict.items():
            threshold_params.append(f"{key}={value}")
        thresholds_str = "pb.Thresholds(" + ", ".join(threshold_params) + ")"
        validate_args.append(f"thresholds={thresholds_str}")

    # Add `actions=` if present: format as `pb.Actions()` for an idiomatic style
    if "actions" in config:
        actions_dict = config["actions"]
        action_params = []
        for key, value in actions_dict.items():
            if key == "highest_only":
                action_params.append(f"{key}={value}")
            elif key in original_actions_expressions:
                # Use the original Python expression for callables
                action_params.append(f"{key}={original_actions_expressions[key]}")
            elif isinstance(value, str):
                action_params.append(f'{key}="{value}"')
            else:
                # For callables or complex expressions, use placeholder
                action_params.append(f"{key}={value}")  # pragma: no cover
        actions_str = "pb.Actions(" + ", ".join(action_params) + ")"
        validate_args.append(f"actions={actions_str}")

    # Add language if present
    if "lang" in config:
        validate_args.append(f'lang="{config["lang"]}"')

    # Add locale if present
    if "locale" in config:
        validate_args.append(f'locale="{config["locale"]}"')

    # Add global brief if present
    if "brief" in config:
        if isinstance(config["brief"], bool):
            validate_args.append(f"brief={str(config['brief'])}")
        else:
            validate_args.append(f'brief="{config["brief"]}"')

    # Add owner if present (governance)
    if "owner" in config:
        validate_args.append(f'owner="{config["owner"]}"')

    # Add consumers if present (governance)
    if "consumers" in config:
        consumers = config["consumers"]
        if isinstance(consumers, list):
            if len(consumers) == 1:
                validate_args.append(f'consumers="{consumers[0]}"')
            else:
                consumers_str = "[" + ", ".join([f'"{c}"' for c in consumers]) + "]"
                validate_args.append(f"consumers={consumers_str}")
        elif isinstance(consumers, str):
            validate_args.append(f'consumers="{consumers}"')

    # Add version if present (governance)
    if "version" in config:
        validate_args.append(f'version="{config["version"]}"')

    # Add reference if present
    if "reference" in config:
        ref_spec = config["reference"]
        if original_reference_expression:
            validate_args.append(f"reference={original_reference_expression}")
        elif isinstance(ref_spec, str):
            ref_code, ref_import = _data_source_code(ref_spec, df_library)
            validate_args.append(f"reference={ref_code}")
            needs_polars_import = needs_polars_import or ref_import == "pl"
            needs_pandas_import = needs_pandas_import or ref_import == "pd"

    # Create the `pb.Validate()` call
    if len(validate_args) == 1:
        # Single argument fits on one line
        code_lines.append(f"    pb.Validate({validate_args[0]})")
    else:
        # Multiple arguments: format each on its own line
        code_lines.append("    pb.Validate(")
        for i, arg in enumerate(validate_args):
            if i == len(validate_args) - 1:
                code_lines.append(f"        {arg},")
            else:
                code_lines.append(f"        {arg},")
        code_lines.append("    )")

    # Add validation steps as chained method calls
    for step_index, step_config in enumerate(config["steps"]):
        # Get original expressions before parsing
        original_expressions = {}

        # Handle string steps (parameterless methods like "rows_distinct")
        if isinstance(step_config, str):
            method_name, parameters = validator._parse_validation_step(
                step_config, namespaces=None, missing_specs=missing_specs, python_mode="source"
            )
            code_lines.append(f"    .{method_name}()")
            continue

        step_method = list(step_config.keys())[
            0
        ]  # Get the method name (conjointly, specially, etc.)
        step_params = step_config[step_method]

        if (
            step_method == "conjointly"
            and isinstance(step_params, dict)
            and "expressions" in step_params
        ):
            original_expressions["expressions"] = step_params["expressions"]

        if step_method == "specially" and isinstance(step_params, dict) and "expr" in step_params:
            if isinstance(step_params["expr"], dict) and "python" in step_params["expr"]:
                original_expressions["expr"] = step_params["expr"]["python"].strip()
            elif isinstance(step_params["expr"], str):
                original_expressions["expr"] = step_params["expr"]

        method_name, parameters = validator._parse_validation_step(
            step_config, namespaces=None, missing_specs=missing_specs, python_mode="source"
        )

        # Apply the original expressions to override the converted lambda functions
        if method_name == "conjointly" and "expressions" in original_expressions:
            # Remove the internal parameter and add expressions as a proper parameter
            if "_conjointly_expressions" in parameters:
                parameters.pop("_conjointly_expressions")
            parameters["expressions"] = original_expressions["expressions"]

        if method_name == "specially" and "expr" in original_expressions:
            parameters["expr"] = original_expressions["expr"]

        # Format parameters
        param_parts = []
        for key, value in parameters.items():
            # Check if we have an original expression for this parameter
            expression_path = f"steps[{step_index}].{list(step_config.keys())[0]}.{key}"

            # Skip using step_expressions for specially/conjointly parameters that we handle specially
            if (
                expression_path in step_expressions
                and not (method_name == "specially" and key == "expr")
                and not (method_name == "conjointly" and key == "expressions")
            ):
                # Use the original Python expression
                param_parts.append(f"{key}={step_expressions[expression_path]}")
            elif (
                key in ("value", "left", "right")
                and isinstance(step_params, dict)
                and isinstance(step_params.get(key), str)
                and _COLUMN_REF_PATTERN.fullmatch(step_params[key])
            ):
                # A column reference written directly (e.g., `value: pb.col("b")`)
                param_parts.append(f"{key}={step_params[key].strip()}")
            elif key == "expressions" and method_name == "conjointly":
                # Handle conjointly expressions list
                if isinstance(value, list):
                    expressions_str = "[" + ", ".join([f'"{expr}"' for expr in value]) + "]"
                    param_parts.append(f"expressions={expressions_str}")
                else:
                    param_parts.append(f"expressions={value}")  # pragma: no cover
            elif key == "expr" and method_name == "specially":
                # Handle specially expr parameter: should be unquoted lambda expression
                if isinstance(value, str):
                    param_parts.append(f"expr={value}")
                else:
                    param_parts.append(f"expr={value}")  # pragma: no cover
            elif key in ["columns", "columns_subset"]:
                if isinstance(value, list):
                    if len(value) == 1:
                        # Single column as string
                        param_parts.append(f'{key}="{value[0]}"')
                    else:
                        # Multiple columns as list
                        columns_str = "[" + ", ".join([f'"{col}"' for col in value]) + "]"
                        param_parts.append(f"{key}={columns_str}")
                else:
                    param_parts.append(f'{key}="{value}"')  # pragma: no cover
            elif key == "missing" and isinstance(value, MissingSpec):
                # Render a resolved MissingSpec as a `pb.MissingSpec(...)` constructor call
                param_parts.append(f"missing={_missing_spec_to_code(value)}")
            elif key == "brief":
                # Handle `brief=` parameter: can be a boolean or a string
                if isinstance(value, bool):
                    param_parts.append(f"brief={str(value)}")
                else:
                    param_parts.append(f'brief="{value}"')
            elif key == "actions":
                # Handle actions parameter: format as `pb.Actions()`
                if isinstance(value, Actions):
                    # Already an `Actions` object, format its attributes
                    action_params = []

                    # Check for original expressions for each action level
                    step_action_base = f"steps[{step_index}].{list(step_config.keys())[0]}.actions"

                    if value.warning is not None:
                        warning_expr_path = f"{step_action_base}.warning"
                        if warning_expr_path in step_expressions:
                            action_params.append(f"warning={step_expressions[warning_expr_path]}")
                        elif isinstance(value.warning, list) and len(value.warning) == 1:
                            action_params.append(f'warning="{value.warning[0]}"')
                        else:
                            action_params.append(f"warning={value.warning}")  # pragma: no cover

                    if value.error is not None:
                        error_expr_path = f"{step_action_base}.error"
                        if error_expr_path in step_expressions:
                            action_params.append(
                                f"error={step_expressions[error_expr_path]}"
                            )  # pragma: no cover
                        elif isinstance(value.error, list) and len(value.error) == 1:
                            action_params.append(f'error="{value.error[0]}"')
                        else:
                            action_params.append(f"error={value.error}")  # pragma: no cover

                    if value.critical is not None:
                        critical_expr_path = f"{step_action_base}.critical"
                        if critical_expr_path in step_expressions:
                            action_params.append(
                                f"critical={step_expressions[critical_expr_path]}"
                            )  # pragma: no cover
                        elif isinstance(value.critical, list) and len(value.critical) == 1:
                            action_params.append(f'critical="{value.critical[0]}"')
                        else:
                            action_params.append(f"critical={value.critical}")  # pragma: no cover

                    if hasattr(value, "highest_only") and value.highest_only is not True:
                        action_params.append(f"highest_only={value.highest_only}")
                    actions_str = "pb.Actions(" + ", ".join(action_params) + ")"
                    param_parts.append(f"actions={actions_str}")
                else:
                    param_parts.append(f"actions={value}")
            elif key == "thresholds":
                # Handle thresholds parameter: format as `pb.Thresholds()`
                if isinstance(value, dict):
                    threshold_params = []
                    for threshold_key, threshold_value in value.items():
                        threshold_params.append(f"{threshold_key}={threshold_value}")
                    thresholds_str = "pb.Thresholds(" + ", ".join(threshold_params) + ")"
                    param_parts.append(f"thresholds={thresholds_str}")
                else:
                    param_parts.append(f"thresholds={value}")
            elif isinstance(value, str):
                param_parts.append(f'{key}="{value}"')
            elif isinstance(value, bool):
                param_parts.append(f"{key}={str(value)}")
            elif isinstance(value, tuple):
                # Handle tuples like `inclusive=(False, True)`
                tuple_str = "(" + ", ".join([_value_code(item) for item in value]) + ")"
                param_parts.append(f"{key}={tuple_str}")
            elif isinstance(value, list):
                # Handle lists/tuples (like `set=` parameter)
                if all(isinstance(item, str) for item in value):
                    list_str = "[" + ", ".join([f'"{item}"' for item in value]) + "]"
                else:
                    list_str = "[" + ", ".join([_value_code(item) for item in value]) + "]"
                param_parts.append(f"{key}={list_str}")
            elif isinstance(value, float):
                param_parts.append(f"{key}={_value_code(value)}")
            else:
                # Handle complex objects (like polars/pandas expressions from python: blocks)
                # For these, we'll use a placeholder since they can't be easily converted back
                param_parts.append(f"{key}={value}")

        if param_parts:
            params_str = ", ".join(param_parts)
            code_lines.append(f"    .{method_name}({params_str})")
        else:
            code_lines.append(f"    .{method_name}()")

    # Add interrogation method call
    code_lines.append("    .interrogate()")
    code_lines.append(")")

    # An `ablation:` block becomes a `pb.Ablation()` of the validation, with its strategies
    if "ablation" in config:
        code_lines[0] = "validation = ("
        ablation_imports, ablation_lines, holdout_import = _ablation_code(
            config["ablation"], df_library
        )
        imports += ablation_imports
        needs_polars_import = needs_polars_import or holdout_import == "pl"
        needs_pandas_import = needs_pandas_import or holdout_import == "pd"
        code_lines += ["", *ablation_lines]

    # Add imports at the beginning
    if needs_polars_import:
        imports.append("import polars as pl")
    if needs_pandas_import:
        imports.append("import pandas as pd")

    # Python code that builds values with `datetime`, `Decimal`, or `ZoneInfo` (e.g.,
    # `Decimal("2.2")`) needs standard-library imports, which go first
    python_sources = [
        *step_expressions.values(),
        *original_actions_expressions.values(),
        original_tbl_expression or "",
        original_reference_expression or "",
    ]
    std_imports = [
        line
        for line, pattern in (
            ("import datetime", r"(?<![\w.])datetime\."),
            ("from decimal import Decimal", r"(?<![\w.])Decimal\("),
            ("from zoneinfo import ZoneInfo", r"(?<![\w.])ZoneInfo\("),
        )
        if any(re.search(pattern, code) for code in python_sources)
    ]
    if std_imports:
        imports = std_imports + [""] + imports

    # Build final code with imports
    final_code_lines = imports + [""] + code_lines

    # Join all code lines and wrap in single markdown code block
    python_code = "\n".join(final_code_lines)
    return f"```python\n{python_code}\n```"
