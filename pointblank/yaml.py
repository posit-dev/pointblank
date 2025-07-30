from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

from pointblank._utils import _is_lib_present
from pointblank.thresholds import Actions
from pointblank.validate import Validate, load_dataset


class YAMLValidationError(Exception):
    """Exception raised for YAML validation errors."""

    pass


def _safe_eval_python_code(code: str) -> Any:
    """Safely evaluate Python code with restricted namespace.

    This function provides a controlled environment for executing Python code embedded in YAML
    configurations. It includes common libraries and functions while restricting access to
    dangerous operations.

    Parameters
    ----------
    code
        The Python code to evaluate.

    Returns
    -------
    Any
        The result of evaluating the Python code.

    Raises
    ------
    YAMLValidationError
        If the code execution fails or contains unsafe operations.
    """
    import ast
    import re
    from pathlib import Path

    from pointblank._utils import _is_lib_present

    # Create a safe namespace with commonly needed imports
    safe_namespace = {
        "Path": Path,  # pathlib.Path
        "__builtins__": {
            # Allow basic built-in functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "print": print,
        },
    }

    # Add pointblank itself to the namespace
    import pointblank as pb

    safe_namespace["pb"] = pb

    # Add polars if available
    if _is_lib_present("polars"):
        import polars as pl

        safe_namespace["pl"] = pl

    # Add pandas if available
    if _is_lib_present("pandas"):
        import pandas as pd

        safe_namespace["pd"] = pd

    # Check for dangerous patterns
    dangerous_patterns = [
        r"import\s+os",
        r"import\s+sys",
        r"import\s+subprocess",
        r"__import__",
        r"exec\s*\(",
        r"eval\s*\(",
        r"open\s*\(",
        r"file\s*\(",
        r"input\s*\(",
        r"raw_input\s*\(",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            raise YAMLValidationError(
                f"Potentially unsafe Python code detected: '{code}'. "
                f"Pattern '{pattern}' is not allowed."
            )

    try:
        # First try to parse as expression for simple cases
        try:
            parsed = ast.parse(code, mode="eval")
            return eval(compile(parsed, "<string>", "eval"), safe_namespace)
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
                    exec(compile(statements_module, "<string>", "exec"), safe_namespace)

                # Then evaluate the last expression and return its value
                expr_module = ast.Expression(body=last_expr)
                return eval(compile(expr_module, "<string>", "eval"), safe_namespace)
            else:
                # No expression at the end, just execute statements
                exec(compile(parsed, "<string>", "exec"), safe_namespace)
                return None

    except Exception as e:
        raise YAMLValidationError(f"Error executing Python code '{code}': {e}")


def _process_python_expressions(value: Any) -> Any:
    """Process Python code snippets embedded in YAML values.

    This function supports the python: block syntax for embedding Python code:

    python: |
      import polars as pl
      pl.scan_csv("data.csv").head(10)

    Note: col_vals_expr() also supports a shortcut syntax where the expr parameter
    can be written directly without the python: wrapper:

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
    >>> _process_python_expressions({"python": "pl.scan_csv('data.csv').head(10)"})
    # Returns the result of the Python expression

    >>> _process_python_expressions({"python": "import polars as pl\\npl.scan_csv('data.csv')"})
    # Returns the result of multiline Python code
    """
    if isinstance(value, dict):
        # Handle python: block syntax
        if "python" in value and len(value) == 1:
            code = value["python"]
            return _safe_eval_python_code(code)

        # Recursively process dictionary values
        return {k: _process_python_expressions(v) for k, v in value.items()}

    elif isinstance(value, list):
        # Recursively process list items
        return [_process_python_expressions(item) for item in value]

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
        "col_vals_not_in_set": "col_vals_not_in_set",
        "col_vals_not_null": "col_vals_not_null",
        "col_vals_null": "col_vals_null",
        "col_vals_expr": "col_vals_expr",
        "rows_distinct": "rows_distinct",
        "rows_complete": "rows_complete",
        "col_count_match": "col_count_match",
        "row_count_match": "row_count_match",
        "col_schema_match": "col_schema_match",
        "conjointly": "conjointly",
        "specially": "specially",
    }

    def __init__(self):
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
        # Check required fields
        if "tbl" not in config:
            raise YAMLValidationError("YAML must contain 'tbl' field")

        if "steps" not in config:
            raise YAMLValidationError("YAML must contain 'steps' field")

        if not isinstance(config["steps"], list):
            raise YAMLValidationError("'steps' must be a list")

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

    def _load_data_source(self, tbl_spec: str, df_library: str = "polars") -> Any:
        """Load data source based on table specification.

        Parameters
        ----------
        tbl_spec
            Data source specification. Can be (1) a dataset name for `load_dataset()`, (2) a CSV file
            path (relative or absolute), (3) a Parquet file path (relative or absolute), or (4) a
            Python code snippet to be executed for dynamic data loading.
        df_library
            DataFrame library to use for loading datasets and CSV files. Options: "polars", "pandas", "duckdb".

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
            processed_tbl_spec = _process_python_expressions(tbl_spec)

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
                return processed_data

        except Exception as e:
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
                from pointblank.validate import _process_data

                return _process_data(file_path)

            else:
                raise YAMLValidationError(
                    f"Unsupported df_library: {df_library}. Use 'polars', 'pandas', or 'duckdb'"
                )

        except Exception as e:
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

    def _parse_validation_step(self, step_config: Union[str, dict]) -> tuple[str, dict]:
        """Parse a single validation step from YAML configuration.

        Parameters
        ----------
        step_config
            Step configuration (string for parameterless steps, dict for others).

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

        # Validate that we know this method
        if method_name not in self.validation_method_map:
            available_methods = list(self.validation_method_map.keys())
            raise YAMLValidationError(
                f"Unknown validation method '{method_name}'. Available methods: {available_methods}"
            )

        # Process Python expressions in all parameters
        processed_parameters = {}
        for key, value in parameters.items():
            # Special case: `col_vals_expr()`'s `expr=` parameter can use shortcut syntax
            if method_name == "col_vals_expr" and key == "expr" and isinstance(value, str):
                # Treat string directly as Python code (shortcut syntax)
                processed_parameters[key] = _safe_eval_python_code(value)
            # Special case: `pre=` parameter can use shortcut syntax (like `expr=`)
            elif key == "pre" and isinstance(value, str):
                # Treat string directly as Python code (shortcut syntax)
                processed_parameters[key] = _safe_eval_python_code(value)
            else:
                # Normal processing (requires python: block syntax)
                processed_parameters[key] = _process_python_expressions(value)
        parameters = processed_parameters

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
                        lambda_expressions.append(_safe_eval_python_code(expr))
                    else:
                        lambda_expressions.append(expr)
                # Pass expressions as positional arguments (stored as special key)
                parameters["_conjointly_expressions"] = lambda_expressions
            else:
                raise YAMLValidationError("conjointly 'expressions' must be a list")

        # Handle `specially()` expr parameter: support shortcut syntax
        if method_name == "specially" and "expr" in parameters:
            expr_value = parameters["expr"]
            if isinstance(expr_value, str):
                # Treat string directly as Python code (shortcut syntax)
                parameters["expr"] = _safe_eval_python_code(expr_value)

        # Convert `actions=` if present (ensure it's an Actions object)
        if "actions" in parameters:
            if isinstance(parameters["actions"], dict):
                parameters["actions"] = Actions(**parameters["actions"])

        # Handle `inclusive=` parameter for `col_vals_[inside|outside]()` (convert list to tuple)
        if "inclusive" in parameters and isinstance(parameters["inclusive"], list):
            parameters["inclusive"] = tuple(parameters["inclusive"])

        return self.validation_method_map[method_name], parameters

    def build_validation(self, config: dict) -> Validate:
        """Convert YAML config to Validate object.

        Parameters
        ----------
        config
            Validated configuration dictionary.

        Returns
        -------
        Validate
            Validate object with configured validation steps.
        """
        # Load data source with specified library
        df_library = config.get("df_library", "polars")
        data = self._load_data_source(config["tbl"], df_library)

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
            processed_actions = _process_python_expressions(config["actions"])
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

        validation = Validate(data, **validate_kwargs)

        # Add validation steps
        for step_config in config["steps"]:
            method_name, parameters = self._parse_validation_step(step_config)

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

    def execute_workflow(self, config: dict) -> Validate:
        """Execute a complete YAML validation workflow.

        Parameters
        ----------
        config
            Validated configuration dictionary.

        Returns
        -------
        Validate
            Interrogated Validate object with results.
        """
        # Build the validation plan
        validation = self.build_validation(config)

        # Execute interrogation to get results
        validation = validation.interrogate()

        return validation


def yaml_interrogate(yaml: Union[str, Path]) -> Validate:
    """Execute a YAML-based validation workflow.

    This is the main entry point for YAML-based validation workflows. It takes YAML configuration
    (as a string or file path) and returns a validated `Validate` object with interrogation results.

    The YAML configuration defines the data source, validation steps, and optional settings like
    thresholds and labels. This function automatically loads the data, builds the validation plan,
    executes all validation steps, and returns the interrogated results.

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.

    Returns
    -------
    Validate
        An instance of the `Validate` class that has been configured based on the YAML input.
        This object contains the results of the validation steps defined in the YAML configuration.
        It includes metadata like table name, label, language, and thresholds if specified.

    Raises
    ------
    YAMLValidationError
        If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing
        required fields, unknown validation methods, or data loading failures.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
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
    `rows_distinct()` validation failed (because there are duplicate rows in the table), while the
    column existence checks passed.

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
    """
    validator = YAMLValidator()
    config = validator.load_config(yaml)
    return validator.execute_workflow(config)


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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
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

    See Also
    --------
    yaml_interrogate : execute YAML-based validation workflows
    """
    validator = YAMLValidator()
    config = validator.load_config(yaml)
    # Only validate, don't execute the workflow
    return None


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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
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

    This function is also useful for educational purposes, helping users understand how YAML
    configurations map to the underlying Python API calls.
    """
    # First, parse the raw YAML to detect Polars/Pandas expressions in the source code
    if isinstance(yaml, Path):
        yaml_content = yaml.read_text()
    elif isinstance(yaml, str):
        # Check if it's a file path (single line, reasonable length, no newlines)
        if len(yaml) < 260 and "\n" not in yaml and Path(yaml).exists():
            yaml_content = Path(yaml).read_text()
        else:
            yaml_content = yaml
    else:
        yaml_content = str(yaml)

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
                    # Special handling for `expr=` and `pre=` parameters that
                    # can use shortcut syntax
                    if key in ["expr", "pre"] and isinstance(value, str):
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
        if tbl_spec.endswith((".csv", ".parquet")):
            # File loading
            validate_args.append(f'data=pb.load_dataset("{tbl_spec}", tbl_type="{df_library}")')
        else:
            # Dataset loading
            validate_args.append(f'data=pb.load_dataset("{tbl_spec}", tbl_type="{df_library}")')
    else:
        # Fallback to placeholder if we couldn't extract the original expression
        validate_args.append("data=<python_expression_result>")

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
                action_params.append(f"{key}={value}")
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

        method_name, parameters = validator._parse_validation_step(step_config)

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
            elif key == "expressions" and method_name == "conjointly":
                # Handle conjointly expressions list
                if isinstance(value, list):
                    expressions_str = "[" + ", ".join([f'"{expr}"' for expr in value]) + "]"
                    param_parts.append(f"expressions={expressions_str}")
                else:
                    param_parts.append(f"expressions={value}")
            elif key == "expr" and method_name == "specially":
                # Handle specially expr parameter: should be unquoted lambda expression
                if isinstance(value, str):
                    param_parts.append(f"expr={value}")
                else:
                    param_parts.append(f"expr={value}")
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
                    param_parts.append(f'{key}="{value}"')
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
                            action_params.append(f"warning={value.warning}")

                    if value.error is not None:
                        error_expr_path = f"{step_action_base}.error"
                        if error_expr_path in step_expressions:
                            action_params.append(f"error={step_expressions[error_expr_path]}")
                        elif isinstance(value.error, list) and len(value.error) == 1:
                            action_params.append(f'error="{value.error[0]}"')
                        else:
                            action_params.append(f"error={value.error}")

                    if value.critical is not None:
                        critical_expr_path = f"{step_action_base}.critical"
                        if critical_expr_path in step_expressions:
                            action_params.append(f"critical={step_expressions[critical_expr_path]}")
                        elif isinstance(value.critical, list) and len(value.critical) == 1:
                            action_params.append(f'critical="{value.critical[0]}"')
                        else:
                            action_params.append(f"critical={value.critical}")

                    if hasattr(value, "highest_only") and value.highest_only is not True:
                        action_params.append(f"highest_only={value.highest_only}")
                    actions_str = "pb.Actions(" + ", ".join(action_params) + ")"
                    param_parts.append(f"actions={actions_str}")
                elif isinstance(value, dict):
                    action_params = []
                    step_action_base = f"steps[{step_index}].{list(step_config.keys())[0]}.actions"
                    for action_key, action_value in value.items():
                        if action_key == "highest_only":
                            action_params.append(f"{action_key}={action_value}")
                        else:
                            # Check if we have an original expression for this action
                            action_expr_path = f"{step_action_base}.{action_key}"
                            if action_expr_path in step_expressions:
                                action_params.append(
                                    f"{action_key}={step_expressions[action_expr_path]}"
                                )
                            elif isinstance(action_value, str):
                                action_params.append(f'{action_key}="{action_value}"')
                            else:
                                # For callables or complex expressions
                                action_params.append(f"{action_key}={action_value}")
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
                tuple_str = "(" + ", ".join([str(item) for item in value]) + ")"
                param_parts.append(f"{key}={tuple_str}")
            elif isinstance(value, list):
                # Handle lists/tuples (like `set=` parameter)
                if all(isinstance(item, str) for item in value):
                    list_str = "[" + ", ".join([f'"{item}"' for item in value]) + "]"
                else:
                    list_str = str(list(value))
                param_parts.append(f"{key}={list_str}")
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

    # Add imports at the beginning
    if needs_polars_import:
        imports.append("import polars as pl")
    if needs_pandas_import:
        imports.append("import pandas as pd")

    # Build final code with imports
    final_code_lines = imports + [""] + code_lines

    # Join all code lines and wrap in single markdown code block
    python_code = "\n".join(final_code_lines)
    return f"```python\n{python_code}\n```"
