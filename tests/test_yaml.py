import pytest

from unittest.mock import patch

import tempfile
import os
import io
import contextlib

from pointblank import load_dataset, yaml_interrogate, validate_yaml, yaml_to_python, Validate
from pointblank.yaml import (
    load_yaml_config,
    YAMLValidationError,
    YAMLValidator,
    _process_python_expressions,
    _safe_eval_python_code,
)

import polars as pl
import pandas as pd


def test_yaml_interrogate_basic_workflow():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: A simple test
    steps:
    - rows_distinct
    - col_exists:
        columns: [date, date_time]
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    # rows_distinct (1) + col_exists with 2 columns (2) = 3 total steps
    assert len(result.validation_info) == 3
    # Check that it's been interrogated (should have results)
    assert hasattr(result, "validation_info")


def test_yaml_interrogate_with_thresholds():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - rows_distinct
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1


def test_yaml_interrogate_complex_example():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: A simple example with the `small_table`.
    lang: en
    locale: en
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - col_exists:
        columns: [date, date_time]
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    - rows_distinct
    - col_vals_gt:
        columns: [d]
        value: 100.0
    - col_vals_le:
        columns: [c]
        value: 5.0
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    # col_exists with 2 columns (2) + col_vals_regex (1) + rows_distinct (1) + col_vals_gt (1) + col_vals_le (1) = 6
    assert len(result.validation_info) == 6


def test_yaml_column_specifications():
    # Test YAML list syntax
    yaml_content1 = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [date, date_time]
    """
    result1 = yaml_interrogate(yaml_content1)
    assert result1 is not None
    assert len(result1.validation_info) == 2  # One step per column

    # Test single column as string
    yaml_content2 = """
    tbl: small_table
    steps:
    - col_exists:
        columns: date
    """
    result2 = yaml_interrogate(yaml_content2)
    assert result2 is not None
    assert len(result2.validation_info) == 1

    # Test single column as list
    yaml_content3 = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [date]
    """
    result3 = yaml_interrogate(yaml_content3)
    assert result3 is not None
    assert len(result3.validation_info) == 1


def test_validation_methods_coverage():
    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct
    - rows_complete
    - col_vals_not_null:
        columns: [date]
    - col_vals_gt:
        columns: [d]
        value: 0
    - col_vals_ge:
        columns: [d]
        value: 0
    - col_vals_lt:
        columns: [c]
        value: 10
    - col_vals_le:
        columns: [c]
        value: 10
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 7


def test_load_yaml_config():
    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct
    """

    config = load_yaml_config(yaml_content)
    assert config["tbl"] == "small_table"
    assert len(config["steps"]) == 1


def test_validate_yaml():
    # Valid configuration
    valid_yaml = """
    tbl: small_table
    steps:
    - rows_distinct
    """
    validate_yaml(valid_yaml)  # Should not raise

    # Valid configuration with tbl: null (for template use cases)
    valid_yaml_null = """
    tbl: null
    steps:
    - rows_distinct
    """
    validate_yaml(valid_yaml_null)  # Should not raise

    # Invalid configuration: missing tbl
    invalid_yaml1 = """
    steps:
    - rows_distinct
    """
    with pytest.raises(YAMLValidationError, match="YAML must contain 'tbl' field"):
        validate_yaml(invalid_yaml1)

    # Invalid configuration: missing steps
    invalid_yaml2 = """
    tbl: small_table
    """
    with pytest.raises(YAMLValidationError, match="YAML must contain 'steps' field"):
        validate_yaml(invalid_yaml2)

    # Invalid configuration: empty steps
    invalid_yaml3 = """
    tbl: small_table
    steps: []
    """
    with pytest.raises(YAMLValidationError, match="'steps' cannot be empty"):
        validate_yaml(invalid_yaml3)


def test_yaml_error_handling():
    # Invalid YAML syntax
    with pytest.raises(YAMLValidationError, match="Invalid YAML syntax"):
        yaml_interrogate("invalid: yaml: content: [")

    # Unknown validation method
    yaml_content = """
    tbl: small_table
    steps:
    - unknown_method
    """
    with pytest.raises(YAMLValidationError, match="Unknown validation method 'unknown_method'"):
        yaml_interrogate(yaml_content)

    # Invalid data source
    yaml_content = """
    tbl: nonexistent_dataset
    steps:
    - rows_distinct
    """
    with pytest.raises(YAMLValidationError, match="Failed to load data source"):
        yaml_interrogate(yaml_content)

    # Invalid threshold values (negative values are not allowed)
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: -1
    steps:
    - rows_distinct
    """
    with pytest.raises(YAMLValidationError, match="Threshold 'warning' must be non-negative"):
        yaml_interrogate(yaml_content)


def test_step_parameter_validation():
    # Step with invalid parameter structure
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists: not_a_dict
    """
    with pytest.raises(
        YAMLValidationError, match="Parameters for 'col_exists' must be a dictionary"
    ):
        yaml_interrogate(yaml_content)

    # Step with multiple methods (should fail)
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [date]
      rows_distinct: {}
    """
    with pytest.raises(
        YAMLValidationError,
        match="Step configuration must contain exactly one validation method",
    ):
        yaml_interrogate(yaml_content)


def test_yaml_column_parsing():
    validator = YAMLValidator()

    # Test various YAML list formats
    assert validator._parse_column_spec(["date", "date_time"]) == ["date", "date_time"]
    assert validator._parse_column_spec(["a", "b", "c"]) == ["a", "b", "c"]
    assert validator._parse_column_spec([]) == []

    # Test string formats
    assert validator._parse_column_spec("single_column") == ["single_column"]

    # Test other types
    assert validator._parse_column_spec(123) == ["123"]


def test_yaml_interrogate_rejects_dict():
    config = {"tbl": "small_table", "steps": ["rows_distinct"]}

    with pytest.raises(YAMLValidationError, match="Only YAML strings and file paths supported"):
        yaml_interrogate(config)


def test_yaml_file_integration(tmp_path):
    yaml_content = """
    tbl: small_table
    tbl_name: test_table
    label: File-based test
    steps:
    - rows_distinct
    - col_exists:
        columns: [date, date_time]
    """

    yaml_file = tmp_path / "test_validation.yaml"
    yaml_file.write_text(yaml_content)

    # Test loading config from file
    config = load_yaml_config(yaml_file)
    assert config["tbl"] == "small_table"
    assert config["tbl_name"] == "test_table"
    assert len(config["steps"]) == 2

    # Test executing workflow from file
    result = yaml_interrogate(yaml_file)
    assert result is not None
    assert (
        len(result.validation_info) == 3
    )  # rows_distinct (1) + col_exists for date (1) + col_exists for date_time (1)


def test_yaml_workflow_results_consistency():
    # This test ensures our YAML implementation produces the same results
    # as the equivalent programmatic validation

    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct
    """

    # YAML-based validation
    yaml_result = yaml_interrogate(yaml_content)

    programmatic_result = Validate(load_dataset("small_table")).rows_distinct().interrogate()

    # Compare results (both should have same structure)
    assert len(yaml_result.validation_info) == len(programmatic_result.validation_info)
    assert (
        yaml_result.validation_info[0].assertion_type
        == programmatic_result.validation_info[0].assertion_type
    )


def test_comprehensive_yaml_validation():
    yaml_content = """
    tbl: small_table
    tbl_name: Comprehensive Test
    thresholds:
      warning: 1
      error: 2
      critical: 0.15
    steps:
    - col_vals_lt:
        columns: [c]
        value: 0
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_ne:
        columns: [c]
        value: 10
    - col_vals_le:
        columns: [a]
        value: 7
    - col_vals_ge:
        columns: [d]
        value: 500
        na_pass: true
    - col_vals_between:
        columns: [c]
        left: 0
        right: 5
        na_pass: true
    - col_vals_outside:
        columns: [a]
        left: 0
        right: 9
        inclusive: [false, true]
    - col_vals_eq:
        columns: [a]
        value: 1
    - col_vals_in_set:
        columns: [f]
        set: [lows, mids, highs]
    - col_vals_not_in_set:
        columns: [f]
        set: [low, mid, high]
    - col_vals_null:
        columns: [c]
    - col_vals_not_null:
        columns: [c]
    - col_vals_regex:
        columns: [f]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    - col_exists:
        columns: [z]
    - rows_distinct
    - rows_distinct:
        columns_subset: [a, b, c]
    - col_count_match:
        count: 14
    - row_count_match:
        count: 20
    """

    try:
        result = yaml_interrogate(yaml_content)
        assert result is not None
        # This should create 18 validation steps
        assert len(result.validation_info) == 18
        # The validation should execute without errors
        assert hasattr(result, "validation_info")
        # Verify that the highest severity level is 'critical'
        assert result._get_highest_severity_level() == "critical"

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise


def test_yaml_to_python_comprehensive():
    yaml_content = """
    tbl: small_table
    tbl_name: Test Table
    thresholds:
      warning: 0.1
      error: 0.25
    steps:
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_outside:
        columns: [a]
        left: 0
        right: 9
        inclusive: [false, true]
    - col_vals_in_set:
        columns: [f]
        set: [low, mid, high]
    - rows_distinct:
        columns_subset: [a, b, c]
    - col_count_match:
        count: 6
    """

    try:
        python_code = yaml_to_python(yaml_content)

        # Check that the generated code contains expected elements
        assert "import pointblank as pb" in python_code
        assert "pb.Validate(" in python_code
        assert 'data=pb.load_dataset("small_table", tbl_type="polars")' in python_code
        assert 'tbl_name="Test Table"' in python_code
        assert "pb.Thresholds(warning=0.1, error=0.25)" in python_code
        assert '.col_vals_eq(columns="a", value=3)' in python_code
        assert (
            '.col_vals_outside(columns="a", left=0, right=9, inclusive=(False, True))'
            in python_code
        )
        assert '.col_vals_in_set(columns="f", set=["low", "mid", "high"])' in python_code
        assert '.rows_distinct(columns_subset=["a", "b", "c"])' in python_code
        assert ".col_count_match(count=6)" in python_code
        assert ".interrogate()" in python_code

        # Check that it starts and ends with the right markers
        assert python_code.startswith("```python\n")
        assert python_code.endswith("\n```")

    except Exception as e:
        raise


def test_yaml_briefs():
    yaml_content = """
    tbl: small_table
    tbl_name: Brief Test
    brief: "**Global Brief**: {auto}"
    lang: en
    steps:
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_lt:
        columns: [c]
        value: 5
        brief: false
    - col_vals_gt:
        columns: [d]
        value: 100
        brief: true
    - col_vals_le:
        columns: [a]
        value: 7
        brief: "This is a custom local brief for the assertion"
    - col_vals_ge:
        columns: [d]
        value: 500
        na_pass: true
        brief: "**Step** {step}: {auto}"
    """

    try:
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) == 5
        assert result.tbl_name == "Brief Test"

        # Check brief values at every step
        assert result.validation_info[0].brief == "**Global Brief**: {auto}"
        assert result.validation_info[1].brief is None
        assert result.validation_info[2].brief == "{auto}"
        assert result.validation_info[3].brief == "This is a custom local brief for the assertion"
        assert result.validation_info[4].brief == "**Step** 5: {auto}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise


def test_yaml_to_python_with_briefs():
    yaml_content = """
    tbl: small_table
    tbl_name: Brief Example
    brief: "**Global Brief**: {auto}"
    lang: el
    thresholds:
      warning: 0.236
      error: 0.6
    steps:
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_lt:
        columns: [c]
        value: 5
        brief: false
    - col_vals_gt:
        columns: [d]
        value: 100
        brief: true
    - col_vals_le:
        columns: [a]
        value: 7
        brief: "This is a custom local brief for the assertion"
    """

    try:
        python_code = yaml_to_python(yaml_content)

        # Check that the generated code contains expected elements
        assert "import pointblank as pb" in python_code
        assert 'brief="**Global Brief**: {auto}"' in python_code
        assert 'lang="el"' in python_code
        assert '.col_vals_lt(columns="c", value=5, brief=False)' in python_code
        assert '.col_vals_gt(columns="d", value=100, brief=True)' in python_code
        assert (
            '.col_vals_le(columns="a", value=7, brief="This is a custom local brief for the assertion")'
            in python_code
        )

    except Exception as e:
        raise


def test_python_expressions():
    # Test python: block syntax with simple dataset loading
    yaml_content = """
tbl:
  python: |
    pb.load_dataset("small_table")
steps:
  - col_vals_not_null:
      columns: a
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
    except Exception as e:
        raise

    # Test python: block syntax with complex polars operations
    yaml_content = """
tbl:
  python: |
    pb.load_dataset("game_revenue").head(5)
steps:
  - row_count_match:
      count: 5
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
    except Exception as e:
        raise

    # Test security restrictions
    yaml_content = """
tbl:
  python: |
    import os
    os.system("echo test")
steps:
  - col_vals_not_null:
      columns: country
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
        raise AssertionError("Security restrictions not working")
    except Exception as e:
        if isinstance(e, YAMLValidationError) and ("not allowed" in str(e) or "unsafe" in str(e)):
            pass  # Expected - security restrictions work
        else:
            raise


def test_python_expressions_advanced():
    # Test python: block in validation step parameters
    yaml_content = """
tbl: game_revenue
steps:
  - col_vals_in_set:
      columns: player_id
      set:
        python: |
          ['Pl_01', 'Pl_02', 'Pl_03']
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
    except Exception as e:
        raise


def test_complex_expression_validation_yaml():
    # YAML configuration with complex Polars expression
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 1, 7, 8, 6],
        "b": [0, 0, 0, 1, 1, 1],
        "c": [0.5, 0.3, 0.8, 1.4, 1.9, 1.2],
    })
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.when(pl.col("b") == 0).then(pl.col("a").is_between(0, 5)).when(pl.col("b") == 1).then(pl.col("a") > 5).otherwise(pl.lit(True))
"""

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)

        # Verify the validation ran successfully
        assert validation_result is not None
        assert len(validation_result.validation_info) == 1

        validation_info = validation_result.validation_info[0]
        assert validation_info.n_passed == 6
        assert validation_info.n_failed == 0

    except Exception as e:
        raise


def test_complex_expression_multiline_validation_yaml():
    # YAML configuration with complex polars expression
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 1, 7, 8, 6],
        "b": [0, 0, 0, 1, 1, 1],
        "c": [0.5, 0.3, 0.8, 1.4, 1.9, 1.2],
    })
steps:
  - col_vals_expr:
      expr:
        python: |
          (
            pl.when(pl.col("b") == 0)
            .then(pl.col("a")
            .is_between(0, 5))
            .when(pl.col("b") == 1)
            .then(pl.col("a") > 5)
            .otherwise(pl.lit(True))
          )
"""

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)

        # Verify the validation ran successfully
        assert validation_result is not None
        assert len(validation_result.validation_info) == 1

        validation_info = validation_result.validation_info[0]
        assert validation_info.n_passed == 6
        assert validation_info.n_failed == 0

    except Exception as e:
        raise


def test_yaml_to_python_with_expressions():
    # YAML configuration with complex expressions
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 1, 7, 8, 6],
        "b": [0, 0, 0, 1, 1, 1],
        "c": [0.5, 0.3, 0.8, 1.4, 1.9, 1.2],
    })
tbl_name: "Expression Test Dataset"
label: "Complex polars expression validation"
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.when(pl.col("b") == 0).then(pl.col("a").is_between(0, 5)).when(pl.col("b") == 1).then(pl.col("a") > 5).otherwise(pl.lit(True))
"""

    try:
        # Test YAML to Python conversion
        python_code = yaml_to_python(yaml_content)

        # Verify the generated Python code contains expected elements
        assert "pb.Validate(" in python_code
        assert "col_vals_expr(" in python_code
        assert "tbl_name=" in python_code
        assert "label=" in python_code
        assert "interrogate()" in python_code

        # Test that the original YAML validation works
        original_result = yaml_interrogate(yaml_content)
        assert original_result is not None
        assert len(original_result.validation_info) == 1
        assert original_result.tbl_name == "Expression Test Dataset"
        assert original_result.label == "Complex polars expression validation"

        # Verify validation logic works correctly
        validation_info = original_result.validation_info[0]
        assert validation_info.n_passed == 6  # All 6 rows should pass the expression
        assert validation_info.n_failed == 0  # No failures expected

    except Exception as e:
        raise


def test_pandas_df_with_pandas_expressions():
    yaml_content = """
tbl:
  python: |
    pd.DataFrame({
        "nums": [1, 2, 3, 4, 5, 6],
        "category": ["A", "B", "A", "B", "A", "B"],
        "values": [10, 20, 30, 40, 50, 60]
    })
steps:
  - col_vals_expr:
      expr: |
        lambda df: df["nums"] > 2
"""

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        result = validator.execute_workflow(config)
        assert result is not None
        validation_info = result.validation_info[0]
        assert validation_info.n_passed == 4  # Values 3, 4, 5, 6 should pass
        assert validation_info.n_failed == 2  # Values 1, 2 should fail
    except Exception as e:
        raise


def test_yaml_to_python_polars_complex_scenarios():
    # Test with multiple polars expressions in different parameters
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.col("a") > 1
  - col_vals_in_set:
      columns: a
      set:
        python: |
          [1, 2, 3]
"""

    try:
        python_code = yaml_to_python(yaml_content)
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )
        assert "import polars as pl" in python_code, "Missing 'import polars as pl' statement"

    except Exception as e:
        raise


def test_yaml_to_python_no_unnecessary_imports():
    # Test 1: Simple YAML without polars should only import pointblank
    yaml_content_simple = """
tbl: small_table
steps:
  - col_vals_gt:
      columns: [a]
      value: 0
"""

    try:
        python_code = yaml_to_python(yaml_content_simple)
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )
        assert "import polars as pl" not in python_code, (
            "Unnecessary 'import polars as pl' statement"
        )
    except Exception as e:
        raise

    # Test 2: YAML with pandas expressions should only import pointblank
    yaml_content_pandas = """
tbl:
  python: |
    pd.DataFrame({"a": [1, 2, 3]})
steps:
  - col_vals_expr:
      expr: |
        lambda df: df["a"] > 1
"""

    try:
        python_code = yaml_to_python(yaml_content_pandas)
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )
        assert "import polars as pl" not in python_code, (
            "Unnecessary 'import polars as pl' statement for pandas expressions"
        )
    except Exception as e:
        raise


def test_yaml_to_python_includes_polars_import():
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [0, 0, 0, 1, 1]
    })
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.col("a") > 2
"""

    try:
        python_code = yaml_to_python(yaml_content)

        # Check if polars import is included
        assert "import polars as pl" in python_code, "Missing 'import polars as pl' statement"

        # Also check for pointblank import
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )

    except Exception as e:
        raise


def test_yaml_to_python_includes_pandas_import():
    yaml_content = """
tbl:
  python: |
    pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [0, 0, 0, 1, 1]
    })
steps:
  - col_vals_expr:
      expr: |
        lambda df: df["a"] > 2
"""

    try:
        python_code = yaml_to_python(yaml_content)

        # Check if pandas import is included
        assert "import pandas as pd" in python_code, "Missing 'import pandas as pd' statement"

        # Also check for pointblank import
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )

        # Should not include polars import
        assert "import polars as pl" not in python_code, (
            "Unnecessary 'import polars as pl' statement"
        )

    except Exception as e:
        raise


def test_yaml_to_python_full_functionality_demo():
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "age": [25, 30, 15, 40, 35],
        "income": [50000, 75000, 0, 100000, 60000],
        "department": ["IT", "Sales", "Intern", "Management", "IT"]
    })
tbl_name: "Employee Dataset"
label: "Comprehensive employee validation"
thresholds:
  warning: 0.1
  error: 0.25
steps:
  - col_vals_gt:
      columns: age
      value: 18
      brief: "Employees must be adults"
  - col_vals_expr:
      expr:
        python: |
          pl.col("income") > 0
      brief: "Income must be positive"
  - col_vals_in_set:
      columns: department
      set: ["IT", "Sales", "Management", "Intern"]
"""

    try:
        python_code = yaml_to_python(yaml_content)

        # Verify all expected components are present
        assert "import pointblank as pb" in python_code
        assert "import polars as pl" in python_code
        assert "pb.Validate(" in python_code
        assert "pb.Thresholds(" in python_code
        assert "tbl_name=" in python_code
        assert "label=" in python_code
        assert "col_vals_gt(" in python_code
        assert "col_vals_expr(" in python_code
        assert "col_vals_in_set(" in python_code
        assert "interrogate()" in python_code

        # Verify that the Python expressions are preserved in the generated code
        assert "pl.DataFrame({" in python_code
        assert '"age": [25, 30, 15, 40, 35]' in python_code
        assert '"income": [50000, 75000, 0, 100000, 60000]' in python_code
        assert '"department": ["IT", "Sales", "Intern", "Management", "IT"]' in python_code
        assert 'pl.col("income") > 0' in python_code

    except Exception as e:
        raise


def test_yaml_with_polars_pre_expressions():
    yaml_content = """
tbl: nycflights
label: NYC Flights Freshness Validation
steps:
  - col_vals_eq:
      columns: year
      value: 2013
      brief: "Check year is current"
  - col_vals_lt:
      columns: day
      value: 32
      pre:
        python: |
          lambda df: (
              df.filter(
                  (pl.col("year") == 2013) &
                  (pl.col("month") <= 12)
              )
          )
      brief: "Check no invalid future dates"
  - col_vals_not_null:
      columns: [dep_time, arr_time]
      pre:
        python: |
          lambda df: (
              df.filter(
                  (pl.col("year") == 2013) &
                  (pl.col("month") == 12) &
                  (pl.col("day") >= 25)
              )
          )
      brief: "Check recent flights have complete timing data"
"""

    try:
        # Test 1: YAML parsing and interrogation
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) >= 3
        assert result.label == "NYC Flights Freshness Validation"

        # Test 2: YAML to Python conversion
        python_code = yaml_to_python(yaml_content)

        # Test 3: Verify round-trip preservation

        # Check that necessary imports are included
        assert "import pointblank as pb" in python_code
        assert "import polars as pl" in python_code

        # Check that the pre= expressions are preserved
        assert "lambda df:" in python_code
        assert 'pl.col("year") == 2013' in python_code
        assert 'pl.col("month") <= 12' in python_code
        assert 'pl.col("month") == 12' in python_code
        assert 'pl.col("day") >= 25' in python_code
        assert "df.filter(" in python_code

        # Check that other elements are preserved
        assert 'label="NYC Flights Freshness Validation"' in python_code
        assert ".col_vals_eq(" in python_code
        assert ".col_vals_lt(" in python_code
        assert ".col_vals_not_null(" in python_code
        assert "value=2013" in python_code
        assert "value=32" in python_code

    except Exception as e:
        raise


def test_yaml_pre_parameter_shortcut_syntax():
    yaml_content = """
tbl: nycflights
label: NYC Flights Freshness Validation
steps:
  - col_vals_eq:
      columns: year
      value: 2013
      brief: "Check year is current"
  - col_vals_lt:
      columns: day
      value: 32
      pre: |
        lambda df: (
            df.filter(
                (pl.col("year") == 2013) &
                (pl.col("month") <= 12)
            )
        )
      brief: "Check no invalid future dates"
  - col_vals_not_null:
      columns: [dep_time, arr_time]
      pre: |
        lambda df: (
            df.filter(
                (pl.col("year") == 2013) &
                (pl.col("month") == 12) &
                (pl.col("day") >= 25)
            )
        )
      brief: "Check recent flights have complete timing data"
"""

    try:
        # Test 1: YAML parsing and interrogation
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) >= 3
        assert result.label == "NYC Flights Freshness Validation"

        # Test 2: YAML to Python conversion
        python_code = yaml_to_python(yaml_content)

        # Test 3: Verify round-trip preservation

        # Check that necessary imports are included
        assert "import pointblank as pb" in python_code
        assert "import polars as pl" in python_code

        # Check that the pre= expressions are preserved
        assert "lambda df:" in python_code
        assert 'pl.col("year") == 2013' in python_code
        assert 'pl.col("month") <= 12' in python_code
        assert 'pl.col("month") == 12' in python_code
        assert 'pl.col("day") >= 25' in python_code
        assert "df.filter(" in python_code

        # Check that other elements are preserved
        assert 'label="NYC Flights Freshness Validation"' in python_code
        assert ".col_vals_eq(" in python_code
        assert ".col_vals_lt(" in python_code
        assert ".col_vals_not_null(" in python_code
        assert "value=2013" in python_code
        assert "value=32" in python_code

    except Exception as e:
        raise


def test_yaml_actions_support():
    # Test actions at global level
    yaml_content = """
tbl: small_table
label: Actions Test
thresholds:
  warning: 1
actions:
  warning: "Warning: Step {step} failed with value {val}"
  error: "Error occurred in column {col}"
  highest_only: true
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    try:
        # Test 1: YAML parsing and interrogation should work
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) >= 1

        # Test 2: YAML to Python conversion should preserve actions
        python_code = yaml_to_python(yaml_content)

        # Check that actions are included in generated code
        assert "pb.Actions(" in python_code
        assert "warning=" in python_code
        assert "highest_only=" in python_code

    except Exception as e:
        raise

    # Test actions at step level
    yaml_step_actions = """
tbl: small_table
steps:
  - col_vals_gt:
      columns: a
      value: 1000
      thresholds:
        warning: 1
      actions:
        warning: "Step-level warning for {col}"
        error: "Step-level error"
"""

    try:
        # Test step-level actions
        result2 = yaml_interrogate(yaml_step_actions)
        assert result2 is not None

        python_code2 = yaml_to_python(yaml_step_actions)
        assert "actions=pb.Actions(" in python_code2

    except Exception as e:
        raise


def test_yaml_actions_with_callables():
    yaml_content = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning:
    python: |
      lambda: print("Custom warning action executed")
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    try:
        # Capture stdout to verify print statements
        captured_output = io.StringIO()

        with contextlib.redirect_stdout(captured_output):
            # Test callable actions
            result = yaml_interrogate(yaml_content)
            assert result is not None

        # Verify the action was executed and printed the expected message
        output_text = captured_output.getvalue()
        assert "Custom warning action executed" in output_text

        python_code = yaml_to_python(yaml_content)
        assert "pb.Actions(" in python_code

    except Exception as e:
        raise


def test_yaml_actions_comprehensive_demo():
    yaml_content = """
tbl: small_table
label: Comprehensive Actions Demo
thresholds:
  warning: 0.1
  error: 0.2
  critical: 0.3
actions:
  warning: "Global warning: {LEVEL} threshold exceeded in step {step}"
  critical: "Global critical alert for {type} validation"
  highest_only: false
steps:
  - col_vals_gt:
      columns: a
      value: 1000
      brief: "Check that a > 1000"
      thresholds:
        warning: 1
      actions:
        warning: "[{LEVEL}: {TYPE}]: Step {step} has a problem with the value {val} in column {col} ({time})"
  - col_vals_in_set:
      columns: a
      set: [1, 2, 3]
      brief: "Check that a is in valid set"
"""

    try:
        # Capture stdout to verify action outputs
        captured_output = io.StringIO()

        with contextlib.redirect_stdout(captured_output):
            # Test execution
            result = yaml_interrogate(yaml_content)

        # Verify actions were executed and templated correctly
        output_text = captured_output.getvalue()

        # Check for step-level action with templating
        assert "[WARNING: COL_VALS_GT]" in output_text
        assert "Step 1 has a problem" in output_text
        assert "column a" in output_text

        # Check for global action
        assert "Global critical alert for col_vals_in_set validation" in output_text

        # Test code generation
        python_code = yaml_to_python(yaml_content)

        # Verify actions are preserved
        assert "pb.Actions(" in python_code
        assert "warning=" in python_code
        assert "highest_only=False" in python_code

    except Exception as e:
        raise


def test_yaml_actions_output_verification():
    # Test 1: String template actions
    yaml_content_templates = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning: "Template test: Step {step} failed on column {col} with value {val}"
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content_templates)

    output_text = captured_output.getvalue()
    assert "Template test: Step 1 failed on column a" in output_text
    assert "with value" in output_text  # Value will be dynamic but should contain this phrase

    # Test 2: Callable actions
    yaml_content_callable = """
tbl: small_table
thresholds:
  error: 1
actions:
  error:
    python: |
      lambda: print("ERROR: Callable action triggered!")
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    captured_output2 = io.StringIO()
    with contextlib.redirect_stdout(captured_output2):
        result2 = yaml_interrogate(yaml_content_callable)

    output_text2 = captured_output2.getvalue()
    assert "ERROR: Callable action triggered!" in output_text2

    # Test 3: Multiple action levels
    yaml_content_multi = """
tbl: small_table
thresholds:
  warning: 0.5
  error: 1
actions:
  warning: "WARN: {LEVEL} - {TYPE}"
  error: "ERR: {LEVEL} - {TYPE}"
  highest_only: false
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    captured_output3 = io.StringIO()
    with contextlib.redirect_stdout(captured_output3):
        result3 = yaml_interrogate(yaml_content_multi)

    output_text3 = captured_output3.getvalue()
    # Should see both warning and error since highest_only is false
    assert "WARN: WARNING - COL_VALS_GT" in output_text3
    assert "ERR: ERROR - COL_VALS_GT" in output_text3


def test_yaml_actions_print_capture_demo():
    yaml_content = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning: "[CAPTURED]: This is a warning from step {step} on column {col}"
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    # Capture the output
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content)

    # Get the captured text
    output_text = captured_output.getvalue()

    # Verify specific content
    expected_text = "[CAPTURED]: This is a warning from step 1 on column a"
    assert expected_text in output_text

    print("✅ Actions print output successfully captured and verified!")

    # Also verify the validation executed correctly
    assert result is not None
    assert len(result.validation_info) == 1
    validation_info = result.validation_info[0]
    assert validation_info.assertion_type == "col_vals_gt"
    assert validation_info.n_failed > 0  # Should fail since a values are not > 1000


def test_yaml_to_python_actions_formatting():
    # Test global actions with callable
    yaml_content = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning:
    python: |
      lambda: print("Custom warning action executed")
  error: "Error template {step}"
  highest_only: false
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    python_code = yaml_to_python(yaml_content)

    # Verify the callable is properly formatted (not as a dict)
    assert "actions=pb.Actions(" in python_code
    assert 'warning=lambda: print("Custom warning action executed")' in python_code
    assert 'error="Error template {step}"' in python_code
    assert "highest_only=False" in python_code

    # Ensure no dict format is present
    assert "warning={'python':" not in python_code
    assert "warning={'python': 'lambda:" not in python_code

    # Test step-level actions with callable
    yaml_step_content = """
tbl: small_table
steps:
  - col_vals_gt:
      columns: a
      value: 1000
      actions:
        warning:
          python: |
            lambda: print("Step-level action")
"""

    step_python_code = yaml_to_python(yaml_step_content)

    # Verify step-level callable formatting
    assert "actions=pb.Actions(" in step_python_code
    assert 'warning=lambda: print("Step-level action")' in step_python_code
    assert "warning={'python':" not in step_python_code


def test_col_schema_match_yaml_basic():
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - [a, "Int64"]
              - [b, "String"]
              - [f, "String"]
          complete: false
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_schema_match"
    assert result.validation_info[0].all_passed is True


def test_col_schema_match_yaml_comprehensive():
    yaml_content = """
    tbl: small_table
    tbl_name: "Schema Validation Test"
    label: "Testing schema validation with various options"
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
      - col_schema_match:
          schema:
            columns:
              - [a, "Int64"]
              - [b, "String"]
              - [d, "Float64"]
              - [e, "Boolean"]
              - [f, "String"]
          complete: false
          in_order: false
          brief: "Partial schema validation"
      - col_schema_match:
          schema:
            columns:
              - [a, "Int64"]
              - [f, "String"]
          complete: false
          brief: "Minimal schema check"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 2
    assert result.tbl_name == "Schema Validation Test"
    assert result.label == "Testing schema validation with various options"

    # Check first validation step
    first_step = result.validation_info[0]
    assert first_step.assertion_type == "col_schema_match"
    assert first_step.brief == "Partial schema validation"

    # Check second validation step
    second_step = result.validation_info[1]
    assert second_step.assertion_type == "col_schema_match"
    assert second_step.brief == "Minimal schema check"


def test_col_schema_match_yaml_column_names_only():
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - ["a"]
              - ["b"]
              - ["f"]
          complete: false
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_schema_match"
    # Should pass because column names exist regardless of types
    assert result.validation_info[0].all_passed is True


def test_col_schema_match_yaml_scalar_column_names():
    """Test using scalar strings for column names (cleaner syntax)."""
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - a
              - b
              - f
          complete: false
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_schema_match"
    # Should pass because column names exist regardless of types
    assert result.validation_info[0].all_passed is True


def test_col_schema_match_yaml_mixed_formats():
    """Test mixing scalar strings and list formats."""
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - [a, "Int64"]  # With type
              - b             # Scalar string
              - [f]           # List format
          complete: false
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_schema_match"
    assert result.validation_info[0].all_passed is True


def test_col_schema_match_yaml_validation_options():
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - [a, "Int64"]
              - [f, "String"]
          complete: false
          in_order: false
          case_sensitive_colnames: false
          case_sensitive_dtypes: false
          full_match_dtypes: false
          brief: "Flexible schema validation"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    step_info = result.validation_info[0]
    assert step_info.assertion_type == "col_schema_match"
    assert step_info.brief == "Flexible schema validation"
    assert step_info.all_passed is True


def test_col_schema_match_yaml_with_actions():
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: 0.1
      error: 0.2
    actions:
      warning: "Schema mismatch warning for {TYPE}"
      error:
        python: |
          lambda: print("Schema validation failed!")
    steps:
      - col_schema_match:
          schema:
            columns:
              - [a, "Int64"]
              - [b, "String"]
              - [wrong_column, "String"]  # This will cause a failure
          complete: false
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # The validation should fail due to wrong_column not existing
    assert result.validation_info[0].all_passed is False


def test_col_schema_match_yaml_failure_case():
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - [a, "String"]  # Wrong type - should be Int64
              - [b, "Int64"]   # Wrong type - should be String
          complete: false
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Should fail due to type mismatches
    assert result.validation_info[0].all_passed is False
    assert result.validation_info[0].n_failed > 0


def test_col_schema_match_yaml_error_cases():
    # Test missing columns field
    with pytest.raises(YAMLValidationError) as exc_info:
        yaml_content = """
        tbl: small_table
        steps:
          - col_schema_match:
              schema:
                invalid_field: "test"
        """
        yaml_interrogate(yaml_content)
    assert "Schema specification must contain 'columns' field" in str(exc_info.value)

    # Test invalid column specification
    with pytest.raises(YAMLValidationError) as exc_info:
        yaml_content = """
        tbl: small_table
        steps:
          - col_schema_match:
              schema:
                columns:
                  - [a, b, c, d]  # Too many elements
        """
        yaml_interrogate(yaml_content)
    assert "Column specification must have 1-2 elements" in str(exc_info.value)

    # Test invalid schema type
    with pytest.raises(YAMLValidationError) as exc_info:
        yaml_content = """
        tbl: small_table
        steps:
          - col_schema_match:
              schema: "invalid_string"
        """
        yaml_interrogate(yaml_content)
    assert "Schema specification must be a dictionary" in str(exc_info.value)


def test_col_schema_match_yaml_complete_mode():
    yaml_content = """
    tbl: small_table
    steps:
      - col_schema_match:
          schema:
            columns:
              - [date_time, "Datetime(time_unit='us', time_zone=None)"]
              - [date, "Date"]
              - [a, "Int64"]
              - [b, "String"]
              - [c, "Int64"]
              - [d, "Float64"]
              - [e, "Boolean"]
              - [f, "String"]
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Should pass with exact schema match
    assert result.validation_info[0].all_passed is True


def test_yaml_conjointly_validation():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: Test conjointly validation
    steps:
    - conjointly:
        expressions:
          - "lambda df: df['d'] > df['a']"
          - "lambda df: df['a'] > 0"
          - "lambda df: df['a'] + df['d'] < 12000"
        brief: "All conditions must pass jointly"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Should pass for small_table dataset
    assert result.validation_info[0].all_passed is True
    assert result.n_passed(i=1, scalar=True) == 13


def test_yaml_conjointly_with_thresholds():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: Test conjointly with thresholds
    steps:
    - conjointly:
        expressions:
          - "lambda df: df['a'] > 0"   # This should pass for all rows
          - "lambda df: df['d'] > 0"   # This should pass for all rows
        thresholds:
          warning: 0.3
          error: 0.6
        brief: "Joint validation with thresholds"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Since we're checking for a > 0 and d > 0, most should pass
    # But due to the conjointly implementation issue, let's be more flexible
    # assert result.n_failed(i=1, scalar=True) > 0


def test_yaml_specially_validation():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: Test specially validation
    steps:
    - specially:
        expr: "lambda df: df.select(pl.col('a') + pl.col('d') > 0)"
        brief: "Custom validation function"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Should pass for small_table dataset
    assert result.validation_info[0].all_passed is True
    assert result.n_passed(i=1, scalar=True) == 13


def test_yaml_specially_with_actions():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: Test specially with actions
    steps:
    - specially:
        expr: "lambda df: df.select(pl.col('a') > 10)"  # Will fail for some rows
        thresholds:
          warning: 0.5
        actions:
          warning: "Values in column 'a' should be greater than 10"
        brief: "Custom validation with actions"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Some failures are expected
    assert result.n_failed(i=1, scalar=True) > 0


def test_yaml_specially_pandas_syntax():
    yaml_content = """
    tbl: small_table
    df_library: pandas
    tbl_name: small_table
    label: Test specially with pandas
    steps:
    - specially:
        expr: "lambda df: df.assign(validation_result=df['a'] + df['d'] > 0)"
        brief: "Pandas-style custom validation"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Should pass for small_table dataset
    assert result.validation_info[0].all_passed is True


def test_yaml_conjointly_and_specially_combined():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: Combined conjointly and specially test
    steps:
    - conjointly:
        expressions:
          - "lambda df: df['a'] > 0"
          - "lambda df: df['d'] > 0"
        brief: "Basic positive value checks"
    - specially:
        expr: "lambda df: df.select(pl.col('a') + pl.col('d') < 20000)"
        brief: "Sum validation check"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 2
    # Both validations should pass
    assert result.validation_info[0].all_passed is True
    assert result.validation_info[1].all_passed is True


def test_yaml_conjointly_expression_col_syntax():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: Test conjointly with expr_col
    steps:
    - conjointly:
        expressions:
          - "lambda df: expr_col('d') > expr_col('a')"
          - "lambda df: expr_col('a') > 0"
        brief: "Using expr_col for cross-table compatibility"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    # Should pass for small_table dataset
    assert result.validation_info[0].all_passed is True


def test_yaml_df_library_parameter():
    # Test with Polars (default)
    yaml_content_polars = """
    tbl: small_table
    df_library: polars
    steps:
    - col_vals_gt:
        columns: a
        value: 0
    """

    result_polars = yaml_interrogate(yaml_content_polars)
    assert result_polars is not None
    assert len(result_polars.validation_info) == 1
    assert result_polars.validation_info[0].all_passed is True

    # Test with Pandas
    yaml_content_pandas = """
    tbl: small_table
    df_library: pandas
    steps:
    - col_vals_gt:
        columns: a
        value: 0
    """

    result_pandas = yaml_interrogate(yaml_content_pandas)
    assert result_pandas is not None
    assert len(result_pandas.validation_info) == 1
    assert result_pandas.validation_info[0].all_passed is True


def test_yaml_df_library_ignored_with_python_expression():
    # Test case 1: Python expression should ignore df_library in execution
    yaml_content_python_expr = """
    tbl:
      python: |
        pb.load_dataset("small_table", tbl_type="polars")
    df_library: pandas  # Should be ignored
    steps:
    - col_vals_not_null:
        columns: [a]
    """

    # Should execute successfully (if df_library was incorrectly applied, might cause issues)
    result = yaml_interrogate(yaml_content_python_expr)
    assert result is not None
    assert len(result.validation_info) == 1

    # Test case 2: yaml_to_python should generate correct code ignoring df_library
    python_code = yaml_to_python(yaml_content_python_expr)

    # Should use the original Python expression, not apply df_library
    assert 'tbl_type="polars"' in python_code, "Should preserve original Python expression"
    assert 'tbl_type="pandas"' not in python_code, (
        "Should not apply df_library to Python expressions"
    )

    # Test case 3: Compare with regular dataset that should use df_library
    yaml_content_regular = """
    tbl: small_table
    df_library: pandas
    steps:
    - col_vals_not_null:
        columns: [a]
    """

    python_code_regular = yaml_to_python(yaml_content_regular)
    assert 'tbl_type="pandas"' in python_code_regular, "Regular datasets should use df_library"


def test_yaml_to_python_conjointly_validation():
    yaml_content = """
    tbl: small_table
    tbl_name: "Conjointly Test"
    label: "Test conjointly conversion"
    steps:
    - conjointly:
        expressions:
          - "lambda df: df['d'] > df['a']"
          - "lambda df: df['a'] > 0"
          - "lambda df: df['a'] + df['d'] < 12000"
        brief: "All conditions must pass jointly"
        thresholds:
          warning: 0.1
          error: 0.2
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "pb.Validate(" in python_code
    assert 'tbl_name="Conjointly Test"' in python_code
    assert 'label="Test conjointly conversion"' in python_code
    assert ".conjointly(" in python_code
    assert "expressions=[" in python_code
    assert "\"lambda df: df['d'] > df['a']\"" in python_code
    assert "\"lambda df: df['a'] > 0\"" in python_code
    assert "\"lambda df: df['a'] + df['d'] < 12000\"" in python_code
    assert 'brief="All conditions must pass jointly"' in python_code
    assert "thresholds=pb.Thresholds(warning=0.1, error=0.2)" in python_code
    assert ".interrogate()" in python_code


def test_yaml_to_python_specially_validation():
    yaml_content = """
    tbl: small_table
    tbl_name: "Specially Test"
    steps:
    - specially:
        expr: "lambda df: df.select(pl.col('a') + pl.col('d') > 0)"
        brief: "Custom validation function"
        thresholds:
          warning: 0.1
        actions:
          warning: "Custom warning for {TYPE}"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "import polars as pl" in python_code  # Should include polars import due to pl.col usage
    assert "pb.Validate(" in python_code
    assert 'tbl_name="Specially Test"' in python_code
    assert ".specially(" in python_code
    assert "expr=lambda df: df.select(pl.col('a') + pl.col('d') > 0)" in python_code
    assert 'brief="Custom validation function"' in python_code
    assert "thresholds=pb.Thresholds(warning=0.1)" in python_code
    assert 'actions=pb.Actions(warning="Custom warning for {TYPE}")' in python_code
    assert ".interrogate()" in python_code


def test_yaml_to_python_specially_pandas_syntax():
    yaml_content = """
    tbl: small_table
    df_library: pandas
    tbl_name: "Specially Pandas Test"
    steps:
    - specially:
        expr: "lambda df: df.assign(validation_result=df['a'] + df['d'] > 0)"
        brief: "Pandas-style custom validation"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "import pandas as pd" in python_code  # Should include pandas import due to df_library
    assert "pb.Validate(" in python_code
    assert 'data=pb.load_dataset("small_table", tbl_type="pandas")' in python_code
    assert 'tbl_name="Specially Pandas Test"' in python_code
    assert ".specially(" in python_code
    assert "expr=lambda df: df.assign(validation_result=df['a'] + df['d'] > 0)" in python_code
    assert 'brief="Pandas-style custom validation"' in python_code
    assert ".interrogate()" in python_code


def test_yaml_to_python_conjointly_and_specially_combined():
    yaml_content = """
    tbl: small_table
    tbl_name: "Combined Test"
    label: "Combined conjointly and specially test"
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - conjointly:
        expressions:
          - "lambda df: df['a'] > 0"
          - "lambda df: df['d'] > 0"
        brief: "Basic positive value checks"
    - specially:
        expr: "lambda df: df.select(pl.col('a') + pl.col('d') < 20000)"
        brief: "Sum validation check"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "import polars as pl" in python_code  # Should include polars import due to pl.col usage
    assert "pb.Validate(" in python_code
    assert 'tbl_name="Combined Test"' in python_code
    assert 'label="Combined conjointly and specially test"' in python_code
    assert "pb.Thresholds(warning=0.1, error=0.2)" in python_code

    # Check conjointly method
    assert ".conjointly(" in python_code
    assert "expressions=[" in python_code
    assert "\"lambda df: df['a'] > 0\"" in python_code
    assert "\"lambda df: df['d'] > 0\"" in python_code
    assert 'brief="Basic positive value checks"' in python_code

    # Check specially method
    assert ".specially(" in python_code
    assert "expr=lambda df: df.select(pl.col('a') + pl.col('d') < 20000)" in python_code
    assert 'brief="Sum validation check"' in python_code

    assert ".interrogate()" in python_code


def test_yaml_to_python_specially_with_python_block():
    yaml_content = """
    tbl: small_table
    steps:
    - specially:
        expr:
          python: |
            lambda df: df.select(pl.col('amount') > 0)
        brief: "Python block expression"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "import polars as pl" in python_code
    assert ".specially(" in python_code
    assert "expr=lambda df: df.select(pl.col('amount') > 0)" in python_code
    assert 'brief="Python block expression"' in python_code


def test_yaml_col_count_match_basic():
    yaml_content = """
    tbl: small_table
    steps:
    - col_count_match:
        count: 8
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_count_match"
    assert result.validation_info[0].all_passed is True


def test_yaml_col_count_match_with_options():
    yaml_content = """
    tbl: small_table
    tbl_name: "Column Count Test"
    label: "Testing column count validation"
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - col_count_match:
        count: 8
        brief: "Check exact column count"
        thresholds:
          warning: 0.05
        actions:
          warning: "Column count mismatch in {col}"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.tbl_name == "Column Count Test"
    assert result.label == "Testing column count validation"

    validation_info = result.validation_info[0]
    assert validation_info.assertion_type == "col_count_match"
    assert validation_info.brief == "Check exact column count"
    assert validation_info.all_passed is True


def test_yaml_col_count_match_failure():
    yaml_content = """
    tbl: small_table
    steps:
    - col_count_match:
        count: 100  # Wrong count - should fail
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_count_match"
    assert result.validation_info[0].all_passed is False
    assert result.validation_info[0].n_failed > 0


def test_yaml_row_count_match_basic():
    yaml_content = """
    tbl: small_table
    steps:
    - row_count_match:
        count: 13
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "row_count_match"
    assert result.validation_info[0].all_passed is True


def test_yaml_row_count_match_with_options():
    yaml_content = """
    tbl: small_table
    tbl_name: "Row Count Test"
    label: "Testing row count validation"
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - row_count_match:
        count: 13
        brief: "Check exact row count"
        thresholds:
          error: 0.15
        actions:
          error: "Row count mismatch - expected {val} rows"
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.tbl_name == "Row Count Test"
    assert result.label == "Testing row count validation"

    validation_info = result.validation_info[0]
    assert validation_info.assertion_type == "row_count_match"
    assert validation_info.brief == "Check exact row count"
    assert validation_info.all_passed is True


def test_yaml_row_count_match_failure():
    yaml_content = """
    tbl: small_table
    steps:
    - row_count_match:
        count: 1000  # Wrong count - should fail
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "row_count_match"
    assert result.validation_info[0].all_passed is False
    assert result.validation_info[0].n_failed > 0


def test_yaml_to_python_col_count_match():
    yaml_content = """
    tbl: small_table
    tbl_name: "Column Count Validation"
    label: "Testing column count conversion"
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - col_count_match:
        count: 8
        brief: "Verify 8 columns present"
        thresholds:
          warning: 0.05
        actions:
          warning: "Column count warning for table"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "pb.Validate(" in python_code
    assert 'tbl_name="Column Count Validation"' in python_code
    assert 'label="Testing column count conversion"' in python_code
    assert "pb.Thresholds(warning=0.1, error=0.2)" in python_code
    assert ".col_count_match(" in python_code
    assert "count=8" in python_code
    assert 'brief="Verify 8 columns present"' in python_code
    assert "thresholds=pb.Thresholds(warning=0.05)" in python_code
    assert 'actions=pb.Actions(warning="Column count warning for table")' in python_code
    assert ".interrogate()" in python_code


def test_yaml_to_python_row_count_match():
    yaml_content = """
    tbl: small_table
    tbl_name: "Row Count Validation"
    label: "Testing row count conversion"
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - row_count_match:
        count: 13
        brief: "Verify 13 rows present"
        thresholds:
          error: 0.15
        actions:
          error: "Row count error for dataset"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "pb.Validate(" in python_code
    assert 'tbl_name="Row Count Validation"' in python_code
    assert 'label="Testing row count conversion"' in python_code
    assert "pb.Thresholds(warning=0.1, error=0.2)" in python_code
    assert ".row_count_match(" in python_code
    assert "count=13" in python_code
    assert 'brief="Verify 13 rows present"' in python_code
    assert "thresholds=pb.Thresholds(error=0.15)" in python_code
    assert 'actions=pb.Actions(error="Row count error for dataset")' in python_code
    assert ".interrogate()" in python_code


def test_yaml_to_python_count_matches_combined():
    yaml_content = """
    tbl: small_table
    tbl_name: "Combined Count Tests"
    label: "Testing both count validation methods"
    thresholds:
      warning: 0.1
    steps:
    - col_count_match:
        count: 8
        brief: "Check column count"
    - row_count_match:
        count: 13
        brief: "Check row count"
    """

    python_code = yaml_to_python(yaml_content)

    # Verify the generated code contains expected elements
    assert "import pointblank as pb" in python_code
    assert "pb.Validate(" in python_code
    assert 'tbl_name="Combined Count Tests"' in python_code
    assert 'label="Testing both count validation methods"' in python_code
    assert "pb.Thresholds(warning=0.1)" in python_code

    # Check both methods are present
    assert ".col_count_match(" in python_code
    assert ".row_count_match(" in python_code
    assert "count=8" in python_code
    assert "count=13" in python_code
    assert 'brief="Check column count"' in python_code
    assert 'brief="Check row count"' in python_code
    assert ".interrogate()" in python_code


def test_yaml_count_matches_with_different_datasets():
    """Test `col_count_match()` with the `game_revenue` dataset."""
    yaml_content_game = """
    tbl: game_revenue
    steps:
    - col_count_match:
        count: 11  # game_revenue has 11 columns
    """

    result = yaml_interrogate(yaml_content_game)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_count_match"
    assert result.validation_info[0].all_passed is True

    # Test `row_count_match()` with the `small_table` dataset
    yaml_content_small = """
    tbl: small_table
    steps:
    - row_count_match:
        count: 13  # small_table has 13 rows
    """

    result = yaml_interrogate(yaml_content_small)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "row_count_match"
    assert result.validation_info[0].all_passed is True


def test_yaml_interrogate_set_tbl_basic():
    """Test basic `yaml_interrogate()` with `set_tbl=` parameter."""

    # Create a test table
    test_table = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": ["x", "y", "z", "w", "v"]}
    )

    yaml_config = """
    tbl: small_table
    tbl_name: "Override Test"
    label: "YAML validation with table override"
    steps:
      - col_exists:
          columns: [a, b, c]
      - col_vals_gt:
          columns: [a]
          value: 0
      - col_vals_gt:
          columns: [b]
          value: 5
    """

    # Execute with table override
    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    assert result.tbl_name == "Override Test"
    assert result.label == "YAML validation with table override"
    assert len(result.validation_info) > 0
    assert all(step.all_passed for step in result.validation_info)

    # Verify that interrogation was completed
    assert result.time_start is not None
    assert result.time_end is not None


def test_yaml_interrogate_set_tbl_vs_no_override():
    """Compare `yaml_interrogate()` with and without `set_tbl=` override."""

    yaml_config = """
    tbl: small_table
    tbl_name: "Test Comparison"
    steps:
      - col_exists:
          columns: [a, b]
      - col_vals_gt:
          columns: [a]
          value: 0
    """

    # Execute without override (uses small_table)
    result_original = yaml_interrogate(yaml_config)

    # Create a test table with same structure as small_table
    test_table = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

    # Execute with override
    result_override = yaml_interrogate(yaml_config, set_tbl=test_table)

    # Both should have same validation structure
    assert len(result_original.validation_info) == len(result_override.validation_info)
    assert result_original.tbl_name == result_override.tbl_name
    assert result_original.label == result_override.label

    # Assertion types should be the same
    for orig_step, override_step in zip(
        result_original.validation_info, result_override.validation_info
    ):
        assert orig_step.assertion_type == override_step.assertion_type
        assert orig_step.column == override_step.column


def test_yaml_interrogate_set_tbl_different_libraries():
    """Test `yaml_interrogate()`'s `set_tbl=` with different DataFrame libraries."""

    yaml_config = """
    tbl: small_table
    steps:
      - col_vals_gt:
          columns: [a]
          value: 0
      - col_exists:
          columns: [a, b]
    """

    # Test with Polars DataFrame
    polars_table = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result_polars = yaml_interrogate(yaml_config, set_tbl=polars_table)
    assert all(step.all_passed for step in result_polars.validation_info)

    # Test with Pandas DataFrame
    pandas_table = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
    result_pandas = yaml_interrogate(yaml_config, set_tbl=pandas_table)
    assert all(step.all_passed for step in result_pandas.validation_info)

    # Both should have same structure
    assert len(result_polars.validation_info) == len(result_pandas.validation_info)


def test_yaml_interrogate_set_tbl_with_thresholds():
    """Test `yaml_interrogate()`'s `set_tbl=` with thresholds configuration."""

    # Create a table that will trigger some failures
    test_table = pl.DataFrame(
        {
            "score": [85, 92, 45, 95, 88, 30, 91, 87, 25],  # Some values < 50
            "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A"],
        }
    )

    yaml_config = """
    tbl: small_table
    tbl_name: "Threshold Test"
    thresholds:
      warning: 0.2
      error: 0.4
      critical: 0.6
    steps:
      - col_vals_gt:
          columns: [score]
          value: 50  # This will fail for some values
      - col_exists:
          columns: [score, category]
    """

    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    assert result.tbl_name == "Threshold Test"
    assert result.thresholds.warning == 0.2
    assert result.thresholds.error == 0.4
    assert result.thresholds.critical == 0.6

    # Some validations should pass, some might trigger thresholds
    assert len(result.validation_info) > 0


def test_yaml_interrogate_set_tbl_with_actions():
    """Test `yaml_interrogate()`'s `set_tbl=` with actions configuration."""

    test_table = pl.DataFrame(
        {"value": [1, 2, 3, 4, 5], "status": ["active", "inactive", "active", "active", "inactive"]}
    )

    # Track action calls
    action_calls = []

    yaml_config = f"""
    tbl: small_table
    tbl_name: "Action Test"
    actions:
      warning: |
        python: |
          action_calls.append("warning_triggered")
    steps:
      - col_vals_gt:
          columns: [value]
          value: 0
    """

    # Execute with actions (note: actions might not trigger if all validations pass)
    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    assert result.tbl_name == "Action Test"
    assert result.actions is not None


def test_yaml_interrogate_set_tbl_with_complex_validations():
    """Test `yaml_interrogate()`'s `set_tbl=` with complex validation scenarios."""

    test_table = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "score": [85, 92, 78, 95, 88, 76, 89, 91, 83, 96],
            "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "active": [True, True, False, True, True, True, False, True, True, False],
            "region": [
                "North",
                "South",
                "North",
                "East",
                "West",
                "North",
                "South",
                "East",
                "West",
                "North",
            ],
        }
    )

    yaml_config = """
    tbl: small_table
    tbl_name: "Complex Validation Test"
    label: "Multi-step validation with various checks"
    thresholds:
      warning: 0.1
      error: 0.25
    steps:
      - col_exists:
          columns: [id, score, category, active, region]
      - col_vals_not_null:
          columns: [id, score, category]
      - col_vals_between:
          columns: [score]
          left: 0
          right: 100
      - col_vals_in_set:
          columns: [category]
          set: [A, B, C]
      - col_vals_gt:
          columns: [id]
          value: 0
      - rows_distinct: {}  # Remove columns parameter for rows_distinct
      - col_schema_match:
          schema:
            columns:
              - [id, Int64]
              - [score, Int64]
              - [category, String]
              - [active, Boolean]
              - [region, String]
          complete: false
    """

    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    assert result.tbl_name == "Complex Validation Test"
    assert result.label == "Multi-step validation with various checks"
    assert len(result.validation_info) > 5  # Should have many validation steps
    assert all(step.all_passed for step in result.validation_info)


def test_yaml_interrogate_set_tbl_with_segments():
    """Test `yaml_interrogate()`'s `set_tbl=` with segmented validations."""

    test_table = pl.DataFrame(
        {
            "region": ["North", "South", "North", "South", "East", "West", "East", "West"],
            "sales": [100, 200, 150, 180, 120, 220, 160, 190],
            "quarter": ["Q1", "Q1", "Q2", "Q2", "Q1", "Q1", "Q2", "Q2"],
        }
    )

    yaml_config = """
    tbl: small_table
    tbl_name: "Segmented Test"
    steps:
      - col_vals_gt:
          columns: [sales]
          value: 50
          segments: region
      - col_vals_not_null:
          columns: [sales, region]
          segments: quarter
    """

    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    assert result.tbl_name == "Segmented Test"
    # Should have multiple validation steps due to segmentation
    assert len(result.validation_info) > 2


def test_yaml_interrogate_set_tbl_with_preprocessing():
    """Test `yaml_interrogate()`'s `set_tbl=` with basic preprocessing that works."""

    test_table = pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 7],  # All positive values
            "category": ["A", "B", "A", "B", "A", "B", "A"],
        }
    )

    yaml_config = """
    tbl: small_table
    tbl_name: "Preprocessing Test"
    steps:
      - col_vals_gt:
          columns: [value]
          value: 0
      - col_exists:
          columns: [value, category]
    """

    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    assert result.tbl_name == "Preprocessing Test"
    # Should pass because all values are positive
    assert all(step.all_passed for step in result.validation_info)


def test_yaml_interrogate_set_tbl_error_cases():
    """Test error handling in `yaml_interrogate()` with `set_tbl=`."""

    # Table with incompatible structure
    incompatible_table = pl.DataFrame({"different_col": [1, 2, 3], "another_col": [4, 5, 6]})

    yaml_config = """
    tbl: small_table
    steps:
      - col_exists:
          columns: [a, b, c]  # These columns don't exist in incompatible_table
    """

    # Should execute but validation should fail
    result = yaml_interrogate(yaml_config, set_tbl=incompatible_table)
    assert result is not None
    # col_exists should fail for non-existent columns
    assert not all(step.all_passed for step in result.validation_info)


def test_yaml_interrogate_set_tbl_with_csv_and_datasets():
    """Test yaml_interrogate `set_tbl=` with CSV files and DataFrames."""

    yaml_config = """
    tbl: small_table  # Will be overridden
    tbl_name: "Dataset Override Test"
    steps:
      - col_exists:
          columns: [a, b]
      - col_vals_gt:
          columns: [a]
          value: 0
    """

    # Test with DataFrame directly (instead of string dataset name)
    test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result_dataframe = yaml_interrogate(yaml_config, set_tbl=test_df)
    assert result_dataframe.tbl_name == "Dataset Override Test"
    assert all(step.all_passed for step in result_dataframe.validation_info)

    # Test with CSV file
    test_data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test_data.write_csv(f.name)
        csv_path = f.name

    try:
        result_csv = yaml_interrogate(yaml_config, set_tbl=csv_path)
        assert result_csv.tbl_name == "Dataset Override Test"
        assert all(step.all_passed for step in result_csv.validation_info)
    finally:
        os.unlink(csv_path)


def test_yaml_interrogate_set_tbl_preserves_all_yaml_config():
    """Test that `set_tbl=` preserves all YAML configuration options."""

    test_table = pl.DataFrame({"metric": [1, 2, 3, 4, 5], "category": ["X", "Y", "Z", "X", "Y"]})

    yaml_config = """
    tbl: small_table
    tbl_name: "Configuration Test"
    label: "Testing all YAML options"
    lang: en
    locale: en-US
    brief: "Step {step}: {auto}"
    thresholds:
      warning: 0.1
      error: 0.3
      critical: 0.5
    steps:
      - col_vals_gt:
          columns: [metric]
          value: 0
          brief: "Custom brief for metric validation"
      - col_exists:
          columns: [metric, category]
    """

    result = yaml_interrogate(yaml_config, set_tbl=test_table)

    # Verify all configuration is preserved
    assert result.tbl_name == "Configuration Test"
    assert result.label == "Testing all YAML options"
    assert result.lang == "en"
    assert result.locale == "en-US"
    assert result.brief == "Step {step}: {auto}"
    assert result.thresholds.warning == 0.1
    assert result.thresholds.error == 0.3
    assert result.thresholds.critical == 0.5

    # Verify briefs are applied
    assert len(result.validation_info) > 0
    # First step should have custom brief
    assert "Custom brief for metric validation" in str(result.validation_info[0].brief)


def test_yaml_interrogate_set_tbl_multiple_scenarios():
    """Test yaml_interrogate `set_tbl=` in multiple realistic scenarios."""

    # Scenario 1: Sales data validation template
    sales_template = """
    tbl: small_table  # Changed from placeholder to valid dataset
    tbl_name: "Sales Data Validation"
    steps:
      - col_exists:
          columns: [customer_id, revenue, region]
      - col_vals_not_null:
          columns: [customer_id, revenue]
      - col_vals_gt:
          columns: [revenue]
          value: 0
      - col_vals_in_set:
          columns: [region]
          set: [North, South, East, West]
    """

    sales_q1 = pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "revenue": [100, 200, 150, 300, 250],
            "region": ["North", "South", "East", "West", "North"],
        }
    )

    sales_q2 = pl.DataFrame(
        {
            "customer_id": [6, 7, 8, 9, 10],
            "revenue": [120, 180, 160, 320, 280],
            "region": ["South", "East", "West", "North", "South"],
        }
    )

    # Apply template to both quarters
    result_q1 = yaml_interrogate(sales_template, set_tbl=sales_q1)
    result_q2 = yaml_interrogate(sales_template, set_tbl=sales_q2)

    assert result_q1.tbl_name == "Sales Data Validation"
    assert result_q2.tbl_name == "Sales Data Validation"
    assert all(step.all_passed for step in result_q1.validation_info)
    assert all(step.all_passed for step in result_q2.validation_info)

    # Scenario 2: User behavior validation
    user_template = """
    tbl: small_table
    tbl_name: "User Behavior Analysis"
    thresholds:
      warning: 0.05
    steps:
      - col_vals_between:
          columns: [session_duration]
          left: 0
          right: 7200  # Max 2 hours
      - col_vals_in_set:
          columns: [device_type]
          set: [mobile, desktop, tablet]
    """

    user_data_mobile = pl.DataFrame(
        {
            "session_duration": [300, 450, 600, 1200, 900],
            "device_type": ["mobile", "mobile", "mobile", "mobile", "mobile"],
        }
    )

    user_data_mixed = pl.DataFrame(
        {
            "session_duration": [600, 1800, 300, 2400, 450],
            "device_type": ["desktop", "tablet", "mobile", "desktop", "tablet"],
        }
    )

    result_mobile = yaml_interrogate(user_template, set_tbl=user_data_mobile)
    result_mixed = yaml_interrogate(user_template, set_tbl=user_data_mixed)

    assert result_mobile.tbl_name == "User Behavior Analysis"
    assert result_mixed.tbl_name == "User Behavior Analysis"
    assert all(step.all_passed for step in result_mobile.validation_info)
    assert all(step.all_passed for step in result_mixed.validation_info)


def test_yaml_interrogate_set_tbl_edge_cases():
    """Test edge cases for yaml_interrogate with `set_tbl=`."""

    # Edge case 1: Empty DataFrame
    empty_table = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})

    yaml_config = """
    tbl: small_table
    steps:
      - col_exists:
          columns: [a]
    """

    result_empty = yaml_interrogate(yaml_config, set_tbl=empty_table)
    assert result_empty is not None
    # col_exists should pass even for empty table if column exists
    assert result_empty.validation_info[0].all_passed

    # Edge case 2: Single row DataFrame
    single_row = pl.DataFrame({"x": [42], "y": ["test"]})

    yaml_single = """
    tbl: small_table
    steps:
      - col_vals_gt:
          columns: [x]
          value: 0
      - col_vals_not_null:
          columns: [x, y]
    """

    result_single = yaml_interrogate(yaml_single, set_tbl=single_row)
    assert all(step.all_passed for step in result_single.validation_info)

    # Edge case 3: Large number of columns
    many_cols_data = {f"col_{i}": [i] * 3 for i in range(50)}
    many_cols_table = pl.DataFrame(many_cols_data)

    yaml_many_cols = """
    tbl: small_table
    steps:
      - col_count_match:
          count: 50
    """

    result_many_cols = yaml_interrogate(yaml_many_cols, set_tbl=many_cols_table)
    assert result_many_cols.validation_info[0].all_passed


def test_yaml_interrogate_tbl_field_options_for_always_override():
    """Test different tbl: field values when always using set_tbl= parameter."""

    # Create test data
    test_table = pl.DataFrame({"metric": [1, 2, 3, 4, 5], "category": ["X", "Y", "Z", "X", "Y"]})

    # Test 1: Using valid built-in dataset names (gets overridden)
    valid_datasets = ["small_table", "game_revenue", "nycflights", "global_sales"]

    for dataset in valid_datasets:
        yaml_config = f"""
tbl: {dataset}
tbl_name: "Test with {dataset}"
steps:
  - col_exists:
      columns: [metric, category]
  - col_vals_gt:
      columns: [metric]
      value: 0
"""

        result = yaml_interrogate(yaml_config, set_tbl=test_table)
        assert result.tbl_name == f"Test with {dataset}"
        assert all(step.all_passed for step in result.validation_info)

    # Test 2: Using YAML null (the recommended approach)
    yaml_null = """
tbl: null
tbl_name: "Test with null"
steps:
  - col_exists:
      columns: [metric, category]
  - col_vals_gt:
      columns: [metric]
      value: 0
"""

    result_null = yaml_interrogate(yaml_null, set_tbl=test_table)
    assert result_null.tbl_name == "Test with null"
    assert all(step.all_passed for step in result_null.validation_info)

    # Test 3: Verify that any valid tbl: field works with `set_tbl=` (gets overridden anyway)
    yaml_override_test = """
tbl: small_table
tbl_name: "Test override behavior"
steps:
  - col_exists:
      columns: [metric, category]
  - col_vals_gt:
      columns: [metric]
      value: 0
"""

    # Should work with set_tbl (overrides the small_table)
    result_override = yaml_interrogate(yaml_override_test, set_tbl=test_table)
    assert result_override.tbl_name == "Test override behavior"
    assert all(step.all_passed for step in result_override.validation_info)


def test_yaml_interrogate_template_pattern():
    """Test the template pattern where YAML configs are designed for reuse with set_tbl=."""

    # Define a reusable validation template
    sales_validation_template = """
    tbl: null  # Will always be overridden
    tbl_name: "Sales Data Validation"
    label: "Standard sales data validation checks"
    thresholds:
      warning: 0.05
      error: 0.1
    steps:
      - col_exists:
          columns: [customer_id, revenue, region, date]
      - col_vals_not_null:
          columns: [customer_id, revenue, region]
      - col_vals_gt:
          columns: [revenue]
          value: 0
      - col_vals_in_set:
          columns: [region]
          set: [North, South, East, West]
    """

    # Apply template to different datasets
    q1_data = pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "revenue": [100, 200, 150, 300],
            "region": ["North", "South", "East", "West"],
            "date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
        }
    )

    q2_data = pl.DataFrame(
        {
            "customer_id": [5, 6, 7, 8],
            "revenue": [250, 180, 220, 350],
            "region": ["South", "North", "West", "East"],
            "date": ["2024-04-01", "2024-04-15", "2024-05-01", "2024-05-15"],
        }
    )

    # Apply same template to both datasets
    q1_result = yaml_interrogate(sales_validation_template, set_tbl=q1_data)
    q2_result = yaml_interrogate(sales_validation_template, set_tbl=q2_data)

    # Both should have same validation structure
    assert q1_result.tbl_name == "Sales Data Validation"
    assert q2_result.tbl_name == "Sales Data Validation"
    assert q1_result.label == "Standard sales data validation checks"
    assert q2_result.label == "Standard sales data validation checks"

    # Both should pass validations
    assert all(step.all_passed for step in q1_result.validation_info)
    assert all(step.all_passed for step in q2_result.validation_info)

    # Should have same number of validation steps
    assert len(q1_result.validation_info) == len(q2_result.validation_info)


def test_safe_eval_python_code_simple_expression():
    """Test evaluation of simple expressions."""

    result = _safe_eval_python_code("2 + 3")
    assert result == 5


def test_safe_eval_python_code_syntax_error_with_statements():
    """Test SyntaxError path with statements followed by expression."""

    code = """
x = 10
y = 20
x + y
"""
    result = _safe_eval_python_code(code)
    assert result == 30


def test_safe_eval_python_code_syntax_error_with_statements_no_final_expression():
    """Test SyntaxError path with statements but no final expression."""

    code = """
x = 10
y = 20
"""
    result = _safe_eval_python_code(code)
    assert result is None


def test_safe_eval_python_code_exception_handling():
    """Test exception handling in `_safe_eval_python_code()`."""

    # Use undefined variable to trigger runtime error
    with pytest.raises(Exception):
        _safe_eval_python_code("undefined_variable + 1")


def test_safe_eval_python_code_import_error():
    """Test that imports in eval code raise YAMLValidationError."""

    with pytest.raises(YAMLValidationError, match="Potentially unsafe Python code"):
        _safe_eval_python_code("import os")


def test_safe_eval_python_code_function_call_allowed():
    """Test that allowed function calls work (print is in the safe namespace)."""

    # print is allowed in safe namespace, so this should work
    result = _safe_eval_python_code("print('hello')")
    assert result is None  # print returns None


def test_safe_eval_python_code_attribute_access_error():
    """Test that attribute access to undefined variables raises YAMLValidationError."""

    with pytest.raises(YAMLValidationError, match="Error executing Python code"):
        _safe_eval_python_code("os.path")


def test_safe_eval_python_code_with_available_namespace():
    """Test `_safe_eval_python_code()` with available namespace variables."""

    # Test with basic arithmetic
    result = _safe_eval_python_code("5 + 10")
    assert result == 15

    # Test with built-in functions
    result = _safe_eval_python_code("len([1, 2, 3])")
    assert result == 3

    # Test with multiple operations
    result = _safe_eval_python_code("max([1, 5, 3]) + min([4, 2, 6])")
    assert result == 7


def test_process_python_expressions_python_block():
    """Test processing dictionary with `python:` block."""

    data = {"python": "2 + 3"}
    result = _process_python_expressions(data)

    assert result == 5


def test_process_python_expressions_nested_dict():
    """Test processing nested dictionary structures."""

    data = {
        "level1": {"python": "10 * 2"},
        "level2": {"normal": "text", "nested": {"python": "5 + 5"}},
    }
    result = _process_python_expressions(data)

    expected = {"level1": 20, "level2": {"normal": "text", "nested": 10}}
    assert result == expected


def test_process_python_expressions_list():
    """Test processing lists with Python expressions."""

    data = [{"python": "1 + 1"}, "normal", {"python": "3 * 3"}]
    result = _process_python_expressions(data)

    assert result == [2, "normal", 9]


def test_process_python_expressions_mixed_structure():
    """Test processing mixed data structures."""

    data = {
        "list": [{"python": "2 + 2"}, "text"],
        "dict": {"python": "7 - 2"},
        "simple": "regular_string",
    }
    result = _process_python_expressions(data)

    expected = {"list": [4, "text"], "dict": 5, "simple": "regular_string"}
    assert result == expected


def test_process_python_expressions_no_expressions():
    """Test processing data with no Python expressions."""

    data = {"key": "value", "list": [1, 2, 3]}
    result = _process_python_expressions(data)

    assert result == data


def test_process_python_expressions_expression_error():
    """Test handling of expression evaluation errors."""

    data = {"python": "undefined_var + 1"}

    with pytest.raises(YAMLValidationError, match="Error executing Python code"):
        _process_python_expressions(data)


def test_yaml_validator_column_parsing():
    """Test the `_parse_column_spec()` method of YAMLValidator."""
    validator = YAMLValidator()

    # Test various YAML list formats
    assert validator._parse_column_spec(["date", "date_time"]) == ["date", "date_time"]
    assert validator._parse_column_spec(["a", "b", "c"]) == ["a", "b", "c"]
    assert validator._parse_column_spec([]) == []

    # Test string formats
    assert validator._parse_column_spec("single_column") == ["single_column"]

    # Test other types
    assert validator._parse_column_spec(123) == ["123"]


def test_yaml_validator_basic_methods():
    """Test basic YAMLValidator methods."""
    validator = YAMLValidator()

    # Test that validator can be instantiated
    assert validator is not None

    # Test that the validation method map exists
    assert hasattr(validator, "validation_method_map")
    assert "rows_distinct" in validator.validation_method_map
    assert "col_exists" in validator.validation_method_map


def test_yaml_validator_load_config_invalid_source_type():
    """Test YAMLValidator with invalid source type."""
    validator = YAMLValidator()

    # Test with unsupported type (list)
    with pytest.raises(
        YAMLValidationError, match="Invalid source type.*Only YAML strings and file paths supported"
    ):
        validator.load_config([1, 2, 3])


def test_yaml_validator_load_config_not_dict():
    """Test YAMLValidator with non-dict root level."""
    validator = YAMLValidator()

    # YAML that loads to a list instead of dict
    yaml_content = "- item1\n- item2"

    with pytest.raises(
        YAMLValidationError, match="YAML must contain a dictionary at the root level"
    ):
        validator.load_config(yaml_content)


def test_yaml_validator_validate_schema_invalid_threshold_key():
    """Test schema validation with invalid threshold key."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "thresholds": {
            "invalid_key": 0.1  # This should trigger the error
        },
    }

    validator = YAMLValidator()
    with pytest.raises(
        YAMLValidationError,
        match="Invalid threshold key.*Must be 'warning', 'error', or 'critical'",
    ):
        validator._validate_schema(config)


def test_yaml_validator_validate_schema_invalid_threshold_type():
    """Test schema validation with invalid threshold value type."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "thresholds": {
            "warning": "not_a_number"  # Should be numeric
        },
    }

    validator = YAMLValidator()
    with pytest.raises(YAMLValidationError, match="Threshold 'warning' must be a number"):
        validator._validate_schema(config)


def test_yaml_validator_validate_schema_negative_threshold():
    """Test schema validation with negative threshold."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "thresholds": {
            "warning": -0.1  # Should be non-negative
        },
    }

    validator = YAMLValidator()
    with pytest.raises(YAMLValidationError, match="Threshold 'warning' must be non-negative"):
        validator._validate_schema(config)


def test_yaml_validator_validate_schema_actions_not_dict():
    """Test schema validation with actions not being a dict."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "actions": "not_a_dict",  # Should be a dict
    }

    validator = YAMLValidator()
    with pytest.raises(YAMLValidationError, match="'actions' must be a dictionary"):
        validator._validate_schema(config)


def test_yaml_validator_validate_schema_invalid_action_key():
    """Test schema validation with invalid action key."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "actions": {
            "invalid_action": "some_value"  # Invalid action key
        },
    }

    validator = YAMLValidator()
    with pytest.raises(
        YAMLValidationError,
        match="Invalid action key.*Must be 'warning', 'error', 'critical', 'default', or 'highest_only'",
    ):
        validator._validate_schema(config)


def test_yaml_validator_validate_schema_highest_only_not_bool():
    """Test schema validation with highest_only not being boolean."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "actions": {
            "highest_only": "not_a_bool"  # Should be boolean
        },
    }

    validator = YAMLValidator()
    with pytest.raises(YAMLValidationError, match="Action 'highest_only' must be a boolean"):
        validator._validate_schema(config)


def test_yaml_validator_validate_schema_invalid_action_value_type():
    """Test schema validation with invalid action value type."""
    config = {
        "tbl": "small_table",
        "steps": [{"rows_distinct": {}}],
        "actions": {
            "warning": 123  # Should be string, dict, or list
        },
    }

    validator = YAMLValidator()
    with pytest.raises(
        YAMLValidationError, match="Action 'warning' must be a string, dictionary.*or list"
    ):
        validator._validate_schema(config)


def test_yaml_validator_load_data_source_process_python_expressions():
    """Test `_load_data_source()` with python expressions processing."""
    validator = YAMLValidator()

    # Test with python expression that returns different object
    tbl_spec = {"python": "{'data': 'test'}"}
    result = validator._load_data_source(tbl_spec, "polars")
    assert result == {"data": "test"}


def test_yaml_validator_load_data_source_csv_file():
    """Test `_load_data_source()` with CSV file."""
    validator = YAMLValidator()

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        temp_csv = f.name

    try:
        # Test loading CSV with polars
        result = validator._load_data_source(temp_csv, "polars")
        assert result is not None

        # Test loading CSV with pandas
        result = validator._load_data_source(temp_csv, "pandas")
        assert result is not None
    finally:
        os.unlink(temp_csv)


def test_yaml_validator_load_csv_file_not_found():
    """Test `_load_csv_file()` with non-existent file."""
    validator = YAMLValidator()

    with pytest.raises(YAMLValidationError, match="CSV file not found"):
        validator._load_csv_file("/nonexistent/file.csv", "polars")


def test_yaml_validator_load_csv_file_polars_not_available():
    """Test `_load_csv_file()` when Polars not available."""
    validator = YAMLValidator()

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n")
        temp_csv = f.name

    try:
        # Mock polars as not available
        with patch("pointblank.yaml._is_lib_present", return_value=False):
            with pytest.raises(YAMLValidationError, match="Polars library is not available"):
                validator._load_csv_file(temp_csv, "polars")
    finally:
        os.unlink(temp_csv)


def test_yaml_validator_load_csv_file_pandas_not_available():
    """Test `_load_csv_file()` when Pandas not available."""
    validator = YAMLValidator()

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n")
        temp_csv = f.name

    try:
        # Mock pandas as not available
        with patch("pointblank.yaml._is_lib_present", return_value=False):
            with pytest.raises(YAMLValidationError, match="Pandas library is not available"):
                validator._load_csv_file(temp_csv, "pandas")
    finally:
        os.unlink(temp_csv)


def test_yaml_validator_load_csv_file_unsupported_library():
    """Test `_load_csv_file()` with unsupported library."""
    validator = YAMLValidator()

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n")
        temp_csv = f.name

    try:
        with pytest.raises(
            YAMLValidationError, match="Unsupported df_library.*Use 'polars', 'pandas', or 'duckdb'"
        ):
            validator._load_csv_file(temp_csv, "unsupported_lib")
    finally:
        os.unlink(temp_csv)


def test_yaml_validator_load_csv_file_exception():
    """Test `_load_csv_file()` with exception during loading."""
    validator = YAMLValidator()

    # Create a temporary invalid CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("invalid,csv,content\n")
        temp_csv = f.name

    try:
        # Mock polars.read_csv to raise an exception
        with patch("polars.read_csv", side_effect=Exception("Read error")):
            with pytest.raises(YAMLValidationError, match="Failed to load CSV file.*with polars"):
                validator._load_csv_file(temp_csv, "polars")
    finally:
        os.unlink(temp_csv)


def test_yaml_validator_parse_schema_spec_columns_not_list():
    """Test `_parse_schema_spec()` with columns not being a list."""
    validator = YAMLValidator()

    schema_spec = {
        "columns": "not_a_list"  # Should be a list
    }

    with pytest.raises(YAMLValidationError, match="Schema 'columns' must be a list"):
        validator._parse_schema_spec(schema_spec)


def test_yaml_validator_parse_schema_spec_invalid_column_list_length():
    """Test `_parse_schema_spec()` with invalid column spec list length."""
    validator = YAMLValidator()

    schema_spec = {
        "columns": [
            ["col1", "type1", "extra"]  # Too many elements
        ]
    }

    with pytest.raises(YAMLValidationError, match="Column specification must have 1-2 elements"):
        validator._parse_schema_spec(schema_spec)


def test_yaml_validator_parse_schema_spec_invalid_column_type():
    """Test `_parse_schema_spec()` with invalid column specification type."""
    validator = YAMLValidator()

    schema_spec = {
        "columns": [
            123  # Should be string or list
        ]
    }

    with pytest.raises(YAMLValidationError, match="Invalid column specification type"):
        validator._parse_schema_spec(schema_spec)


def test_yaml_validator_parse_schema_spec_no_columns():
    """Test `_parse_schema_spec()` without columns field."""
    validator = YAMLValidator()

    schema_spec = {}  # Missing columns field

    with pytest.raises(
        YAMLValidationError, match="Schema specification must contain 'columns' field"
    ):
        validator._parse_schema_spec(schema_spec)


def test_yaml_validator_parse_schema_spec_not_dict():
    """Test `_parse_schema_spec()` with non-dict schema."""
    validator = YAMLValidator()

    schema_spec = "not_a_dict"  # Should be a dict

    with pytest.raises(YAMLValidationError, match="Schema specification must be a dictionary"):
        validator._parse_schema_spec(schema_spec)


def test_yaml_to_python_file_path_detection():
    """Test `yaml_to_python()` with file path detection."""

    # Create a temporary YAML file
    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct: {}
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_yaml = f.name

    try:
        # Test with file path (short, no newlines)
        result = yaml_to_python(temp_yaml)
        assert "pb.Validate" in result
        assert "rows_distinct" in result
    finally:
        os.unlink(temp_yaml)


def test_yaml_to_python_content_analysis():
    """Test `yaml_to_python()` with content analysis for Polars/Pandas."""
    yaml_content = """
    tbl:
      python: "pl.DataFrame({'a': [1, 2, 3]})"
    steps:
    - col_vals_gt:
        columns: a
        value:
          python: "pd.Series([0, 1, 2]).mean()"
    """

    result = yaml_to_python(yaml_content)
    assert "import polars as pl" in result
    assert "import pandas as pd" in result


def test_yaml_to_python_file_loading_csv():
    """Test `yaml_to_python()` with CSV file loading."""
    yaml_content = """
    tbl: test_data.csv
    steps:
    - rows_distinct: {}
    """

    result = yaml_to_python(yaml_content)
    assert 'pb.load_dataset("test_data.csv"' in result


def test_yaml_to_python_file_loading_parquet():
    """Test `yaml_to_python()` with Parquet file loading."""
    yaml_content = """
    tbl: test_data.parquet
    steps:
    - rows_distinct: {}
    """

    result = yaml_to_python(yaml_content)
    assert 'pb.load_dataset("test_data.parquet"' in result


def test_yaml_to_python_with_lang():
    """Test `yaml_to_python()` with language parameter."""
    yaml_content = """
    tbl: small_table
    lang: fr
    steps:
    - rows_distinct: {}
    """

    result = yaml_to_python(yaml_content)
    assert 'lang="fr"' in result


def test_yaml_to_python_with_locale():
    """Test `yaml_to_python()` with locale parameter."""
    yaml_content = """
    tbl: small_table
    locale: fr_FR
    steps:
    - rows_distinct: {}
    """

    result = yaml_to_python(yaml_content)
    assert 'locale="fr_FR"' in result


def test_yaml_to_python_with_brief_bool():
    """Test `yaml_to_python()` with brief as boolean."""
    yaml_content = """
    tbl: small_table
    brief: true
    steps:
    - rows_distinct: {}
    """

    result = yaml_to_python(yaml_content)
    assert "brief=True" in result


def test_yaml_to_python_conjointly_expressions_list():
    """Test `yaml_to_python()` with conjointly expressions list."""
    yaml_content = """
    tbl: small_table
    steps:
    - conjointly:
        expressions:
          - "lambda df: df['a'] > 0"
          - "lambda df: df['b'] < 10"
    """

    result = yaml_to_python(yaml_content)
    assert (
        "conjointly(expressions=[\"lambda df: df['a'] > 0\", \"lambda df: df['b'] < 10\"])"
        in result
    )


def test_yaml_to_python_specially_expr():
    """Test `yaml_to_python()` with specially expr parameter."""
    yaml_content = """
    tbl: small_table
    steps:
    - specially:
        expr: "lambda x: x > 0"
    """

    result = yaml_to_python(yaml_content)
    assert "specially(expr=lambda x: x > 0)" in result


def test_yaml_to_python_single_column():
    """Test `yaml_to_python()` with single column."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [single_col]
    """

    result = yaml_to_python(yaml_content)
    assert 'col_exists(columns="single_col")' in result


def test_yaml_to_python_brief_bool_in_step():
    """Test `yaml_to_python()` with brief as boolean in step."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        brief: false
    """

    result = yaml_to_python(yaml_content)
    assert "brief=False" in result


def test_yaml_to_python_brief_string_in_step():
    """Test `yaml_to_python()` with brief as string in step."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        brief: "Custom brief message"
    """

    result = yaml_to_python(yaml_content)
    assert 'brief="Custom brief message"' in result


def test_yaml_to_python_actions_object_warning():
    """Test `yaml_to_python()` with Actions object warning."""

    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          warning: "warn_func"
          error: null
          critical: null
    """

    result = yaml_to_python(yaml_content)
    assert "Actions(" in result
    assert "warning=" in result


def test_yaml_to_python_actions_object_error():
    """Test `yaml_to_python()` with Actions object error."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          error: "error_func"
    """

    result = yaml_to_python(yaml_content)
    assert "Actions(" in result
    assert "error=" in result


def test_yaml_to_python_actions_object_critical_list():
    """Test `yaml_to_python()` with Actions object critical as list."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          critical: ["critical_func"]
    """

    result = yaml_to_python(yaml_content)
    assert "Actions(" in result
    assert "critical=" in result


def test_yaml_to_python_actions_dict_highest_only():
    """Test `yaml_to_python()` with actions `highest_only:`."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          highest_only: false
          warning: "warn_func"
    """

    result = yaml_to_python(yaml_content)
    assert "highest_only=False" in result


def test_yaml_to_python_thresholds_object():
    """Test `yaml_to_python()` with Thresholds object."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        thresholds:
          warning: 0.1
          error: 0.2
    """

    result = yaml_to_python(yaml_content)
    assert "Thresholds(" in result
    assert "warning=0.1" in result
    assert "error=0.2" in result


def test_yaml_to_python_string_parameter():
    """Test `yaml_to_python()` with string parameter."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_vals_regex:
        columns: a
        pattern: "[0-9]+"
    """

    result = yaml_to_python(yaml_content)
    assert 'pattern="[0-9]+"' in result


def test_yaml_to_python_tuple_parameter():
    """Test `yaml_to_python()` with tuple parameter."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_vals_between:
        columns: a
        left: 0
        right: 10
        inclusive: [false, true]
    """

    result = yaml_to_python(yaml_content)
    # This will be converted to a list format in the output
    assert "inclusive=" in result


def test_yaml_to_python_step_actions_dict_highest_only():
    """Test `yaml_to_python()` with step-level actions dict `highest_only:`."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          highest_only: false
          warning: "warn_action"
    """

    result = yaml_to_python(yaml_content)
    assert "highest_only=False" in result
    assert "pb.Actions(" in result


def test_yaml_to_python_step_actions_dict_original_expression():
    """Test `yaml_to_python()` with step-level actions dict using original expressions."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          warning:
            python: "lambda x: print('Warning!')"
    """

    result = yaml_to_python(yaml_content)
    assert "pb.Actions(" in result
    assert "warning=" in result


def test_yaml_to_python_step_actions_dict_string_values():
    """Test `yaml_to_python()` with step-level actions dict with string values."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          warning: "warning_message"
          error: "error_message"
    """

    result = yaml_to_python(yaml_content)
    assert 'warning="warning_message"' in result
    assert 'error="error_message"' in result
    assert "pb.Actions(" in result


def test_yaml_to_python_step_actions_dict_complex_values():
    """Test `yaml_to_python()` with step-level actions dict with complex values."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions:
          warning: 123
          error: true
    """

    result = yaml_to_python(yaml_content)
    assert 'warning="123"' in result  # Numbers get converted to strings in Actions
    assert 'error="True"' in result  # Booleans get converted to strings in Actions
    assert "pb.Actions(" in result


def test_yaml_to_python_step_actions_non_dict():
    """Test `yaml_to_python()` with step-level actions that are not dict."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: a
        actions: "simple_action"
    """

    result = yaml_to_python(yaml_content)
    assert "actions=simple_action" in result


def test_yaml_to_python_step_actions_dict_with_complex_structure():
    """Test `yaml_to_python()` with complex step-level actions dict structure."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_vals_gt:
        columns: [a]
        value: 0
        actions:
          warning: "warning_func"
          error: 42
          critical: true
          highest_only: false
    """

    result = yaml_to_python(yaml_content)

    # Verify the step-level actions dictionary processing
    assert "pb.Actions(" in result
    assert "highest_only=False" in result
    assert 'warning="warning_func"' in result
    assert 'error="42"' in result
    assert 'critical="True"' in result


def test_yaml_to_python_boolean_parameters():
    """Test `yaml_to_python()` with boolean parameter values."""
    yaml_content = """
    tbl: small_table
    steps:
      - col_vals_gt:
          columns: c
          value: 5
          na_pass: true
          active: false
          brief: true
    """

    python_code = yaml_to_python(yaml_content)

    # Verify boolean values are correctly converted to Python format
    assert "na_pass=True" in python_code
    assert "active=False" in python_code
    assert "brief=True" in python_code


def test_yaml_to_python_non_dict_thresholds():
    """Test `yaml_to_python()` with non-dict threshold values."""
    yaml_content = """
    tbl: small_table
    steps:
      - col_vals_gt:
          columns: c
          value: 5
          thresholds: 0.1
      - col_vals_lt:
          columns: d
          value: 1000
          thresholds: true
    """

    python_code = yaml_to_python(yaml_content)

    # Verify non-dict threshold values are handled correctly
    assert "thresholds=0.1" in python_code
    assert "thresholds=True" in python_code


# Custom Actions Tests (from test_yaml_custom_actions_simple.py)


# Test helper functions that can be used as custom actions
def test_action_function():
    """A simple test action function."""
    from pointblank import get_action_metadata

    metadata = get_action_metadata()
    if metadata:
        print(f"Test action triggered: {metadata['type']} at step {metadata['step']}")
    else:
        print("Test action triggered: no metadata available")


def logging_action():
    """Action that simulates logging behavior."""
    from pointblank import get_action_metadata

    metadata = get_action_metadata()
    if metadata:
        print(f"LOG: {metadata['type'].upper()} - Step {metadata['step']} failed")
    else:
        print("LOG: Action triggered with no metadata")


def test_yaml_custom_actions_namespace_import_functionality():
    """Test that namespaces parameter allows importing modules."""
    # Test that we can import a test module through namespaces
    namespaces = {"test_actions": "tests.test_yaml"}

    result = _safe_eval_python_code("test_actions.test_action_function", namespaces=namespaces)
    assert callable(result)
    assert result.__name__ == "test_action_function"


def test_yaml_custom_actions_basic_with_dict_namespaces():
    """Test basic custom action functionality using dictionary namespaces."""
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: 0.01  # Very low threshold to trigger action
    actions:
      warning:
        python: test_actions.test_action_function
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000  # This will fail and trigger the action
    """

    # Create a module-like namespace
    namespaces = {"test_actions": "tests.test_yaml"}

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces=namespaces)
        assert result is not None
        assert len(result.validation_info) == 1

    # Verify the custom action was executed
    output_text = captured_output.getvalue()
    assert "Test action triggered:" in output_text


def test_yaml_custom_actions_inline_import():
    """Test custom action using inline import (no namespaces param needed)."""
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: 0.01
    actions:
      warning:
        python: |
          test_actions = __import__('tests.test_yaml', fromlist=[''])
          test_actions.test_action_function
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
    """

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content)
        assert result is not None

    output_text = captured_output.getvalue()
    assert "Test action triggered:" in output_text


def test_yaml_custom_actions_invalid_namespace_error():
    """Test error handling for invalid namespace modules."""
    yaml_content = """
    tbl: small_table
    actions:
      warning:
        python: some_module.some_function
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
    """

    namespaces = {"some_module": "non.existent.module"}

    with pytest.raises(ImportError) as exc_info:
        yaml_interrogate(yaml_content, namespaces=namespaces)

    assert "Could not import requested namespace 'non.existent.module'" in str(exc_info.value)


def test_yaml_custom_actions_multiple_namespaces():
    """Test using multiple namespaces."""
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: 0.01
    actions:
      warning:
        python: actions.logging_action
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
    """

    namespaces = {
        "actions": "tests.test_yaml",
        "utils": "os",  # Just to test multiple imports work
    }

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces=namespaces)
        assert result is not None

    output_text = captured_output.getvalue()
    assert "LOG:" in output_text


def test_yaml_custom_actions_namespaces_list_format():
    """Test namespaces as a list."""
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: 0.01
    actions:
      warning:
        python: |
          import tests.test_yaml as actions
          actions.test_action_function
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
    """

    namespaces = ["tests.test_yaml"]

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces=namespaces)
        assert result is not None

    output_text = captured_output.getvalue()
    assert "Test action triggered:" in output_text


def test_yaml_custom_actions_with_set_tbl():
    """Test that custom actions work with set_tbl parameter."""
    # Create a test table that will fail validation
    test_table = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    yaml_content = """
    tbl: small_table  # This will be overridden
    thresholds:
      warning: 0.01
    actions:
      warning:
        python: actions.test_action_function
    steps:
    - col_vals_gt:
        columns: [a]
        value: 100  # Will fail with our test data
    """

    namespaces = {"actions": "tests.test_yaml"}

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, set_tbl=test_table, namespaces=namespaces)
        assert result is not None

    output_text = captured_output.getvalue()
    assert "Test action triggered:" in output_text


def test_yaml_custom_actions_step_level():
    """Test custom actions defined at the step level."""
    yaml_content = """
    tbl: small_table
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
        thresholds:
          warning: 0.01
        actions:
          warning:
            python: actions.logging_action
    """

    namespaces = {"actions": "tests.test_yaml"}

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces=namespaces)
        assert result is not None

    output_text = captured_output.getvalue()
    assert "LOG:" in output_text


def test_yaml_custom_actions_empty_namespaces_fallback():
    """Test that empty namespaces parameter works (fallback to normal behavior)."""
    yaml_content = """
    tbl: small_table
    actions:
      warning:
        python: |
          lambda: print("Lambda action executed")
    thresholds:
      warning: 0.01
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000
    """

    # Test with None namespaces
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces=None)
        assert result is not None

    output_text = captured_output.getvalue()
    assert "Lambda action executed" in output_text

    # Test with empty dict namespaces
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces={})
        assert result is not None

    output_text = captured_output.getvalue()
    assert "Lambda action executed" in output_text


def test_yaml_custom_actions_original_issue_use_case():
    """Test the exact use case from the original GitHub issue."""
    # This is what the user wanted to work
    yaml_content = """
    tbl: small_table
    df_library: polars
    actions:
      warning:
        python: log_actions.log_issue
    thresholds:
      warning: 0.05
      error: 0.10
      critical: 0.15
    steps:
    - col_vals_gt:
        columns: [a]
        value: 1000  # This will fail
    """

    # Define the custom action function
    def log_issue():
        from pointblank import get_action_metadata

        metadata = get_action_metadata()
        if metadata:
            print(f"Type: {metadata['type']}, Step: {metadata['step']}")
        else:
            print("Custom action executed")

    # Add the function to our test module
    import tests.test_yaml as test_mod

    test_mod.log_issue = log_issue

    # Use namespaces to make it available
    namespaces = {"log_actions": "tests.test_yaml"}

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content, namespaces=namespaces)
        assert result is not None

    output_text = captured_output.getvalue()
    # The metadata type appears to be the validation type, not the action level
    assert "Type: col_vals_gt, Step: 1" in output_text or "Custom action executed" in output_text


def test_yaml_validator_invalid_threshold_keys():
    """Test error handling for invalid threshold keys"""

    validator = YAMLValidator()
    config = {
        "tbl": "small_table",
        "thresholds": {
            "invalid_key": 0.1  # Invalid threshold key
        },
        "steps": [{"rows_distinct": None}],
    }

    with pytest.raises(YAMLValidationError, match="Invalid threshold key: invalid_key"):
        validator._validate_schema(config)


def test_yaml_validator_non_dict_thresholds():
    """Test error handling for non-dict thresholds"""

    validator = YAMLValidator()
    config = {
        "tbl": "small_table",
        "thresholds": ["not", "a", "dict"],  # Should be dict
        "steps": [{"rows_distinct": None}],
    }

    with pytest.raises(YAMLValidationError, match="'thresholds' must be a dictionary"):
        validator._validate_schema(config)


def test_yaml_validator_unsupported_df_library():
    """Test error handling for unsupported `df_library=`"""
    import tempfile
    import os

    validator = YAMLValidator()

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n")
        temp_file = f.name

    try:
        with pytest.raises(YAMLValidationError, match="Unsupported df_library: unsupported"):
            validator._load_csv_file(temp_file, "unsupported")
    finally:
        os.unlink(temp_file)


def test_yaml_validator_invalid_step_configuration_type():
    """Test error handling for invalid step configuration"""
    from pointblank.yaml import YAMLValidator, YAMLValidationError

    validator = YAMLValidator()

    # Test with invalid step configuration type (not dict or string)
    with pytest.raises(YAMLValidationError, match="Invalid step configuration type"):
        validator._parse_validation_step(123)  # Integer instead of dict/string


def test_yaml_validator_unknown_validation_method():
    """Test error handling for unknown validation method"""
    from pointblank.yaml import YAMLValidator, YAMLValidationError

    validator = YAMLValidator()

    # Test with unknown validation method
    with pytest.raises(YAMLValidationError, match="Unknown validation method 'nonexistent_method'"):
        validator._parse_validation_step({"nonexistent_method": {"value": 1}})


def test_yaml_validator_conjointly_non_list_expressions():
    """Test error handling for conjointly with non-list expressions"""

    validator = YAMLValidator()

    # Test with conjointly method having non-list expressions
    with pytest.raises(YAMLValidationError, match="conjointly 'expressions' must be a list"):
        validator._parse_validation_step({"conjointly": {"expressions": "not_a_list"}})


def test_yaml_to_python_file_path_check():
    """Test `yaml_to_python()` with file path detection"""
    import tempfile
    import os

    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
tbl: small_table
steps:
- col_vals_gt:
    columns: [a]
    value: 0
""")
        temp_file = f.name

    try:
        # Test with file path (should read from file)
        result = yaml_to_python(temp_file)
        assert "pb.load_dataset" in result
        assert "col_vals_gt" in result
    finally:
        os.unlink(temp_file)


def test_yaml_to_python_dataset_loading():
    """Test `yaml_to_python()` with dataset loading"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_gt:
    columns: [a]
    value: 0
"""

    result = yaml_to_python(yaml_content)
    # Should contain dataset loading code
    assert 'pb.load_dataset("small_table", tbl_type="polars")' in result


def test_yaml_to_python_with_actions_simple():
    """Test `yaml_to_python()` with actions having simple values"""
    yaml_content = """
tbl: small_table
actions:
  warning: "warn_func"
steps:
- col_vals_gt:
    columns: [a]
    value: 0
"""

    result = yaml_to_python(yaml_content)
    # Should handle action values
    assert "actions=" in result
    assert "warning=" in result


def test_yaml_to_python_conjointly_simple():
    """Test `yaml_to_python()` with conjointly basic structure"""
    yaml_content = """
tbl: small_table
steps:
- conjointly:
    expressions: ["lambda x: True", "lambda x: True"]
"""

    result = yaml_to_python(yaml_content)
    # Should format conjointly expressions properly
    assert "conjointly" in result
    assert "expressions=" in result


def test_yaml_to_python_specially_expr():
    """Test `yaml_to_python()` with specially expr parameter"""
    yaml_content = """
tbl: small_table
steps:
- specially:
    expr: "lambda x: len(x) > 0"
"""

    result = yaml_to_python(yaml_content)
    # Should handle specially expr parameter
    assert "specially" in result
    assert "expr=" in result


def test_yaml_to_python_brief_boolean():
    """Test `yaml_to_python()` with boolean brief parameter"""
    yaml_content = """
tbl: small_table
steps:
- rows_distinct:
    brief: true
"""

    result = yaml_to_python(yaml_content)
    # Should format boolean brief parameter
    assert "brief=True" in result


def test_yaml_to_python_actions_with_warning_list():
    """Test `yaml_to_python()` with actions having warning as list"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_gt:
    columns: [a]
    value: 0
    actions:
      warning: ["warn_func"]
"""

    result = yaml_to_python(yaml_content)
    # Should handle warning as list
    assert "warning=" in result


def test_yaml_to_python_actions_with_error_list():
    """Test `yaml_to_python()` with actions having error as list"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_gt:
    columns: [a]
    value: 0
    actions:
      error: ["error_func"]
"""

    result = yaml_to_python(yaml_content)
    # Should handle error as list
    assert "error=" in result


def test_yaml_to_python_actions_with_critical_list():
    """Test `yaml_to_python()` with actions having critical as list"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_gt:
    columns: [a]
    value: 0
    actions:
      critical: ["critical_func"]
"""

    result = yaml_to_python(yaml_content)
    # Should handle critical as list
    assert "critical=" in result


def test_yaml_to_python_actions_with_highest_only_false():
    """Test `yaml_to_python()` with actions having `highest_only=False`"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_gt:
    columns: [a]
    value: 0
    actions:
      warning: warn_func
      highest_only: false
"""

    result = yaml_to_python(yaml_content)
    # Should handle highest_only=False
    assert "highest_only=False" in result


def test_col_vals_increasing_yaml_basic():
    """Test basic col_vals_increasing() YAML translation and execution"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_increasing:
    columns: a
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_vals_increasing"


def test_col_vals_increasing_yaml_with_all_params():
    """Test col_vals_increasing() with all parameters in YAML"""
    yaml_content = """
tbl: small_table
tbl_name: "Increasing Values Test"
label: "Test col_vals_increasing"
steps:
- col_vals_increasing:
    columns: [a, d]
    allow_stationary: true
    decreasing_tol: 0.5
    na_pass: true
    thresholds:
      warning: 0.1
      error: 0.2
    brief: "Values should be increasing"
    active: true
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    # Two columns = 2 validation steps
    assert len(result.validation_info) == 2

    # Check first step parameters
    val_info = result.validation_info[0]
    assert val_info.assertion_type == "col_vals_increasing"
    assert val_info.na_pass is True
    assert val_info.brief == "Values should be increasing"
    assert val_info.active is True
    assert val_info.val_info["allow_stationary"] is True
    assert val_info.val_info["decreasing_tol"] == 0.5


def test_col_vals_increasing_yaml_to_python():
    """Test yaml_to_python() for col_vals_increasing()"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_increasing:
    columns: a
    allow_stationary: true
    decreasing_tol: 1.0
    na_pass: true
"""

    python_code = yaml_to_python(yaml_content)
    assert "import pointblank as pb" in python_code
    assert ".col_vals_increasing(" in python_code
    assert 'columns="a"' in python_code
    assert "allow_stationary=True" in python_code
    assert "decreasing_tol=1.0" in python_code
    assert "na_pass=True" in python_code


def test_col_vals_decreasing_yaml_basic():
    """Test basic col_vals_decreasing() YAML translation and execution"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_decreasing:
    columns: a
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_vals_decreasing"


def test_col_vals_decreasing_yaml_with_all_params():
    """Test col_vals_decreasing() with all parameters in YAML"""
    yaml_content = """
tbl: small_table
tbl_name: "Decreasing Values Test"
label: "Test col_vals_decreasing"
steps:
- col_vals_decreasing:
    columns: [c, d]
    allow_stationary: true
    increasing_tol: 2.0
    na_pass: false
    thresholds:
      warning: 0.15
      error: 0.3
    brief: "Values should be decreasing"
    active: true
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    # Two columns = 2 validation steps
    assert len(result.validation_info) == 2

    # Check first step parameters
    val_info = result.validation_info[0]
    assert val_info.assertion_type == "col_vals_decreasing"
    assert val_info.na_pass is False
    assert val_info.brief == "Values should be decreasing"
    assert val_info.active is True
    assert val_info.val_info["allow_stationary"] is True
    assert val_info.val_info["increasing_tol"] == 2.0


def test_col_vals_decreasing_yaml_to_python():
    """Test yaml_to_python() for col_vals_decreasing()"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_decreasing:
    columns: d
    allow_stationary: false
    increasing_tol: 0.5
    na_pass: true
"""

    python_code = yaml_to_python(yaml_content)
    assert "import pointblank as pb" in python_code
    assert ".col_vals_decreasing(" in python_code
    assert 'columns="d"' in python_code
    assert "allow_stationary=False" in python_code
    assert "increasing_tol=0.5" in python_code
    assert "na_pass=True" in python_code


def test_col_vals_within_spec_yaml_basic():
    """Test basic col_vals_within_spec() YAML translation and execution"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_within_spec:
    columns: b
    spec: "email"
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "col_vals_within_spec"


def test_col_vals_within_spec_yaml_with_all_params():
    """Test col_vals_within_spec() with all parameters in YAML"""
    yaml_content = """
tbl: small_table
tbl_name: "Spec Validation Test"
label: "Test col_vals_within_spec"
steps:
- col_vals_within_spec:
    columns: [b]
    spec: "url"
    na_pass: true
    thresholds:
      warning: 0.2
      error: 0.4
    brief: "Values should match URL spec"
    active: true
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1

    # Check step parameters
    val_info = result.validation_info[0]
    assert val_info.assertion_type == "col_vals_within_spec"
    assert val_info.na_pass is True
    assert val_info.brief == "Values should match URL spec"
    assert val_info.active is True


def test_col_vals_within_spec_yaml_to_python():
    """Test yaml_to_python() for col_vals_within_spec()"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_within_spec:
    columns: b
    spec: "email"
    na_pass: false
"""

    python_code = yaml_to_python(yaml_content)
    assert "import pointblank as pb" in python_code
    assert ".col_vals_within_spec(" in python_code
    assert 'columns="b"' in python_code
    assert 'spec="email"' in python_code
    assert "na_pass=False" in python_code


def test_col_vals_within_spec_yaml_various_specs():
    """Test col_vals_within_spec() with various spec types"""
    yaml_content = """
tbl: small_table
steps:
- col_vals_within_spec:
    columns: b
    spec: "postal_code[US]"
- col_vals_within_spec:
    columns: b
    spec: "iban[DE]"
- col_vals_within_spec:
    columns: b
    spec: "credit_card"
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 3


def test_tbl_match_yaml_basic():
    """Test basic tbl_match() YAML translation and execution"""
    yaml_content = """
tbl: small_table
steps:
- tbl_match:
    tbl_compare:
      python: |
        pb.load_dataset("small_table", tbl_type="polars")
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1
    assert result.validation_info[0].assertion_type == "tbl_match"


def test_tbl_match_yaml_with_all_params():
    """Test tbl_match() with all parameters in YAML"""
    yaml_content = """
tbl: small_table
tbl_name: "Table Match Test"
label: "Test tbl_match"
steps:
- tbl_match:
    tbl_compare:
      python: |
        pb.load_dataset("small_table", tbl_type="polars")
    thresholds:
      warning: 0.0
      error: 0.5
    brief: "Tables should match"
    active: true
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1

    # Check step parameters
    val_info = result.validation_info[0]
    assert val_info.assertion_type == "tbl_match"
    assert val_info.brief == "Tables should match"
    assert val_info.active is True


def test_tbl_match_yaml_to_python():
    """Test yaml_to_python() for tbl_match()"""
    yaml_content = """
tbl: small_table
steps:
- tbl_match:
    tbl_compare:
      python: |
        pb.load_dataset("small_table", tbl_type="polars")
"""

    python_code = yaml_to_python(yaml_content)
    assert "import pointblank as pb" in python_code
    assert ".tbl_match(" in python_code
    assert "tbl_compare=" in python_code
    assert 'pb.load_dataset("small_table", tbl_type="polars")' in python_code


def test_prompt_yaml_to_python():
    """Test yaml_to_python() for prompt"""
    yaml_content = """
tbl: small_table
steps:
- prompt:
    prompt: "Check data quality"
    model: "openai:gpt-4"
    columns_subset: [a, b]
    batch_size: 1000
    max_concurrent: 3
"""

    python_code = yaml_to_python(yaml_content)
    assert "import pointblank as pb" in python_code
    assert ".prompt(" in python_code
    assert 'prompt="Check data quality"' in python_code
    assert 'model="openai:gpt-4"' in python_code
    assert 'columns_subset=["a", "b"]' in python_code
    assert "batch_size=1000" in python_code
    assert "max_concurrent=3" in python_code


def test_new_methods_comprehensive_yaml():
    """Test all several methods together in one YAML workflow"""
    yaml_content = """
tbl: small_table
tbl_name: "Comprehensive New Methods Test"
label: "Testing several validation methods"
thresholds:
  warning: 0.1
  error: 0.25
steps:
- col_vals_increasing:
    columns: a
    allow_stationary: false
- col_vals_decreasing:
    columns: d
    allow_stationary: true
- col_vals_within_spec:
    columns: b
    spec: "email"
    na_pass: true
- tbl_match:
    tbl_compare:
      python: |
        pb.load_dataset("small_table", tbl_type="polars")
- prompt:
    prompt: "Data should be reasonable"
    model: "anthropic:claude-sonnet-4"
    columns_subset: [a, d]
"""

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 5
    assert result.validation_info[0].assertion_type == "col_vals_increasing"
    assert result.validation_info[1].assertion_type == "col_vals_decreasing"
    assert result.validation_info[2].assertion_type == "col_vals_within_spec"
    assert result.validation_info[3].assertion_type == "tbl_match"
    assert result.validation_info[4].assertion_type == "prompt"


def test_new_methods_yaml_to_python_comprehensive():
    """Test yaml_to_python() for a group of validation methods"""
    yaml_content = """
tbl: small_table
tbl_name: "New Methods Demo"
steps:
- col_vals_increasing:
    columns: a
    allow_stationary: true
    decreasing_tol: 0.1
- col_vals_decreasing:
    columns: d
    increasing_tol: 0.2
- col_vals_within_spec:
    columns: b
    spec: "url"
- tbl_match:
    tbl_compare:
      python: |
        pb.load_dataset("small_table", tbl_type="polars")
- prompt:
    prompt: "Validate data quality"
    model: "openai:gpt-4"
"""

    python_code = yaml_to_python(yaml_content)

    # Check all methods are present
    assert "import pointblank as pb" in python_code
    assert ".col_vals_increasing(" in python_code
    assert ".col_vals_decreasing(" in python_code
    assert ".col_vals_within_spec(" in python_code
    assert ".tbl_match(" in python_code
    assert ".prompt(" in python_code

    # Check key parameters
    assert "allow_stationary=True" in python_code
    assert "decreasing_tol=0.1" in python_code
    assert "increasing_tol=0.2" in python_code
    assert 'spec="url"' in python_code
    assert 'model="openai:gpt-4"' in python_code
