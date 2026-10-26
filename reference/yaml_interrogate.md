# yaml_interrogate()

Execute a YAML-based validation workflow.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/yaml.py#L998-L1342)

``` python
yaml_interrogate(
    yaml,
    set_tbl=None,
    namespaces=None,
)
```

This is the main entry point for YAML-based validation workflows. It takes YAML configuration (as a string or file path) and returns a validated [Validate](../reference/Validate.md#pointblank.Validate) object with interrogation results.

The YAML configuration defines the data source, validation steps, and optional settings like thresholds and labels. This function automatically loads the data, builds the validation plan, executes all validation steps, and returns the interrogated results.

## Parameters

`yaml: Union[str, Path]`  
YAML configuration as string or file path. Can be: (1) a YAML string containing the validation configuration, or (2) a Path object or string path to a YAML file.

`set_tbl: Any = None`  
An optional table to override the table specified in the YAML configuration. This allows you to apply a YAML-defined validation workflow to a different table than what's specified in the configuration. If provided, this table is used in place of the table defined in the YAML's `tbl` field (which is then not loaded, so it can be a placeholder or a source that isn't available). This can be any supported table type including DataFrame objects, Ibis table objects, CSV file paths, Parquet file paths, GitHub URLs, or database connection strings.

`namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None`  
Optional module namespaces to make available for Python code execution in YAML configurations. Can be a dictionary mapping aliases to module names or a list of module names. See the "Using Namespaces" section below for detailed examples.

## Returns

`Validate`  
An instance of the [Validate](../reference/Validate.md#pointblank.Validate) class that has been configured based on the YAML input. This object contains the results of the validation steps defined in the YAML configuration. It includes metadata like table name, label, language, and thresholds if specified.

## Raises

`YAMLValidationError`  
If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing required fields, unknown validation methods, or data loading failures.

## Using Namespaces

The `namespaces=` parameter enables custom Python modules and functions in YAML configurations. This is particularly useful for custom action functions and advanced Python expressions.

**Namespace formats:**

- Dictionary format: `{"alias": "module.name"}` maps aliases to module names
- List format: `["module.name", "another.module"]` imports modules directly

**Option 1: Inline expressions (no namespaces needed)**

``` python
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

result = pb.yaml_interrogate(yaml_config)
result
```

    Custom warning triggered

[TABLE]

**Option 2: External functions with namespaces**

``` python
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
result = pb.yaml_interrogate(yaml_config, namespaces={'actions': '__main__'})
result
```

    Data validation failed: please check your data.

[TABLE]

This approach enables modular, reusable validation workflows with custom business logic.

## Examples

For the examples here, we'll use YAML configurations to define validation workflows. Let's start with a basic YAML workflow that validates the built-in `small_table` dataset.

``` python
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

[TABLE]

The validation table shows the results of our YAML-defined workflow. We can see that the [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) validation failed (because there are duplicate rows in the table), while the column existence checks passed.

Now let's create a more comprehensive validation workflow with thresholds and metadata:

``` python
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

    Table name: small_table_demo
    Label: Comprehensive data validation
    Total validation steps: 4

The validation results now include our custom table name and label. The thresholds we defined will determine when validation steps are marked as warnings, errors, or critical failures.

You can also load YAML configurations from files. Here's how you would work with a YAML file:

``` python
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

[TABLE]

This approach is particularly useful for storing validation configurations as part of your data pipeline or version control system, allowing you to maintain validation rules alongside your code.

### Governance Metadata

YAML workflows support governance metadata via `owner`, `consumers`, and `version` top-level keys. These are forwarded to the [Validate](../reference/Validate.md#pointblank.Validate) constructor and embedded in the validation report:

``` python
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

    Owner: Data Engineering
    Consumers: ['Analytics', 'Finance', 'Compliance']
    Version: 2.1.0

### Aggregate Validations

YAML supports aggregate validation methods for checking column-level statistics. These methods validate that a column's sum, average, or standard deviation meets a threshold:

``` python
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

[TABLE]

The 15 available aggregate methods follow the pattern `col_{stat}_{comparator}` where `{stat}` is `sum`, `avg`, or `sd` and `{comparator}` is `gt`, `lt`, `ge`, `le`, or `eq`.

### Data Freshness

Check that a date/datetime column has recent data using [data_freshness](../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness):

``` yaml
tbl: events.csv
steps:
- data_freshness:
    columns: event_date
    freshness: "24h"
```

### Active Parameter Shortcut

The `active=` parameter controls whether a validation step runs. It supports boolean values and Python expression shortcuts:

``` yaml
steps:
- col_vals_gt:
    columns: [d]
    value: 100
    active: false            # Skip this step

- col_vals_not_null:
    columns: [a]
    active: true             # Always run (default)
```

### Null Percentage Check

Use [col_pct_null](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null) to validate that the percentage of null values in a column is within bounds:

``` yaml
steps:
- col_pct_null:
    columns: [a, b]
    value: 0.05
```

### Using `set_tbl=` to Override the Table

The `set_tbl=` parameter allows you to override the table specified in the YAML configuration. This is useful when you have a template validation workflow but want to apply it to different tables:

``` python
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

    Validation applied to: Test Table

[TABLE]

This feature makes YAML configurations more reusable and flexible, allowing you to define validation logic once and apply it to multiple similar tables.
