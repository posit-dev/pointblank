## validate_yaml()


Validate YAML configuration against the expected structure.


Usage

``` python
validate_yaml(yaml)
```


This function validates that a YAML configuration conforms to the expected structure for validation workflows. It checks for required fields, proper data types, and valid validation method names. This is useful for validating configurations before execution or for building configuration editors and validators.

The function performs comprehensive validation including:

- required fields ('tbl' and 'steps')
- proper data types for all fields
- valid threshold configurations
- known validation method names
- proper step configuration structure


## Parameters


`yaml: Union[str, Path]`  
YAML configuration as string or file path. Can be: (1) a YAML string containing the validation configuration, or (2) a Path object or string path to a YAML file.


## Raises


`YAMLValidationError`  
If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing required fields, unknown validation methods, or data loading failures.


## Examples

For the examples here, we'll demonstrate how to validate YAML configurations before using them with validation workflows. This is particularly useful for building robust data validation systems where you want to catch configuration errors early.

Let's start with validating a basic configuration:


``` python
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


    Basic YAML configuration is valid


The function completed without raising an exception, which means our configuration is valid and follows the expected structure.

Now let's validate a more complex configuration with thresholds and metadata:


``` python
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


    Complex YAML configuration is valid
    Configuration has 3 validation steps


This configuration includes all the optional metadata fields and complex validation steps, demonstrating that the validation handles the full range of supported options.

Let's see what happens when we try to validate an invalid configuration:


``` python
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


    Validation failed: Error loading YAML configuration: YAML must contain 'tbl' field


The validation correctly identifies that our configuration is missing the required `'tbl'` field.

Here's a practical example of using validation in a workflow builder:


``` python
def safe_yaml_interrogate(yaml_config):
    """Safely execute a YAML configuration after validation."""
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


    ✓ YAML configuration is valid
    Validation completed with 1 steps


This pattern of validating before executing helps build more reliable data validation pipelines by catching configuration errors early in the process.

Note that this function only validates the structure and does not check if the specified data source ('tbl') exists or is accessible. Data source validation occurs during execution with [yaml_interrogate()](yaml_interrogate.md#pointblank.yaml_interrogate).


## Supported Top-Level Keys

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

Unknown top-level keys are rejected, which catches typos like `tbl_nmae` or `step`.


## Supported Validation Methods

In addition to all standard validation methods (e.g., [col_vals_gt](Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [rows_distinct](Validate.rows_distinct.md#pointblank.Validate.rows_distinct), [col_schema_match](Validate.col_schema_match.md#pointblank.Validate.col_schema_match)), the following methods are also supported:

- [col_pct_null](Validate.col_pct_null.md#pointblank.Validate.col_pct_null): check the percentage of null values in a column
- [data_freshness](Validate.data_freshness.md#pointblank.Validate.data_freshness): check that data is recent
- aggregate methods: [col_sum_gt](Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [col_sum_lt](Validate.col_sum_lt.md#pointblank.Validate.col_sum_lt), [col_sum_ge](Validate.col_sum_ge.md#pointblank.Validate.col_sum_ge), [col_sum_le](Validate.col_sum_le.md#pointblank.Validate.col_sum_le), [col_sum_eq](Validate.col_sum_eq.md#pointblank.Validate.col_sum_eq), [col_avg_gt](Validate.col_avg_gt.md#pointblank.Validate.col_avg_gt), [col_avg_lt](Validate.col_avg_lt.md#pointblank.Validate.col_avg_lt), [col_avg_ge](Validate.col_avg_ge.md#pointblank.Validate.col_avg_ge), [col_avg_le](Validate.col_avg_le.md#pointblank.Validate.col_avg_le), [col_avg_eq](Validate.col_avg_eq.md#pointblank.Validate.col_avg_eq), [col_sd_gt](Validate.col_sd_gt.md#pointblank.Validate.col_sd_gt), [col_sd_lt](Validate.col_sd_lt.md#pointblank.Validate.col_sd_lt), [col_sd_ge](Validate.col_sd_ge.md#pointblank.Validate.col_sd_ge), [col_sd_le](Validate.col_sd_le.md#pointblank.Validate.col_sd_le), [col_sd_eq](Validate.col_sd_eq.md#pointblank.Validate.col_sd_eq)


#### See Also

- [yaml_interrogate()](yaml_interrogate.md): execute YAML-based validation workflows
