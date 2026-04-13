## yaml_to_python()


Convert YAML validation configuration to equivalent Python code.


Usage

``` python
yaml_to_python(yaml)
```


This function takes a YAML validation configuration and generates the equivalent Python code that would produce the same validation workflow. This is useful for documentation, code generation, or learning how to translate YAML workflows into programmatic workflows.

The generated Python code includes all necessary imports, data loading, validation steps, and interrogation execution, formatted as executable Python code.


## Parameters


`yaml: Union[str, Path]`  
YAML configuration as string or file path. Can be: (1) a YAML string containing the validation configuration, or (2) a Path object or string path to a YAML file.


## Returns


`str`  
A formatted Python code string enclosed in markdown code blocks that replicates the YAML workflow. The code includes import statements, data loading, validation method calls, and interrogation execution.


## Raises


`YAMLValidationError`  
If the YAML is invalid, malformed, or contains unknown validation methods.


## Examples

Convert a basic YAML configuration to Python code:


``` python
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


    ```python
    import pointblank as pb

    (
        pb.Validate(
            data=pb.load_dataset("small_table", tbl_type="polars"),
            tbl_name="Data Quality Check",
        )
        .col_vals_not_null(columns=["a", "b"])
        .col_vals_gt(columns="c", value=0)
        .interrogate()
    )
    ```


The generated Python code shows exactly how to replicate the YAML workflow programmatically. This is particularly useful when transitioning from YAML-based workflows to code-based workflows, or when generating documentation that shows both YAML and Python approaches.

For more complex workflows with thresholds and metadata:


``` python
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


    ```python
    import pointblank as pb

    (
        pb.Validate(
            data=pb.load_dataset("small_table", tbl_type="polars"),
            tbl_name="Advanced Validation",
            label="Production data check",
            thresholds=pb.Thresholds(warning=0.1, error=0.2),
        )
        .col_vals_between(columns="c", left=1, right=10)
        .col_vals_regex(columns="b", pattern="[0-9]-[a-z]{3}-[0-9]{3}")
        .interrogate()
    )
    ```


The generated code includes all configuration parameters, thresholds, and maintains the exact same validation logic as the original YAML workflow.

Governance metadata (`owner`, `consumers`, `version`) and `reference` are also rendered in the generated Python code:


``` python
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


    ```python
    import pointblank as pb

    (
        pb.Validate(
            data=pb.load_dataset("small_table", tbl_type="polars"),
            tbl_name="Sales Pipeline",
            owner="Data Engineering",
            consumers=["Analytics", "Finance"],
            version="2.1.0",
        )
        .col_vals_not_null(columns="a")
        .col_sum_gt(columns="d", value=0)
        .interrogate()
    )
    ```


This function is also useful for educational purposes, helping users understand how YAML configurations map to the underlying Python API calls.
