# Validation Serialization Test Infrastructure

This directory contains test files and utilities for ensuring serialization compatibility of pointblank validation objects across versions.

## Overview

The serialization functionality in pointblank allows validation objects to be saved to disk and reloaded later. To ensure this works correctly across different versions and with various types of preprocessing functions, we maintain a collection of reference validation files for regression testing.

## Files

### Preprocessing Functions (`preprocessing_functions.py`)

Contains preprocessing functions used in validation examples:

- `double_column_a()` - Simple column transformation
- `add_computed_column()` - Creates computed columns
- `filter_by_d_gt_100()` - Filtering operations
- `narwhals_median_transform()` - Cross-backend compatible functions using narwhals
- `complex_preprocessing()` - Complex multi-step transformations
- `pandas_compatible_transform()` - Functions that work with both pandas and polars

### Test File Generator (`generate_test_files.py`)

Script that creates reference validation objects with various preprocessing functions:

- Creates test datasets
- Defines validation objects with different preprocessing scenarios
- Saves both pickle (`.pkl`) and JSON (`.json`) files
- Each validation object is interrogated to populate results

### Test Cases (`tests/test_serialization_compat.py`)

Comprehensive tests for serialization functionality located in the main tests directory:

- **Roundtrip testing**: Pickle and unpickle validation objects
- **Preprocessing preservation**: Verify functions are correctly serialized
- **Cross-backend compatibility**: Test narwhals functions work after deserialization
- **Complex workflows**: Multi-step validation with different preprocessing functions

### Generated Files

The following validation files are generated for regression testing:

#### Basic Validation Examples

- `no_preprocessing.pkl/.json` - Control case without preprocessing
- `simple_preprocessing.pkl/.json` - Basic single-function preprocessing

#### Advanced Validation Examples

- `complex_preprocessing.pkl/.json` - Multi-step transformations
- `multiple_steps.pkl/.json` - Different preprocessing per validation step
- `narwhals_function.pkl/.json` - Cross-backend compatible functions
- `pandas_compatible.pkl/.json` - Functions that work with multiple backends

## Usage

### Running Tests

```bash
# Run all serialization compatibility tests
python -m pytest tests/test_serialization_compat.py -v

# Generate new test files (if functions change)
cd pointblank/data/validations
python generate_test_files.py
```

### Adding New Test Cases

1. Add new preprocessing functions to `preprocessing_functions.py`
2. Update `generate_test_files.py` to create validations using the new functions
3. Add corresponding test cases in `tests/test_serialization_compat.py`
4. Regenerate test files: `python generate_test_files.py`

## Version Compatibility

These reference files serve as regression tests to ensure:

- New versions can load validation files created with previous versions
- Preprocessing functions are correctly preserved across serialization
- Cross-backend compatibility is maintained
- Complex workflows continue to work after deserialization

The pickle files are the authoritative test cases, while JSON files provide human-readable versions for debugging.

## Best Practices

### For Preprocessing Functions

- Always use proper function definitions (not lambdas) for serializable functions
- Import required libraries inside functions for self-contained serialization
- Use narwhals for cross-backend compatibility when possible
- Test functions work with both polars and pandas DataFrames

### For Test Coverage

- Include examples of each type of preprocessing function
- Test both simple and complex multi-step workflows
- Verify roundtrip serialization (pickle → unpickle → pickle again)
- Check that deserialized functions produce expected results

### For Maintenance

- Regenerate test files when adding new preprocessing function types
- Keep test functions focused and well-documented
- Update tests when validation object structure changes
- Document any breaking changes that affect serialization compatibility
