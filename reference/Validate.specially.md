# Validate.specially()

Perform a specialized validation with customized logic.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15939-L16295)

``` python
Validate.specially(
    expr,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [specially()](../reference/Validate.specially.md#pointblank.Validate.specially) validation method allows for the creation of specialized validation expressions that can be used to validate specific conditions or logic in the data. This method provides maximum flexibility by accepting a custom callable that encapsulates your validation logic.

The callable function can have one of two signatures:

- a function accepting a single parameter (the data table): `def validate(data): ...`
- a function with no parameters: `def validate(): ...`

The second form is particularly useful for environment validations that don't need to inspect the data table.

The callable function must ultimately return one of:

1.  a single boolean value or boolean list
2.  a table where the final column contains boolean values (column name is unimportant)

The validation will operate over the number of test units that is equal to the number of rows in the data table (if returning a table with boolean values). If returning a scalar boolean value, the validation will operate over a single test unit. For a return of a list of boolean values, the length of the list constitutes the number of test units.

## Parameters

`expr: Callable`  
A callable function that defines the specialized validation logic. This function should:

1)  accept the target data table as its single argument (though it may ignore it), or
2)  take no parameters at all (for environment validations). The function must ultimately return boolean values representing validation results. Design your function to incorporate any custom parameters directly within the function itself using closure variables or default parameters.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like [`has_columns()`](../reference/has_columns.md#pointblank.has_columns) and [`has_rows()`](../reference/has_rows.md#pointblank.has_rows) can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](../reference/Validate.md#pointblank.Validate) object or used in subsequent validation steps.

## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).

## Examples

The [specially()](../reference/Validate.specially.md#pointblank.Validate.specially) method offers maximum flexibility for validation, allowing you to create custom validation logic that fits your specific needs. The following examples demonstrate different patterns and use cases for this powerful validation approach.

### Simple validation with direct table access

This example shows the most straightforward use case where we create a function that directly checks if the sum of two columns is positive.

``` python
import pointblank as pb
import polars as pl

simple_tbl = pl.DataFrame({
    "a": [5, 7, 1, 3, 9, 4],
    "b": [6, 3, 0, 5, 8, 2]
})

# Simple function that validates directly on the table
def validate_sum_positive(data):
    return data.select(pl.col("a") + pl.col("b") > 0)

(
    pb.Validate(data=simple_tbl)
    .specially(expr=validate_sum_positive)
    .interrogate()
)
```

[TABLE]

The function returns a Polars DataFrame with a single boolean column indicating whether the sum of columns `a` and `b` is positive for each row. Each row in the resulting DataFrame is a distinct test unit. This pattern works well for simple validations where you don't need configurable parameters.

### Advanced validation with closure variables for parameters

When you need to make your validation configurable, you can use the function factory pattern (also known as closures) to create parameterized validations:

``` python
# Create a parameterized validation function using closures
def make_column_ratio_validator(col1, col2, min_ratio):
    def validate_column_ratio(data):
        return data.select((pl.col(col1) / pl.col(col2)) > min_ratio)
    return validate_column_ratio

(
    pb.Validate(data=simple_tbl)
    .specially(
        expr=make_column_ratio_validator(col1="a", col2="b", min_ratio=0.5)
    )
    .interrogate()
)
```

[TABLE]

This approach allows you to create reusable validation functions that can be configured with different parameters without modifying the function itself.

### Validation function returning a list of booleans

This example demonstrates how to create a validation function that returns a list of boolean values, where each element represents a separate test unit:

``` python
import pointblank as pb
import polars as pl
import random

# Create sample data
transaction_tbl = pl.DataFrame({
    "transaction_id": [f"TX{i:04d}" for i in range(1, 11)],
    "amount": [120.50, 85.25, 50.00, 240.75, 35.20, 150.00, 85.25, 65.00, 210.75, 90.50],
    "category": ["food", "shopping", "entertainment", "travel", "utilities",
                "food", "shopping", "entertainment", "travel", "utilities"]
})

# Define a validation function that returns a list of booleans
def validate_transaction_rules(data):
    # Create a list to store individual test results
    test_results = []

    # Check each row individually against multiple business rules
    for row in data.iter_rows(named=True):
        # Rule: transaction IDs must start with "TX" and be 6 chars long
        valid_id = row["transaction_id"].startswith("TX") and len(row["transaction_id"]) == 6

        # Rule: Amounts must be appropriate for their category
        valid_amount = True
        if row["category"] == "food" and (row["amount"] < 10 or row["amount"] > 200):
            valid_amount = False
        elif row["category"] == "utilities" and (row["amount"] < 20 or row["amount"] > 300):
            valid_amount = False
        elif row["category"] == "entertainment" and row["amount"] > 100:
            valid_amount = False

        # A transaction passes if it satisfies both rules
        test_results.append(valid_id and valid_amount)

    return test_results

(
    pb.Validate(data=transaction_tbl)
    .specially(
        expr=validate_transaction_rules,
        brief="Validate transaction IDs and amounts by category."
    )
    .interrogate()
)
```

[TABLE]

This example shows how to create a validation function that applies multiple business rules to each row and returns a list of boolean results. Each boolean in the list represents a separate test unit, and a test unit passes only if all rules are satisfied for a given row.

The function iterates through each row in the data table, checking:

1.  if transaction IDs follow the required format
2.  if transaction amounts are appropriate for their respective categories

This approach is powerful when you need to apply complex, conditional logic that can't be easily expressed using the built-in validation functions.

### Table-level validation returning a single boolean

Sometimes you need to validate properties of the entire table rather than row-by-row. In these cases, your function can return a single boolean value:

``` python
def validate_table_properties(data):
    # Check if table has at least one row with column 'a' > 10
    has_large_values = data.filter(pl.col("a") > 10).height > 0

    # Check if mean of column 'b' is positive
    has_positive_mean = data.select(pl.mean("b")).item() > 0

    # Return a single boolean for the entire table
    return has_large_values and has_positive_mean

(
    pb.Validate(data=simple_tbl)
    .specially(expr=validate_table_properties)
    .interrogate()
)
```

[TABLE]

This example demonstrates how to perform multiple checks on the table as a whole and combine them into a single validation result.

### Environment validation that doesn't use the data table

The [specially()](../reference/Validate.specially.md#pointblank.Validate.specially) validation method can even be used to validate aspects of your environment that are completely independent of the data:

``` python
def validate_pointblank_version():
    try:
        import importlib.metadata
        version = importlib.metadata.version("pointblank")
        version_parts = version.split(".")

        # Get major and minor components regardless of how many parts there are
        major = int(version_parts[0])
        minor = int(version_parts[1])

        # Check both major and minor components for version `0.9+`
        return (major > 0) or (major == 0 and minor >= 9)

    except Exception as e:
        # More specific error handling could be added here
        print(f"Version check failed: {e}")
        return False

(
    pb.Validate(data=simple_tbl)
    .specially(
        expr=validate_pointblank_version,
        brief="Check Pointblank version `>=0.9.0`."
    )
    .interrogate()
)
```

[TABLE]

This pattern shows how to validate external dependencies or environment conditions as part of your validation workflow. Notice that the function doesn't take any parameters at all, which makes it cleaner when the validation doesn't need to access the data table.

By combining these patterns, you can create sophisticated validation workflows that address virtually any data quality requirement in your organization.
