# Validate.col_vals_in_table()

Validate that column values exist in a reference table (referential integrity).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15167-L15344)

``` python
Validate.col_vals_in_table(
    columns,
    ref_table,
    ref_column,
    na_pass=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The `col_vals_in_table()` validation method checks whether each value (or composite key) in the specified column(s) of this table exists in the corresponding column(s) of a reference table. This is a referential integrity check: it catches orphaned foreign keys, broken references, and values that fall outside a controlled vocabulary maintained in another table.

Each row is a test unit. Rows whose key values are found in the reference table pass. Rows with orphaned values fail. Failing rows are extractable via [get_data_extracts()](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) just like any other column-level validation.

Cross-backend matching is supported: the reference table can be from a different backend (e.g., a Polars DataFrame checked against a DuckDB table, or a MySQL table checked against a SQLite table). The lighter table is automatically materialized to match the heavier table's backend before comparison.

## Parameters

`columns: str | list[str]`  
The column (or list of columns for composite keys) in this table that should reference the other table. For composite keys, supply a list of column names.

`ref_table: Any`  
The reference table containing the valid values. Can be a DataFrame, a database table, a file path (e.g., to a CSV or Parquet file), a database connection string (e.g., `"duckdb:///path/to/db.ddb::table_name"`), or a callable that returns one of these. A path or connection string is loaded at interrogation time, as with [`tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match); if the reference table can't be loaded, the step is marked as having an evaluation error.

`ref_column: str | list[str]`  
The column (or list of columns for composite keys) in the reference table to check against. Must have the same length as `columns`.

`na_pass: bool = False`  
If `True`, rows where the key column(s) are null will pass. If `False` (default), null key values are treated as failing (not found in reference).

`pre: Callable | None = None`  
An optional preprocessing function to apply to the data table before validation.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Failure-condition thresholds for this step (overrides global thresholds).

`actions: Actions | None = None`  
Actions to take when thresholds are exceeded.

`brief: str | bool | None = None`  
A brief description for this validation step.

`active: bool | Callable = True`  
Whether this step is active (can be a bool or callable returning bool).

`dimension: str | None = None`  
The data quality dimension tag for this step.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with this step added (for method chaining).

## Examples

Check that every `customer_id` in orders exists in the customers table:

``` python
import pointblank as pb
import polars as pl

customers = pl.DataFrame({"id": [1, 2, 3, 4, 5]})
orders = pl.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 3, 99],
})

validation = (
    pb.Validate(data=orders)
    .col_vals_in_table(
        columns="customer_id",
        ref_table=customers,
        ref_column="id",
    )
    .interrogate()
)

validation
```

[TABLE]

The last row fails because `customer_id=99` does not exist in the customers table.

Composite foreign keys are also supported:

``` python
catalog = pl.DataFrame({
    "region": ["US", "US", "EU"],
    "sku": ["A1", "B2", "A1"],
})

orders = pl.DataFrame({
    "region": ["US", "EU", "US"],
    "sku": ["A1", "A1", "C3"],
})

validation = (
    pb.Validate(data=orders)
    .col_vals_in_table(
        columns=["region", "sku"],
        ref_table=catalog,
        ref_column=["region", "sku"],
    )
    .interrogate()
)

validation
```

[TABLE]
