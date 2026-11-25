# to_validate()

Build a validation plan for a table from the metadata.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L742-L836)

``` python
to_validate(
    data,
    **kwargs,
)
```

This turns the declared types and constraints into validation steps, so that a table can be checked against its own documentation. The first step is a [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) with the schema from [`to_schema()`](../reference/pointblank.MetadataImport.to_schema.md). Then, for each variable in turn:

- [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) if it's `required`
- [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) on the column if it's `unique`
- [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between) if it has both `min_val` and `max_val`, or [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge) or [`col_vals_le()`](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le) if it has only one of them
- [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set) if it has `allowed_values`
- [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) if it has a `pattern`
- [`col_vals_not_in_set()`](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set) with its `missing_values`, if it's a string variable that declares some

The last step checks that the declared missing-value markers don't appear in the data, as when a file was read without converting markers like `"NA"` to nulls. The `min_length` and `max_length` of a variable aren't checked. None of the steps pass null values, so a null in an optional variable fails its range, set and pattern checks. To change any of the steps, get the plan as code with [`to_code()`](../reference/Validate.to_code.md#pointblank.Validate.to_code) and edit it.

## Parameters

`data: Any`  
The table to validate.

`**kwargs: Any`  
Arguments for [`Validate`](../reference/Validate.md#pointblank.Validate), such as `tbl_name=`, `thresholds=` or `actions=`. A `label=` replaces the default label, which is `"Validation: <dataset name> (<format>)"`, or `"Validation from <format> metadata"` if the metadata has no dataset name.

## Returns

`Validate`  
A [Validate](../reference/Validate.md#pointblank.Validate) object with the steps added. It hasn't been interrogated yet.

## Examples

The bundled Frictionless Data Package describes a table of sales transactions. Let's check a few transactions against it:

``` python
import datetime
import polars as pl
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

transactions = pl.DataFrame(
    {
        "transaction_id": ["TXN-00001", "TXN-00002", "TXN-00003", "TXN-00004"],
        "customer_id": ["CUST-101", "CUST-102", "CUST-101", "CUST-103"],
        "amount": [25.50, 1200.00, 0.00, 89.99],
        "quantity": [1, 3, 2, 1],
        "category": ["food", "electronics", "home", "toys"],
        "sale_date": [datetime.date(2024, 5, d) for d in (1, 1, 2, 3)],
        "discount_pct": [0.0, 10.0, 0.0, 5.0],
        "email": ["a@example.com", "b@example.com", "c@example.com", "not-an-email"],
    }
)

validation = meta.to_validate(data=transactions, tbl_name="transactions").interrogate()

validation
```

[TABLE]

The metadata gave 17 steps. The table matches the declared schema, and three of the constraint checks find a failing row: the `amount` of `0.00` is below the minimum of `0.01`, `"toys"` isn't an allowed `category`, and `"not-an-email"` doesn't match the pattern for `email`. To see how the plan was built, or to start from it and change some steps, print it as code:

``` python
print(meta.to_validate(data=transactions).to_code())
```

    import pointblank as pb

    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
            label="Validation: transactions (frictionless)",
        )
        .col_schema_match(schema=pb.Schema(columns=[("transaction_id", "String"), ("customer_id", "String"), ("amount", "Float64"), ("quantity", "Int64"), ("category", "String"), ("sale_date", "Date"), ("discount_pct", "Float64"), ("email", "String")]))
        .col_vals_not_null(columns="transaction_id")
        .rows_distinct(columns_subset="transaction_id")
        .col_vals_not_in_set(columns="transaction_id", set=["NA", "N/A"])
        .col_vals_not_null(columns="customer_id")
        .col_vals_not_in_set(columns="customer_id", set=["NA", "N/A"])
        .col_vals_not_null(columns="amount")
        .col_vals_between(columns="amount", left=0.01, right=99999.99)
        .col_vals_not_null(columns="quantity")
        .col_vals_between(columns="quantity", left=1.0, right=1000.0)
        .col_vals_not_null(columns="category")
        .col_vals_in_set(columns="category", set=["electronics", "clothing", "food", "home", "sports"])
        .col_vals_not_in_set(columns="category", set=["NA", "N/A"])
        .col_vals_not_null(columns="sale_date")
        .col_vals_between(columns="discount_pct", left=0.0, right=50.0)
        .col_vals_regex(columns="email", pattern="^[^@]+@[^@]+\\.[^@]+$")
        .col_vals_not_in_set(columns="email", set=["NA", "N/A"])
    )

    validation

Each step can be traced back to the metadata. For instance, the [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step for `transaction_id` comes from its `unique` constraint, and the [col_vals_not_in_set()](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set) steps check the `"NA"` and `"N/A"` markers declared for the string columns.
