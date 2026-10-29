# Validate.add_steps()

Add the steps from one or more [Steps](../reference/Steps.md#pointblank.Steps) collections to the validation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L6368-L6576)

``` python
Validate.add_steps(
    *steps,
    active=None,
    thresholds=None,
    exclude=None,
    columns_map=None,
)
```

This method appends previously recorded validation steps to the validation plan. The main source of steps is a [`Steps`](../reference/Steps.md#pointblank.Steps) collection: a reusable group of steps built with the same methods as [Validate](../reference/Validate.md#pointblank.Validate) but not attached to any table. Each recorded step is applied to this validation as if its method had been called directly, in the order the steps were recorded. Several collections can be added in one call, and the result can be chained with other validation methods, so shared steps and steps specific to this table can be combined in one plan. A [Validate](../reference/Validate.md#pointblank.Validate) object can also be given, in which case its steps are copied into this validation.

Steps written for one table don't always fit another one exactly. The optional arguments adapt the added steps without changing the collections themselves: `columns_map=` renames columns, `thresholds=` and `active=` replace those settings on every added step, and `exclude=` leaves out steps that don't apply.

## Parameters

`*steps: Steps | Validate`  
One or more [`Steps`](../reference/Steps.md#pointblank.Steps) or [Validate](../reference/Validate.md#pointblank.Validate) objects whose steps should be added, in the order given.

`active: bool | Callable | None = None`  
Replace the `active=` setting of every added step. If `None` (the default), each step keeps its own setting. Use `False` to add the steps in an inactive state, or a callable (such as [`has_columns()`](../reference/has_columns.md#pointblank.has_columns)) to decide at interrogation time.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Replace the `thresholds=` setting of every added step. If `None` (the default), each step keeps its own thresholds and otherwise uses the thresholds of this validation. Accepts the same values as the `thresholds=` argument of [Validate](../reference/Validate.md#pointblank.Validate).

`exclude: list[str | int] | None = None`  
Steps to leave out, as a list of method names (e.g., `["col_vals_regex"]` skips every regex step) or of 1-based positions (e.g., `[2]` skips the second step). Positions count within each collection, so `[2]` skips the second step of every collection given.

`columns_map: dict[str, str] | None = None`  
A dictionary that renames columns in the added steps, mapping the column names used in the collection to the names in this table (e.g., `{"customer_id": "cust_id"}`). It applies to column names given as strings or lists of strings in a step's `columns=`, `column=`, or `columns_subset=` argument. Column selectors such as [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) are left as they are.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the steps added, for further method chaining.

## Raises

`ValueError`  
If no [Steps](../reference/Steps.md#pointblank.Steps) or [Validate](../reference/Validate.md#pointblank.Validate) objects are given.

`TypeError`  
If an argument in `*steps` isn't a [Steps](../reference/Steps.md#pointblank.Steps) or [Validate](../reference/Validate.md#pointblank.Validate) object, if an item in `exclude=` isn't a string or an integer, or if a step's arguments don't fit its validation method.

## Examples

Let's define two collections of steps for order data: one for completeness and one for valid amounts.

``` python
import pointblank as pb
import polars as pl

completeness = (
    pb.Steps()
    .col_vals_not_null(columns="order_id")
    .col_vals_not_null(columns="email")
)

valid_amounts = (
    pb.Steps()
    .col_vals_gt(columns="amount", value=0)
    .col_vals_le(columns="amount", value=10_000)
)
```

Both collections can be added to a validation of an orders table, along with a check that only applies to this table (the `region` values):

``` python
orders = pl.DataFrame(
    {
        "order_id": ["A1", "A2", None, "A4"],
        "email": ["a@x.com", None, "c@z.com", "d@w.net"],
        "amount": [25.0, 120.0, 0.0, 18_500.0],
        "region": ["east", "west", "east", "north"],
    }
)

(
    pb.Validate(data=orders, tbl_name="orders")
    .add_steps(completeness, valid_amounts)
    .col_vals_in_set(columns="region", set=["east", "west"])
    .interrogate()
)
```

[TABLE]

The first four steps come from the two collections, in order, and the fifth is the table-specific check. Each step finds one failing row.

A table of refunds uses the same rules, but with different column names, no email column, and a stricter tolerance for failures. Here, `columns_map=` renames the columns, `exclude=` skips the second completeness step (the email check), and `thresholds=` makes any failure an error.

``` python
refunds = pl.DataFrame(
    {"refund_ref": ["R1", "R2", "R3"], "refund_amt": [15.0, -4.0, 60.0]}
)

(
    pb.Validate(data=refunds, tbl_name="refunds")
    .add_steps(
        completeness,
        valid_amounts,
        columns_map={"order_id": "refund_ref", "amount": "refund_amt"},
        exclude=[2],
        thresholds=pb.Thresholds(error=1),
    )
    .interrogate()
)
```

[TABLE]

Because positions count within each collection, `exclude=[2]` also skips the second step of `valid_amounts` (the upper limit on the amount). The negative refund fails the remaining amount check and reaches the error threshold.
