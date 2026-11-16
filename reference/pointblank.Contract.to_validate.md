# to_validate()

Compile the contract into a validation plan for a table, without running it.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L719-L836)

``` python
to_validate(data)
```

This method turns the contract into a [`Validate`](../reference/Validate.md#pointblank.Validate) object for the given table. The validation is set up with the contract's name, label, thresholds, and governance metadata (owner, consumers, and version). Then the schema check (if the contract has a schema) and each of the contract's steps are added in order. Nothing is checked yet: the plan runs when you call [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) on it.

Use this method instead of [`validate()`](../reference/pointblank.Contract.validate.md) when you want to do something with the plan before it runs. Most often that's adding checks that are local to one use of the data, or ones a contract can't express, such as [`col_vals_str_len()`](../reference/Validate.col_vals_str_len.md#pointblank.Validate.col_vals_str_len) or [`specially()`](../reference/Validate.specially.md#pointblank.Validate.specially). You can also attach [`Actions`](../reference/Actions.md#pointblank.Actions) and final actions this way, or combine the contract with a reusable [`Steps`](../reference/Steps.md#pointblank.Steps) library via [`add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps). The contract itself isn't changed, so these additions don't affect anyone else using it.

## Parameters

`data: IntoDataFrame`  
The table to validate against the contract. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

## Returns

`Validate`  
A [Validate](../reference/Validate.md#pointblank.Validate) object holding the contract's checks, not yet interrogated.

## Raises

`TypeError`  
If a step's arguments don't fit its validation method (for example, a misspelled parameter name). Step arguments are only checked when the contract is compiled.

## Examples

Let's define a small contract for a table of products: a two-column schema, plus rules that product IDs are present and prices are positive.

``` python
import pointblank as pb
import polars as pl

products_contract = pb.Contract(
    name="products",
    version="2.0.0",
    schema=pb.Schema(product_id="String", price="Float64"),
    steps=[
        pb.Step("col_vals_not_null", columns="product_id"),
        pb.Step("col_vals_gt", columns="price", value=0),
    ],
)

products = pl.DataFrame(
    {
        "product_id": ["P-100", "P-101", "P-102", "X-103"],
        "price": [19.99, 5.49, 120.00, 7.25],
    }
)
```

Our team also wants product IDs to start with `"P-"`. That rule isn't part of the shared contract, so we compile the contract with `to_validate()`, add a [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step, and then interrogate.

``` python
(
    products_contract.to_validate(products)
    .col_vals_regex(columns="product_id", pattern=r"^P-")
    .interrogate()
)
```

[TABLE]

The first three steps come from the contract (the schema check, then the two rules) and all pass. The fourth step is our addition, and it flags the ID `"X-103"`.
