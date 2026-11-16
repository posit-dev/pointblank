# Steps

A reusable collection of validation step definitions.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L22-L2189)

``` python
Steps(steps=None)
```

A [Steps](../reference/Steps.md#pointblank.Steps) object is a list of validation steps that isn't attached to any table. You build it with the same methods you'd call on a [`Validate`](../reference/Validate.md#pointblank.Validate) object ([col_vals_not_null()](../reference/pointblank.Steps.col_vals_not_null.md), [rows_distinct()](../reference/pointblank.Steps.rows_distinct.md), [col_vals_regex()](../reference/pointblank.Steps.col_vals_regex.md), and so on), but each call only records the step. Nothing is checked until the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and that validation is interrogated.

This makes it possible to write a group of checks once and use it in many places. Many tables share the same rules: every table with a customer ID needs that ID to be present and unique, and every table with an email column needs valid addresses. Without [Steps](../reference/Steps.md#pointblank.Steps), those rules get copied into each validation and drift apart over time. With [Steps](../reference/Steps.md#pointblank.Steps), you define them in one place (a shared module, for example), name them for what they check (`id_checks`, `contact_checks`), and add them to whichever validations need them. Each validation keeps its own table, thresholds, actions, and metadata, and can mix shared steps with its own.

Since the methods have the same signatures as the [Validate](../reference/Validate.md#pointblank.Validate) methods they mirror, you get the same argument names, defaults, and editor completion, and a misspelled argument raises a `TypeError` as soon as the step is recorded.

## Parameters

`steps: list[Step] | None = None`  
An optional list of [`Step`](../reference/Step.md#pointblank.Step) objects to start the collection with. This is useful when step definitions come from data (a configuration file, for example) rather than from method calls.

## Returns

`Steps`  
A [Steps](../reference/Steps.md#pointblank.Steps) object. Add steps to it with its validation methods, and apply it with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Recording Steps

Each validation method records one step and returns the same [Steps](../reference/Steps.md#pointblank.Steps) object, so calls can be chained. All of the row-based and table-level validation methods are available, from [col_vals_gt()](../reference/pointblank.Steps.col_vals_gt.md) to [tbl_reconcile()](../reference/pointblank.Steps.tbl_reconcile.md), including [col_vals_scored()](../reference/pointblank.Steps.col_vals_scored.md). The aggregation methods (`col_sum_*()`, `col_avg_*()`, and `col_sd_*()`), [specially()](../reference/Validate.specially.md#pointblank.Validate.specially), and [prompt()](../reference/Validate.prompt.md#pointblank.Validate.prompt) aren't, so add those directly to the validation.

Because each method adds to the object it's called on, a collection isn't copied when you add to it. Calling a method on a shared collection (e.g., `extended = id_checks.col_exists(...)`) adds the step to `id_checks` itself, and `extended` is the same object. To build variations, keep the shared parts in separate collections and add several of them to a validation.

Printing a [Steps](../reference/Steps.md#pointblank.Steps) object lists its steps with the arguments that differ from the defaults, and `len()` gives the number of steps.

## Applying Steps To A Validation

[`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) appends the recorded steps to a validation, in order, as if each method had been called on it directly. It takes any number of [Steps](../reference/Steps.md#pointblank.Steps) objects (and also [Validate](../reference/Validate.md#pointblank.Validate) objects, whose steps it copies), and has options for adapting the steps to the validation at hand:

- `columns_map=` renames columns in the steps, so that the same collection works on tables that name a column differently (e.g., `{"customer_id": "cust_id"}`)
- `thresholds=` and `active=` replace the thresholds and active state of every added step
- `exclude=` skips steps by method name or by their 1-based position in a collection

## Steps, Step, And Contract

A [`Step`](../reference/Step.md#pointblank.Step) is a single step stored as data: a method name and a dictionary of arguments. A [Steps](../reference/Steps.md#pointblank.Steps) object holds a list of them, built with method calls instead. A [`Contract`](../reference/Contract.md#pointblank.Contract) also holds a list of [Step](../reference/Step.md#pointblank.Step) objects, together with a schema and metadata (an owner, a version, consumers), and can be saved as YAML. Use [Steps](../reference/Steps.md#pointblank.Steps) for reusable building blocks inside Python code, and a [Contract](../reference/Contract.md#pointblank.Contract) for an agreement about a table that's shared, versioned, and enforced at a pipeline boundary. The two work together: a contract compiled with [`to_validate()`](../reference/pointblank.Contract.to_validate.md) is an ordinary validation, so collections can be added to it with [add_steps()](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Examples

Let's build two collections of checks for customer data. The first covers identifiers (each customer ID must be present and appear only once), and the second covers contact details (email addresses must look valid, and the country must be one of three codes).

``` python
import pointblank as pb
import polars as pl

id_checks = (
    pb.Steps()
    .col_vals_not_null(columns="customer_id")
    .rows_distinct(columns_subset="customer_id")
)

contact_checks = (
    pb.Steps()
    .col_vals_regex(columns="email", pattern=r"^[^@]+@[^@]+\.[a-z]+$")
    .col_vals_in_set(columns="country", set=["CA", "MX", "US"])
)

id_checks
```

Steps -- 2 steps

1.col_vals_not_null(columns='customer_id')

2.rows_distinct(columns_subset='customer_id')

Displaying a collection lists its steps and their arguments. No table is involved yet.

Now let's apply both collections to a table of customers from a CRM system. The validation is created as usual, and [add_steps()](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) appends the four recorded steps before interrogation.

``` python
crm_customers = pl.DataFrame(
    {
        "customer_id": [1, 2, 3, 3, 5],
        "email": ["a@x.com", "b@y.org", "c-at-z.com", "d@w.net", "e@v.io"],
        "country": ["US", "CA", "US", "MX", "BR"],
    }
)

(
    pb.Validate(data=crm_customers, tbl_name="crm_customers")
    .add_steps(id_checks, contact_checks)
    .interrogate()
)
```

[TABLE]

The report has one step per recorded step, in the order they were added. Customer ID `3` appears twice, so both of its rows fail the [rows_distinct](../reference/pointblank.Steps.rows_distinct.md) step. The address `"c-at-z.com"` fails the email pattern, and the country `"BR"` isn't in the allowed set.

The same collections can check a table from a billing system, where the ID column is called `cust_id`. The `columns_map=` argument renames the column in the added steps, and `thresholds=` gives every added step a warning threshold of one failing test unit.

``` python
billing_accounts = pl.DataFrame(
    {
        "cust_id": [10, 11, None, 13],
        "email": ["f@x.com", "g@y.com", "h@z.com", "i@w"],
        "country": ["US", "US", "CA", "US"],
    }
)

(
    pb.Validate(data=billing_accounts, tbl_name="billing_accounts")
    .add_steps(
        id_checks,
        contact_checks,
        columns_map={"customer_id": "cust_id"},
        thresholds=pb.Thresholds(warning=1),
    )
    .interrogate()
)
```

[TABLE]

The ID steps now check `cust_id`, and one account is missing its ID. The address `"i@w"` has no domain suffix, so it fails the email pattern. Both failing steps reach the warning threshold.

## See Also

[add_steps()](../reference/Validate.add_steps.md#pointblank.Validate.add_steps)  
Add the steps in one or more collections to a validation.

[Step](../reference/Step.md#pointblank.Step)  
A single validation step stored as data.

[Contract](../reference/Contract.md#pointblank.Contract)  
A schema, steps, and metadata that make up an agreement about a table.

## Functions

| Name | Description |
|----|----|
| [col_vals_gt()](pointblank.Steps.col_vals_gt.md) | Add a `col_vals_gt()` step to the collection. |
| [col_vals_lt()](pointblank.Steps.col_vals_lt.md) | Add a `col_vals_lt()` step to the collection. |
| [col_vals_ge()](pointblank.Steps.col_vals_ge.md) | Add a `col_vals_ge()` step to the collection. |
| [col_vals_le()](pointblank.Steps.col_vals_le.md) | Add a `col_vals_le()` step to the collection. |
| [col_vals_eq()](pointblank.Steps.col_vals_eq.md) | Add a `col_vals_eq()` step to the collection. |
| [col_vals_ne()](pointblank.Steps.col_vals_ne.md) | Add a `col_vals_ne()` step to the collection. |
| [col_vals_between()](pointblank.Steps.col_vals_between.md) | Add a `col_vals_between()` step to the collection. |
| [col_vals_outside()](pointblank.Steps.col_vals_outside.md) | Add a `col_vals_outside()` step to the collection. |
| [col_vals_in_set()](pointblank.Steps.col_vals_in_set.md) | Add a `col_vals_in_set()` step to the collection. |
| [col_vals_not_in_set()](pointblank.Steps.col_vals_not_in_set.md) | Add a `col_vals_not_in_set()` step to the collection. |
| [col_vals_increasing()](pointblank.Steps.col_vals_increasing.md) | Add a `col_vals_increasing()` step to the collection. |
| [col_vals_decreasing()](pointblank.Steps.col_vals_decreasing.md) | Add a `col_vals_decreasing()` step to the collection. |
| [col_vals_null()](pointblank.Steps.col_vals_null.md) | Add a `col_vals_null()` step to the collection. |
| [col_vals_not_null()](pointblank.Steps.col_vals_not_null.md) | Add a `col_vals_not_null()` step to the collection. |
| [col_vals_regex()](pointblank.Steps.col_vals_regex.md) | Add a `col_vals_regex()` step to the collection. |
| [col_vals_within_spec()](pointblank.Steps.col_vals_within_spec.md) | Add a `col_vals_within_spec()` step to the collection. |
| [col_vals_str_len()](pointblank.Steps.col_vals_str_len.md) | Add a `col_vals_str_len()` step to the collection. |
| [col_vals_scored()](pointblank.Steps.col_vals_scored.md) | Add a `col_vals_scored()` step to the collection. |
| [col_vals_chat_format()](pointblank.Steps.col_vals_chat_format.md) | Add a `col_vals_chat_format()` step to the collection. |
| [col_vals_text_quality()](pointblank.Steps.col_vals_text_quality.md) | Add a `col_vals_text_quality()` step to the collection. |
| [col_vals_in_language()](pointblank.Steps.col_vals_in_language.md) | Add a `col_vals_in_language()` step to the collection. |
| [col_vals_token_len()](pointblank.Steps.col_vals_token_len.md) | Add a `col_vals_token_len()` step to the collection. |
| [col_vals_no_pii()](pointblank.Steps.col_vals_no_pii.md) | Add a `col_vals_no_pii()` step to the collection. |
| [col_vals_in_table()](pointblank.Steps.col_vals_in_table.md) | Add a `col_vals_in_table()` step to the collection. |
| [col_vals_expr()](pointblank.Steps.col_vals_expr.md) | Add a `col_vals_expr()` step to the collection. |
| [col_exists()](pointblank.Steps.col_exists.md) | Add a `col_exists()` step to the collection. |
| [col_pct_null()](pointblank.Steps.col_pct_null.md) | Add a `col_pct_null()` step to the collection. |
| [col_pct_missing()](pointblank.Steps.col_pct_missing.md) | Add a `col_pct_missing()` step to the collection. |
| [col_missing_coded()](pointblank.Steps.col_missing_coded.md) | Add a `col_missing_coded()` step to the collection. |
| [col_missing_only_coded()](pointblank.Steps.col_missing_only_coded.md) | Add a `col_missing_only_coded()` step to the collection. |
| [col_missing_consistent()](pointblank.Steps.col_missing_consistent.md) | Add a `col_missing_consistent()` step to the collection. |
| [rows_distinct()](pointblank.Steps.rows_distinct.md) | Add a `rows_distinct()` step to the collection. |
| [rows_near_distinct()](pointblank.Steps.rows_near_distinct.md) | Add a `rows_near_distinct()` step to the collection. |
| [rows_not_contaminated()](pointblank.Steps.rows_not_contaminated.md) | Add a `rows_not_contaminated()` step to the collection. |
| [rows_complete()](pointblank.Steps.rows_complete.md) | Add a `rows_complete()` step to the collection. |
| [col_schema_match()](pointblank.Steps.col_schema_match.md) | Add a `col_schema_match()` step to the collection. |
| [row_count_match()](pointblank.Steps.row_count_match.md) | Add a `row_count_match()` step to the collection. |
| [col_count_match()](pointblank.Steps.col_count_match.md) | Add a `col_count_match()` step to the collection. |
| [data_freshness()](pointblank.Steps.data_freshness.md) | Add a `data_freshness()` step to the collection. |
| [tbl_match()](pointblank.Steps.tbl_match.md) | Add a `tbl_match()` step to the collection. |
| [tbl_reconcile()](pointblank.Steps.tbl_reconcile.md) | Add a `tbl_reconcile()` step to the collection. |
| [conjointly()](pointblank.Steps.conjointly.md) | Add a `conjointly()` step to the collection. |
