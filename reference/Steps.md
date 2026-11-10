# Steps

A reusable collection of validation step definitions.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L22-L2189)

``` python
Steps(steps=None)
```

A [Steps](../reference/Steps.md#pointblank.Steps) object is a list of validation steps that isn't attached to any table. You build it with the same methods you'd call on a [`Validate`](../reference/Validate.md#pointblank.Validate) object ([col_vals_not_null()](../reference/Steps.md#pointblank.Steps.col_vals_not_null), [rows_distinct()](../reference/Steps.md#pointblank.Steps.rows_distinct), [col_vals_regex()](../reference/Steps.md#pointblank.Steps.col_vals_regex), and so on), but each call only records the step. Nothing is checked until the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and that validation is interrogated.

This makes it possible to write a group of checks once and use it in many places. Many tables share the same rules: every table with a customer ID needs that ID to be present and unique, and every table with an email column needs valid addresses. Without [Steps](../reference/Steps.md#pointblank.Steps), those rules get copied into each validation and drift apart over time. With [Steps](../reference/Steps.md#pointblank.Steps), you define them in one place (a shared module, for example), name them for what they check (`id_checks`, `contact_checks`), and add them to whichever validations need them. Each validation keeps its own table, thresholds, actions, and metadata, and can mix shared steps with its own.

Since the methods have the same signatures as the [Validate](../reference/Validate.md#pointblank.Validate) methods they mirror, you get the same argument names, defaults, and editor completion, and a misspelled argument raises a `TypeError` as soon as the step is recorded.

## Parameters

`steps: list[Step] | None = None`  
An optional list of [`Step`](../reference/Step.md#pointblank.Step) objects to start the collection with. This is useful when step definitions come from data (a configuration file, for example) rather than from method calls.

## Returns

`Steps`  
A [Steps](../reference/Steps.md#pointblank.Steps) object. Add steps to it with its validation methods, and apply it with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Recording Steps

Each validation method records one step and returns the same [Steps](../reference/Steps.md#pointblank.Steps) object, so calls can be chained. All of the row-based and table-level validation methods are available, from [col_vals_gt()](../reference/Steps.md#pointblank.Steps.col_vals_gt) to [tbl_reconcile()](../reference/Steps.md#pointblank.Steps.tbl_reconcile), including [col_vals_scored()](../reference/Steps.md#pointblank.Steps.col_vals_scored). The aggregation methods (`col_sum_*()`, `col_avg_*()`, and `col_sd_*()`), [specially()](../reference/Validate.specially.md#pointblank.Validate.specially), and [prompt()](../reference/Validate.prompt.md#pointblank.Validate.prompt) aren't, so add those directly to the validation.

Because each method adds to the object it's called on, a collection isn't copied when you add to it. Calling a method on a shared collection (e.g., `extended = id_checks.col_exists(...)`) adds the step to `id_checks` itself, and `extended` is the same object. To build variations, keep the shared parts in separate collections and add several of them to a validation.

Printing a [Steps](../reference/Steps.md#pointblank.Steps) object lists its steps with the arguments that differ from the defaults, and `len()` gives the number of steps.

## Applying Steps To A Validation

[`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) appends the recorded steps to a validation, in order, as if each method had been called on it directly. It takes any number of [Steps](../reference/Steps.md#pointblank.Steps) objects (and also [Validate](../reference/Validate.md#pointblank.Validate) objects, whose steps it copies), and has options for adapting the steps to the validation at hand:

- `columns_map=` renames columns in the steps, so that the same collection works on tables that name a column differently (e.g., `{"customer_id": "cust_id"}`)
- `thresholds=` and `active=` replace the thresholds and active state of every added step
- `exclude=` skips steps by method name or by their 1-based position in a collection

## Steps, Step, And Contract

A [`Step`](../reference/Step.md#pointblank.Step) is a single step stored as data: a method name and a dictionary of arguments. A [Steps](../reference/Steps.md#pointblank.Steps) object holds a list of them, built with method calls instead. A [`Contract`](../reference/Contract.md#pointblank.Contract) also holds a list of [Step](../reference/Step.md#pointblank.Step) objects, together with a schema and metadata (an owner, a version, consumers), and can be saved as YAML. Use [Steps](../reference/Steps.md#pointblank.Steps) for reusable building blocks inside Python code, and a [Contract](../reference/Contract.md#pointblank.Contract) for an agreement about a table that's shared, versioned, and enforced at a pipeline boundary. The two work together: a contract compiled with [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) is an ordinary validation, so collections can be added to it with [add_steps()](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

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

The report has one step per recorded step, in the order they were added. Customer ID `3` appears twice, so both of its rows fail the [rows_distinct](../reference/Steps.md#pointblank.Steps.rows_distinct) step. The address `"c-at-z.com"` fails the email pattern, and the country `"BR"` isn't in the allowed set.

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

## Methods

| Name | Description |
|----|----|
| [col_vals_gt()](#col_vals_gt) | Add a `col_vals_gt()` step to the collection. |
| [col_vals_lt()](#col_vals_lt) | Add a `col_vals_lt()` step to the collection. |
| [col_vals_ge()](#col_vals_ge) | Add a `col_vals_ge()` step to the collection. |
| [col_vals_le()](#col_vals_le) | Add a `col_vals_le()` step to the collection. |
| [col_vals_eq()](#col_vals_eq) | Add a `col_vals_eq()` step to the collection. |
| [col_vals_ne()](#col_vals_ne) | Add a `col_vals_ne()` step to the collection. |
| [col_vals_between()](#col_vals_between) | Add a `col_vals_between()` step to the collection. |
| [col_vals_outside()](#col_vals_outside) | Add a `col_vals_outside()` step to the collection. |
| [col_vals_in_set()](#col_vals_in_set) | Add a `col_vals_in_set()` step to the collection. |
| [col_vals_not_in_set()](#col_vals_not_in_set) | Add a `col_vals_not_in_set()` step to the collection. |
| [col_vals_increasing()](#col_vals_increasing) | Add a `col_vals_increasing()` step to the collection. |
| [col_vals_decreasing()](#col_vals_decreasing) | Add a `col_vals_decreasing()` step to the collection. |
| [col_vals_null()](#col_vals_null) | Add a `col_vals_null()` step to the collection. |
| [col_vals_not_null()](#col_vals_not_null) | Add a `col_vals_not_null()` step to the collection. |
| [col_vals_regex()](#col_vals_regex) | Add a `col_vals_regex()` step to the collection. |
| [col_vals_within_spec()](#col_vals_within_spec) | Add a `col_vals_within_spec()` step to the collection. |
| [col_vals_str_len()](#col_vals_str_len) | Add a `col_vals_str_len()` step to the collection. |
| [col_vals_scored()](#col_vals_scored) | Add a `col_vals_scored()` step to the collection. |
| [col_vals_chat_format()](#col_vals_chat_format) | Add a `col_vals_chat_format()` step to the collection. |
| [col_vals_text_quality()](#col_vals_text_quality) | Add a `col_vals_text_quality()` step to the collection. |
| [col_vals_in_language()](#col_vals_in_language) | Add a `col_vals_in_language()` step to the collection. |
| [col_vals_token_len()](#col_vals_token_len) | Add a `col_vals_token_len()` step to the collection. |
| [col_vals_no_pii()](#col_vals_no_pii) | Add a `col_vals_no_pii()` step to the collection. |
| [col_vals_in_table()](#col_vals_in_table) | Add a `col_vals_in_table()` step to the collection. |
| [col_vals_expr()](#col_vals_expr) | Add a `col_vals_expr()` step to the collection. |
| [col_exists()](#col_exists) | Add a `col_exists()` step to the collection. |
| [col_pct_null()](#col_pct_null) | Add a `col_pct_null()` step to the collection. |
| [col_pct_missing()](#col_pct_missing) | Add a `col_pct_missing()` step to the collection. |
| [col_missing_coded()](#col_missing_coded) | Add a `col_missing_coded()` step to the collection. |
| [col_missing_only_coded()](#col_missing_only_coded) | Add a `col_missing_only_coded()` step to the collection. |
| [col_missing_consistent()](#col_missing_consistent) | Add a `col_missing_consistent()` step to the collection. |
| [rows_distinct()](#rows_distinct) | Add a `rows_distinct()` step to the collection. |
| [rows_near_distinct()](#rows_near_distinct) | Add a `rows_near_distinct()` step to the collection. |
| [rows_not_contaminated()](#rows_not_contaminated) | Add a `rows_not_contaminated()` step to the collection. |
| [rows_complete()](#rows_complete) | Add a `rows_complete()` step to the collection. |
| [col_schema_match()](#col_schema_match) | Add a `col_schema_match()` step to the collection. |
| [row_count_match()](#row_count_match) | Add a `row_count_match()` step to the collection. |
| [col_count_match()](#col_count_match) | Add a `col_count_match()` step to the collection. |
| [data_freshness()](#data_freshness) | Add a `data_freshness()` step to the collection. |
| [tbl_match()](#tbl_match) | Add a `tbl_match()` step to the collection. |
| [tbl_reconcile()](#tbl_reconcile) | Add a `tbl_reconcile()` step to the collection. |
| [conjointly()](#conjointly) | Add a `conjointly()` step to the collection. |

### col_vals_gt()

Add a `col_vals_gt()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L286-L328)

``` python
col_vals_gt(
    columns,
    value,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), which checks whether column values are greater than a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_lt()

Add a `col_vals_lt()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L330-L372)

``` python
col_vals_lt(
    columns,
    value,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt), which checks whether column values are less than a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_ge()

Add a `col_vals_ge()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L462-L505)

``` python
col_vals_ge(
    columns,
    value,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge), which checks whether column values are greater than or equal to a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_le()

Add a `col_vals_le()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L507-L550)

``` python
col_vals_le(
    columns,
    value,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_le()`](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le), which checks whether column values are less than or equal to a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_eq()

Add a `col_vals_eq()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L374-L416)

``` python
col_vals_eq(
    columns,
    value,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_eq()`](../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq), which checks whether column values are equal to a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_ne()

Add a `col_vals_ne()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L418-L460)

``` python
col_vals_ne(
    columns,
    value,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_ne()`](../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne), which checks whether column values are not equal to a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_between()

Add a `col_vals_between()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L552-L599)

``` python
col_vals_between(
    columns,
    left,
    right,
    inclusive=(True, True),
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between), which checks whether column values lie between two bounds, given as fixed values or other columns. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_outside()

Add a `col_vals_outside()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L601-L648)

``` python
col_vals_outside(
    columns,
    left,
    right,
    inclusive=(True, True),
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_outside()`](../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside), which checks whether column values lie outside two bounds, given as fixed values or other columns. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_in_set()

Add a `col_vals_in_set()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L650-L691)

``` python
col_vals_in_set(
    columns,
    set,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set), which checks whether column values belong to a set of allowed values. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_not_in_set()

Add a `col_vals_not_in_set()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L693-L736)

``` python
col_vals_not_in_set(
    columns,
    set,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_not_in_set()`](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set), which checks whether column values avoid a set of disallowed values. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_increasing()

Add a `col_vals_increasing()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L738-L785)

``` python
col_vals_increasing(
    columns,
    allow_stationary=False,
    decreasing_tol=None,
    na_pass=False,
    order_by=None,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_increasing()`](../reference/Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing), which checks whether column values increase from row to row. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_decreasing()

Add a `col_vals_decreasing()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L787-L834)

``` python
col_vals_decreasing(
    columns,
    allow_stationary=False,
    increasing_tol=None,
    na_pass=False,
    order_by=None,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_decreasing()`](../reference/Validate.col_vals_decreasing.md#pointblank.Validate.col_vals_decreasing), which checks whether column values decrease from row to row. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_null()

Add a `col_vals_null()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L836-L873)

``` python
col_vals_null(
    columns,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_null()`](../reference/Validate.col_vals_null.md#pointblank.Validate.col_vals_null), which checks whether column values are all Null. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_not_null()

Add a `col_vals_not_null()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L875-L912)

``` python
col_vals_not_null(
    columns,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null), which checks whether column values are free of Nulls. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_regex()

Add a `col_vals_regex()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L914-L959)

``` python
col_vals_regex(
    columns,
    pattern,
    na_pass=False,
    inverse=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), which checks whether column values match a regular expression. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_within_spec()

Add a `col_vals_within_spec()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L961-L1004)

``` python
col_vals_within_spec(
    columns,
    spec,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_within_spec()`](../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec), which checks whether column values fit a specification such as an email address, URL, or postal code. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_str_len()

Add a `col_vals_str_len()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1006-L1051)

``` python
col_vals_str_len(
    columns,
    min_val=None,
    max_val=None,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_str_len()`](../reference/Validate.col_vals_str_len.md#pointblank.Validate.col_vals_str_len), which checks whether the lengths of string values fall within given bounds. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_scored()

Add a `col_vals_scored()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1053-L1102)

``` python
col_vals_scored(
    columns,
    scorer,
    min_val=None,
    max_val=None,
    na_pass=False,
    cache=True,
    keep_scores=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_scored()`](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored), which checks whether each row's score from a model falls within given bounds. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_chat_format()

Add a `col_vals_chat_format()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1104-L1157)

``` python
col_vals_chat_format(
    columns,
    roles=("system", "user", "assistant"),
    alternate=True,
    first_role=None,
    last_role=None,
    allow_empty=False,
    role_key="role",
    content_key="content",
    na_pass=False,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_chat_format()`](../reference/Validate.col_vals_chat_format.md#pointblank.Validate.col_vals_chat_format), which checks whether each value is a well-formed conversation (a list of messages with roles). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_text_quality()

Add a `col_vals_text_quality()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1159-L1221)

``` python
col_vals_text_quality(
    columns,
    min_words=50,
    max_words=100000,
    min_mean_word_length=3,
    max_mean_word_length=10,
    max_symbol_ratio=0.1,
    max_bullet_lines=0.9,
    max_ellipsis_lines=0.3,
    min_alpha_words=0.8,
    min_stop_words=2,
    max_repeated_lines=0.3,
    stop_words=None,
    na_pass=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_text_quality()`](../reference/Validate.col_vals_text_quality.md#pointblank.Validate.col_vals_text_quality), which checks text with the heuristic quality filters of web text (adding a validation step for each filter when the collection is added to a validation). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_in_language()

Add a `col_vals_in_language()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1223-L1274)

``` python
col_vals_in_language(
    columns,
    expected=None,
    expected_column=None,
    min_confidence=0.5,
    backend="langid",
    model=None,
    na_pass=False,
    cache=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_in_language()`](../reference/Validate.col_vals_in_language.md#pointblank.Validate.col_vals_in_language), which checks whether text is in the expected language. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_token_len()

Add a `col_vals_token_len()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1276-L1323)

``` python
col_vals_token_len(
    columns,
    tokenizer,
    min_val=None,
    max_val=None,
    na_pass=False,
    cache=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_token_len()`](../reference/Validate.col_vals_token_len.md#pointblank.Validate.col_vals_token_len), which checks whether the length of text in tokens is within bounds. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_no_pii()

Add a `col_vals_no_pii()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1325-L1367)

``` python
col_vals_no_pii(
    columns,
    entities=("email", "phone", "credit_card"),
    na_pass=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_no_pii()`](../reference/Validate.col_vals_no_pii.md#pointblank.Validate.col_vals_no_pii), which checks whether text has no personal information (adding a validation step for each kind of information when the collection is added to a validation). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_in_table()

Add a `col_vals_in_table()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2004-L2045)

``` python
col_vals_in_table(
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

The step records a call to [`Validate.col_vals_in_table()`](../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table), which checks whether column values exist in a column of a reference table (referential integrity). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_vals_expr()

Add a `col_vals_expr()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1369-L1406)

``` python
col_vals_expr(
    expr,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_expr()`](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr), which checks column values with a custom expression. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_exists()

Add a `col_exists()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1408-L1440)

``` python
col_exists(
    columns,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_exists()`](../reference/Validate.col_exists.md#pointblank.Validate.col_exists), which checks whether one or more columns exist in the table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_pct_null()

Add a `col_pct_null()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1442-L1478)

``` python
col_pct_null(
    columns,
    p,
    tol=0,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_pct_null()`](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null), which checks whether the percentage of Null values in a column matches an expected value. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_pct_missing()

Add a `col_pct_missing()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1480-L1521)

``` python
col_pct_missing(
    columns,
    missing,
    max_pct,
    reason=None,
    category=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_pct_missing()`](../reference/Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing), which checks whether the percentage of structured missing values in a column stays within a limit. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_missing_coded()

Add a `col_missing_coded()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1523-L1562)

``` python
col_missing_coded(
    columns,
    missing,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_missing_coded()`](../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded), which checks whether all missing values in a column are coded (no uncoded Nulls). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_missing_only_coded()

Add a `col_missing_only_coded()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1564-L1609)

``` python
col_missing_only_coded(
    columns,
    missing,
    allowed=None,
    min_val=None,
    max_val=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_missing_only_coded()`](../reference/Validate.col_missing_only_coded.md#pointblank.Validate.col_missing_only_coded), which checks whether a column contains only documented missing-value codes and legitimate values. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_missing_consistent()

Add a `col_missing_consistent()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1787-L1828)

``` python
col_missing_consistent(
    columns,
    missing,
    when_reason,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_missing_consistent()`](../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent), which checks whether related columns share a consistent pattern of missingness for a given reason. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### rows_distinct()

Add a `rows_distinct()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1611-L1648)

``` python
rows_distinct(
    columns_subset=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct), which checks whether rows are distinct, across all columns or a subset of them. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### rows_near_distinct()

Add a `rows_near_distinct()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1650-L1695)

``` python
rows_near_distinct(
    columns,
    method="minhash",
    threshold=0.85,
    ngram=5,
    num_perm=128,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.rows_near_distinct()`](../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct), which checks that no row's text is a near duplicate of another's. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### rows_not_contaminated()

Add a `rows_not_contaminated()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1697-L1746)

``` python
rows_not_contaminated(
    columns,
    reference,
    reference_columns=None,
    method="ngram",
    ngram=None,
    threshold=0.8,
    num_perm=128,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.rows_not_contaminated()`](../reference/Validate.rows_not_contaminated.md#pointblank.Validate.rows_not_contaminated), which checks that no row's text overlaps with a reference set of texts (such as a benchmark). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### rows_complete()

Add a `rows_complete()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1748-L1785)

``` python
rows_complete(
    columns_subset=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.rows_complete()`](../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete), which checks whether rows are complete, with no missing values across all columns or a subset of them. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_schema_match()

Add a `col_schema_match()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1830-L1875)

``` python
col_schema_match(
    schema,
    complete=True,
    in_order=True,
    case_sensitive_colnames=True,
    case_sensitive_dtypes=True,
    full_match_dtypes=True,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), which checks whether the table's columns and their types match a schema. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### row_count_match()

Add a `row_count_match()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1877-L1916)

``` python
row_count_match(
    count,
    tol=0,
    inverse=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.row_count_match()`](../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match), which checks whether the table's row count matches an expected count or the count of another table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### col_count_match()

Add a `col_count_match()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1965-L2002)

``` python
col_count_match(
    count,
    inverse=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_count_match()`](../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match), which checks whether the table's column count matches an expected count or the count of another table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### data_freshness()

Add a `data_freshness()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1918-L1963)

``` python
data_freshness(
    column,
    max_age,
    reference_time=None,
    timezone=None,
    allow_tz_mismatch=False,
    allow_future=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.data_freshness()`](../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness), which checks whether the latest value in a datetime column is recent enough. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### tbl_match()

Add a `tbl_match()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2047-L2101)

``` python
tbl_match(
    tbl_compare,
    keys=None,
    columns=None,
    column_map=None,
    tolerance=None,
    normalize=None,
    null_equal=True,
    schema="strict",
    dup_keys="flag",
    order_by=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), which checks whether the table matches a comparison table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### tbl_reconcile()

Add a `tbl_reconcile()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2103-L2150)

``` python
tbl_reconcile(
    tbl_compare,
    keys=None,
    measures=None,
    agg="sum",
    tolerance=None,
    column_map=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.tbl_reconcile()`](../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile), which checks whether aggregates of the table reconcile with those of a comparison table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.

### conjointly()

Add a `conjointly()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2152-L2189)

``` python
conjointly(
    *exprs,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.conjointly()`](../reference/Validate.conjointly.md#pointblank.Validate.conjointly), which checks whether several row-wise conditions all hold for each row. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

The expressions are recorded together in the step's `exprs=` argument and passed back as positional arguments when the step is applied.

#### Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
