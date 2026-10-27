# Validate.col_vals_within_spec()

Validate whether column values fit within a specification.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L10867-L11165)

``` python
Validate.col_vals_within_spec(
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

The [col_vals_within_spec()](../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec) validation method checks whether column values in a table correspond to a specification (`spec=`) type (details of which are available in the *Specifications* section). Specifications include common data types like email addresses, URLs, postal codes, vehicle identification numbers (VINs), International Bank Account Numbers (IBANs), and more. This validation will operate over the number of test units that is equal to the number of rows in the table.

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`spec: str`  
A specification string for defining the specification type. Examples are `"email"`, `"url"`, and `"postal_code[USA]"`. See the *Specifications* section for all available options.

`na_pass: bool = False`  
Should any encountered None, NA, or Null values be considered as passing test units? By default, this is `False`. Set to `True` to pass test units with missing values.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list). Read the *Segmentation* section for usage information.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step(s) meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like [`has_columns()`](../reference/has_columns.md#pointblank.has_columns) and [`has_rows()`](../reference/has_rows.md#pointblank.has_rows) can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Specifications

A specification type must be used with the `spec=` argument. This is a string-based keyword that corresponds to the type of data in the specified columns. The following keywords can be used:

- `"isbn"`: The International Standard Book Number (ISBN) is a unique numerical identifier for books. This keyword validates both 10-digit and 13-digit ISBNs.

- `"vin"`: A vehicle identification number (VIN) is a unique code used by the automotive industry to identify individual motor vehicles.

- `"postal_code[<country_code>]"`: A postal code (also known as postcodes, PIN, or ZIP codes) is a series of letters, digits, or both included in a postal address. Because the coding varies by country, a country code in either the 2-letter (ISO 3166-1 alpha-2) or 3-letter (ISO 3166-1 alpha-3) format needs to be supplied (e.g., `"postal_code[US]"` or `"postal_code[USA]"`). The keyword alias `"zip"` can be used for US ZIP codes.

- `"credit_card"`: A credit card number can be validated across a variety of issuers. The validation uses the Luhn algorithm.

- `"iban[<country_code>]"`: The International Bank Account Number (IBAN) is a system of identifying bank accounts across countries. Because the length and coding varies by country, a country code needs to be supplied (e.g., `"iban[DE]"` or `"iban[DEU]"`).

- `"swift"`: Business Identifier Codes (also known as SWIFT-BIC, BIC, or SWIFT code) are unique identifiers for financial and non-financial institutions.

- `"phone"`, `"email"`, `"url"`, `"ipv4"`, `"ipv6"`, `"mac"`: Phone numbers, email addresses, Internet URLs, IPv4 or IPv6 addresses, and MAC addresses can be validated with their respective keywords.

Only a single `spec=` value should be provided per function call.

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that you can refer to a column via `columns=` that is expected to be present in the transformed table, but may not exist in the table before preprocessing. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](../reference/Validate.md#pointblank.Validate) object or used in subsequent validation steps.

## Segmentation

The `segments=` argument allows for the segmentation of a validation step into multiple segments. This is useful for applying the same validation step to different subsets of the data. The segmentation can be done based on a single column or specific fields within a column.

Providing a single column name will result in a separate validation step for each unique value in that column. For example, if you have a column called `"region"` with values `"North"`, `"South"`, and `"East"`, the validation step will be applied separately to each region.

Alternatively, you can provide a tuple that specifies a column name and its corresponding values to segment on. For example, if you have a column called `"date"` and you want to segment on only specific dates, you can provide a tuple like `("date", ["2023-01-01", "2023-01-02"])`. Any other values in the column will be disregarded (i.e., no validation steps will be created for them).

A list with a combination of column names and tuples can be provided as well. This allows for more complex segmentation scenarios. The following inputs are both valid:

    # Segments from all unique values in the `region` column
    # and specific dates in the `date` column
    segments=["region", ("date", ["2023-01-01", "2023-01-02"])]

    # Segments from all unique values in the `region` and `date` columns
    segments=["region", "date"]

The segmentation is performed during interrogation, and the resulting validation steps will be numbered sequentially. Each segment will have its own validation step, and the results will be reported separately. This allows for a more granular analysis of the data and helps identify issues within specific segments.

Importantly, the segmentation process will be performed after any preprocessing of the data table. Because of this, one can conceivably use the `pre=` argument to generate a column that can be used for segmentation. For example, you could create a new column called `"segment"` through use of `pre=` and then use that column for segmentation.

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

For the examples here, we'll use a simple Polars DataFrame with an email column. The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "email": [
            "user@example.com",
            "admin@test.org",
            "invalid-email",
            "contact@company.co.uk",
        ],
    }
)

pb.preview(tbl)
```

|     | email *String*        |
|-----|-----------------------|
| 1   | user@example.com      |
| 2   | admin@test.org        |
| 3   | invalid-email         |
| 4   | contact@company.co.uk |

Let's validate that all of the values in the `email` column are valid email addresses. We'll determine if this validation had any failing test units (there are four test units, one for each row).

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_within_spec(columns="email", spec="email")
    .interrogate()
)

validation
```

[TABLE]

The validation table shows that one test unit failed (the invalid email address in row 3).
