# Validate.col_vals_no_pii()

Validate whether text has no personal information.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L16450-L16606)

``` python
Validate.col_vals_no_pii(
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

Text collected for training or evaluating a model shouldn't have personal information in it, such as email addresses, phone numbers, or credit card numbers. The `col_vals_no_pii()` validation method adds a [col_vals_regex()](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step (with `inverse=True`) for each column and each kind of information: a row fails if its text has something that looks like one. Each row is a test unit. The patterns use only the features of regular expressions that every backend has, so the checks run in the database (or in PySpark) like any other [col_vals_regex()](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step.

The patterns find likely matches, not certain ones: a number with the shape of a credit card number matches, with no checksum, and an unusual format of a phone number can be missed. For a thorough check, use a model for named entities (e.g., with [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored)).

The entities are:

- `"email"`: email addresses
- `"phone"`: phone numbers of North America, with or without a country code (like `(555) 123-4567` or `+1 555.123.4567`)
- `"credit_card"`: credit card numbers (Visa, Mastercard, American Express, and Discover), with or without spaces or dashes
- `"ip_address"`: IPv4 addresses
- `"us_ssn"`: US Social Security numbers (like `123-45-6789`)
- `"iban"`: International Bank Account Numbers

## Parameters

`columns: str | list[str]`  
A single column or a list of columns to validate. There will be a separate set of validation steps for each column (one for each entity).

`entities: Sequence[str] | Mapping[str, str | None] = ("email", "phone", "credit_card")`  
The kinds of information to look for (by default, `"email"`, `"phone"`, and `"credit_card"`), or a dictionary of patterns of your own, by name (which can be mixed with the names above, as in `{"email": None, "employee_id": r"EMP-[0-9]{6}"}`, where `None` is the pattern of an entity above).

`na_pass: bool = True`  
Should a missing value pass? The default is `True`, as a missing text has no personal information.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
The brief of each validation step, displayed in the reporting table. By default (`None` or `True`), each step's brief says what it checks (e.g., "No email addresses in `text`."). A string is used for every step, with `"{auto}"` replaced by that description (and `"{step}"` by the step number). Use `False` for no brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation steps should be active. Using `False` will make the validation steps inactive (still reporting their presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing.

`dimension: str | None = None`  
An optional data quality dimension to categorize the validation steps for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation steps.

## Raises

`TypeError`  
If `entities=` isn't a list of entities or a dictionary of patterns, or a pattern isn't a string.

`ValueError`  
If `entities=` is empty or has an entity that isn't one of those above.

## Examples

Let's check a table of support tickets for personal information.

``` python
import pointblank as pb
import polars as pl

tickets = pl.DataFrame(
    {
        "text": [
            "The app crashes when I open the settings page.",
            "Please email me at jo.smith@example.com when it's fixed.",
            "Call me back on (555) 123-4567 after five.",
            "I was charged twice on card 4111 1111 1111 1111.",
        ]
    }
)

pb.Validate(data=tickets).col_vals_no_pii(columns="text").interrogate()
```

[TABLE]

Each of the last three tickets fails one of the steps: the email address in step 1, the phone number in step 2, and the card number in step 3.
