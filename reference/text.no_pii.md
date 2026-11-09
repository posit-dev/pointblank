# text.no_pii()

Make validation steps that check that text has no personal information.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/text.py#L531-L669)

``` python
text.no_pii(
    columns,
    entities=DEFAULT_PII,
    na_pass=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    active=True,
)
```

Text collected for training or evaluating a model shouldn't have personal information in it, such as email addresses, phone numbers, or credit card numbers. This function makes a [col_vals_regex()](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step (with `inverse=True`) for each column and each kind of information: a row fails if its text has something that looks like one. The patterns use only the features of regular expressions that every backend has, so the checks run in the database (or in PySpark) like any other [col_vals_regex()](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step.

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
The column with the text, or a list of them.

`entities: Sequence[str] | Mapping[str, str] = DEFAULT_PII`  
The kinds of information to look for (by default, `"email"`, `"phone"`, and `"credit_card"`), or a dictionary of patterns of your own, by name (which can be mixed with the names above, as in `{"email": None, "employee_id": r"EMP-[0-9]{6}"}`, where `None` is the pattern of an entity above).

`na_pass: bool = True`  
Whether a row with a missing value passes (it does by default, as a missing text has no personal information).

`pre: Callable | None = None`  
A function that transforms the table before each step, as in the validation methods.

`segments: SegmentSpec | None = None`  
The segments of the steps, as in the validation methods.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
The thresholds of the steps (by default, those of the validation).

`actions: Actions | None = None`  
The actions of the steps.

`active: bool | Callable = True`  
Whether the steps are active.

## Returns

`Steps`  
The steps, to be added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

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

pb.Validate(data=tickets).add_steps(pb.text.no_pii("text")).interrogate()
```

[TABLE]

Each of the last three tickets fails one of the steps: the email address in step 1, the phone number in step 2, and the card number in step 3.
