# Validate.data_freshness()

Validate that data in a datetime column is not older than a specified maximum age.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L14597-L14971)

``` python
Validate.data_freshness(
    column,
    max_age,
    reference_time=None,
    timezone=None,
    allow_tz_mismatch=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The `data_freshness()` validation method checks whether the most recent timestamp in the specified datetime column is within the allowed `max_age=` from the `reference_time=` (which defaults to the current time). This is useful for ensuring data pipelines are delivering fresh data and for enforcing data SLAs.

This method helps detect stale data by comparing the maximum (most recent) value in a datetime column against an expected freshness threshold.

## Parameters

`column: str`  
The name of the datetime column to check for freshness. This column should contain date or datetime values.

`max_age: str | datetime.timedelta`  
The maximum allowed age of the data. Can be specified as: (1) a string with a human-readable duration like `"24 hours"`, `"1 day"`, `"30 minutes"`, `"2 weeks"`, etc. (supported units: `seconds`, `minutes`, `hours`, `days`, `weeks`), or (2) a `datetime.timedelta` object for precise control.

`reference_time: datetime.datetime | str | None = None`  
The reference point in time to compare against. Defaults to `None`, which uses the current time (UTC if `timezone=` is not specified). Can be: (1) a `datetime.datetime` object (timezone-aware recommended), (2) a string in ISO 8601 format (e.g., `"2024-01-15T10:30:00"` or `"2024-01-15T10:30:00+05:30"`), or (3) `None` to use the current time.

`timezone: str | None = None`  
The timezone to use for interpreting the data and reference time. Accepts IANA timezone names (e.g., `"America/New_York"`), hour offsets (e.g., `"-7"`), or ISO 8601 offsets (e.g., `"-07:00"`). When `None` (default), naive datetimes are treated as UTC. See the *The `timezone=` Parameter* section for details.

`allow_tz_mismatch: bool = False`  
Whether to allow timezone mismatches between the column data and reference time. By default (`False`), a warning note is added when comparing timezone-naive with timezone-aware datetimes. Set to `True` to suppress these warnings.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

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

## How Timezones Affect Freshness Checks

Freshness validation involves comparing two times: the **data time** (the most recent timestamp in your column) and the **execution time** (when and where the validation runs). Timezone confusion typically arises because these two times may originate from different contexts.

Consider these common scenarios:

- your data timestamps are stored in UTC (common for databases), but you're running validation on your laptop in New York (Eastern Time)
- you develop and test validation locally, then deploy it to a cloud workflow that runs in UTC--suddenly your 'same' validation behaves differently
- your data comes from servers in multiple regions, each recording timestamps in their local timezone

The `timezone=` parameter exists to solve this problem by establishing a single, explicit timezone context for the freshness comparison. When you specify a timezone, Pointblank interprets both the data timestamps (if naive) and the execution time in that timezone, ensuring consistent behavior whether you run validation on your laptop or in a cloud workflow.

**Scenario 1: Data has timezone-aware datetimes**

``` python
# Your data column has values like: 2024-01-15 10:30:00+00:00 (UTC)
# Comparison is straightforward as both sides have explicit timezones
.data_freshness(column="updated_at", max_age="24 hours")
```

**Scenario 2: Data has naive datetimes (no timezone)**

``` python
# Your data column has values like: 2024-01-15 10:30:00 (no timezone)
# Specify the timezone the data was recorded in:
.data_freshness(column="updated_at", max_age="24 hours", timezone="America/New_York")
```

**Scenario 3: Ensuring consistent behavior across environments**

``` python
# Pin the timezone to ensure identical results whether running locally or in the cloud
.data_freshness(
    column="updated_at",
    max_age="24 hours",
    timezone="UTC",  # Explicit timezone removes environment dependence
)
```

## The `timezone=` Parameter

The `timezone=` parameter accepts several convenient formats, making it easy to specify timezones in whatever way is most natural for your use case. The following examples illustrate the three supported input styles.

**IANA Timezone Names** (recommended for regions with daylight saving time):

``` python
timezone="America/New_York"   # Eastern Time (handles DST automatically)
timezone="Europe/London"      # UK time
timezone="Asia/Tokyo"         # Japan Standard Time
timezone="Australia/Sydney"   # Australian Eastern Time
timezone="UTC"                # Coordinated Universal Time
```

**Simple Hour Offsets** (quick and easy):

``` python
timezone="-7"    # UTC-7 (e.g., Mountain Standard Time)
timezone="+5"    # UTC+5 (e.g., Pakistan Standard Time)
timezone="0"     # UTC
timezone="-12"   # UTC-12
```

**ISO 8601 Offset Format** (precise, including fractional hours):

``` python
timezone="-07:00"   # UTC-7
timezone="+05:30"   # UTC+5:30 (e.g., India Standard Time)
timezone="+00:00"   # UTC
timezone="-09:30"   # UTC-9:30
```

When a timezone is specified:

- naive datetime values in the column are assumed to be in this timezone.
- the reference time (if naive) is assumed to be in this timezone.
- the validation report will show times in this timezone.

When `None` (default):

- if your column has timezone-aware datetimes, those timezones are used
- if your column has naive datetimes, they're treated as UTC
- the current time reference uses UTC

Note that IANA timezone names are preferred when daylight saving time transitions matter, as they automatically handle the offset changes. Fixed offsets like `"-7"` or `"-07:00"` do not account for DST.

## Recommendations For Working With Timestamps

When working with datetime data, storing timestamps in UTC in your databases is strongly recommended since it provides a consistent reference point regardless of where your data originates or where it's consumed. Using timezone-aware datetimes whenever possible helps avoid ambiguity--when a datetime has an explicit timezone, there's no guessing about what time it actually represents.

If you're working with naive datetimes (which lack timezone information), always specify the `timezone=` parameter so Pointblank knows how to interpret those values. When providing `reference_time=` as a string, use ISO 8601 format with the timezone offset included (e.g., `"2024-01-15T10:30:00+00:00"`) to ensure unambiguous parsing. Finally, prefer IANA timezone names (like `"America/New_York"`) over fixed offsets (like `"-05:00"`) when daylight saving time transitions matter, since IANA names automatically handle the twice-yearly offset changes. To see all available IANA timezone names in Python, use `zoneinfo.available_timezones()` from the standard library's `zoneinfo` module.

## Examples

The simplest use of `data_freshness()` requires just two arguments: the `column=` containing your timestamps and `max_age=` specifying how old the data can be. In this first example, we create sample data with an `"updated_at"` column containing timestamps from 1, 12, and 20 hours ago. By setting `max_age="24 hours"`, we're asserting that the most recent timestamp should be within 24 hours of the current time. Since the newest record is only 1 hour old, this validation passes.

``` python
import pointblank as pb
import polars as pl
from datetime import datetime, timedelta

# Create sample data with recent timestamps
recent_data = pl.DataFrame({
    "id": [1, 2, 3],
    "updated_at": [
        datetime.now() - timedelta(hours=1),
        datetime.now() - timedelta(hours=12),
        datetime.now() - timedelta(hours=20),
    ]
})

validation = (
    pb.Validate(data=recent_data)
    .data_freshness(column="updated_at", max_age="24 hours")
    .interrogate()
)

validation
```

[TABLE]

The `max_age=` parameter accepts human-readable strings with various time units. You can chain multiple `data_freshness()` calls to check different freshness thresholds simultaneously--useful for tiered SLAs where you might want warnings at 30 minutes but errors at 2 days.

``` python
# Check data is fresh within different time windows
validation = (
    pb.Validate(data=recent_data)
    .data_freshness(column="updated_at", max_age="30 minutes")  # Very fresh
    .data_freshness(column="updated_at", max_age="2 days")      # Reasonably fresh
    .data_freshness(column="updated_at", max_age="1 week")      # Within a week
    .interrogate()
)

validation
```

[TABLE]

When your data contains naive datetimes (timestamps without timezone information), use the `timezone=` parameter to specify what timezone those values represent. Here we have event data recorded in Eastern Time, so we set `timezone="America/New_York"` to ensure the freshness comparison is done correctly.

``` python
# Data with naive datetimes (assume they're in Eastern Time)
eastern_data = pl.DataFrame({
    "event_time": [
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=5),
    ]
})

validation = (
    pb.Validate(data=eastern_data)
    .data_freshness(
        column="event_time",
        max_age="12 hours",
        timezone="America/New_York"  # Interpret times as Eastern
    )
    .interrogate()
)

validation
```

[TABLE]

For reproducible validations or historical checks, you can use `reference_time=` to compare against a specific point in time instead of the current time. This is particularly useful for testing or when validating data snapshots. The reference time should include a timezone offset (like `+00:00` for UTC) to avoid ambiguity.

``` python
validation = (
    pb.Validate(data=recent_data)
    .data_freshness(
        column="updated_at",
        max_age="24 hours",
        reference_time="2024-01-15T12:00:00+00:00"
    )
    .interrogate()
)

validation
```

[TABLE]
