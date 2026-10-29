# date_field()

Create a date column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1297-L1415)

``` python
date_field(
    min_date=None,
    max_date=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
)
```

The [date_field()](../reference/date_field.md#pointblank.date_field) function defines the constraints and behavior for a date column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the date range with `min_date=` and `max_date=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`.

Dates are generated uniformly within the specified range. If no range is provided, the default range is 2000-01-01 to 2030-12-31. Both `min_date=` and `max_date=` accept either `datetime.date` objects or ISO 8601 date strings (e.g., `"2024-06-15"`).

## Parameters

`min_date: str | date | None = None`  
Minimum date (inclusive). Can be an ISO format string (e.g., `"2020-01-01"`) or a `datetime.date` object. Default is `None` (defaults to `2000-01-01`).

`max_date: str | date | None = None`  
Maximum date (inclusive). Can be an ISO format string (e.g., `"2024-12-31"`) or a `datetime.date` object. Default is `None` (defaults to `2030-12-31`).

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. When `True`, the generator will retry until it produces `n` distinct dates. Ensure the date range is large enough to accommodate the requested number of unique dates.

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of the other constraints, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single `datetime.date` value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

## Returns

`DateField`  
A date field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `min_date` is later than `max_date`, or if a date string cannot be parsed.

## Examples

The `min_date=` and `max_date=` parameters accept `datetime.date` objects to define date ranges:

``` python
import pointblank as pb
from datetime import date

schema = pb.Schema(
    birth_date=pb.date_field(
        min_date=date(1960, 1, 1),
        max_date=date(2005, 12, 31),
    ),
    hire_date=pb.date_field(
        min_date=date(2020, 1, 1),
        max_date=date(2024, 12, 31),
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns2 |                   |                  |
|-----------------------|-------------------|------------------|
|                       | birth_date *Date* | hire_date *Date* |
| 1                     | 1986-01-03        | 2024-05-15       |
| 2                     | 1967-06-30        | 2021-08-16       |
| 3                     | 1961-07-13        | 2024-08-26       |
| 4                     | 1987-07-09        | 2020-06-20       |
| 5                     | 1998-01-06        | 2020-02-04       |
| 96                    | 1969-04-14        | 2023-01-29       |
| 97                    | 1975-03-23        | 2021-03-23       |
| 98                    | 1981-05-29        | 2021-06-13       |
| 99                    | 1982-09-14        | 2020-11-02       |
| 100                   | 1968-12-21        | 2020-08-07       |

For convenience, ISO format strings can be used instead of `date` objects:

``` python
schema = pb.Schema(
    event_date=pb.date_field(min_date="2024-01-01", max_date="2024-12-31"),
    signup_date=pb.date_field(min_date="2023-06-01", max_date="2024-06-01"),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns2 |                   |                    |
|----------------------|-------------------|--------------------|
|                      | event_date *Date* | signup_date *Date* |
| 1                    | 2024-05-28        | 2023-10-27         |
| 2                    | 2024-02-12        | 2023-07-13         |
| 3                    | 2024-01-09        | 2023-06-09         |
| 4                    | 2024-10-30        | 2024-03-30         |
| 5                    | 2024-06-06        | 2023-11-05         |
| 46                   | 2024-02-13        | 2023-07-14         |
| 47                   | 2024-08-30        | 2024-01-29         |
| 48                   | 2024-03-31        | 2023-08-30         |
| 49                   | 2024-09-04        | 2024-02-03         |
| 50                   | 2024-10-29        | 2024-03-29         |

We can introduce missing dates with `nullable=True` and enforce distinct values using `unique=True`:

``` python
schema = pb.Schema(
    order_date=pb.date_field(
        min_date="2024-01-01", max_date="2024-03-31",
        unique=True,
    ),
    cancel_date=pb.date_field(
        min_date="2024-01-01", max_date="2024-12-31",
        nullable=True, null_probability=0.5,
    ),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=7))
```

| PolarsRows30Columns2 |                   |                    |
|----------------------|-------------------|--------------------|
|                      | order_date *Date* | cancel_date *Date* |
| 1                    | 2024-02-11        | None               |
| 2                    | 2024-01-20        | 2024-03-18         |
| 3                    | 2024-02-20        | None               |
| 4                    | 2024-03-24        | 2024-11-29         |
| 5                    | 2024-01-07        | None               |
| 26                   | 2024-03-14        | 2024-04-24         |
| 27                   | 2024-01-06        | None               |
| 28                   | 2024-03-12        | 2024-11-17         |
| 29                   | 2024-01-18        | None               |
| 30                   | 2024-02-07        | 2024-02-01         |

All 30 `order_date` values are distinct, which is possible because the first quarter of 2024 has 91 days to choose from. About half of the `cancel_date` values are missing (20 of 30 here). Each column is generated independently, so a few cancellation dates fall before the order date in the same row.
