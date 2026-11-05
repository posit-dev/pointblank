# date_field()

Create a date column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1302-L1420)

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
| 1                     | 1983-05-26        | 2023-12-11       |
| 2                     | 1987-05-07        | 2024-02-27       |
| 3                     | 2002-04-20        | 2022-06-14       |
| 4                     | 1978-09-08        | 2023-11-03       |
| 5                     | 1973-03-31        | 2022-03-09       |
| 96                    | 1999-10-20        | 2022-04-15       |
| 97                    | 1982-10-12        | 2023-01-15       |
| 98                    | 1964-05-24        | 2022-07-25       |
| 99                    | 1986-08-27        | 2020-03-05       |
| 100                   | 1963-08-21        | 2022-10-06       |

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
| 1                    | 2024-06-28        | 2024-05-07         |
| 2                    | 2024-08-11        | 2023-07-17         |
| 3                    | 2024-02-06        | 2023-12-16         |
| 4                    | 2024-10-08        | 2024-02-09         |
| 5                    | 2024-12-29        | 2023-10-13         |
| 46                   | 2024-04-25        | 2024-05-06         |
| 47                   | 2024-09-27        | 2023-12-25         |
| 48                   | 2024-05-04        | 2023-09-04         |
| 49                   | 2024-12-23        | 2024-04-27         |
| 50                   | 2024-03-13        | 2023-06-13         |

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
| 1                    | 2024-01-11        | 2024-08-27         |
| 2                    | 2024-02-06        | None               |
| 3                    | 2024-03-15        | None               |
| 4                    | 2024-03-07        | None               |
| 5                    | 2024-01-03        | 2024-08-14         |
| 26                   | 2024-02-26        | 2024-11-09         |
| 27                   | 2024-03-10        | None               |
| 28                   | 2024-01-04        | None               |
| 29                   | 2024-03-13        | 2024-05-10         |
| 30                   | 2024-01-12        | 2024-03-26         |

All 30 `order_date` values are distinct, which is possible because the first quarter of 2024 has 91 days to choose from. About half of the `cancel_date` values are missing (20 of 30 here). Each column is generated independently, so a few cancellation dates fall before the order date in the same row.
