# datetime_field()

Create a datetime column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1519-L1642)

``` python
datetime_field(
    min_date=None,
    max_date=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
)
```

The [datetime_field()](../reference/datetime_field.md#pointblank.datetime_field) function defines the constraints and behavior for a datetime column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the datetime range with `min_date=` and `max_date=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`.

Datetime values are generated uniformly (at second-level resolution) within the specified range. If no range is provided, the default range is 2000-01-01T00:00:00 to 2030-12-31T23:59:59. Both `min_date=` and `max_date=` accept `datetime` objects, `date` objects (which are converted to datetimes at midnight), or ISO 8601 datetime strings.

## Parameters

`min_date: str | datetime | None = None`  
Minimum datetime (inclusive). Can be an ISO format string (e.g., `"2024-01-01T00:00:00"`), a `datetime.datetime` object, or a `datetime.date` object. Default is `None` (defaults to `2000-01-01 00:00:00`).

`max_date: str | datetime | None = None`  
Maximum datetime (inclusive). Can be an ISO format string, a `datetime.datetime` object, or a `datetime.date` object. Default is `None` (defaults to `2030-12-31 23:59:59`).

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. With second-level resolution over a wide range, collisions are unlikely for moderate dataset sizes.

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of the other constraints, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single `datetime.datetime` value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

## Returns

`DatetimeField`  
A datetime field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `min_date` is later than `max_date`, or if a datetime string cannot be parsed.

## Examples

The `min_date=` and `max_date=` parameters accept `datetime` objects for precise range definitions:

``` python
import pointblank as pb
from datetime import datetime

schema = pb.Schema(
    created_at=pb.datetime_field(
        min_date=datetime(2024, 1, 1),
        max_date=datetime(2024, 12, 31),
    ),
    updated_at=pb.datetime_field(
        min_date=datetime(2024, 6, 1),
        max_date=datetime(2024, 12, 31),
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns2 |                       |                       |
|-----------------------|-----------------------|-----------------------|
|                       | created_at *Datetime* | updated_at *Datetime* |
| 1                     | 2024-02-17 16:33:53   | 2024-10-18 16:58:07   |
| 2                     | 2024-11-14 21:21:34   | 2024-12-27 22:05:31   |
| 3                     | 2024-12-29 14:40:05   | 2024-10-18 17:45:31   |
| 4                     | 2024-12-05 21:08:23   | 2024-12-27 07:09:44   |
| 5                     | 2024-02-14 09:38:39   | 2024-11-07 05:03:11   |
| 96                    | 2024-12-10 00:21:07   | 2024-09-24 01:12:48   |
| 97                    | 2024-11-05 10:23:02   | 2024-07-31 01:21:45   |
| 98                    | 2024-04-07 14:52:22   | 2024-07-21 11:49:28   |
| 99                    | 2024-02-07 05:34:29   | 2024-06-21 02:52:24   |
| 100                   | 2024-04-10 21:16:58   | 2024-11-26 08:05:46   |

For a quick setup, ISO format strings work just as well:

``` python
schema = pb.Schema(
    event_time=pb.datetime_field(
        min_date="2024-03-01T08:00:00",
        max_date="2024-03-01T18:00:00",
    ),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=23))
```

| PolarsRows30Columns1 |                       |
|----------------------|-----------------------|
|                      | event_time *Datetime* |
| 1                    | 2024-03-01 17:43:40   |
| 2                    | 2024-03-01 11:59:13   |
| 3                    | 2024-03-01 09:55:19   |
| 4                    | 2024-03-01 09:19:42   |
| 5                    | 2024-03-01 14:05:41   |
| 26                   | 2024-03-01 11:29:59   |
| 27                   | 2024-03-01 11:46:57   |
| 28                   | 2024-03-01 16:47:38   |
| 29                   | 2024-03-01 14:56:30   |
| 30                   | 2024-03-01 09:21:50   |

Optional timestamps can be simulated with `nullable=True`, and datetime fields work nicely alongside other field types:

``` python
schema = pb.Schema(
    order_id=pb.int_field(min_val=1000, max_val=9999, unique=True),
    placed_at=pb.datetime_field(
        min_date=datetime(2024, 1, 1),
        max_date=datetime(2024, 12, 31),
    ),
    shipped_at=pb.datetime_field(
        min_date=datetime(2024, 1, 2),
        max_date=datetime(2025, 1, 15),
        nullable=True, null_probability=0.3,
    ),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=7))
```

| PolarsRows30Columns3 |  |  |  |
|----|----|----|----|
|  | order_id *Int64* | placed_at *Datetime* | shipped_at *Datetime* |
| 1 | 6166 | 2024-01-16 18:32:08 | 2024-07-21 20:24:26 |
| 2 | 9373 | 2024-11-24 12:56:24 | 2024-02-21 20:29:32 |
| 3 | 7458 | 2024-04-01 12:23:24 | 2024-04-12 09:02:35 |
| 4 | 6794 | 2024-12-02 14:53:31 | 2024-12-21 13:26:10 |
| 5 | 1134 | 2024-11-19 08:33:56 | 2025-01-11 21:54:07 |
| 26 | 7005 | 2024-10-06 18:14:09 | 2024-06-21 02:55:08 |
| 27 | 1190 | 2024-11-24 19:46:42 | 2024-11-18 17:42:24 |
| 28 | 3488 | 2024-10-17 14:06:03 | 2024-12-12 05:32:56 |
| 29 | 7086 | 2024-11-09 12:39:59 | None |
| 30 | 4109 | 2024-11-02 02:08:30 | 2024-08-08 13:14:56 |

About 30% of the `shipped_at` values are missing (13 of 30 here). Since each column is generated independently, some `shipped_at` values come before `placed_at` in the same row, so these columns aren't suited for checks of date ordering without some extra processing.
