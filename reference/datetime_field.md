# datetime_field()

Create a datetime column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1476-L1593)

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
Custom callable that generates values. When provided, this overrides all other constraints. The callable should take no arguments and return a single `datetime.datetime` value.

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
| 1                     | 2024-12-25 04:22:08   | 2024-09-21 14:13:08   |
| 2                     | 2024-10-29 16:22:23   | 2024-07-03 10:44:55   |
| 3                     | 2024-04-22 14:13:08   | 2024-06-07 15:09:55   |
| 4                     | 2024-12-12 14:04:53   | 2024-09-28 03:03:37   |
| 5                     | 2024-11-18 04:49:47   | 2024-11-12 13:36:43   |
| 96                    | 2024-07-29 13:15:44   | 2024-11-28 23:02:29   |
| 97                    | 2024-04-28 08:49:29   | 2024-12-30 22:17:11   |
| 98                    | 2024-12-13 09:42:37   | 2024-08-16 08:51:48   |
| 99                    | 2024-10-28 23:35:39   | 2024-11-26 06:03:26   |
| 100                   | 2024-06-25 14:22:27   | 2024-09-23 19:35:59   |

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
| 1                    | 2024-03-01 13:16:39   |
| 2                    | 2024-03-01 09:31:15   |
| 3                    | 2024-03-01 08:18:39   |
| 4                    | 2024-03-01 13:35:02   |
| 5                    | 2024-03-01 15:42:50   |
| 26                   | 2024-03-01 11:44:35   |
| 27                   | 2024-03-01 08:38:34   |
| 28                   | 2024-03-01 14:43:09   |
| 29                   | 2024-03-01 11:07:47   |
| 30                   | 2024-03-01 11:55:58   |

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
| 1 | 6305 | 2024-05-05 18:20:24 | None |
| 2 | 3471 | 2024-02-28 14:00:58 | 2025-01-04 02:44:43 |
| 3 | 7468 | 2024-06-02 08:01:18 | None |
| 4 | 1791 | 2024-09-09 19:08:56 | 2024-06-03 08:01:18 |
| 5 | 2186 | 2024-01-19 18:03:43 | None |
| 26 | 4622 | 2024-11-17 02:49:13 | 2024-01-24 22:56:59 |
| 27 | 1763 | 2024-08-07 14:24:37 | 2024-11-18 02:49:13 |
| 28 | 3181 | 2024-02-18 01:55:04 | 2024-08-08 14:24:37 |
| 29 | 5744 | 2024-03-27 16:44:16 | 2024-02-19 01:55:04 |
| 30 | 7867 | 2024-09-01 21:51:34 | 2025-01-03 22:52:16 |
