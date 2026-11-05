# time_field()

Create a time column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1738-L1854)

``` python
time_field(
    min_time=None,
    max_time=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
)
```

The [time_field()](../reference/time_field.md#pointblank.time_field) function defines the constraints and behavior for a time-of-day column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the time range with `min_time=` and `max_time=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`.

Time values are generated uniformly (at second-level resolution) within the specified range. If no range is provided, the default range is 00:00:00 to 23:59:59. Both `min_time=` and `max_time=` accept `datetime.time` objects or ISO format time strings (e.g., `"09:30:00"`). The column that [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) creates holds `datetime.time` values: a `Time` column in Polars, and an `object` column in pandas (which has no time dtype).

## Parameters

`min_time: str | time | None = None`  
Minimum time (inclusive). Can be an ISO format string (e.g., `"08:00:00"`) or a `datetime.time` object. Default is `None` (defaults to `00:00:00`).

`max_time: str | time | None = None`  
Maximum time (inclusive). Can be an ISO format string (e.g., `"17:30:00"`) or a `datetime.time` object. Default is `None` (defaults to `23:59:59`).

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. With second-level resolution within a time range, uniqueness is feasible for moderate dataset sizes.

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of the other constraints, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

## Returns

`TimeField`  
A time field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `min_time` is later than `max_time`, or if a time string cannot be parsed.

## Examples

The `min_time=` and `max_time=` parameters accept `datetime.time` objects, making it easy to define business-hours ranges:

``` python
import pointblank as pb
from datetime import time

schema = pb.Schema(
    start_time=pb.time_field(
        min_time=time(9, 0, 0),
        max_time=time(12, 0, 0),
    ),
    end_time=pb.time_field(
        min_time=time(13, 0, 0),
        max_time=time(17, 0, 0),
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns2 |                   |                 |
|-----------------------|-------------------|-----------------|
|                       | start_time *Time* | end_time *Time* |
| 1                     | 09:33:30          | 16:07:53        |
| 2                     | 10:30:01          | 15:38:28        |
| 3                     | 11:52:49          | 16:18:39        |
| 4                     | 10:20:55          | 14:52:47        |
| 5                     | 09:54:28          | 16:04:02        |
| 96                    | 09:57:08          | 16:43:44        |
| 97                    | 11:18:21          | 14:37:24        |
| 98                    | 10:05:24          | 14:13:00        |
| 99                    | 10:48:11          | 14:24:35        |
| 100                   | 10:04:47          | 16:36:16        |

ISO format strings can also be used for convenience:

``` python
schema = pb.Schema(
    login_time=pb.time_field(min_time="06:00:00", max_time="23:59:59"),
    alarm_time=pb.time_field(min_time="05:00:00", max_time="09:00:00"),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=23))
```

| PolarsRows30Columns2 |                   |                   |
|----------------------|-------------------|-------------------|
|                      | login_time *Time* | alarm_time *Time* |
| 1                    | 12:14:22          | 05:40:14          |
| 2                    | 18:51:03          | 08:03:57          |
| 3                    | 23:33:02          | 05:46:05          |
| 4                    | 12:05:09          | 05:52:36          |
| 5                    | 16:46:16          | 08:18:50          |
| 26                   | 14:23:20          | 06:46:04          |
| 27                   | 15:21:40          | 05:40:46          |
| 28                   | 13:20:58          | 05:48:26          |
| 29                   | 13:36:16          | 07:26:45          |
| 30                   | 06:56:22          | 08:16:30          |

It's possible to introduce optional time values with `nullable=True` and combine them with other field types:

``` python
schema = pb.Schema(
    employee_id=pb.int_field(min_val=100, max_val=999, unique=True),
    check_in=pb.time_field(min_time="07:00:00", max_time="10:00:00"),
    check_out=pb.time_field(
        min_time="16:00:00", max_time="20:00:00",
        nullable=True, null_probability=0.15,
    ),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=7))
```

| PolarsRows30Columns3 |                     |                 |                  |
|----------------------|---------------------|-----------------|------------------|
|                      | employee_id *Int64* | check_in *Time* | check_out *Time* |
| 1                    | 214                 | 07:10:21        | 17:38:42         |
| 2                    | 470                 | 09:56:57        | 17:53:38         |
| 3                    | 439                 | 07:23:59        | 19:24:20         |
| 4                    | 732                 | 08:12:40        | 16:47:22         |
| 5                    | 384                 | 08:10:25        | None             |
| 26                   | 589                 | 07:37:17        | 18:27:40         |
| 27                   | 729                 | 07:06:01        | 19:17:30         |
| 28                   | 466                 | 07:14:24        | None             |
| 29                   | 629                 | 07:36:53        | 17:11:09         |
| 30                   | 631                 | 08:14:52        | None             |

Every `check_in` time falls in the morning window and every non-missing `check_out` time in the late afternoon or evening one, with 7 of the 30 `check_out` values missing. The `employee_id` values are distinct three-digit numbers.
