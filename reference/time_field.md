# time_field()

Create a time column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1689-L1796)

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

Time values are generated uniformly (at second-level resolution) within the specified range. If no range is provided, the default range is 00:00:00 to 23:59:59. Both `min_time=` and `max_time=` accept `datetime.time` objects or ISO format time strings (e.g., `"09:30:00"`).

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
Custom callable that generates values. When provided, this overrides all other constraints. The callable should take no arguments and return a single value.

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

| PolarsRows100Columns2 |                     |                   |
|-----------------------|---------------------|-------------------|
|                       | start_time *String* | end_time *String* |
| 1                     | 10:19:09            | 16:32:49          |
| 2                     | 09:22:48            | 14:19:09          |
| 3                     | 09:04:39            | 16:46:32          |
| 4                     | 11:41:39            | 13:22:48          |
| 5                     | 10:23:45            | 13:04:39          |
| 96                    | 10:53:39            | 15:29:54          |
| 97                    | 10:47:25            | 13:59:36          |
| 98                    | 09:06:22            | 14:10:33          |
| 99                    | 09:44:23            | 13:40:49          |
| 100                   | 10:01:36            | 13:29:15          |

ISO format strings can also be used for convenience:

``` python
schema = pb.Schema(
    login_time=pb.time_field(min_time="06:00:00", max_time="23:59:59"),
    alarm_time=pb.time_field(min_time="05:00:00", max_time="09:00:00"),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=23))
```

| PolarsRows30Columns2 |                     |                     |
|----------------------|---------------------|---------------------|
|                      | login_time *String* | alarm_time *String* |
| 1                    | 22:50:11            | 08:32:49            |
| 2                    | 20:11:17            | 06:19:09            |
| 3                    | 23:16:07            | 08:46:32            |
| 4                    | 11:16:39            | 05:22:48            |
| 5                    | 22:14:46            | 05:04:39            |
| 26                   | 07:55:40            | 06:55:01            |
| 27                   | 07:33:41            | 05:05:20            |
| 28                   | 23:51:48            | 07:19:58            |
| 29                   | 18:55:31            | 07:51:23            |
| 30                   | 15:04:12            | 08:20:29            |

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

| PolarsRows30Columns3 |  |  |  |
|----|----|----|----|
|  | employee_id *Int64* | check_in *String* | check_out *String* |
| 1 | 431 | 08:28:25 | 17:28:25 |
| 2 | 254 | 07:41:11 | 16:41:11 |
| 3 | 504 | 08:47:48 | None |
| 4 | 766 | 09:57:44 | 18:57:44 |
| 5 | 149 | 07:13:11 | None |
| 26 | 679 | 08:00:57 | 18:34:24 |
| 27 | 226 | 09:52:12 | 16:33:48 |
| 28 | 328 | 09:51:19 | 17:00:57 |
| 29 | 745 | 09:39:11 | 18:52:12 |
| 30 | 742 | 07:16:53 | 18:51:19 |
