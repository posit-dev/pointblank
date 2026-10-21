# duration_field()

Create a duration column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1904-L2018)

``` python
duration_field(
    min_duration=None,
    max_duration=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
)
```

The [duration_field()](../reference/duration_field.md#pointblank.duration_field) function defines the constraints and behavior for a duration (timedelta) column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the duration range with `min_duration=` and `max_duration=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`.

Duration values are generated uniformly (at second-level resolution) within the specified range. If no range is provided, the default range is 0 seconds to 30 days. Both `min_duration=` and `max_duration=` accept `datetime.timedelta` objects or colon-separated strings in `"HH:MM:SS"` or `"MM:SS"` format.

## Parameters

`min_duration: str | timedelta | None = None`  
Minimum duration (inclusive). Can be a `"HH:MM:SS"` or `"MM:SS"` string, or a `datetime.timedelta` object. Default is `None` (defaults to 0 seconds).

`max_duration: str | timedelta | None = None`  
Maximum duration (inclusive). Can be a `"HH:MM:SS"` or `"MM:SS"` string, or a `datetime.timedelta` object. Default is `None` (defaults to 30 days).

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. With second-level resolution within a duration range, uniqueness is feasible for moderate dataset sizes.

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, this overrides all other constraints. The callable should take no arguments and return a single `datetime.timedelta` value.

## Returns

`DurationField`  
A duration field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `min_duration` is greater than `max_duration`, or if a duration string cannot be parsed.

## Examples

The `min_duration=` and `max_duration=` parameters accept `timedelta` objects for defining duration ranges:

``` python
import pointblank as pb
from datetime import timedelta

schema = pb.Schema(
    session_length=pb.duration_field(
        min_duration=timedelta(minutes=5),
        max_duration=timedelta(hours=2),
    ),
    wait_time=pb.duration_field(
        min_duration=timedelta(seconds=30),
        max_duration=timedelta(minutes=15),
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns2 |                           |                      |
|-----------------------|---------------------------|----------------------|
|                       | session_length *Duration* | wait_time *Duration* |
| 1                     | 1:51:24                   | 0:13:48              |
| 2                     | 0:44:34                   | 0:05:26              |
| 3                     | 1:58:16                   | 0:14:39              |
| 4                     | 0:16:24                   | 0:01:55              |
| 5                     | 0:07:19                   | 0:00:47              |
| 96                    | 0:34:48                   | 0:04:13              |
| 97                    | 0:40:16                   | 0:04:54              |
| 98                    | 0:25:24                   | 0:03:03              |
| 99                    | 0:19:37                   | 0:02:19              |
| 100                   | 1:29:36                   | 0:11:04              |

Colon-separated strings can also be used for quick duration definitions:

``` python
schema = pb.Schema(
    call_duration=pb.duration_field(min_duration="0:01:00", max_duration="1:30:00"),
    break_time=pb.duration_field(min_duration="0:05:00", max_duration="0:30:00"),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=23))
```

| PolarsRows30Columns2 |                          |                       |
|----------------------|--------------------------|-----------------------|
|                      | call_duration *Duration* | break_time *Duration* |
| 1                    | 0:40:34                  | 0:14:53               |
| 2                    | 0:12:24                  | 0:07:51               |
| 3                    | 0:03:19                  | 0:05:34               |
| 4                    | 1:21:49                  | 0:25:12               |
| 5                    | 0:42:52                  | 0:15:28               |
| 26                   | 0:59:53                  | 0:22:29               |
| 27                   | 0:50:00                  | 0:26:25               |
| 28                   | 0:08:51                  | 0:19:43               |
| 29                   | 0:29:04                  | 0:17:15               |
| 30                   | 0:05:49                  | 0:06:57               |

Optional durations can be created with `nullable=True`, and duration fields work well alongside other field types:

``` python
schema = pb.Schema(
    task_id=pb.int_field(min_val=1, max_val=500, unique=True),
    time_spent=pb.duration_field(
        min_duration=timedelta(minutes=1),
        max_duration=timedelta(hours=8),
    ),
    overtime=pb.duration_field(
        min_duration=timedelta(0),
        max_duration=timedelta(hours=4),
        nullable=True, null_probability=0.6,
    ),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=7))
```

| PolarsRows30Columns3 |  |  |  |
|----|----|----|----|
|  | task_id *Int64* | time_spent *Duration* | overtime *Duration* |
| 1 | 166 | 2:57:51 | None |
| 2 | 486 | 1:23:23 | 0:41:11 |
| 3 | 78 | 3:36:37 | None |
| 4 | 203 | 5:56:29 | 2:57:44 |
| 5 | 334 | 0:27:22 | None |
| 26 | 31 | 5:09:48 | 2:34:24 |
| 27 | 424 | 1:08:36 | None |
| 28 | 290 | 2:02:55 | 1:00:57 |
| 29 | 64 | 5:45:24 | None |
| 30 | 115 | 5:43:39 | 2:51:19 |
