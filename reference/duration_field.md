# duration_field()

Create a duration column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1962-L2082)

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
Custom callable that generates values. When provided, it's used instead of the other constraints, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single `datetime.timedelta` value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

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
| 1                     | 1:36:08                   | 0:08:55              |
| 2                     | 1:25:07                   | 0:04:42              |
| 3                     | 0:10:20                   | 0:11:09              |
| 4                     | 0:08:03                   | 0:05:40              |
| 5                     | 1:33:16                   | 0:04:30              |
| 96                    | 0:23:10                   | 0:10:55              |
| 97                    | 1:31:24                   | 0:03:15              |
| 98                    | 1:11:52                   | 0:07:40              |
| 99                    | 1:13:05                   | 0:09:55              |
| 100                   | 1:58:22                   | 0:10:56              |

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
| 1                    | 0:42:46                  | 0:07:00               |
| 2                    | 1:27:53                  | 0:23:55               |
| 3                    | 0:33:12                  | 0:29:15               |
| 4                    | 0:29:35                  | 0:05:33               |
| 5                    | 0:30:59                  | 0:08:32               |
| 26                   | 0:33:01                  | 0:12:12               |
| 27                   | 1:27:59                  | 0:18:13               |
| 28                   | 0:56:46                  | 0:19:53               |
| 29                   | 0:37:15                  | 0:17:08               |
| 30                   | 0:52:53                  | 0:22:33               |

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
| 1 | 126 | 4:11:34 | None |
| 2 | 314 | 5:02:27 | 2:24:24 |
| 3 | 87 | 6:06:30 | None |
| 4 | 40 | 1:51:03 | None |
| 5 | 340 | 4:43:48 | 2:27:00 |
| 26 | 352 | 7:07:54 | None |
| 27 | 252 | 5:24:11 | None |
| 28 | 286 | 1:56:47 | 2:44:22 |
| 29 | 9 | 6:03:07 | 0:27:10 |
| 30 | 30 | 2:18:33 | None |

The `time_spent` values range from one minute to eight hours, while the `overtime` column is missing in most rows (20 of 30 here, in line with the `null_probability=` of `0.6`). Both columns have a duration dtype, so they can take part in arithmetic with dates and datetimes.
