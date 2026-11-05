# int_field()

Create an integer column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L299-L437)

``` python
int_field(
    min_val=None,
    max_val=None,
    allowed=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
    dtype="Int64",
)
```

The [int_field()](../reference/int_field.md#pointblank.int_field) function defines the constraints and behavior for an integer column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the range of values with `min_val=` and `max_val=`, restrict values to a specific set with `allowed=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`. The `dtype=` parameter sets the integer type in the schema (e.g., `"Int8"`, `"UInt16"`, `"Int64"`), which also limits the range of values.

When no constraints are specified, values are drawn uniformly from the full range of the chosen integer dtype. If both `min_val=` and `max_val=` are provided, values are drawn uniformly from that range. If `allowed=` is provided, values are sampled from that specific list.

## Parameters

`min_val: int | None = None`  
Minimum value (inclusive). Default is `None` (no minimum, uses dtype lower bound).

`max_val: int | None = None`  
Maximum value (inclusive). Default is `None` (no maximum, uses dtype upper bound).

`allowed: list[int] | None = None`  
List of allowed values (categorical constraint). When provided, values are sampled from this list. Cannot be combined with `min_val=`/`max_val=`.

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. When `True`, the generator will retry until it produces `n` distinct values (subject to retry limits).

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of `min_val=`, `max_val=` and `allowed=`, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single integer value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

`dtype: str = "Int64"`  
Integer dtype. Default is `"Int64"`. Options: `"Int8"`, `"Int16"`, `"Int32"`, `"Int64"`, `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"`. The dtype limits the range of values (e.g., `0` to `255` for `"UInt8"`) and is the dtype of the column that [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) creates. With `output="pandas"`, the column is a NumPy integer (e.g., `int32`), or a pandas nullable integer (e.g., `Int32`) when `nullable=True`.

## Returns

`IntField`  
An integer field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `min_val` is greater than `max_val`, if `allowed` is an empty list, if `null_probability` is not between `0.0` and `1.0`, or if `dtype` is not a valid integer type.

## Examples

The `min_val=` and `max_val=` parameters constrain generated ranges, while `allowed=` restricts values to a specific set:

``` python
import pointblank as pb

schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    age=pb.int_field(min_val=0, max_val=120),
    rating=pb.int_field(allowed=[1, 2, 3, 4, 5]),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |                     |             |                |
|-----------------------|---------------------|-------------|----------------|
|                       | user_id *Int64*     | age *Int64* | rating *Int64* |
| 1                     | 5702568738186988586 | 24          | 5              |
| 2                     | 2878223714068359465 | 113         | 5              |
| 3                     | 2827316211584448787 | 97          | 1              |
| 4                     | 4882856902780995319 | 84          | 2              |
| 5                     | 5672268219098327547 | 49          | 5              |
| 96                    | 8515820166226230863 | 20          | 1              |
| 97                    | 1273514605515757248 | 100         | 5              |
| 98                    | 302880528361637242  | 98          | 4              |
| 99                    | 7613481963472661909 | 33          | 4              |
| 100                   | 8634311640077946044 | 83          | 1              |

It's possible to introduce missing values with `nullable=True` and `null_probability=`, and to limit the range of values with a smaller `dtype=`:

``` python
schema = pb.Schema(
    score=pb.int_field(min_val=0, max_val=255, dtype="UInt8"),
    optional_val=pb.int_field(
        min_val=1, max_val=50,
        nullable=True, null_probability=0.3,
    ),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns2 |               |                      |
|----------------------|---------------|----------------------|
|                      | score *UInt8* | optional_val *Int64* |
| 1                    | 18            | None                 |
| 2                    | 141           | 39                   |
| 3                    | 135           | 34                   |
| 4                    | 193           | None                 |
| 5                    | 247           | 34                   |
| 46                   | 178           | 18                   |
| 47                   | 35            | 26                   |
| 48                   | 228           | None                 |
| 49                   | 17            | None                 |
| 50                   | 134           | None                 |

We can also enforce uniqueness with `unique=True` to produce distinct identifiers within a range:

``` python
schema = pb.Schema(
    record_id=pb.int_field(min_val=1000, max_val=9999, unique=True),
    priority=pb.int_field(allowed=[1, 2, 3]),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=10))
```

| PolarsRows30Columns2 |                   |                  |
|----------------------|-------------------|------------------|
|                      | record_id *Int64* | priority *Int64* |
| 1                    | 6210              | 1                |
| 2                    | 3668              | 1                |
| 3                    | 6795              | 1                |
| 4                    | 5212              | 1                |
| 5                    | 5567              | 2                |
| 26                   | 7514              | 3                |
| 27                   | 7629              | 2                |
| 28                   | 8762              | 1                |
| 29                   | 7175              | 2                |
| 30                   | 8944              | 1                |

For complete control, a custom `generator=` callable can be provided:

``` python
import random

rng = random.Random(0)

schema = pb.Schema(
    even_numbers=pb.int_field(generator=lambda: rng.choice(range(0, 100, 2))),
)

pb.preview(pb.generate_dataset(schema, n=20, seed=5))
```

| PolarsRows20Columns1 |                      |
|----------------------|----------------------|
|                      | even_numbers *Int64* |
| 1                    | 48                   |
| 2                    | 96                   |
| 3                    | 52                   |
| 4                    | 4                    |
| 5                    | 32                   |
| 16                   | 36                   |
| 17                   | 16                   |
| 18                   | 96                   |
| 19                   | 12                   |
| 20                   | 78                   |

The generator is called once for each row, so every value is an even number below 100. The `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) doesn't reach the generator, which is why it uses its own `random.Random(0)`: that makes the data the same on every run.
