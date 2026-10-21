# int_field()

Create an integer column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L299-L428)

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

The [int_field()](../reference/int_field.md#pointblank.int_field) function defines the constraints and behavior for an integer column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the range of values with `min_val=` and `max_val=`, restrict values to a specific set with `allowed=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`. The `dtype=` parameter lets you choose the specific integer type (e.g., `"Int8"`, `"UInt16"`, `"Int64"`), which also determines the valid range of values.

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
Custom callable that generates values. When provided, this overrides all other constraints (`min_val=`, `max_val=`, `allowed=`, etc.). The callable should take no arguments and return a single integer value.

`dtype: str = "Int64"`  
Integer dtype. Default is `"Int64"`. Options: `"Int8"`, `"Int16"`, `"Int32"`, `"Int64"`, `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"`.

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
| 1                     | 7188536481533917197 | 118         | 3              |
| 2                     | 2674009078779859984 | 99          | 1              |
| 3                     | 7652102777077138151 | 37          | 1              |
| 4                     | 157503859921753049  | 114         | 5              |
| 5                     | 2829213282471975080 | 106         | 3              |
| 96                    | 7027508096731143831 | 36          | 2              |
| 97                    | 6055996548456656575 | 69          | 1              |
| 98                    | 3822709996092631588 | 39          | 2              |
| 99                    | 1522653102058131295 | 114         | 1              |
| 100                   | 5690877051669225499 | 99          | 5              |

It's possible to introduce missing values with `nullable=True` and `null_probability=`, and to select a smaller dtype with `dtype=`:

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
|                      | score *Int64* | optional_val *Int64* |
| 1                    | 148           | 50                   |
| 2                    | 42            | 19                   |
| 3                    | 8             | None                 |
| 4                    | 157           | 2                    |
| 5                    | 216           | None                 |
| 46                   | 218           | 24                   |
| 47                   | 19            | None                 |
| 48                   | 141           | None                 |
| 49                   | 143           | 43                   |
| 50                   | 169           | 21                   |

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
| 1                    | 1533              | 3                |
| 2                    | 8026              | 1                |
| 3                    | 8906              | 2                |
| 4                    | 1243              | 2                |
| 5                    | 4376              | 3                |
| 26                   | 3861              | 2                |
| 27                   | 5966              | 2                |
| 28                   | 6940              | 2                |
| 29                   | 3178              | 3                |
| 30                   | 8486              | 2                |

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
