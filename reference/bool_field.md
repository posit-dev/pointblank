# bool_field()

Create a boolean column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L1094-L1199)

``` python
bool_field(
    p_true=0.5,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
)
```

The [bool_field()](../reference/bool_field.md#pointblank.bool_field) function defines the constraints and behavior for a boolean column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). The `p_true=` parameter controls the probability of generating `True` values, which is useful for simulating real-world distributions where events may be rare or common (e.g., 5% fraud rate, 80% active users).

By default, `True` and `False` are equally likely (`p_true=0.5`). Setting `p_true=0.0` produces all `False` values, and `p_true=1.0` produces all `True` values.

## Parameters

`p_true: float = 0.5`  
Probability of generating `True`. Default is `0.5` (equal probability). Must be between `0.0` and `1.0`.

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. Note that boolean columns can only have 2 unique non-null values, so `n` must be `<= 2` when `unique=True` (or `<= 3` with `nullable=True`).

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of the other constraints, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single boolean value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

## Returns

`BoolField`  
A boolean field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `p_true` is not between `0.0` and `1.0`, or if `null_probability` is not between `0.0` and `1.0`.

## Examples

The `p_true=` parameter controls the distribution of `True`/`False` values, allowing you to simulate different probabilities:

``` python
import pointblank as pb

schema = pb.Schema(
    is_active=pb.bool_field(p_true=0.8),
    is_premium=pb.bool_field(p_true=0.2),
    is_verified=pb.bool_field(),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | is_active *Boolean* | is_premium *Boolean* | is_verified *Boolean* |
| 1 | False | False | False |
| 2 | False | False | False |
| 3 | False | False | False |
| 4 | True | True | True |
| 5 | True | False | False |
| 96 | True | False | True |
| 97 | True | False | True |
| 98 | False | False | False |
| 99 | False | False | False |
| 100 | False | False | False |

Optional boolean flags can be simulated by combining `nullable=True` with `null_probability=`:

``` python
schema = pb.Schema(
    opted_in=pb.bool_field(p_true=0.6),
    has_referral=pb.bool_field(
        p_true=0.3,
        nullable=True, null_probability=0.25,
    ),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns2 |                    |                        |
|----------------------|--------------------|------------------------|
|                      | opted_in *Boolean* | has_referral *Boolean* |
| 1                    | False              | False                  |
| 2                    | False              | False                  |
| 3                    | False              | None                   |
| 4                    | True               | True                   |
| 5                    | True               | None                   |
| 46                   | True               | True                   |
| 47                   | True               | None                   |
| 48                   | False              | None                   |
| 49                   | True               | False                  |
| 50                   | True               | False                  |

Boolean fields can be combined with other field types in a realistic schema:

``` python
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, max_val=9999, unique=True),
    name=pb.string_field(preset="name"),
    email_verified=pb.bool_field(p_true=0.9),
    is_admin=pb.bool_field(p_true=0.05),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=10))
```

| PolarsRows30Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | user_id *Int64* | name *String* | email_verified *Boolean* | is_admin *Boolean* |
| 1 | 9362 | Joshua Robinson | True | False |
| 2 | 534 | Marie Young | True | False |
| 3 | 7027 | Melissa Atkins | True | False |
| 4 | 7907 | Paisley Jackson | True | False |
| 5 | 9472 | Martha White | True | False |
| 26 | 4647 | Zachary Ramirez | False | False |
| 27 | 4298 | Sarah Gallagher | True | False |
| 28 | 7485 | Kimberly Tran | True | True |
| 29 | 2862 | John Marshall | True | True |
| 30 | 4967 | Linda Garner | True | False |

Here, `email_verified` is `True` in 26 of the 30 rows and `is_admin` in 3 of them. These counts are close to, but not exactly, the `p_true=` values of `0.9` and `0.05`, since each value is drawn independently.
