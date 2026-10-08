# Field reference

All field types for synthetic data generation.

## Common parameters (all field types)

| Parameter          | Type               | Default  | Description                  |
|--------------------|--------------------|----------|------------------------------|
| `nullable`         | `bool`             | `False`  | Allow null values            |
| `null_probability` | `float`            | `0.0`    | Fraction of nulls (0.0-1.0)  |
| `unique`           | `bool`             | `False`  | All values must be distinct  |
| `generator`        | `Callable \| None` | `None`   | Custom generator function    |

## IntField / int_field()

```python
pb.int_field(
    min_val: int | None = None,
    max_val: int | None = None,
    allowed: list[int] | None = None,
    dtype: str = "Int64",         # Int8/16/32/64, UInt8/16/32/64
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## FloatField / float_field()

```python
pb.float_field(
    min_val: float | None = None,
    max_val: float | None = None,
    allowed: list[float] | None = None,
    precision: int | None = None,   # decimal places
    dtype: str = "Float64",         # Float32 or Float64
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## StringField / string_field()

```python
pb.string_field(
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,     # regex pattern to match
    preset: str | None = None,      # named preset (e.g., "email")
    allowed: list[str] | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

Only one of `preset`, `pattern`, or `allowed` may be set.

## BoolField / bool_field()

```python
pb.bool_field(
    p_true: float = 0.5,           # probability of True
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## DateField / date_field()

```python
pb.date_field(
    min_date: str | date | None = None,    # "2024-01-01" or date()
    max_date: str | date | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## DatetimeField / datetime_field()

```python
pb.datetime_field(
    min_date: str | datetime | None = None,
    max_date: str | datetime | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## TimeField / time_field()

```python
pb.time_field(
    min_time: str | time | None = None,
    max_time: str | time | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## DurationField / duration_field()

```python
pb.duration_field(
    min_duration: str | timedelta | None = None,
    max_duration: str | timedelta | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## profile_fields()

```python
pb.profile_fields(
    set: str = "standard",        # "standard" or "extended"
    split_name: bool = True,      # first_name + last_name vs name
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    prefix: str | None = None,
) -> dict[str, StringField]
```

Returns a dict of StringField objects suitable for `**`-unpacking
into `Schema()`.

Standard set includes: name (or first_name + last_name), email,
phone_number, address, city, country.

Extended set adds: company, job, url, and more.
