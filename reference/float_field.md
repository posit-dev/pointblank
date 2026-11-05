# float_field()

Create a floating-point column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L539-L692)

``` python
float_field(
    min_val=None,
    max_val=None,
    allowed=None,
    precision=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
    dtype="Float64",
)
```

The [float_field()](../reference/float_field.md#pointblank.float_field) function defines the constraints and behavior for a floating-point column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the range of values with `min_val=` and `max_val=`, restrict values to a specific set with `allowed=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`. The `dtype=` parameter sets the float type in the schema (`"Float32"` or `"Float64"`).

When both `min_val=` and `max_val=` are provided, values are drawn from a uniform distribution across that range. If neither is specified, values are drawn uniformly from a large default range. If `allowed=` is provided, values are sampled from that specific list.

## Parameters

`min_val: float | None = None`  
Minimum value (inclusive). Default is `None` (no minimum).

`max_val: float | None = None`  
Maximum value (inclusive). Default is `None` (no maximum).

`allowed: list[float] | None = None`  
List of allowed values (categorical constraint). When provided, values are sampled from this list. Cannot be combined with `min_val=`/`max_val=`.

`precision: int | None = None`  
Number of decimal places to round generated values to. Default is `None` (no rounding). Must be a non-negative integer. Has no effect when `allowed=` or `generator=` is used.

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. When `True`, the generator will retry until it produces `n` distinct values.

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of `min_val=`, `max_val=`, `allowed=` and `precision=`, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single float value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

`dtype: str = "Float64"`  
Float dtype. Default is `"Float64"`. Options: `"Float32"`, `"Float64"`. It's the dtype of the column that [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) creates.

## Returns

`FloatField`  
A float field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If `min_val` is greater than `max_val`, if `allowed` is an empty list, if `null_probability` is not between `0.0` and `1.0`, if `precision` is negative, or if `dtype` is not a valid float type.

## Examples

The `min_val=` and `max_val=` parameters define the generated value ranges:

``` python
import pointblank as pb

schema = pb.Schema(
    price=pb.float_field(min_val=0.01, max_val=9999.99),
    probability=pb.float_field(min_val=0.0, max_val=1.0),
    temperature=pb.float_field(min_val=-40.0, max_val=50.0),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | price *Float64* | probability *Float64* | temperature *Float64* |
| 1 | 5938.004549741373 | 0.8799156974355079 | -5.157722860993047 |
| 2 | 7605.283132107856 | 0.14435757239986746 | 31.296623312850457 |
| 3 | 1728.6007069298096 | 0.7470355821054965 | 39.77800136715432 |
| 4 | 8941.257026606254 | 0.7226387800366582 | -12.968123980080339 |
| 5 | 6057.495421314872 | 0.017779160149829498 | -2.867472818974349 |
| 96 | 8056.4659030644625 | 0.6697716258081057 | -23.090891067448077 |
| 97 | 363.53767483767353 | 0.8729074363705923 | 25.794353646053793 |
| 98 | 8536.596555606327 | 0.10074496550443779 | 5.683426744810603 |
| 99 | 375.10627637757153 | 0.0802829629623063 | 42.06519501931237 |
| 100 | 3828.1379737802536 | 0.825305531548672 | 16.95315561462961 |

It's also possible to restrict values to a discrete set with `allowed=`, which is useful for fixed pricing tiers or measurement levels:

``` python
schema = pb.Schema(
    discount=pb.float_field(allowed=[0.05, 0.10, 0.15, 0.20, 0.25]),
    weight_kg=pb.float_field(min_val=0.5, max_val=100.0),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns2 |                    |                     |
|----------------------|--------------------|---------------------|
|                      | discount *Float64* | weight_kg *Float64* |
| 1                    | 0.25               | 73.22386971863972   |
| 2                    | 0.25               | 78.89100603758705   |
| 3                    | 0.15               | 66.18540144593624   |
| 4                    | 0.2                | 85.28332950401631   |
| 5                    | 0.1                | 9.437798842836877   |
| 46                   | 0.25               | 54.94434699269292   |
| 47                   | 0.05               | 84.30281543783005   |
| 48                   | 0.25               | 83.05195258036858   |
| 49                   | 0.15               | 5.997813792849301   |
| 50                   | 0.1                | 76.69053821491201   |

We can simulate missing measurements by introducing null values:

``` python
schema = pb.Schema(
    reading=pb.float_field(
        min_val=0.0, max_val=500.0,
        nullable=True, null_probability=0.2,
    ),
    calibration=pb.float_field(min_val=0.9, max_val=1.1),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=23))
```

| PolarsRows30Columns2 |                    |                       |
|----------------------|--------------------|-----------------------|
|                      | reading *Float64*  | calibration *Float64* |
| 1                    | 42.7404746224444   | 0.9839786717043394    |
| 2                    | 143.7033480997308  | 1.0137963699650125    |
| 3                    | 141.68548764039124 | 0.9571151553193716    |
| 4                    | 320.87527875793967 | 0.9443715373140421    |
| 5                    | 51.2887560208391   | 0.9744119101525103    |
| 26                   | None               | 0.9649956220992375    |
| 27                   | 122.9112380710341  | 1.0448877279482989    |
| 28                   | 178.03479442978855 | 1.0261184992342618    |
| 29                   | 118.05390826822676 | 1.0785441477257058    |
| 30                   | None               | 0.937538519570002     |

Use `precision=` to round generated values to a fixed number of decimal places. This is useful for prices, scores, or any measurement where full floating-point precision is unwanted:

``` python
schema = pb.Schema(
    price=pb.float_field(min_val=1.0, max_val=200.0, precision=2),
    score=pb.float_field(min_val=0.0, max_val=100.0, precision=1),
    probability=pb.float_field(min_val=0.0, max_val=1.0, precision=4),
)

pb.preview(pb.generate_dataset(schema, n=20, seed=23))
```

| PolarsRows20Columns3 |                 |                 |                       |
|----------------------|-----------------|-----------------|-----------------------|
|                      | price *Float64* | score *Float64* | probability *Float64* |
| 1                    | 119.17          | 62.3            | 0.8799                |
| 2                    | 152.35          | 27.7            | 0.1444                |
| 3                    | 35.4            | 55.9            | 0.747                 |
| 4                    | 178.93          | 48.4            | 0.7226                |
| 5                    | 121.54          | 45.2            | 0.0178                |
| 16                   | 1.49            | 95.3            | 0.3973                |
| 17                   | 190.14          | 92.6            | 0.4119                |
| 18                   | 59.56           | 2.7             | 0.3355                |
| 19                   | 169.02          | 45.2            | 0.9446                |
| 20                   | 195.32          | 86.9            | 0.5298                |

A custom `generator=` provides full control over value generation, such as values from a distribution other than the uniform one:

``` python
import random, math

rng = random.Random(0)

schema = pb.Schema(
    sensor_value=pb.float_field(min_val=-10.0, max_val=10.0),
    log_value=pb.float_field(generator=lambda: math.log(rng.uniform(1, 1000))),
)

pb.preview(pb.generate_dataset(schema, n=20, seed=23))
```

| PolarsRows20Columns2 |                        |                     |
|----------------------|------------------------|---------------------|
|                      | sensor_value *Float64* | log_value *Float64* |
| 1                    | -0.08615076131600574   | 6.738836419047254   |
| 2                    | -2.9847483771133065    | 6.630942519000257   |
| 3                    | 2.8777641630224373     | 6.042991461173114   |
| 4                    | 5.0084456481351936     | 5.559364739458459   |
| 5                    | 3.4974316227831608     | 6.237862500009073   |
| 16                   | 4.957328889320658      | 5.526471683068103   |
| 17                   | 2.146448046995115      | 6.813264923713322   |
| 18                   | -2.9744841993093996    | 6.890408378292458   |
| 19                   | 2.3890556663818057     | 6.697536613756536   |
| 20                   | 8.41986500151721       | 6.804906921310479   |

The `log_value` column holds the logarithm of a uniform value between 1 and 1000, so its values lie between 0 and about 6.9 and are bunched toward the top of that range. The generator has its own seeded `random.Random(0)`, since the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) doesn't reach it.
