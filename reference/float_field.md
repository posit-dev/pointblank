# float_field()

Create a floating-point column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L530-L674)

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

The [float_field()](../reference/float_field.md#pointblank.float_field) function defines the constraints and behavior for a floating-point column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). You can control the range of values with `min_val=` and `max_val=`, restrict values to a specific set with `allowed=`, enforce uniqueness with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`. The `dtype=` parameter lets you choose between `"Float32"` and `"Float64"` precision.

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
Custom callable that generates values. When provided, this overrides all other constraints. The callable should take no arguments and return a single float value.

`dtype: str = "Float64"`  
Float dtype. Default is `"Float64"`. Options: `"Float32"`, `"Float64"`.

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
| 1 | 9248.64401895442 | 0.9248652516259452 | 43.23787264633508 |
| 2 | 9486.04880781621 | 0.9486057779931771 | 45.37452001938594 |
| 3 | 8924.325591818912 | 0.8924333440485793 | 40.31900096437214 |
| 4 | 835.5150972932996 | 0.08355067683068362 | -32.48043908523847 |
| 5 | 5920.270428312815 | 0.5920272268857353 | 13.282450419716177 |
| 96 | 4446.926385790886 | 0.4446925279641446 | 0.022327516773010814 |
| 97 | 3427.7653590611476 | 0.3427762214585577 | -9.150140068729808 |
| 98 | 8923.280842563525 | 0.8923288689140904 | 40.309598202268134 |
| 99 | 8137.5531808932155 | 0.8137559456012128 | 33.238035104109144 |
| 100 | 8951.80870117522 | 0.8951816604808429 | 40.56634944327587 |

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
| 1                    | 0.15               | 92.52409253678155   |
| 2                    | 0.05               | 94.88627491032112   |
| 3                    | 0.05               | 89.29711773283364   |
| 4                    | 0.25               | 8.813292344653021   |
| 5                    | 0.15               | 59.406709075130664  |
| 46                   | 0.25               | 27.918663919265157  |
| 47                   | 0.2                | 57.49577854139957   |
| 48                   | 0.1                | 82.15598649681618   |
| 49                   | 0.1                | 33.41508237533323   |
| 50                   | 0.1                | 37.28056623460687   |

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
| 1                    | 462.4326258129726  | 1.084973050325189     |
| 2                    | 474.3028889965886  | 1.0897211555986355    |
| 3                    | None               | 1.0784866688097159    |
| 4                    | 41.775338415341814 | 0.9167101353661368    |
| 5                    | None               | 1.0184054453771472    |
| 26                   | None               | 0.9725585036956217    |
| 27                   | 379.8933061217655  | 1.0519573224487062    |
| 28                   | 184.3419073841034  | 0.9737367629536414    |
| 29                   | 286.0233905958904  | 1.0144093562383563    |
| 30                   | 330.26605462515903 | 1.0321064218500637    |

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
| 1                    | 185.05          | 92.5            | 0.9249                |
| 2                    | 189.77          | 94.9            | 0.9486                |
| 3                    | 178.59          | 89.2            | 0.8924                |
| 4                    | 17.63           | 8.4             | 0.0836                |
| 5                    | 118.81          | 59.2            | 0.592                 |
| 16                   | 84.83           | 42.1            | 0.4212                |
| 17                   | 103.01          | 51.3            | 0.5126                |
| 18                   | 147.11          | 73.4            | 0.7342                |
| 19                   | 72.44           | 35.9            | 0.359                 |
| 20                   | 12.45           | 5.8             | 0.0576                |

Setting `dtype="Float32"` gives reduced precision, and a custom `generator=` provides full control over value generation:

``` python
import random, math

rng = random.Random(0)

schema = pb.Schema(
    sensor_value=pb.float_field(min_val=-10.0, max_val=10.0, dtype="Float32"),
    log_value=pb.float_field(generator=lambda: math.log(rng.uniform(1, 1000))),
)

pb.preview(pb.generate_dataset(schema, n=20, seed=23))
```

| PolarsRows20Columns2 |                        |                     |
|----------------------|------------------------|---------------------|
|                      | sensor_value *Float64* | log_value *Float64* |
| 1                    | 8.497305032518906      | 6.738836419047254   |
| 2                    | 8.972115559863543      | 6.630942519000257   |
| 3                    | 7.848666880971585      | 6.042991461173114   |
| 4                    | -8.328986463386327     | 5.559364739458459   |
| 5                    | 1.840544537714706      | 6.237862500009073   |
| 16                   | -1.5750327677161184    | 5.526471683068103   |
| 17                   | 0.25267848784712044    | 6.813264923713322   |
| 18                   | 4.684181575669911      | 6.890408378292458   |
| 19                   | -2.8200728042760055    | 6.697536613756536   |
| 20                   | -8.848909407015631     | 6.804906921310479   |
