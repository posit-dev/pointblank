# The pytest Fixture

When Pointblank is installed, a [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) **pytest fixture** is automatically available in all your test files. There is no need to import anything or add configuration to `conftest.py`: the fixture is registered via pytest's plugin system.

The fixture works identically to `pb.generate_dataset()`, but with one key difference: when you don't supply a `seed=` parameter, a deterministic seed is automatically derived from the test's fully-qualified name. This means:

- the **same test** always produces the **same data**: no manual seed management required.
- *different tests* get different seeds, so they exercise different datasets.
- you can still pass an explicit `seed=` to override the automatic seed when needed.

## Basic Usage

Use it by adding [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) to your test function's parameter list:

``` python
import pointblank as pb

def test_etl_handles_nulls(generate_dataset):
    schema = pb.Schema(
        user_id=pb.int_field(unique=True),
        email=pb.string_field(preset="email", nullable=True, null_probability=0.3),
        age=pb.int_field(min_val=0, max_val=120),
    )

    df = generate_dataset(schema, n=500)
    result = my_etl_pipeline(df)
    assert result.filter(pl.col("email").is_null()).shape[0] == 0
```

All parameters from [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset) are supported: `n=`, `seed=`, `output=`, and `country=`:

``` python
def test_german_data(generate_dataset):
    schema = pb.Schema(
        name=pb.string_field(preset="name"),
        city=pb.string_field(preset="city"),
    )

    df = generate_dataset(schema, n=200, country="DE", output="pandas")
    assert len(df) == 200
```

## Multiple Datasets in One Test

Calling the fixture multiple times within the same test produces different (but still deterministic) data on each call:

``` python
def test_merge_pipeline(generate_dataset):
    customers = generate_dataset(customer_schema, n=1000, country="US")
    orders = generate_dataset(order_schema, n=5000)

    # Each call gets a unique seed derived from the test name + call index,
    # so both DataFrames are deterministic and different from each other.
    result = merge_pipeline(customers, orders)
    assert result.shape[0] > 0
```

## Testing Across Locales

The fixture makes locale testing particularly concise when combined with `pytest.mark.parametrize`:

``` python
import pytest
import pointblank as pb

@pytest.mark.parametrize("country", ["US", "DE", "JP", "BR"])
def test_name_normalizer(generate_dataset, country):
    schema = pb.Schema(name=pb.string_field(preset="name_full"))
    df = generate_dataset(schema, n=100, country=country)
    result = normalize_names(df)
    assert result["name"].str.len_chars().min() > 0
```

## Sharing Schemas Across Tests

Define schemas as fixtures in `conftest.py` and compose them with [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset):

``` python
import pytest
import pointblank as pb

@pytest.fixture
def customer_schema():
    return pb.Schema(
        id=pb.int_field(unique=True),
        name=pb.string_field(preset="name"),
        email=pb.string_field(preset="email"),
        city=pb.string_field(preset="city"),
    )
```

``` python
def test_customer_validation(generate_dataset, customer_schema):
    df = generate_dataset(customer_schema, n=200, country="DE")
    validation = pb.Validate(df).col_vals_not_null(columns="email").interrogate()
    assert validation.all_passed()
```

``` python
def test_customer_export(generate_dataset, customer_schema):
    df = generate_dataset(customer_schema, n=50, country="JP")
    exported = export_to_parquet(df)
    assert exported.exists()
```

## Debugging with Seed Introspection

The fixture callable exposes two attributes that make debugging failed tests straightforward:

- `generate_dataset.default_seed`: the base seed derived from the test name (available before any call)
- `generate_dataset.last_seed`: the seed actually used for the most recent call (accounts for the call counter and explicit overrides)

Include `.last_seed` in assertion messages so failures are immediately reproducible:

``` python
def test_age_range(generate_dataset):
    schema = pb.Schema(age=pb.int_field(min_val=18, max_val=100))
    df = generate_dataset(schema, n=500)
    min_age = df["age"].min()
    assert min_age >= 18, (
        f"Expected min age >= 18, got {min_age} (seed={generate_dataset.last_seed})"
    )
```

You can also use `.default_seed` to reproduce the exact dataset outside of pytest:

``` python
# In a REPL or notebook, reproduce the data from a failed test:
import pointblank as pb
df = pb.generate_dataset(schema, n=500, seed=<default_seed_from_output>)
```

## Seed Stability

A given seed (whether explicit or auto-derived) is guaranteed to produce identical output **within the same Pointblank version**. Across versions, changes to country data files or generator logic may alter the output for a given seed.

For CI pipelines that require bit-exact data across library upgrades, we recommend saving generated DataFrames as Parquet or CSV snapshot files rather than relying on cross-version seed stability. This is the same approach used by snapshot-testing tools like `pytest-snapshot` and `syrupy`.
