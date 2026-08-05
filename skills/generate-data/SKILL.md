---
name: generate-data
description: >
  Generate synthetic datasets with Pointblank using Schema and field
  classes. Covers IntField, FloatField, StringField, BoolField,
  DateField, DatetimeField, and more. Supports presets (name, email,
  address, etc.), country-specific data, nullable columns, unique
  constraints, and profile_fields() for person data. Use when creating
  test data, fixtures, or synthetic datasets for validation testing.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-generation
    - synthetic-data
    - test-data
    - schema
    - faker
---

# Generate Data

Skill for creating synthetic datasets from schema definitions and
field specifications. Useful for testing validation rules, creating
fixtures, generating demo data, and prototyping pipelines.

## Quick start

```python
import pointblank as pb

schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=10000, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=95),
    score=pb.float_field(min_val=0.0, max_val=100.0, precision=2),
    active=pb.bool_field(p_true=0.8),
)

df = schema.generate(n=1000, seed=42)
```

## Skill directory structure

```
skills/generate-data/
+-- SKILL.md                    <- This file
+-- references/
    +-- field-reference.md      <- All field types and parameters
    +-- presets-reference.md    <- Available string presets
```

## When to use what

| I want to...                        | Use                                   |
| ----------------------------------- | ------------------------------------- |
| Generate a dataset from a schema    | `schema.generate()`                   |
| Generate without creating a Schema  | `pb.generate_dataset()`               |
| Define integer columns              | `int_field()`                         |
| Define float columns                | `float_field()`                       |
| Define string columns with patterns | `string_field(pattern=)`              |
| Define string columns with presets  | `string_field(preset=)`               |
| Define boolean columns              | `bool_field()`                        |
| Define date columns                 | `date_field()`                        |
| Define datetime columns             | `datetime_field()`                    |
| Define time columns                 | `time_field()`                        |
| Define duration columns             | `duration_field()`                    |
| Add person profile fields           | `profile_fields()`                    |
| Generate country-specific data      | `generate(country="DE")`              |
| Make columns nullable               | `nullable=True, null_probability=0.1` |
| Ensure unique values                | `unique=True`                         |
| Use a custom generator function     | `generator=my_func`                   |

## Core concepts

### Schema-based generation

Define columns using field classes, then generate:

```python
schema = pb.Schema(
    order_id=pb.int_field(min_val=1, max_val=99999, unique=True),
    product=pb.string_field(allowed=["Widget A", "Widget B", "Gadget"]),
    quantity=pb.int_field(min_val=1, max_val=100),
    price=pb.float_field(min_val=0.99, max_val=999.99, precision=2),
    shipped=pb.bool_field(p_true=0.7),
    order_date=pb.date_field(min_date="2024-01-01", max_date="2024-12-31"),
)

df = schema.generate(n=500, seed=42, output="polars")
```

### generate() parameters

| Parameter  | Default    | Description                             |
| ---------- | ---------- | --------------------------------------- |
| `n`        | `100`      | Number of rows to generate              |
| `seed`     | `None`     | Random seed for reproducibility         |
| `output`   | `"polars"` | Output format: `"polars"` or `"pandas"` |
| `country`  | `"US"`     | Country code for locale-aware data      |
| `shuffle`  | `True`     | Shuffle rows after generation           |
| `weighted` | `True`     | Use weighted distributions              |

### generate_dataset() convenience function

```python
df = pb.generate_dataset(schema, n=500, seed=42)
```

### Nullable columns

Any field type supports nulls:

```python
pb.int_field(min_val=0, max_val=100, nullable=True, null_probability=0.1)
pb.string_field(preset="email", nullable=True, null_probability=0.05)
```

### Unique constraints

Ensure all generated values are distinct:

```python
pb.int_field(min_val=1, max_val=10000, unique=True)
pb.string_field(preset="email", unique=True)
```

### Allowed values (categorical)

Restrict to a specific set of values:

```python
pb.int_field(allowed=[1, 2, 3, 5, 8, 13])
pb.float_field(allowed=[0.5, 1.0, 1.5, 2.0])
pb.string_field(allowed=["low", "medium", "high"])
```

### String patterns

Generate strings matching a pattern:

```python
pb.string_field(pattern=r"[A-Z]{3}-\d{4}")    # "ABC-1234"
pb.string_field(pattern=r"INV-\d{6}")          # "INV-003847"
pb.string_field(pattern=r"[a-z]{5,10}")        # random lowercase
```

### String presets

Use built-in presets for realistic data:

```python
pb.string_field(preset="name")           # full names
pb.string_field(preset="email")          # email addresses
pb.string_field(preset="address")        # street addresses
pb.string_field(preset="city")           # city names
pb.string_field(preset="phone_number")   # phone numbers
pb.string_field(preset="company")        # company names
pb.string_field(preset="job")            # job titles
pb.string_field(preset="url")            # URLs
pb.string_field(preset="uuid4")          # UUIDs
pb.string_field(preset="iban")           # IBANs
pb.string_field(preset="ssn")            # SSNs
```

Presets produce country-specific data when `country` is set.

### Profile fields

Generate person-related fields as a group:

```python
fields = pb.profile_fields(
    set="standard",        # "standard" or "extended"
    split_name=True,       # first_name + last_name vs full name
    include=None,          # specific fields to include
    exclude=None,          # specific fields to exclude
    prefix=None,           # prefix for field names
)

schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=99999, unique=True),
    **fields,
)

df = schema.generate(n=100, country="US")
```

### Custom generators

Supply your own generator function:

```python
import random

def custom_sku():
    return f"SKU-{random.randint(1000, 9999)}"

schema = pb.Schema(
    sku=pb.string_field(generator=custom_sku),
)
```

### Country-specific generation

Over 100 countries supported:

```python
# German names, addresses, phone numbers
df = schema.generate(n=100, country="DE")

# Japanese
df = schema.generate(n=100, country="JP")

# Brazilian
df = schema.generate(n=100, country="BR")
```

## Workflows

### Creating test data for validation rules

1. Define the schema matching your production table.
2. Use field constraints to generate realistic ranges.
3. Add some nullable columns to test null handling.
4. Generate the dataset.
5. Run your validation plan against it.

```python
schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=10000, unique=True),
    amount=pb.float_field(min_val=-10, max_val=1000, precision=2),
    status=pb.string_field(allowed=["active", "inactive", "INVALID"]),
    email=pb.string_field(preset="email", nullable=True, null_probability=0.1),
)

test_df = schema.generate(n=500, seed=42)

validation = (
    pb.Validate(data=test_df)
    .col_vals_gt(columns="amount", value=0)
    .col_vals_in_set(columns="status", set=["active", "inactive"])
    .col_vals_not_null(columns="email")
    .interrogate()
)
```

### Generating fixtures from an existing table

```python
# Infer schema from real data
schema = pb.Schema.from_table(
    production_df,
    infer_constraints=True,
    categorical_threshold=20,
)

# Generate synthetic version
fixture = schema.generate(n=100, seed=1)
```

## Gotchas

1. **`unique=True` needs a large enough range.** If `max_val - min_val`
   < `n`, generation will fail for integer fields.
2. **Only one of `preset`, `pattern`, `allowed` per StringField.**
   They are mutually exclusive.
3. **Presets require `faker` to be installed.** Install with
   `pip install pointblank[faker]` or `pip install faker`.
4. **`seed` makes generation reproducible** but the same seed with
   different `n` produces different data (not a prefix of larger).
5. **`output` only supports `"polars"` and `"pandas"`.** For other
   formats, convert after generation.
6. **`null_probability=0` with `nullable=True`** generates no nulls.
   Set the probability to get actual null values.
